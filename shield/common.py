"""Shared pieces: device pick, strength table, quality metrics, EOT transforms, ffmpeg helpers. Torch + stdlib only."""
import json
import math
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# --- device: picked once -----------------------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    torch.set_flush_denormal(True)  # denormal floats make CPU convolutions crawl
ON_GPU = DEVICE.type == "cuda"
FAST = DEVICE.type != "cpu"  # GPU-sized budgets vs CPU-sized budgets


def autocast():
    """fp16 forward/backward on CUDA only. δ and the loss stay fp32 (callers cast back with .float())."""
    return torch.autocast("cuda", dtype=torch.float16, enabled=ON_GPU)


class ShieldError(Exception):
    """A failure the user should read as-is (plain language)."""


# --- strength table ----------------------------------------------------------------------------------------------
# eps: L∞ bound. Images: fraction of the 0..1 pixel range. Audio: fraction of the clip's peak amplitude.
# steps = (GPU, CPU). CPU budgets are sized so a photo finishes in ~10-25 s and a 20 s voice note in < 30 s on 4 cores.
STRENGTHS = {
    "light":    dict(img_eps=7 / 255,  img_steps=(60, 30),  aud_eps=0.008, aud_steps=(100, 30), vid_eps=6 / 255,  vid_steps=(20, 4)),
    "balanced": dict(img_eps=9 / 255,  img_steps=(120, 30), aud_eps=0.012, aud_steps=(160, 36), vid_eps=9 / 255,  vid_steps=(30, 6)),
    "strong":   dict(img_eps=13 / 255, img_steps=(200, 40), aud_eps=0.024, aud_steps=(240, 40), vid_eps=12 / 255, vid_steps=(40, 9)),
}
EOT_SAMPLES = 4 if FAST else 2  # random transforms per PGD step, run as one batch


def strength(name: str) -> dict:
    if name not in STRENGTHS:
        raise ShieldError("Unknown strength. Use light, balanced or strong.")
    s = STRENGTHS[name]
    pick = 0 if FAST else 1
    return {k: (v[pick] if isinstance(v, tuple) else v) for k, v in s.items()}


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


# --- quality metrics ---------------------------------------------------------------------------------------------
def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """a, b in 0..1. 10·log10(1 / MSE)."""
    mse = (a.float() - b.float()).pow(2).mean().item()
    return 99.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)


def _gauss1d(size: int, sigma: float, device) -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-x.pow(2) / (2 * sigma * sigma))
    return g / g.sum()


def gaussian_blur(x: torch.Tensor, sigma: float, size: int = 5) -> torch.Tensor:
    """Separable gaussian on [N,C,H,W] via two grouped conv2d."""
    c = x.shape[1]
    g = _gauss1d(size, sigma, x.device).to(x.dtype)
    x = F.conv2d(F.pad(x, (size // 2, size // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, size // 2, size // 2), mode="replicate"), g.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean SSIM (Wang et al. 2004): 11×11 gaussian window σ=1.5, on luma, inputs [N,3,H,W] or [3,H,W] in 0..1."""
    if a.dim() == 3:
        a, b = a[None], b[None]
    w = torch.tensor([0.299, 0.587, 0.114], device=a.device).view(1, 3, 1, 1)
    a, b = (a.float() * w).sum(1, keepdim=True), (b.float() * w).sum(1, keepdim=True)
    g = _gauss1d(11, 1.5, a.device)
    win = (g[:, None] * g[None, :])[None, None]
    mu_a, mu_b = F.conv2d(a, win), F.conv2d(b, win)
    var_a = F.conv2d(a * a, win) - mu_a * mu_a
    var_b = F.conv2d(b * b, win) - mu_b * mu_b
    cov = F.conv2d(a * b, win) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2))
    return s.mean().item()


# --- EOT transforms for images / video frames (all differentiable, all batched) -----------------------------------
_RGB2YCC = torch.tensor([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])


def yuv420_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """What a video codec / JPEG does to colour: RGB→YCbCr, average chroma 2×2, upsample, back to RGB."""
    m = _RGB2YCC.to(x.device, x.dtype)
    ycc = torch.einsum("ij,njhw->nihw", m, x)
    y, c = ycc[:, :1], ycc[:, 1:]
    h, w = c.shape[-2:]
    c = F.interpolate(F.avg_pool2d(c, 2, ceil_mode=True), size=(h, w), mode="bilinear", align_corners=False)
    return torch.einsum("ij,njhw->nihw", torch.linalg.inv(m.float()).to(x.dtype), torch.cat([y, c], 1))


def eot_image(x: torch.Tensor, scale: float) -> torch.Tensor:
    """One random draw of 'what happens to a picture after you post it', applied to the whole batch.
    resize by `scale` (antialiased when shrinking) → maybe light blur → maybe chroma subsampling → tiny noise."""
    if abs(scale - 1) > 1e-3:
        h, w = x.shape[-2:]
        size = (max(16, round(h * scale)), max(16, round(w * scale)))
        x = F.interpolate(x, size=size, mode="area") if scale < 1 else F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    if torch.rand(()) < 0.5:
        x = gaussian_blur(x, sigma=0.4 + 0.5 * torch.rand(()).item())
    if torch.rand(()) < 0.6:
        x = yuv420_roundtrip(x)
    return x + torch.randn_like(x) * (1.0 / 255)


# --- ffmpeg helpers ----------------------------------------------------------------------------------------------
def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, **kw)
    if p.returncode != 0:
        tail = p.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["unknown error"]
        raise ShieldError(f"Could not read that file ({tail[0][:160]}).")
    return p


def ffprobe(path) -> dict:
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]).stdout
    return json.loads(out)


def notify(on_progress, **kw) -> None:
    if on_progress:
        on_progress(kw)
