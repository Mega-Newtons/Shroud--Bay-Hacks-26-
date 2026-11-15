"""Photo shield: PGD in full-resolution pixel space against a face-identity ensemble (ArcFace w600k_r50 + FaceNet).

    python -m shield.image in.jpg out.png --strength balanced [--decoy other_face.jpg]
    python -m shield.image --calibrate
"""
import argparse
import io
import os
import time
import warnings as _warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .common import (DEVICE, EOT_SAMPLES, FAST, MODELS, ROOT, ShieldError, autocast, cos, eot_image, gaussian_blur, notify, psnr, ssim)
from .common import strength as strength_cfg

os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))  # facenet-pytorch reads its weights from here (no download)

# Cosine thresholds for "same person", per encoder. See calibrate() for the measured distributions behind them.
THRESHOLDS = {"arcface": 0.30, "facenet": 0.50}
WEIGHTS = {"arcface": 1.0, "facenet": 0.5}     # ArcFace is what open-source face swappers read identity from
SIZES = {"arcface": 112, "facenet": 160}
TARGET = {"arcface": 0.05, "facenet": 0.20}    # early-stop once the EOT-averaged similarity is below these
MAX_SIDE = 1600 if FAST else 1280              # photos are downscaled to this long side before shielding
MAX_FACES = 5
# Standard ArcFace 112×112 five-point template (eyes, nose, mouth corners).
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], np.float64)
_M: dict = {}


def load_models() -> dict:
    if _M:
        return _M
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        import onnx2torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
        arc_path = MODELS / "w600k_r50.onnx"
        if not arc_path.exists():
            raise ShieldError("Face models are missing. Run scripts/setup.sh first.")
        _M["mtcnn"] = MTCNN(keep_all=True, device=DEVICE, min_face_size=24)
        _M["arcface"] = onnx2torch.convert(str(arc_path)).to(DEVICE).eval().requires_grad_(False)
        _M["facenet"] = InceptionResnetV1(pretrained="vggface2").to(DEVICE).eval().requires_grad_(False)
    return _M


# --- geometry ----------------------------------------------------------------------------------------------------
def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama 1991): 2×3 M with dst ≈ s·R·src + t."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src))        # cross-covariance
    d = np.array([1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])   # no reflections
    R = U @ np.diag(d) @ Vt
    scale = (S * d).sum() / (s0 ** 2).sum(1).mean()
    return np.concatenate([scale * R, (mu_d - scale * R @ mu_s)[:, None]], 1)


def arc_matrix(lm: np.ndarray) -> np.ndarray:
    """Image pixels → ArcFace 112×112 crop."""
    return umeyama(lm, TEMPLATE)


def box_matrix(box: np.ndarray, size: int = 160) -> np.ndarray:
    """Image pixels → FaceNet crop: the detector box stretched to size×size (what facenet-pytorch's extract_face does)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    sx, sy = size / max(x2 - x1, 1.0), size / max(y2 - y1, 1.0)
    return np.array([[sx, 0, (0.5 - x1) * sx - 0.5], [0, sy, (0.5 - y1) * sy - 0.5]])


def warp(img: torch.Tensor, M: torch.Tensor, size: int) -> torch.Tensor:
    """Differentiable crop. img [N,3,H,W]; M [N,2,3] maps source pixels → output pixels; returns [N,3,size,size].
    affine_grid wants the opposite direction in normalised coords, so:  θ = N_src · M⁻¹ · N_out⁻¹
    with pixel i ↔ normalised (2i+1)/W − 1 (align_corners=False)."""
    n, _, h, w = img.shape
    M = M.to(torch.float64)
    A = torch.cat([M, torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64, device=M.device).expand(n, 1, 3)], 1)
    out_inv = torch.tensor([[size / 2, 0, (size - 1) / 2], [0, size / 2, (size - 1) / 2], [0, 0, 1]], dtype=torch.float64, device=M.device)
    src = torch.tensor([[2 / w, 0, 1 / w - 1], [0, 2 / h, 1 / h - 1], [0, 0, 1]], dtype=torch.float64, device=M.device)
    theta = (src @ torch.linalg.inv(A) @ out_inv)[:, :2].to(img.dtype)
    grid = F.affine_grid(theta, (n, 3, size, size), align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def rescale_matrix(M: torch.Tensor, f: float) -> torch.Tensor:
    """M for an image that was resized by factor f (pixel centres: x' = (x+0.5)·f − 0.5)."""
    A, t = M[:, :, :2], M[:, :, 2]
    return torch.cat([A / f, (t + A.sum(-1) * (0.5 / f - 0.5))[:, :, None]], 2)


def shift_matrix(M: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """M for a crop of the image whose top-left corner is (x0, y0)."""
    M = M.copy()
    M[:, 2] += M[:, :2] @ np.array([x0, y0])
    return M


def jitter(n: int, size: int, device) -> torch.Tensor:
    """Random small similarity in output space [n,3,3]: ±2° rotation, ±3 % scale, ±1.5 % shift.
    Stands in for a different detector putting the landmarks in slightly different places."""
    ang = (torch.rand(n, device=device, dtype=torch.float64) - 0.5) * 2 * 0.035
    s = 1 + (torch.rand(n, device=device, dtype=torch.float64) - 0.5) * 0.06
    t = (torch.rand(n, 2, device=device, dtype=torch.float64) - 0.5) * 2 * 0.015 * size
    c = (size - 1) / 2
    J = torch.zeros(n, 3, 3, device=device, dtype=torch.float64)
    J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1], J[:, 2, 2] = s * ang.cos(), -s * ang.sin(), s * ang.sin(), s * ang.cos(), 1
    J[:, 0, 2] = c - J[:, 0, 0] * c - J[:, 0, 1] * c + t[:, 0]
    J[:, 1, 2] = c - J[:, 1, 0] * c - J[:, 1, 1] * c + t[:, 1]
    return J


# --- detection + embeddings (no grad) ---------------------------------------------------------------------------------
def detect(x: torch.Tensor, max_side: int = 640) -> list[list[dict]]:
    """x [B,3,H,W] in 0..1 → per image, faces sorted largest first: {"box": [4], "lm": [5,2]} in full-res pixels.
    MTCNN runs on a ≤640 px copy (fast); coordinates are scaled back."""
    m = load_models()
    if x.dim() == 3:
        x = x[None]
    h, w = x.shape[-2:]
    f = min(1.0, max_side / max(h, w))
    with torch.inference_mode():
        small = F.interpolate(x, scale_factor=f, mode="bilinear", antialias=True, align_corners=False) if f < 1 else x
        fh, fw = h / small.shape[-2], w / small.shape[-1]
        boxes, probs, points = m["mtcnn"].detect((small * 255).permute(0, 2, 3, 1), landmarks=True)
    out = []
    for b, p, l in zip(boxes, probs, points):
        faces = []
        if b is not None:
            for bb, pp, ll in zip(b, p, l):
                if pp is None or pp < 0.92:
                    continue
                # pixel-centre correct scale-back: x_full = (x_small + 0.5)·f − 0.5
                box = np.array([bb[0] * fw, bb[1] * fh, bb[2] * fw, bb[3] * fh], np.float64)
                lm = (np.asarray(ll, np.float64) + 0.5) * np.array([fw, fh]) - 0.5
                faces.append({"box": box, "lm": lm})
        faces.sort(key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
        out.append(faces)
    return out


def _norm(name: str, crop: torch.Tensor) -> torch.Tensor:
    # ArcFace (InsightFace): RGB, (px−127.5)/127.5.  FaceNet: (px−127.5)/128.
    return (crop - 0.5) / 0.5 if name == "arcface" else (crop * 255 - 127.5) / 128.0


def encode(name: str, crop: torch.Tensor) -> torch.Tensor:
    with autocast():
        e = load_models()[name](_norm(name, crop))
    return e.float()


def face_matrices(face: dict) -> dict:
    return {"arcface": arc_matrix(face["lm"]), "facenet": box_matrix(face["box"])}


def embed_faces(x: torch.Tensor, faces: list[dict]) -> dict:
    """Reference/verification embeddings for faces of ONE image x [1,3,H,W] → {"arcface": [n,512], "facenet": [n,512]}.
    ArcFace: bilinear warp straight from full res (what InsightFace's cv2.warpAffine does).
    FaceNet: antialiased shrink first (facenet-pytorch resizes with area interpolation)."""
    out = {}
    with torch.inference_mode():
        for name, size in SIZES.items():
            crops = []
            for fc in faces:
                M = torch.from_numpy(face_matrices(fc)[name])[None].to(DEVICE)
                img, sc = x, float(np.sqrt(abs(np.linalg.det(M[0, :, :2].cpu().numpy()))))
                if name == "facenet" and sc < 0.75:
                    img = F.interpolate(x, scale_factor=sc, mode="bilinear", antialias=True, align_corners=False)
                    M = rescale_matrix(M, img.shape[-1] / x.shape[-1])
                crops.append(warp(img, M, size))
            out[name] = encode(name, torch.cat(crops)).clone()
    return out


# --- the attack ------------------------------------------------------------------------------------------------------
def face_roi(face: dict, h: int, w: int, pad: float = 0.18) -> tuple[int, int, int, int]:
    """Integer box (x0,y0,x1,y1) covering everything either encoder looks at, plus padding."""
    Minv = np.linalg.inv(np.vstack([arc_matrix(face["lm"]), [0, 0, 1]]))
    quad = (Minv @ np.array([[0, 0, 1], [112, 0, 1], [0, 112, 1], [112, 112, 1]], np.float64).T).T[:, :2]
    lo = np.minimum(quad.min(0), face["box"][:2])
    hi = np.maximum(quad.max(0), face["box"][2:])
    p = pad * (hi - lo)
    x0, y0 = np.floor(np.maximum(lo - p, 0)).astype(int)
    x1, y1 = np.ceil(np.minimum(hi + p, [w, h])).astype(int)
    return int(x0), int(y0), int(x1), int(y1)


def soft_mask(h: int, w: int, device, feather: float = 0.10) -> torch.Tensor:
    """[1,1,h,w] mask that is 1 inside and eases to 0 over the outer `feather` of each side (no visible seam)."""
    def ramp(n):
        t = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / max(1.0, feather * n)
        t = torch.minimum(t, t.flip(0)).clamp(0, 1)
        return t * t * (3 - 2 * t)
    return (ramp(h)[:, None] * ramp(w)[None, :])[None, None]


TEX_FLOOR = float(os.environ.get("SHROUD_TEX_FLOOR", 0.3))   # share of ε allowed on perfectly smooth areas
LUMA_ONLY = os.environ.get("SHROUD_LUMA", "0") == "1"         # experiment: one grey δ instead of three colour channels


def texture_bound(roi: torch.Tensor, floor: float = TEX_FLOOR) -> torch.Tensor:
    """Perceptual masking, [N,1,h,w] in TEX_FLOOR..1. The eye forgives changes in busy areas (hair, eyes, beard,
    edges) and spots them on smooth skin, so the allowed ε is scaled by local contrast: std of luma in a 7×7 window."""
    luma = (roi * roi.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    mean = F.avg_pool2d(luma, 7, 1, 3, count_include_pad=False)
    std = (F.avg_pool2d(luma * luma, 7, 1, 3, count_include_pad=False) - mean * mean).clamp_min(0).sqrt()
    tex = gaussian_blur(F.max_pool2d((std / 0.05).clamp(0, 1), 5, 1, 2), 2.0, 9)
    return floor + (1 - floor) * tex


def pgd_rois(roi: torch.Tensor, mats: dict, e0: dict, eps: float, steps: int, mask: torch.Tensor, decoy: dict | None = None,
             delta_init: torch.Tensor | None = None, group: torch.Tensor | None = None, per_step: int | None = None,
             on_step=None, k: int = EOT_SAMPLES, min_steps: int = 2, tex_floor: float = TEX_FLOOR) -> tuple[torch.Tensor, dict]:
    """Core PGD. roi [N,3,h,w] face regions (0..1); mats[name] [N,2,3] roi pixels → encoder crop; e0[name] [N,512].
    group [N] maps each roi to the δ it shares (default: one δ per roi; video on CPU shares one δ per few frames).
    per_step: use a random subset of this many rois per step (None = all). Returns (δ [n_delta,3,h,w], last sims)."""
    n = roi.shape[0]
    group = torch.arange(n, device=roi.device) if group is None else group
    n_delta = int(group.max()) + 1
    # per-δ perceptual bound (mean over the rois sharing that δ); the pixel change is δ·tex with |δ| ≤ ε
    tex = torch.zeros(n_delta, 1, *roi.shape[2:], device=roi.device).index_add_(0, group, texture_bound(roi, tex_floor))
    tex = tex / torch.bincount(group, minlength=n_delta).view(-1, 1, 1, 1).clamp_min(1)
    ch = 1 if LUMA_ONLY else 3
    if delta_init is None:   # random start: cos(f(x), e0) is at its maximum at δ=0, so the gradient there is ~0
        delta = (torch.rand(n_delta, ch, *roi.shape[2:], device=roi.device) - 0.5) * eps
    else:                    # warm start arrives as a pixel change: undo the bound to get back to δ
        delta = (delta_init[:, :ch] / tex).clamp(-eps, eps)
    delta.requires_grad_(True)
    mom = torch.zeros_like(delta)
    alpha = eps * max(2.5 / steps, 0.08)                     # step size
    # Resize so the face is ~2× the encoder input before the random 0.75–1.25× factor (large faces are shrunk
    # with antialiasing, which spreads the gradient over every pixel instead of the few a bilinear tap touches).
    face_scale = float(torch.linalg.det(mats["arcface"][:, :, :2].double()).abs().sqrt().mean())
    base = min(1.0, 2.0 * face_scale)
    sims = {name: 1.0 for name in SIZES}
    for step in range(1, steps + 1):
        sel = torch.randperm(n, device=roi.device)[:per_step] if per_step and per_step < n else torch.arange(n, device=roi.device)
        adv = (roi[sel] + (delta * tex)[group[sel]] * mask).clamp(0, 1)          # x + δ, clipped to valid pixels
        crops = {name: [] for name in SIZES}
        for _ in range(k):                                                # EOT: k random "post-upload" versions
            f = 1.0 if torch.rand(()) < 0.25 else min(1.25, base * (0.75 + 0.5 * torch.rand(()).item()))
            t = eot_image(adv, f)
            for name, size in SIZES.items():
                Mk = rescale_matrix(mats[name][sel], t.shape[-1] / adv.shape[-1])
                Mk = (jitter(len(sel), size, roi.device) @ torch.cat([Mk, Mk.new_tensor([[[0, 0, 1]]]).expand(len(sel), 1, 3)], 1))[:, :2]
                crops[name].append(warp(t, Mk, size))                     # small shifts live in the jitter
        # Loss = Σ_encoders w · mean cos(f(T(x+δ)), e0)   [− cos(f(T(x+δ)), e_decoy) in targeted mode]
        loss, cur = 0.0, {}
        for name in SIZES:
            emb = encode(name, torch.cat(crops[name]))                    # one batch of k·N crops per encoder
            c = cos(emb, e0[name][sel].repeat(k, 1))
            cur[name] = c.detach()
            loss = loss + WEIGHTS[name] * c.mean()
            if decoy is not None:
                loss = loss - WEIGHTS[name] * cos(emb, decoy[name].expand_as(emb)).mean()
        sims = {name: float(v.mean()) for name, v in cur.items()}
        worst = {name: float(v.view(k, -1).mean(0).max()) for name, v in cur.items()}
        if on_step:
            on_step(step, sims["arcface"])
        if step >= min_steps and decoy is None and all(worst[nm] < TARGET[nm] for nm in SIZES):
            # Early stop: every face already reads as someone else. If it is far past the target (warm starts
            # tend to overshoot), give some of the perturbation back instead of spending it on margin nobody needs.
            if all(worst[nm] < TARGET[nm] - 0.12 for nm in SIZES):
                with torch.no_grad():
                    delta.mul_(0.88)
            break
        (g,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            # Momentum PGD: δ ← Π_{‖δ‖∞ ≤ ε} ( δ − α · sign(μ·m + g/‖g‖₁) )
            mom.mul_(0.9).add_(g / g.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12))
            delta.sub_(alpha * mom.sign()).clamp_(-eps, eps)
    return (delta.detach() * tex).expand(-1, 3, -1, -1).contiguous(), sims


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, np.uint8).copy()).to(DEVICE).permute(2, 0, 1).float().div(255)[None]


def to_image(x: torch.Tensor) -> Image.Image:
    return Image.fromarray((x[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy())


def load_image(src, max_side: int = MAX_SIDE) -> Image.Image:
    try:
        img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    except Exception:
        raise ShieldError("Could not read that image.")
    if max(img.size) > max_side:
        f = max_side / max(img.size)
        img = img.resize((round(img.width * f), round(img.height * f)), Image.LANCZOS)
    return img


def match_faces(ref: list[dict], found: list[dict]) -> list[dict]:
    """For each reference face, the re-detected face whose box centre is nearest (falls back to the reference)."""
    out = []
    for r in ref:
        c = (r["box"][:2] + r["box"][2:]) / 2
        size = max(r["box"][2] - r["box"][0], 1.0)
        best = min(found, key=lambda f: np.linalg.norm((f["box"][:2] + f["box"][2:]) / 2 - c), default=None)
        ok = best is not None and np.linalg.norm((best["box"][:2] + best["box"][2:]) / 2 - c) < 0.5 * size
        out.append(best if ok else r)
    return out


def decoy_embedding(path) -> dict:
    x = to_tensor(load_image(path))
    faces = detect(x)[0]
    if not faces:
        raise ShieldError("No face found in the decoy photo.")
    return embed_faces(x, faces[:1])


def measure(x: torch.Tensor, ref_faces: list[dict], e0: dict) -> dict:
    """Re-detect faces in x and return, per encoder, the cosine to e0 for each reference face."""
    faces = match_faces(ref_faces, detect(x)[0])
    e = embed_faces(x, faces)
    return {name: cos(e[name], e0[name]).tolist() for name in SIZES}


def shield_image(src, dst, strength: str = "balanced", decoy=None, on_progress=None, original_out=None) -> dict:
    t0 = time.time()
    cfg = strength_cfg(strength)
    eps, steps = cfg["img_eps"], cfg["img_steps"]
    warnings: list[str] = []
    notify(on_progress, phase="preparing", step=0, total_steps=steps, current_similarity=None)
    load_models()
    img = load_image(src)
    x = to_tensor(img)
    if original_out:
        img.save(original_out)
    h, w = x.shape[-2:]
    faces = detect(x)[0]
    if not faces:
        raise ShieldError("No face found. Try a clearer, front-facing photo.")
    if len(faces) > MAX_FACES:
        warnings.append(f"Found {len(faces)} faces; shielded the {MAX_FACES} largest.")
        faces = faces[:MAX_FACES]
    e0 = embed_faces(x, faces)                                    # reference identity per face
    e_decoy = decoy_embedding(decoy) if decoy else None
    total = steps * len(faces)
    thr = THRESHOLDS["arcface"]
    notify(on_progress, phase="shielding", step=0, total_steps=total, current_similarity=1.0, before=1.0, threshold=thr)

    delta = torch.zeros_like(x)                                   # δ lives in full-res pixel space
    done = 0
    for i, fc in enumerate(faces):
        x0, y0, x1, y1 = face_roi(fc, h, w)
        mats = {n_: torch.from_numpy(shift_matrix(M, x0, y0))[None].to(DEVICE) for n_, M in face_matrices(fc).items()}
        mask = soft_mask(y1 - y0, x1 - x0, DEVICE)
        d, _ = pgd_rois(x[:, :, y0:y1, x0:x1], mats, {n_: e[i:i + 1] for n_, e in e0.items()}, eps, steps, mask, decoy=e_decoy,
                        on_step=lambda s, sim, b=done: notify(on_progress, phase="shielding", step=b + s, total_steps=total,
                                                              current_similarity=sim, before=1.0, threshold=thr))
        cur, new = delta[:, :, y0:y1, x0:x1], d * mask
        delta[:, :, y0:y1, x0:x1] = torch.where(new.abs() >= cur.abs(), new, cur)   # overlapping faces: keep the larger
        done += steps

    notify(on_progress, phase="verifying", step=total, total_steps=total, current_similarity=None, before=1.0, threshold=thr)
    to_image(x + delta).save(dst)                                 # 8-bit PNG
    # Honest numbers: reload the saved file from disk, RE-DETECT the faces, and measure that.
    y = to_tensor(Image.open(dst).convert("RGB"))
    after = measure(y, faces, e0)
    buf = io.BytesIO()
    Image.open(dst).convert("RGB").save(buf, "JPEG", quality=90)
    jpeg = measure(to_tensor(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")), faces, e0)
    before = {n_: cos(e0[n_], e0[n_]).tolist() for n_ in SIZES}   # the untouched file against itself
    # Headline = ArcFace, worst (highest) face.
    per_encoder = {n_: {"before": round(max(before[n_]), 4), "after": round(max(after[n_]), 4), "threshold": THRESHOLDS[n_],
                        "after_jpeg_q90": round(max(jpeg[n_]), 4)} for n_ in SIZES}
    if per_encoder["arcface"]["after"] >= thr:
        warnings.append("Still recognizable. Try a stronger setting.")
    x0, y0, x1, y1 = face_roi(faces[0], h, w)
    scores = {"before": per_encoder["arcface"]["before"], "after": per_encoder["arcface"]["after"], "threshold": thr,
              "quality": {"psnr_db": round(psnr(x, y), 2), "ssim": round(ssim(x, y), 4),
                          "psnr_face_db": round(psnr(x[:, :, y0:y1, x0:x1], y[:, :, y0:y1, x0:x1]), 2)},
              "per_encoder": per_encoder, "after_jpeg_q90": per_encoder["arcface"]["after_jpeg_q90"],
              "faces": len(faces), "epsilon": round(eps, 4), "size": [w, h]}
    if e_decoy is not None:
        ey = embed_faces(y, match_faces(faces, detect(y)[0]))
        scores["decoy_similarity"] = round(float(cos(ey["arcface"], e_decoy["arcface"]).max()), 4)
    return {"scores": scores, "warnings": warnings, "elapsed_s": round(time.time() - t0, 2)}


def attack_faces(x: torch.Tensor, landmarks: list, eps: float, steps: int, e0: dict | None = None, decoy_emb: dict | None = None,
                 delta_init: torch.Tensor | None = None, on_step=None) -> tuple[torch.Tensor, dict]:
    """Convenience wrapper: x [B,3,H,W], landmarks[b] = list of face dicts from detect(). Returns (δ [B,3,H,W], sims)."""
    delta = torch.zeros_like(x) if delta_init is None else delta_init.clone()
    sims = {}
    for b, faces in enumerate(landmarks):
        xb = x[b:b + 1]
        eb = embed_faces(xb, faces) if e0 is None else e0[b]
        for i, fc in enumerate(faces):
            x0, y0, x1, y1 = face_roi(fc, *x.shape[-2:])
            mats = {n_: torch.from_numpy(shift_matrix(M, x0, y0))[None].to(DEVICE) for n_, M in face_matrices(fc).items()}
            mask = soft_mask(y1 - y0, x1 - x0, DEVICE)
            init = delta[b:b + 1, :, y0:y1, x0:x1] if delta_init is not None else None
            d, sims = pgd_rois(xb[:, :, y0:y1, x0:x1], mats, {n_: e[i:i + 1] for n_, e in eb.items()}, eps, steps, mask,
                               decoy=decoy_emb, delta_init=init, on_step=on_step)
            delta[b:b + 1, :, y0:y1, x0:x1] = d * mask
    return delta, sims


def calibrate(folder=ROOT / "samples") -> None:
    """Print same-person vs different-person cosine per encoder on the sample photos.
    Files named <prefix>_<person>_<n>.jpg are grouped by <person>."""
    files = sorted(Path(folder).glob("cal_*_*.jpg"))
    if len(files) < 3:
        print("No calibration photos (samples/cal_<person>_<n>.jpg).")
        return
    embs, people = {n_: [] for n_ in SIZES}, []
    for f in files:
        x = to_tensor(load_image(f))
        faces = detect(x)[0]
        if not faces:
            print("  no face:", f.name)
            continue
        e = embed_faces(x, faces[:1])
        for n_ in SIZES:
            embs[n_].append(e[n_][0])
        people.append(f.stem.split("_")[1])
    for n_ in SIZES:
        E = F.normalize(torch.stack(embs[n_]), dim=-1)
        S = (E @ E.T).cpu().numpy()
        same = [S[i, j] for i in range(len(people)) for j in range(i) if people[i] == people[j]]
        diff = [S[i, j] for i in range(len(people)) for j in range(i) if people[i] != people[j]]
        print(f"{n_:8s} same person: n={len(same)} min {min(same):.3f} mean {np.mean(same):.3f} | "
              f"different: n={len(diff)} mean {np.mean(diff):.3f} max {max(diff):.3f} | threshold in use {THRESHOLDS[n_]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Shroud photo shield")
    ap.add_argument("src", nargs="?"), ap.add_argument("dst", nargs="?")
    ap.add_argument("--strength", default="balanced", choices=["light", "balanced", "strong"])
    ap.add_argument("--decoy")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
        raise SystemExit
    if not a.src or not a.dst:
        ap.error("src and dst are required")
    r = shield_image(a.src, a.dst, a.strength, decoy=a.decoy,
                     on_progress=lambda p: print(f"  {p['phase']:<10} step {p['step']:>3}/{p['total_steps']}  match {p['current_similarity']}"))
    s = r["scores"]
    print(f"device {DEVICE} · {s['faces']} face(s) · {s['size'][0]}×{s['size'][1]} · ε {s['epsilon']}")
    for n_, v in s["per_encoder"].items():
        print(f"  {n_:8s} match before {v['before']:.3f} → after {v['after']:.3f} (JPEG q90: {v['after_jpeg_q90']:.3f}) threshold {v['threshold']}")
    q = s["quality"]
    print(f"PSNR {q['psnr_db']:.1f} dB (face region {q['psnr_face_db']:.1f}) · SSIM {q['ssim']:.4f} · {r['elapsed_s']} s")
    for w_ in r["warnings"]:
        print("warning:", w_)
