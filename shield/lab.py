"""Attack lab: we attack our own output with the face swapper most open-source deepfake tools use.

inswapper_128 takes (a) a 128×128 aligned target face and (b) the ArcFace identity embedding of a source
face, and repaints the target with the source identity. We run it twice on the same target — once with
the identity read from the ORIGINAL photo, once from the SHIELDED photo — and measure how closely each
fake matches the real person (ArcFace cosine). If the shield works, the second fake is someone else.

    python -m shield.lab original.png shielded.png target.jpg outdir/
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import image as im
from .common import DEVICE, MODELS, ROOT, ShieldError, cos, gaussian_blur

SWAPPER = MODELS / "inswapper_128.onnx"
DEFAULT_TARGETS = [ROOT / "samples" / "lab_target.jpg", ROOT / "samples" / "fn_bradley_cooper.jpg"]
_M: dict = {}


def available() -> bool:
    return SWAPPER.exists()


def load_models() -> dict:
    if _M:
        return _M
    import onnx
    import onnx2torch
    model = onnx.load(str(SWAPPER))
    # "emap": a 512×512 matrix InsightFace stores as the last graph initializer; the swapper expects
    # the ArcFace embedding projected through it and re-normalised.
    _M["emap"] = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[-1]).copy()).to(DEVICE)
    _M["swapper"] = onnx2torch.convert(model).to(DEVICE).eval().requires_grad_(False)
    im.load_models()
    return _M


def warp_to(img: torch.Tensor, M: np.ndarray, h: int, w: int) -> torch.Tensor:
    """Like image.warp but to an h×w canvas: M (2×3) maps source pixels → output pixels."""
    _, _, sh, sw = img.shape
    A = np.vstack([M, [0, 0, 1]])
    out_inv = np.array([[w / 2, 0, (w - 1) / 2], [0, h / 2, (h - 1) / 2], [0, 0, 1]])
    src = np.array([[2 / sw, 0, 1 / sw - 1], [0, 2 / sh, 1 / sh - 1], [0, 0, 1]])
    theta = torch.from_numpy((src @ np.linalg.inv(A) @ out_inv)[:2]).to(img)[None]
    return F.grid_sample(img, F.affine_grid(theta, (1, img.shape[1], h, w), align_corners=False), mode="bilinear",
                         padding_mode="zeros", align_corners=False)


def identity(path) -> tuple[torch.Tensor, torch.Tensor]:
    """ArcFace embedding of the largest face in a photo, exactly as a swap tool would read it."""
    x = im.to_tensor(im.load_image(path))
    faces = im.detect(x)[0]
    if not faces:
        raise ShieldError("No face found in the source photo.")
    return im.embed_faces(x, faces[:1])["arcface"][0], x


@torch.inference_mode()
def swap(target: torch.Tensor, face: dict, emb: torch.Tensor) -> torch.Tensor:
    """Repaint `face` in target [1,3,H,W] with identity `emb` [512]; returns the full image."""
    m = load_models()
    h, w = target.shape[-2:]
    M = im.umeyama(face["lm"], im.TEMPLATE * (128 / 112))        # image → 128×128 aligned crop
    crop = warp_to(target, M, 128, 128)
    latent = F.normalize(emb[None].float(), dim=1) @ m["emap"]
    latent = latent / latent.norm(dim=1, keepdim=True)
    fake = m["swapper"](crop, latent).clamp(0, 1)                # [1,3,128,128], RGB 0..1
    # paste back: warp the fake and a feathered mask through the inverse transform, then blend
    Minv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
    mask = torch.zeros(1, 1, 128, 128, device=target.device)
    mask[..., 10:118, 10:118] = 1
    mask = gaussian_blur(mask, 5.0, 21)
    back, mback = warp_to(fake, Minv, h, w), warp_to(mask, Minv, h, w)
    return target * (1 - mback) + back * mback


def faceswap(original, shielded, target=None, out_dir=None) -> dict:
    """Swap the original and the shielded identity onto the same target. Returns files + real measured scores."""
    load_models()
    if target is None:
        target = next((p for p in DEFAULT_TARGETS if p.exists()), None)
        if target is None:
            raise ShieldError("No target face available. Add one as samples/lab_target.jpg.")
    out_dir = Path(out_dir or ".")
    e_real, _ = identity(original)
    e_shield, _ = identity(shielded)
    t = im.to_tensor(im.load_image(target, 768))
    faces = im.detect(t)[0]
    if not faces:
        raise ShieldError("No face found in the target photo.")
    face = faces[0]
    # show a square around the target face rather than the whole photo
    x1, y1, x2, y2 = face["box"]
    cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, 1.1 * max(x2 - x1, y2 - y1)
    h, w = t.shape[-2:]
    r = min(r, cx, cy, w - cx, h - cy)
    box = (int(cx - r), int(cy - r), int(cx + r), int(cy + r))

    files, sims = {}, {}
    im.to_image(t).crop(box).save(out_dir / "lab_target.png")
    files["target_url"] = out_dir / "lab_target.png"
    for key, emb in (("original", e_real), ("shielded", e_shield)):
        out = swap(t, face, emb)
        found = im.detect(out)[0]                                 # honest: re-detect on the finished fake
        e_fake = im.embed_faces(out, found[:1] or [face])["arcface"][0]
        sims[key] = round(float(cos(e_fake, e_real)), 3)
        p = out_dir / f"lab_swap_{key}.png"
        im.to_image(out).crop(box).save(p)
        files[f"swap_{key}_url"] = p
    return {"scores": {"similarity_original": sims["original"], "similarity_shielded": sims["shielded"],
                       "source_similarity": round(float(cos(e_shield, e_real)), 3), "threshold": im.THRESHOLDS["arcface"]},
            "files": files}


if __name__ == "__main__":
    a = sys.argv[1:]
    r = faceswap(a[0], a[1], a[2] if len(a) > 2 else None, a[3] if len(a) > 3 else ".")
    print(r["scores"])
