"""Alignment tests: Umeyama recovers a known similarity; the differentiable affine_grid/grid_sample crop matches an
independent numpy inverse-warp with bilinear sampling. Run: python -m tests.test_align"""
import numpy as np
import torch

from shield.image import TEMPLATE, arc_matrix, umeyama, warp


def reference_warp(img: np.ndarray, M: np.ndarray, size: int) -> np.ndarray:
    """img [H,W,3] float; for every output pixel, map back through M⁻¹ and sample bilinearly (zeros outside)."""
    h, w, _ = img.shape
    Minv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))
    out = np.zeros((size, size, 3))
    for v in range(size):
        for u in range(size):
            x, y, _ = Minv @ [u, v, 1.0]
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            for dy in (0, 1):
                for dx in (0, 1):
                    xi, yi = x0 + dx, y0 + dy
                    if 0 <= xi < w and 0 <= yi < h:
                        out[v, u] += img[yi, xi] * (1 - abs(x - xi)) * (1 - abs(y - yi))
    return out


def test_umeyama():
    rng = np.random.default_rng(0)
    ang, s, t = 0.3, 1.7, np.array([12.0, -5.0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    src = rng.normal(size=(5, 2)) * 40
    M = umeyama(src, (s * (R @ src.T)).T + t)
    assert np.allclose(M[:, :2], s * R, atol=1e-8) and np.allclose(M[:, 2], t, atol=1e-8), M


def test_crop_matches_reference():
    rng = np.random.default_rng(1)
    img = rng.random((180, 240, 3))
    # landmarks: the template under a known rotation/scale/shift, plus noise
    ang, s = -0.2, 1.4
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    lm = (s * (R @ (TEMPLATE - 56).T)).T + [120, 90] + rng.normal(size=(5, 2))
    M = arc_matrix(lm)
    ours = warp(torch.from_numpy(img).permute(2, 0, 1)[None].double(), torch.from_numpy(M)[None], 112)[0].permute(1, 2, 0).numpy()
    ref = reference_warp(img, M, 112)
    err = np.abs(ours - ref).max()
    assert err < 1e-4, f"crop differs from reference: max abs err {err}"
    return err


def test_gradient_reaches_pixels():
    x = torch.rand(1, 3, 180, 240, requires_grad=True)
    M = torch.from_numpy(arc_matrix(TEMPLATE * 1.2 + 30))[None]
    warp(x, M, 112).sum().backward()
    assert x.grad.abs().sum() > 0


if __name__ == "__main__":
    test_umeyama()
    print("umeyama ok")
    print(f"crop vs numpy reference ok (max abs err {test_crop_matches_reference():.2e})")
    test_gradient_reaches_pixels()
    print("gradient ok")
