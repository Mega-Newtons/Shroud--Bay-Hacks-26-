"""Video shield: the photo attack on the face region of every frame, warm-started frame to frame.

    python -m shield.video in.mp4 out.mp4 --strength balanced [--decoy face.jpg] [--shield-audio]

Frames go in and out of ffmpeg as raw rgb24 over pipes (no OpenCV). The reported scores are measured on frames
DECODED from the encoded mp4, with faces re-detected — i.e. on what a deepfake tool would actually receive.
"""
import os
import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import image as I
from .common import DEVICE, FAST, ShieldError, cos, ffprobe, notify, psnr, run, ssim
from .common import strength as strength_cfg

MAX_SECONDS, MAX_HEIGHT = (10.0, 720) if FAST else (5.0, 480)
# GPU: one δ per frame, 16 frames per PGD batch, 2 EOT draws.  CPU: one δ shared by 5 consecutive frames (it follows
# the face), 2 of those frames sampled per step, 1 EOT draw — ~8× less encoder work for a small loss in strength.
CHUNK, SHARE, PER_STEP, EOT_K = (16, 1, None, 2) if FAST else (5, 5, 2, 1)
BT709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
X264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", "-preset", "veryfast"]
TEX_FLOOR = float(os.environ.get("SHROUD_VID_FLOOR", 0.5))   # video compression eats fine detail, so smooth areas keep more of ε than in photos


def normalise(src, dst, max_seconds: float, max_height: int) -> None:
    """Trim / scale / cap fps into an H.264 mp4 with known colour tags. This is the 'original' shown in the UI and
    the exact frames the shield works on, so before/after line up pixel for pixel."""
    vf = (f"fps='min(30,source_fps)',scale=-2:'trunc(min({max_height},ih)/2)*2':flags=bicubic:out_color_matrix=bt709:out_range=tv,"
          "format=yuv420p")
    run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", str(max_seconds), "-vf", vf, *X264, *BT709,
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dst)])


def decode(path) -> tuple[np.ndarray, str, bool]:
    """mp4 → uint8 frames [T,H,W,3], fps string, has_audio."""
    streams = ffprobe(path)["streams"]
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise ShieldError("No video found in that file.")
    w, h = int(v["width"]), int(v["height"])
    raw = run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", "scale=in_color_matrix=bt709:in_range=tv,format=rgb24",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return frames, v.get("r_frame_rate", "30/1"), any(s.get("codec_type") == "audio" for s in streams)


def encode(frames: np.ndarray, fps: str, dst, audio_from=None, audio_copy: bool = True) -> None:
    """Raw rgb24 frames → libx264 yuv420p CRF 14 veryfast; audio copied from `audio_from` (or encoded if it is a WAV)."""
    t, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", fps, "-i", "-"]
    if audio_from:
        cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy" if audio_copy else "aac"]
        cmd += [] if audio_copy else ["-b:a", "192k"]
    cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", *X264, *BT709, "-movflags", "+faststart", str(dst)]
    run(cmd, input=frames.tobytes())


def to_float(frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames).to(DEVICE).permute(0, 3, 1, 2).float().div(255)


def track_faces(frames: np.ndarray, batch: int = 16) -> list:
    """Main face per frame (or None): the largest face in the first frame that has one, then the nearest to it."""
    out, prev = [], None
    for i in range(0, len(frames), batch):
        for faces in I.detect(to_float(frames[i:i + batch])):
            if not faces:
                out.append(None)
                continue
            if prev is None:
                best = faces[0]
            else:
                c = (prev["box"][:2] + prev["box"][2:]) / 2
                best = min(faces, key=lambda f: np.linalg.norm((f["box"][:2] + f["box"][2:]) / 2 - c))
                if np.linalg.norm((best["box"][:2] + best["box"][2:]) / 2 - c) > 1.5 * (prev["box"][2] - prev["box"][0]):
                    best = faces[0]            # scene cut: start over from the largest face
            out.append(best)
            prev = best
    return out


def embed_frames(frames: np.ndarray, faces: list, batch: int = 16) -> dict:
    """Embeddings of the tracked face in each frame → {"arcface": [T,512], "facenet": [T,512]} (rows of NaN where no face)."""
    out = {n: torch.full((len(frames), 512), float("nan"), device=DEVICE) for n in I.SIZES}
    idx = [t for t, f in enumerate(faces) if f is not None]
    with torch.inference_mode():
        for i in range(0, len(idx), batch):
            sel = idx[i:i + batch]
            x = to_float(frames[sel])
            for n, size in I.SIZES.items():
                M = torch.from_numpy(np.stack([I.face_matrices(faces[t])[n] for t in sel])).to(DEVICE)
                out[n][sel] = I.encode(n, I.warp(x, M, size))
    return out


def shield_video(src, dst, strength: str = "balanced", decoy=None, on_progress=None, original_out=None,
                 shield_audio_track: bool = False, max_seconds: float | None = None, max_height: int | None = None) -> dict:
    t0 = time.time()
    cfg = strength_cfg(strength)
    eps, steps = cfg["vid_eps"], cfg["vid_steps"]
    max_seconds, max_height = max_seconds or MAX_SECONDS, max_height or MAX_HEIGHT
    warnings: list[str] = []
    thr = I.THRESHOLDS["arcface"]
    notify(on_progress, phase="preparing", step=0, total_steps=0, current_similarity=None)
    I.load_models()
    tmp = tempfile.TemporaryDirectory()
    work = Path(tmp.name)
    orig = Path(original_out) if original_out else work / "original.mp4"

    info = ffprobe(src)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if v is None:
        raise ShieldError("No video found in that file.")
    if float(info.get("format", {}).get("duration") or 0) > max_seconds + 0.25:
        warnings.append(f"Trimmed to the first {max_seconds:g} seconds.")
    if min(int(v["width"]), int(v["height"])) > max_height and int(v["height"]) > max_height:
        warnings.append(f"Scaled down to {max_height}p.")
    normalise(src, orig, max_seconds, max_height)
    frames, fps, has_audio = decode(orig)
    T, H, W, _ = frames.shape
    if T == 0:
        raise ShieldError("Could not read any frames from that video.")

    faces = track_faces(frames)
    with_face = [t for t, f in enumerate(faces) if f is not None]
    if not with_face:
        raise ShieldError("No face found in the video. Try a clip where the face is clearly visible.")
    # Identity reference: the mean embedding over up to 24 evenly spaced original frames.
    ref_idx = [with_face[i] for i in np.linspace(0, len(with_face) - 1, min(24, len(with_face))).round().astype(int)]
    ref = embed_frames(frames[ref_idx], [faces[t] for t in ref_idx])
    e_id = {n: F.normalize(F.normalize(e, dim=-1).mean(0, keepdim=True), dim=-1) for n, e in ref.items()}
    before = {n: float(cos(ref[n], e_id[n]).mean()) for n in I.SIZES}
    e_decoy = I.decoy_embedding(decoy) if decoy else None
    notify(on_progress, phase="shielding", step=0, total_steps=T, current_similarity=before["arcface"],
           before=before["arcface"], threshold=thr)

    # One fixed-size window that follows the face: δ is stored in window coordinates, so it moves with the face and
    # can be reused (warm start) from one frame to the next without resampling.
    rois = np.array([I.face_roi(faces[t], H, W) for t in with_face])
    rw = int(min(W, np.percentile(rois[:, 2] - rois[:, 0], 95)))
    rh = int(min(H, np.percentile(rois[:, 3] - rois[:, 1], 95)))
    mask = I.soft_mask(rh, rw, DEVICE)
    # Anchor the window on the landmark centroid (steadier than the detector box), at the median box-centre offset.
    cents = np.array([faces[t]["lm"].mean(0) for t in with_face])
    off = np.median(np.stack([(rois[:, 0] + rois[:, 2]) / 2, (rois[:, 1] + rois[:, 3]) / 2], 1) - cents, axis=0)
    origin = {}
    for t, c in zip(with_face, cents + off):
        origin[t] = (int(np.clip(round(c[0] - rw / 2), 0, W - rw)), int(np.clip(round(c[1] - rh / 2), 0, H - rh)))

    out = frames.copy()
    prev_delta, sim, hard = None, before["arcface"], False
    for c0 in range(0, T, CHUNK):
        ts = [t for t in range(c0, min(c0 + CHUNK, T)) if faces[t] is not None]
        if ts:
            roi = torch.stack([to_float(frames[t:t + 1, origin[t][1]:origin[t][1] + rh, origin[t][0]:origin[t][0] + rw])[0] for t in ts])
            mats = {n: torch.from_numpy(np.stack([I.shift_matrix(I.face_matrices(faces[t])[n], *origin[t]) for t in ts])).to(DEVICE)
                    for n in I.SIZES}
            group = torch.arange(len(ts), device=DEVICE) // SHARE
            n_delta = int(group.max()) + 1
            init = prev_delta[-1:].expand(n_delta, -1, -1, -1) if prev_delta is not None else None   # warm start
            # The first chunk starts cold and gets 3× the steps; later chunks only have to track small changes.
            d, sims = I.pgd_rois(roi, mats, {n: e_id[n].expand(len(ts), -1) for n in I.SIZES}, eps,
                                 steps * (3 if prev_delta is None else 2 if hard else 1), mask, decoy=e_decoy, delta_init=init,
                                 group=group, per_step=PER_STEP, k=EOT_K, min_steps=1, tex_floor=TEX_FLOOR)
            prev_delta, sim = d, sims["arcface"]
            hard = sim > I.TARGET["arcface"] + 0.1          # did not converge: give the next chunk twice the steps
            adv = ((roi + d[group] * mask).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            for j, t in enumerate(ts):
                out[t, origin[t][1]:origin[t][1] + rh, origin[t][0]:origin[t][0] + rw] = adv[j]
        notify(on_progress, phase="shielding", step=min(c0 + CHUNK, T), total_steps=T, current_similarity=sim,
               before=before["arcface"], threshold=thr)

    notify(on_progress, phase="encoding", step=T, total_steps=T, current_similarity=sim, before=before["arcface"], threshold=thr)
    audio_scores, audio_from, audio_copy = None, (orig if has_audio else None), True
    if shield_audio_track and has_audio:
        from . import audio as A
        wav = work / "shielded.wav"
        try:
            audio_scores = A.shield_audio(orig, wav, strength, max_seconds=max_seconds)["scores"]
            audio_from, audio_copy = wav, False
        except ShieldError as e:      # e.g. a silent track: keep the original audio and say so
            warnings.append(f"Audio track left as is: {e}")
    elif shield_audio_track:
        warnings.append("The video has no audio track to shield.")
    encode(out, fps, dst, audio_from, audio_copy)

    # Honest verification: decode what was just encoded, re-detect, re-embed.
    notify(on_progress, phase="verifying", step=T, total_steps=T, current_similarity=sim, before=before["arcface"], threshold=thr)
    dec, _, _ = decode(dst)
    n = min(len(dec), T)
    dec_faces = track_faces(dec[:n])
    emb = embed_frames(dec[:n], dec_faces)
    per = {m: cos(emb[m], e_id[m]) for m in I.SIZES}                       # NaN where no face
    found = ~torch.isnan(per["arcface"])
    if not bool(found.any()):
        raise ShieldError("No face could be read back from the shielded video.")
    after = {m: float(per[m][found].mean()) for m in I.SIZES}
    per_frame = [round(float(s), 4) if bool(ok) else None for s, ok in zip(per["arcface"], found)]
    ps, ss = [], []
    for i in range(0, n, 16):
        a, b = to_float(frames[i:i + 16]), to_float(dec[i:i + 16])
        ps.append(psnr(a, b) * len(a)), ss.append(ssim(a, b) * len(a))
    below = int((per["arcface"][found] < thr).sum())
    if after["arcface"] >= thr or below < 0.9 * int(found.sum()):
        warnings.append(f"Still recognizable in {int(found.sum()) - below} of {int(found.sum())} frames. Try a stronger setting.")
    per_encoder = {m: {"before": round(before[m], 4), "after": round(after[m], 4), "threshold": I.THRESHOLDS[m]} for m in I.SIZES}
    scores = {"before": per_encoder["arcface"]["before"], "after": per_encoder["arcface"]["after"], "threshold": thr,
              "quality": {"psnr_db": round(sum(ps) / n, 2), "ssim": round(sum(ss) / n, 4)},
              "per_encoder": per_encoder, "per_frame": per_frame, "frames": n, "fps": round(eval_fps(fps), 3),
              "faces": int(found.sum()), "audio": audio_scores, "epsilon": round(eps, 4), "size": [W, H],
              "worst_frame": round(float(per["arcface"][found].max()), 4),
              "frames_below_threshold": int((per["arcface"][found] < thr).sum())}
    if e_decoy is not None:
        scores["decoy_similarity"] = round(float(cos(emb["arcface"][found], e_decoy["arcface"]).mean()), 4)
    tmp.cleanup()
    return {"scores": scores, "warnings": warnings, "elapsed_s": round(time.time() - t0, 2)}


def eval_fps(s: str) -> float:
    a, _, b = s.partition("/")
    return float(a) / float(b or 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Shroud video shield")
    ap.add_argument("src"), ap.add_argument("dst")
    ap.add_argument("--strength", default="balanced", choices=["light", "balanced", "strong"])
    ap.add_argument("--decoy")
    ap.add_argument("--shield-audio", action="store_true")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--max-height", type=int)
    ap.add_argument("--original-out")
    a = ap.parse_args()
    last = [None]

    def show(p):
        key = (p["phase"], p["step"] * 10 // max(1, p["total_steps"]))
        if key != last[0]:
            last[0] = key
            print(f"  {p['phase']:<10} frame {p['step']:>4}/{p['total_steps']}  match {p['current_similarity']}")

    r = shield_video(a.src, a.dst, a.strength, decoy=a.decoy, on_progress=show, original_out=a.original_out,
                     shield_audio_track=a.shield_audio, max_seconds=a.max_seconds, max_height=a.max_height)
    s = r["scores"]
    print(f"device {DEVICE} · {s['frames']} frames @ {s['fps']} fps · {s['size'][0]}×{s['size'][1]} · face read back in {s['faces']} frames")
    for n_, v_ in s["per_encoder"].items():
        print(f"  {n_:8s} mean match before {v_['before']:.3f} → after {v_['after']:.3f} (threshold {v_['threshold']})")
    print(f"  worst frame {s['worst_frame']:.3f} · {s['frames_below_threshold']}/{s['faces']} frames below threshold")
    print(f"PSNR {s['quality']['psnr_db']:.1f} dB · SSIM {s['quality']['ssim']:.4f} · {r['elapsed_s']} s")
    if s["audio"]:
        print(f"  audio: match {s['audio']['before']:.3f} → {s['audio']['after']:.3f} · SNR {s['audio']['quality']['snr_db']} dB")
    for w_ in r["warnings"]:
        print("warning:", w_)
