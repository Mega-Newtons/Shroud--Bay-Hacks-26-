"""Voice shield: PGD on the raw waveform against the SpeechBrain ECAPA-TDNN speaker encoder.

    python -m shield.audio in.wav out.wav --strength balanced [--decoy other_voice.wav]
"""
import argparse
import math
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from .common import DEVICE, EOT_SAMPLES, MODELS, ShieldError, autocast, cos, ffprobe, notify, run
from .common import strength as strength_cfg

THRESHOLD = 0.25      # SpeechBrain's default cosine threshold for "same speaker" (SpeakerRecognition.verify_batch)
TARGET = 0.12         # stop once the full clip scores below this (margin under the threshold)
CROP_SECONDS = 2.0    # PGD looks at random windows of this length: cheap steps + robust to trimming
_M: dict = {}


def load_models() -> dict:
    """ECAPA built from its SpeechBrain modules (no hyperpyyaml / hub fetch); weights from models/ecapa/."""
    if _M:
        return _M
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    ckpt = MODELS / "ecapa" / "embedding_model.ckpt"
    if not ckpt.exists():
        raise ShieldError("Voice model is missing. Run scripts/setup.sh first.")
    ecapa = ECAPA_TDNN(80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
                       dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192)
    ecapa.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    _M["fbank"] = Fbank(n_mels=80).to(DEVICE).eval()
    _M["ecapa"] = ecapa.to(DEVICE).eval().requires_grad_(False)
    return _M


def embed(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """[B,T] waveform at `sr` → [B,192] speaker embedding. Differentiable all the way back to the samples:
    resample → log-mel fbank → per-utterance mean normalisation → ECAPA (the same chain as encode_batch)."""
    m = load_models()
    if sr != 16000:
        wav = AF.resample(wav, sr, 16000)
    feats = m["fbank"](wav)                             # fp32: STFT + log-mel
    feats = feats - feats.mean(dim=1, keepdim=True)     # InputNormalization(norm_type="sentence", std_norm=False)
    with autocast():
        emb = m["ecapa"](feats)
    return emb.float().squeeze(1)


# --- I/O: ffmpeg → mono 16-bit WAV, stdlib wave ---------------------------------------------------------------------
def to_wav(src, dst, max_seconds: float | None = None) -> None:
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    run(cmd + ["-c:a", "pcm_s16le", "-f", "wav", str(dst)])


def read_wav(path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype="<i2")
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0), sr


def write_wav(path, x: torch.Tensor, sr: int) -> None:
    pcm = (x.detach().clamp(-1, 1) * 32767.0).round().cpu().numpy().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def load_any(src, max_seconds: float | None = None) -> tuple[torch.Tensor, int]:
    with tempfile.TemporaryDirectory() as d:
        to_wav(src, Path(d) / "a.wav", max_seconds)
        return read_wav(Path(d) / "a.wav")


def snr_db(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    noise = (noisy - clean).pow(2).sum().item()
    return 99.0 if noise < 1e-12 else 10 * math.log10(clean.pow(2).sum().item() / noise)


# --- EOT for audio ---------------------------------------------------------------------------------------------------
def eot_audio(x: torch.Tensor, peak: float) -> torch.Tensor:
    """[K,T] → random light low-pass (conv1d) on half the batch + light noise. The time shift comes from the random crops."""
    k = torch.tensor([0.25, 0.5, 0.25], device=x.device).view(1, 1, 3)       # binomial low-pass
    lp = F.conv1d(F.pad(x[:, None], (1, 1), mode="replicate"), k)[:, 0]
    pick = (torch.rand(x.shape[0], 1, device=x.device) < 0.5).float()
    x = pick * lp + (1 - pick) * x
    return x + torch.randn_like(x) * (peak * 0.003)                           # ≈ -50 dB re. peak


def envelope_bound(x: torch.Tensor, sr: int) -> torch.Tensor:
    """Per-sample share of ε, between 0.15 and 1, following the loudness envelope: the perturbation is allowed
    to be loudest where the voice is loud (where it is masked) and stays near-silent in the pauses."""
    win = max(3, int(sr * 0.02) | 1)
    env = F.max_pool1d(x.abs()[None, None], win, stride=1, padding=win // 2)
    env = F.avg_pool1d(env, win, stride=1, padding=win // 2)[0, 0, : x.numel()]
    return 0.15 + 0.85 * (env / env.max().clamp_min(1e-6)).clamp(0, 1)


def shield_audio(src, dst, strength: str = "balanced", decoy=None, on_progress=None, original_out=None,
                 max_seconds: float = 20.0) -> dict:
    t0 = time.time()
    cfg = strength_cfg(strength)
    warnings: list[str] = []
    notify(on_progress, phase="preparing", step=0, total_steps=cfg["aud_steps"], current_similarity=None)
    load_models()
    info = ffprobe(src)
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        raise ShieldError("No audio found in that file.")
    if float(info.get("format", {}).get("duration") or 0) > max_seconds + 0.25:
        warnings.append(f"Trimmed to the first {max_seconds:g} seconds.")
    x, sr = load_any(src, max_seconds)
    x = x.to(DEVICE)
    peak = x.abs().max().item()
    if x.numel() < sr * 0.5 or peak < 1e-3:
        raise ShieldError("That recording is too short or silent. Record at least a second of speech.")
    if original_out:
        write_wav(original_out, x, sr)

    with torch.inference_mode():
        e0 = embed(x[None], sr).clone()                           # reference identity
        e_decoy = None
        if decoy:                                                 # optional: identity to steer toward
            dw, dsr = load_any(decoy, max_seconds)
            e_decoy = embed(dw.to(DEVICE)[None], dsr).clone()

    steps, eps_max = cfg["aud_steps"], cfg["aud_eps"] * peak
    bound = envelope_bound(x, sr)                                 # [T] in 0.15..1
    crop = min(x.numel(), int(CROP_SECONDS * sr))
    K = EOT_SAMPLES
    delta = torch.zeros_like(x).uniform_(-1, 1) * bound * eps_max * 0.1
    delta.requires_grad_(True)
    grad_avg = torch.zeros_like(x)
    check_every = 3
    sim, eps = 1.0, eps_max * 0.4
    notify(on_progress, phase="shielding", step=0, total_steps=steps, current_similarity=1.0, before=1.0, threshold=THRESHOLD)

    step = 0
    for step in range(1, steps + 1):
        # ε ramps geometrically from 0.4·ε_max to ε_max over the first 60 % of the budget. Because we stop as soon as
        # the target is met, the clip ends up with (about) the quietest ε on that ramp that works.
        eps = eps_max * min(1.0, 0.4 * 2.5 ** (step / (0.6 * steps)))
        starts = torch.randint(0, x.numel() - crop + 1, (K,), device=DEVICE)
        idx = starts[:, None] + torch.arange(crop, device=DEVICE)[None]       # [K,crop] random windows (time-shift EOT)
        adv = (x[idx] + delta[idx]).clamp(-1, 1)
        emb = embed(eot_audio(adv, peak), sr)
        # Loss: push the embedding away from the speaker (cosine to e0 ↓); with a decoy, also pull toward it.
        loss = cos(emb, e0).mean()
        if e_decoy is not None:
            loss = loss - cos(emb, e_decoy).mean()
        (g,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            # Momentum PGD: normalise the gradient, accumulate, step by sign, project back into the
            # envelope-shaped L∞ ball |δ(t)| ≤ ε·bound(t).
            grad_avg.mul_(0.8).add_(g / g.abs().mean().clamp_min(1e-12))
            delta.sub_(eps / 3 * bound * grad_avg.sign())
            delta.copy_(torch.maximum(torch.minimum(delta, eps * bound), -eps * bound))
            delta.copy_((x + delta).clamp(-1, 1) - x)             # keep the output a valid waveform
        if step % check_every == 0 or step == steps:
            with torch.inference_mode():
                sim = cos(embed((x + delta)[None], sr), e0).item()  # full clip, no EOT
            notify(on_progress, phase="shielding", step=step, total_steps=steps, current_similarity=sim,
                   before=1.0, threshold=THRESHOLD)
            if sim < TARGET and (e_decoy is None or step >= steps // 2):
                break                                              # early stop

    notify(on_progress, phase="verifying", step=step, total_steps=steps, current_similarity=sim, before=1.0, threshold=THRESHOLD)
    write_wav(dst, x + delta.detach(), sr)
    # Honest numbers: re-read the 16-bit file that was just written and measure that.
    y, _ = read_wav(dst)
    y = y.to(DEVICE)
    with torch.inference_mode():
        e1 = embed(y[None], sr)
        after = cos(e1, e0).item()
        before = cos(embed(x[None], sr), e0).item()
        decoy_sim = cos(e1, e_decoy).item() if e_decoy is not None else None
    if after >= THRESHOLD:
        warnings.append("Still above the threshold. Try a stronger setting.")
    scores = {"before": round(before, 4), "after": round(after, 4), "threshold": THRESHOLD,
              "quality": {"snr_db": round(snr_db(x, y), 2)},
              "per_encoder": {"ecapa": {"before": round(before, 4), "after": round(after, 4), "threshold": THRESHOLD}},
              "epsilon": round(eps / max(peak, 1e-9), 4), "steps": step, "sample_rate": sr,
              "seconds": round(x.numel() / sr, 2)}
    if decoy_sim is not None:
        scores["decoy_similarity"] = round(decoy_sim, 4)
    return {"scores": scores, "warnings": warnings, "elapsed_s": round(time.time() - t0, 2)}


def grad_check() -> float:
    """Gradient-norm check: d(embedding)/d(waveform) must be non-zero and finite."""
    w = (torch.randn(1, 16000, device=DEVICE) * 0.1).requires_grad_(True)
    embed(w, 16000).pow(2).sum().backward()
    n = w.grad.norm().item()
    assert math.isfinite(n) and n > 0, "no gradient reaches the waveform"
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Shroud voice shield")
    ap.add_argument("src"), ap.add_argument("dst")
    ap.add_argument("--strength", default="balanced", choices=["light", "balanced", "strong"])
    ap.add_argument("--decoy")
    ap.add_argument("--max-seconds", type=float, default=20.0)
    a = ap.parse_args()
    print(f"device {DEVICE} · grad-norm check {grad_check():.3f}")
    r = shield_audio(a.src, a.dst, a.strength, decoy=a.decoy, max_seconds=a.max_seconds,
                     on_progress=lambda p: print(f"  {p['phase']:<10} step {p['step']:>3}/{p['total_steps']}  match {p['current_similarity']}"))
    s = r["scores"]
    print(f"match before {s['before']:.3f} → after {s['after']:.3f}  (threshold {s['threshold']})")
    print(f"SNR {s['quality']['snr_db']:.1f} dB · ε {s['epsilon']} of peak · {s['steps']} steps · {r['elapsed_s']} s")
    for w_ in r["warnings"]:
        print("warning:", w_)
