# Shroud

**Your selfie shouldn't be a deepfake kit.**

Anyone can clone your voice from seconds of audio or fake your face from one selfie. Deepfake protection
exists for celebrities, not for everyone else. Shroud adds invisible changes to your photos, voice notes
and videos that scramble the fingerprint AI pulls from you. You see the same file; deepfake tools see
someone else. The ordinary, elevated.

Everything runs on your own machine. No accounts, no uploads, no hosted APIs.

## Run it

```bash
./scripts/setup.sh      # once: venv, dependencies, model weights, fonts (needs Python 3.11 and ffmpeg)
./scripts/run.sh        # starts the real server → http://localhost:8000
```

`run.sh` starts `backend/app.py` (real models, real numbers). `backend/mock_app.py` exists only for UI
development; it marks every response `"mock": true` and the page shows a MOCK DATA banner when it sees one.

Command line:

```bash
.venv/bin/python -m shield.audio in.wav out.wav --strength balanced
.venv/bin/python -m shield.image in.jpg out.png --strength balanced     # add --decoy other.jpg to steer toward someone
.venv/bin/python -m shield.video in.mp4 out.mp4 --strength balanced
.venv/bin/python -m shield.image --calibrate                            # same-person vs different-person scores
.venv/bin/python -m shield.lab original.png shielded.png target.jpg outdir/   # face-swap attack on our own output
.venv/bin/python -m tests.test_align && .venv/bin/python -m tests.test_audio_grad
```

Chrome extension: `chrome://extensions` → Developer mode → Load unpacked → `extension/`. With the server
running, any photo or audio file you attach on any site is shielded first and swapped into the upload.
`http://localhost:8000/ext/test.html` is a plain upload page for trying it.

## How it works

Clone and swap tools do not look at you the way people do. They run your face or voice through an
*identity encoder* and keep a few hundred numbers: your fingerprint. Shroud attacks that step.

1. **Measure.** Compute the identity embedding `e0` of the original with the encoders these tools use:
   ArcFace (w600k_r50, from InsightFace's buffalo_l, the identity model of most open-source face
   swappers) and FaceNet (VGGFace2) for faces; SpeechBrain ECAPA-TDNN for voices.
2. **Nudge.** Projected gradient descent on the file itself: find a perturbation δ with ‖δ‖∞ ≤ ε that
   minimises `cos(f(x+δ), e0)`, summed over the encoder ensemble. With a decoy, it also maximises
   `cos(f(x+δ), e_decoy)`, so fakes come out as the decoy.
3. **Make it survive.** Every step averages the loss over random transforms (EOT): resize 0.75–1.25×,
   small shifts, light blur, a YUV 4:2:0 chroma round-trip for images and video; time shifts, low-pass and
   noise for audio. Scores are re-measured from the saved file, and for video from frames decoded back
   out of the encoded mp4, with faces re-detected.

Details worth knowing:

- Face alignment is differentiable end to end: MTCNN landmarks → Umeyama similarity transform (numpy) →
  `affine_grid` / `grid_sample` crop from the full-resolution tensor. δ lives in full-resolution pixels
  and only on the face region.
- ArcFace is converted from ONNX to a torch module with onnx2torch so gradients flow; no onnxruntime.
- Voice: δ is optimised at the file's own sample rate and resampled to 16 kHz inside the loss.
- Video: frames are piped through ffmpeg as raw RGB; δ is warm-started from the previous frame.
- Metrics (PSNR, SSIM, SNR), transforms and alignment are a few lines of torch each. No OpenCV, scipy,
  kornia or scikit-image.

## Measured on this build

MEASURED_TABLE

## The attack lab

The page can run the face swapper used by most open-source deepfake tools (inswapper_128, converted to
torch) twice on the same target: once with the identity read from your original photo, once from the
shielded one. It shows both fakes and how closely each matches the real you.

## Limits, plainly

- Shroud targets specific encoders (ArcFace, FaceNet, ECAPA). Tools built on them are affected most.
  Transfer to other or commercial systems is not guaranteed and we have not measured it.
- Heavy compression, rescaling, denoising or deliberate "purification" can weaken the layer.
  The JPEG q90 and encoded-video numbers on the certificate show how much survives mild cases.
- Stronger settings are more visible. "Invisible" is honest at Light, mostly honest at Balanced, and
  a trade at Strong; the certificate reports PSNR/SSIM or SNR for every file so you can judge.
- It protects the file you shield, not photos of you that are already out there.
- It raises the cost of cloning you. It is not a guarantee.

## Layout

```
shield/     common.py (device, metrics, EOT)  audio.py  image.py  video.py  lab.py
backend/    app.py (real server)  mock_app.py (UI development only)  contract.md (the API)
frontend/   index.html  style.css  app.js  guilloche.js  fonts/     hand-written, no build step
extension/  Manifest V3, vanilla JS
scripts/    setup.sh  run.sh  fetch_assets.py
tests/      alignment vs. a reference crop, gradient reaches the raw waveform
```

Dependencies are listed and justified line by line in `requirements.txt`.
