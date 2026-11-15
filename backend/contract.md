# Shroud API contract (locked)

One process, one port. The mock server (`mock_app.py`) and the real server (`app.py`) implement exactly this.
The only difference: every mock response carries `"mock": true`, and the UI shows a MOCK DATA banner when it sees it.

## GET /api/health
```json
{ "status": "ok", "device": "cuda", "models": ["ecapa-tdnn", "facenet-vggface2", "arcface-w600k-r50", "mtcnn"], "lab": true }
```

## POST /api/protect/{audio|image|video}
Multipart form: `file` (required), `strength` = `light` | `balanced` | `strong` (default `balanced`),
`decoy` (optional file: a face photo / voice clip of the identity to steer toward),
`shield_audio` (video only, `"1"` to also run the voice shield on the audio track).
```json
{ "job_id": "a3f9c2d41b7e" }
```
Errors: 400 `{ "detail": "plain-language message" }`.

## GET /api/jobs/{id}
Always the same shape. Fields that are not known yet are `null`.

Queued:
```json
{ "job_id": "a3f9c2d41b7e", "kind": "image", "status": "queued", "phase": "queued", "progress": 0.0,
  "step": 0, "total_steps": 0, "current_similarity": null, "result_url": null, "original_url": null,
  "scores": null, "elapsed_s": 0.0, "error": null, "serial": 417, "strength": "balanced", "warnings": [] }
```

Running (polled every ~300 ms; `current_similarity` falls as PGD proceeds; `scores.before` and
`scores.threshold` are filled as soon as the original has been measured):
```json
{ "job_id": "a3f9c2d41b7e", "kind": "image", "status": "running", "phase": "shielding", "progress": 0.53,
  "step": 64, "total_steps": 120, "current_similarity": 0.31, "result_url": null,
  "original_url": "/api/files/a3f9c2d41b7e/original.png",
  "scores": { "before": 1.0, "after": null, "threshold": 0.3, "quality": null, "per_encoder": null },
  "elapsed_s": 4.2, "error": null, "serial": 417, "strength": "balanced", "warnings": [] }
```
`phase` is one of `queued`, `preparing`, `shielding`, `encoding`, `verifying`, `done`, `error`.

Done — image:
```json
{ "job_id": "a3f9c2d41b7e", "kind": "image", "status": "done", "phase": "done", "progress": 1.0,
  "step": 120, "total_steps": 120, "current_similarity": 0.12,
  "result_url": "/api/files/a3f9c2d41b7e/shielded.png", "original_url": "/api/files/a3f9c2d41b7e/original.png",
  "scores": {
    "before": 1.0, "after": 0.12, "threshold": 0.3,
    "quality": { "psnr_db": 38.4, "ssim": 0.967 },
    "per_encoder": { "arcface": { "before": 1.0, "after": 0.09, "threshold": 0.3 },
                     "facenet": { "before": 1.0, "after": 0.17, "threshold": 0.45 } },
    "after_jpeg_q90": 0.16, "faces": 1, "epsilon": 0.031 },
  "elapsed_s": 9.8, "error": null, "serial": 417, "strength": "balanced", "warnings": [] }
```
`before`/`after` are the headline numbers (the highest-weighted encoder: ArcFace for faces, ECAPA for voice):
cosine similarity between the identity embedding of the file and that of the ORIGINAL. An untouched
copy scores 1.0; below `threshold` the encoder reads it as a different person.

Done — audio (`quality` is SNR instead of PSNR/SSIM):
```json
{ "kind": "audio", "status": "done", "result_url": "/api/files/<id>/shielded.wav", "original_url": "/api/files/<id>/original.wav",
  "scores": { "before": 1.0, "after": 0.14, "threshold": 0.25, "quality": { "snr_db": 31.2 },
              "per_encoder": { "ecapa": { "before": 1.0, "after": 0.14, "threshold": 0.25 } }, "epsilon": 0.004 },
  "...": "all other fields as above" }
```

Done — video (adds `per_frame`, measured on frames DECODED from the encoded mp4):
```json
{ "kind": "video", "status": "done", "result_url": "/api/files/<id>/shielded.mp4", "original_url": "/api/files/<id>/original.mp4",
  "scores": { "before": 1.0, "after": 0.21, "threshold": 0.3, "quality": { "psnr_db": 37.1, "ssim": 0.958 },
              "per_encoder": { "arcface": { "before": 1.0, "after": 0.21, "threshold": 0.3 },
                               "facenet": { "before": 1.0, "after": 0.33, "threshold": 0.45 } },
              "per_frame": [0.24, 0.19, 0.22, null, 0.2], "frames": 150, "fps": 30.0, "faces": 148,
              "audio": null },
  "...": "all other fields as above" }
```
`per_frame[i]` is `null` where no face was detected in the decoded frame. `scores.audio` is the audio
score object (same shape as an audio job's `scores`) when `shield_audio` was set.

Error:
```json
{ "job_id": "a3f9c2d41b7e", "kind": "image", "status": "error", "phase": "error", "progress": 0.0, "step": 0,
  "total_steps": 0, "current_similarity": null, "result_url": null, "original_url": null, "scores": null,
  "elapsed_s": 0.4, "error": "No face found. Try a clearer, front-facing photo.", "serial": 417,
  "strength": "balanced", "warnings": [] }
```
Unknown id: 404 `{ "detail": "Unknown job." }`.

## GET /api/files/{job_id}/{name}
The original (normalised to a browser-friendly format at the processed size) and the shielded file.

## POST /api/lab/faceswap  (attack lab, real server only when `health.lab` is true)
Form: `job_id` of a finished image job, optional `target` image. Returns
```json
{ "target_url": "...", "swap_original_url": "...", "swap_shielded_url": "...",
  "similarity_original": 0.62, "similarity_shielded": 0.08, "threshold": 0.3 }
```
`similarity_*` = ArcFace cosine between each swapped face and the real person.
