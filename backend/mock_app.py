"""MOCK server for UI development only. Follows backend/contract.md exactly, but every number is fake.

Every response carries "mock": true and the UI shows a MOCK DATA banner when it sees that flag.
The demo runs backend/app.py, never this file.
    python -m uvicorn backend.mock_app:app --port 8000
"""
import math
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "backend" / "jobs_mock"
JOBS: dict[str, dict] = {}
SERIAL = [416]
DURATION = 8.0  # seconds a fake job takes
THRESH = {"audio": 0.25, "image": 0.30, "video": 0.30}
EXT = {"audio": "wav", "image": "png", "video": "mp4"}

app = FastAPI(title="Shroud MOCK")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def run(job: dict, fail: bool) -> None:
    t0 = time.time()
    total = {"light": 60, "balanced": 120, "strong": 200}[job["strength"]]
    job.update(status="running", phase="preparing")
    time.sleep(0.6)
    if fail:
        job.update(status="error", phase="error", elapsed_s=round(time.time() - t0, 1),
                   error="No face found. Try a clearer, front-facing photo.")
        return
    thr = THRESH[job["kind"]]
    job.update(phase="shielding", total_steps=total, original_url=job["_url"],
               scores={"before": 1.0, "after": None, "threshold": thr, "quality": None, "per_encoder": None})
    while (t := time.time() - t0 - 0.6) < DURATION:
        p = t / DURATION
        sim = 0.12 + 0.88 * math.exp(-4.5 * p) + 0.015 * math.sin(t * 9)  # fake, plausible decay
        job.update(progress=round(p, 3), step=int(p * total), current_similarity=round(sim, 3),
                   elapsed_s=round(time.time() - t0, 1))
        time.sleep(0.15)
    after = 0.13
    scores = {"before": 1.0, "after": after, "threshold": thr, "epsilon": 0.031}
    if job["kind"] == "audio":
        scores |= {"quality": {"snr_db": 31.2}, "per_encoder": {"ecapa": {"before": 1.0, "after": after, "threshold": thr}}}
    else:
        scores |= {"quality": {"psnr_db": 38.4, "ssim": 0.967},
                   "per_encoder": {"arcface": {"before": 1.0, "after": after, "threshold": 0.30},
                                   "facenet": {"before": 1.0, "after": 0.21, "threshold": 0.45}}, "faces": 1}
    if job["kind"] == "image":
        scores["after_jpeg_q90"] = 0.17
    if job["kind"] == "video":
        scores |= {"per_frame": [None if i == 37 else round(0.18 + 0.08 * math.sin(i / 6) + 0.03 * math.sin(i * 1.7), 3)
                                 for i in range(120)], "frames": 120, "fps": 30.0, "faces": 119, "audio": None}
    job.update(status="done", phase="done", progress=1.0, step=total, current_similarity=after,
               result_url=job["_url"], scores=scores, elapsed_s=round(time.time() - t0, 1))


@app.get("/api/health")
def health():
    return {"status": "ok", "device": "mock", "models": [], "lab": False, "mock": True}


@app.post("/api/protect/{kind}")
async def protect(kind: str, file: UploadFile = File(...), strength: str = Form("balanced"),
                  decoy: UploadFile | None = File(None), shield_audio: str = Form("0")):
    if kind not in THRESH:
        raise HTTPException(400, "Unknown media type.")
    if strength not in ("light", "balanced", "strong"):
        raise HTTPException(400, "Unknown strength.")
    jid = uuid.uuid4().hex[:12]
    d = JOBS_DIR / jid
    d.mkdir(parents=True, exist_ok=True)
    # The mock returns the upload itself as the "result", under its own extension so the browser can show it.
    ext = Path(file.filename or "").suffix.lstrip(".") or EXT[kind]
    (d / f"upload.{ext}").write_bytes(await file.read())
    SERIAL[0] += 1
    job = dict(job_id=jid, kind=kind, status="queued", phase="queued", progress=0.0, step=0, total_steps=0,
               current_similarity=None, result_url=None, original_url=None, scores=None, elapsed_s=0.0,
               error=None, serial=SERIAL[0], strength=strength, warnings=[], _url=f"/api/files/{jid}/upload.{ext}")
    JOBS[jid] = job
    threading.Thread(target=run, args=(job, "fail" in (file.filename or "")), daemon=True).start()
    return {"job_id": jid, "mock": True}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "Unknown job.")
    return {k: v for k, v in JOBS[jid].items() if not k.startswith("_")} | {"mock": True}


@app.get("/api/files/{jid}/{name}")
def files(jid: str, name: str):
    p = JOBS_DIR / jid / Path(name).name
    if not p.is_file():
        raise HTTPException(404, "No such file.")
    return FileResponse(p)


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
