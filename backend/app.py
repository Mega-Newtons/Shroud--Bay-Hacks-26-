"""Shroud server: the real one. One process, one port, serves the API (backend/contract.md) and frontend/.

    python -m uvicorn backend.app:app --port 8000

Every number returned here comes from running the models in shield/. Nothing is hardcoded;
if a job fails, the error is reported as-is.
"""
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from shield import audio, image, video
from shield.common import DEVICE, ShieldError

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "backend" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
SERIAL_FILE = JOBS_DIR / "serial.txt"
OUT_EXT = {"audio": "wav", "image": "png", "video": "mp4"}
STRENGTHS = ("light", "balanced", "strong")

JOBS: dict[str, dict] = {}
GPU_LOCK = threading.Lock()      # one shielding job at a time; others wait as "queued"
SERIAL_LOCK = threading.Lock()
LOADED: list[str] = []

try:                             # attack lab is optional (needs models/inswapper_128.onnx)
    from shield import lab
except Exception:                # noqa: BLE001 - lab is a stretch feature; the core must still start
    lab = None

app = FastAPI(title="Shroud")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def load_models() -> None:
    """Load every model once, before the first request."""
    audio.load_models(); LOADED.append("ecapa-tdnn")
    image.load_models(); LOADED.extend(["mtcnn", "arcface-w600k-r50", "facenet-vggface2"])
    if lab is not None and lab.available():
        lab.load_models(); LOADED.append("inswapper-128")


def next_serial() -> int:
    with SERIAL_LOCK:
        n = int(SERIAL_FILE.read_text()) + 1 if SERIAL_FILE.exists() else 417
        SERIAL_FILE.write_text(str(n))
        return n


def run(job: dict, src: Path, decoy: Path | None, shield_audio: bool) -> None:
    jid, kind = job["job_id"], job["kind"]
    d = JOBS_DIR / jid
    ext = OUT_EXT[kind]
    dst, orig = d / f"shielded.{ext}", d / f"original.{ext}"
    with GPU_LOCK:
        t0 = time.time()
        job.update(status="running", phase="preparing", _t0=t0)

        def on_progress(p: dict) -> None:
            # p comes straight from the PGD loop: phase, step, total_steps, current_similarity, before, threshold
            step, total = p.get("step", job["step"]), p.get("total_steps", job["total_steps"])
            job.update(phase=p.get("phase", job["phase"]), step=step, total_steps=total,
                       progress=round(min(0.99, step / total), 3) if total else job["progress"],
                       elapsed_s=round(time.time() - t0, 1))
            if p.get("current_similarity") is not None:
                job["current_similarity"] = round(float(p["current_similarity"]), 3)
            if p.get("before") is not None:
                job["scores"] = {"before": round(float(p["before"]), 3), "after": None, "threshold": p.get("threshold"),
                                 "quality": None, "per_encoder": None}
                if orig.exists():
                    job["original_url"] = f"/api/files/{jid}/{orig.name}"

        try:
            if kind == "audio":
                out = audio.shield_audio(src, dst, strength=job["strength"], decoy=decoy, on_progress=on_progress, original_out=orig)
            elif kind == "image":
                out = image.shield_image(src, dst, strength=job["strength"], decoy=decoy, on_progress=on_progress, original_out=orig)
            else:
                out = video.shield_video(src, dst, strength=job["strength"], decoy=decoy, on_progress=on_progress,
                                         original_out=orig, shield_audio_track=shield_audio)
            job.update(status="done", phase="done", progress=1.0, scores=out["scores"], warnings=out.get("warnings", []),
                       current_similarity=out["scores"]["after"], result_url=f"/api/files/{jid}/{dst.name}",
                       original_url=f"/api/files/{jid}/{orig.name}", elapsed_s=round(time.time() - t0, 1))
            if job["total_steps"]:
                job["step"] = min(job["step"], job["total_steps"])
        except ShieldError as e:      # expected, user-facing ("No face found…")
            job.update(status="error", phase="error", error=str(e), elapsed_s=round(time.time() - t0, 1))
        except Exception as e:        # noqa: BLE001 - unexpected: show it, never paper over it
            traceback.print_exc()
            job.update(status="error", phase="error", error=f"Shielding failed: {e}", elapsed_s=round(time.time() - t0, 1))


@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "models": LOADED, "lab": "inswapper-128" in LOADED}


async def save(upload: UploadFile, d: Path, stem: str) -> Path:
    ext = Path(upload.filename or "").suffix.lower()[:8] or ".bin"
    p = d / f"{stem}{ext}"
    p.write_bytes(await upload.read())
    return p


@app.post("/api/protect/{kind}")
async def protect(kind: str, file: UploadFile = File(...), strength: str = Form("balanced"),
                  decoy: UploadFile | None = File(None), shield_audio: str = Form("0")):
    if kind not in OUT_EXT:
        raise HTTPException(400, "Unknown media type. Use audio, image or video.")
    if strength not in STRENGTHS:
        raise HTTPException(400, "Unknown strength. Use light, balanced or strong.")
    jid = uuid.uuid4().hex[:12]
    d = JOBS_DIR / jid
    d.mkdir(parents=True)
    src = await save(file, d, "input")
    if src.stat().st_size == 0:
        raise HTTPException(400, "That file is empty.")
    decoy_path = await save(decoy, d, "decoy") if decoy is not None and decoy.filename else None
    job = dict(job_id=jid, kind=kind, status="queued", phase="queued", progress=0.0, step=0, total_steps=0,
               current_similarity=None, result_url=None, original_url=None, scores=None, elapsed_s=0.0,
               error=None, serial=next_serial(), strength=strength, warnings=[])
    JOBS[jid] = job
    threading.Thread(target=run, args=(job, src, decoy_path, shield_audio == "1"), daemon=True).start()
    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "Unknown job.")
    job = JOBS[jid]
    if job["status"] == "running" and "_t0" in job:      # keep the clock moving between progress callbacks
        job["elapsed_s"] = round(time.time() - job["_t0"], 1)
    return {k: v for k, v in job.items() if not k.startswith("_")}


@app.get("/api/files/{jid}/{name}")
def files(jid: str, name: str):
    p = JOBS_DIR / Path(jid).name / Path(name).name
    if not p.is_file():
        raise HTTPException(404, "No such file.")
    return FileResponse(p, headers={"Cache-Control": "no-store"})


@app.post("/api/lab/faceswap")
async def faceswap(job_id: str = Form(...), target: UploadFile | None = File(None)):
    """Attack lab: swap the original and the shielded identity onto the same target face."""
    if "inswapper-128" not in LOADED:
        raise HTTPException(400, "The face swapper isn't installed on this server.")
    job = JOBS.get(job_id)
    if not job or job["kind"] != "image" or job["status"] != "done":
        raise HTTPException(400, "Shield a photo first.")
    d = JOBS_DIR / job_id
    target_path = await save(target, d, "lab_target_in") if target is not None and target.filename else None
    try:
        with GPU_LOCK:
            out = lab.faceswap(d / "original.png", d / "shielded.png", target_path, d)
    except ShieldError as e:
        raise HTTPException(400, str(e))
    return {**out["scores"], **{k: f"/api/files/{job_id}/{Path(v).name}?t={int(time.time())}" for k, v in out["files"].items()}}


# The extension's upload test page (stands in for "any website" in the demo): http://localhost:8000/ext/test.html
app.mount("/ext", StaticFiles(directory=ROOT / "extension", html=True), name="extension")
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
