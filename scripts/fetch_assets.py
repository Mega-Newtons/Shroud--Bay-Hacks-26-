"""One-time download of model weights and fonts (stdlib only). Safe to re-run; skips files that exist.

Nothing here runs at app runtime: the app only ever reads from models/ and frontend/fonts/.
"""
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FONTS = ROOT / "frontend" / "fonts"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have  {dest.relative_to(ROOT)}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = get(url)
    except Exception as e:  # report and keep going so one blocked host doesn't stop the rest
        print(f"  FAIL  {dest.relative_to(ROOT)}  <- {url}  ({e})")
        return False
    dest.write_bytes(data)
    print(f"  got   {dest.relative_to(ROOT)}  ({len(data) / 1e6:.1f} MB)")
    return True


def fonts() -> None:
    """Latin subsets of the three families, straight from Google Fonts, saved locally as woff2."""
    specs = {
        "newsreader": "family=Newsreader:ital,opsz,wght@0,6..72,300..700;1,6..72,300..700",
        "plex-sans": "family=IBM+Plex+Sans:wght@400;500;600",
        "plex-mono": "family=IBM+Plex+Mono:wght@400;500",
    }
    for name, q in specs.items():
        try:
            css = get(f"https://fonts.googleapis.com/css2?{q}&display=swap").decode()
        except Exception as e:
            print(f"  FAIL  font css {name} ({e})")
            continue
        # Each @font-face block is preceded by a /* subset */ comment; keep only "latin".
        for subset, block in re.findall(r"/\* ([\w-]+) \*/\s*@font-face\s*{(.*?)}", css, re.S):
            if subset != "latin":
                continue
            style = re.search(r"font-style:\s*(\w+)", block).group(1)
            weight = re.search(r"font-weight:\s*([\d ]+)", block).group(1).strip().replace(" ", "-")
            url = re.search(r"url\((https://[^)]+\.woff2)\)", block).group(1)
            fetch(url, FONTS / f"{name}-{style}-{weight}.woff2")


def models(lab: bool) -> None:
    # FaceNet (InceptionResnetV1, VGGFace2). Placed where facenet-pytorch looks when TORCH_HOME=models/torch.
    fetch("https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt",
          MODELS / "torch" / "checkpoints" / "20180402-114759-vggface2.pt")
    # ArcFace w600k_r50 lives inside InsightFace's buffalo_l release zip; keep only that one file.
    arc = MODELS / "w600k_r50.onnx"
    if arc.exists():
        print(f"  have  {arc.relative_to(ROOT)}")
    else:
        try:
            z = zipfile.ZipFile(io.BytesIO(get("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")))
            arc.write_bytes(z.read("w600k_r50.onnx"))
            print(f"  got   {arc.relative_to(ROOT)}")
        except Exception as e:
            print(f"  FAIL  w600k_r50.onnx ({e})")
    # ECAPA-TDNN speaker encoder weights (SpeechBrain, VoxCeleb). Only the embedding model is needed.
    fetch("https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt",
          MODELS / "ecapa" / "embedding_model.ckpt")
    if lab:  # Attack lab: the identity-conditioned face swapper used by most open-source swap tools.
        fetch("https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx",
              MODELS / "inswapper_128.onnx")


if __name__ == "__main__":
    print("fonts:")
    fonts()
    print("models:")
    models(lab="--no-lab" not in sys.argv)
