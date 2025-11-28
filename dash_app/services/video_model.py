# dash_app/services/video_model.py
from __future__ import annotations
import base64, sys, importlib.util, tempfile, os
from pathlib import Path
from typing import Dict, Any

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
INFER_PATH = PROJECT_ROOT / "utils" / "videos" / "infer_video.py"  # <-- video-only

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Strictly video-only imports
try:
    from utils.videos.infer_video import load_model as _load_model, predict_video as _predict_video
except Exception:
    if not INFER_PATH.exists():
        raise RuntimeError(f"Cannot find inference module at {INFER_PATH}")
    spec = importlib.util.spec_from_file_location("infer_video_local", INFER_PATH)
    infer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(infer)
    _load_model = infer.load_model
    _predict_video = infer.predict_video

_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL

def _bytes_from_contents(contents: str) -> bytes:
    _, b64data = contents.split(",", 1)
    return base64.b64decode(b64data)

class _TempPath:
    def __init__(self, data: bytes, suffix: str):
        self.data = data
        self.suffix = suffix if suffix else ""
        self.path = None
    def __enter__(self) -> str:
        fd, tmp = tempfile.mkstemp(suffix=self.suffix)
        os.close(fd)
        with open(tmp, "wb") as f:
            f.write(self.data)
        self.path = tmp
        return self.path
    def __exit__(self, exc_type, exc, tb):
        try:
            if self.path and os.path.exists(self.path):
                os.remove(self.path)
        except Exception:
            pass

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".3gp"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def predict_from_upload(contents: str, filename: str, explain: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Accepts dcc.Upload contents + filename and returns model result.
    Video-only. Always returns a dict (never raises).

    Required by renderer:
      - fake_prob (or prob_fake), real_prob (optional)
      - label: "FAKE" | "REAL"
      - type: "video"
    Optional keys (only if explain=True):
      - window_probs, windows (for per-window scores)
      - attn_weights, explain (list), path, low_confidence
    """
    res: Dict[str, Any] = {}

    try:
        get_model()
        ext = (Path(filename).suffix or "").lower() if filename else ""
        data = _bytes_from_contents(contents)

        is_video = (ext in VIDEO_EXTS) or (isinstance(contents, str) and contents.startswith("data:video/"))
        if not is_video:
            return {"type": "unknown", "error": f"Unsupported file type for video model: {ext or 'unknown'}"}

        suffix = ext if ext else ".mp4"
        with _TempPath(data, suffix=suffix) as tmp_path:
            # Try to pass explain; fall back if infer fn doesn't accept it
            try:
                res = _predict_video(tmp_path, explain=explain)
            except TypeError:
                res = _predict_video(tmp_path)

        # Normalize/annotate result for the renderer
        if not isinstance(res, dict):
            res = {"error": "Invalid result from video predictor (expected dict)"}

        res.setdefault("type", "video")
        res["path"] = filename or res.get("path", "")

        # Ensure minimum fields exist so UI doesn't crash
        if "fake_prob" in res or "prob_fake" in res:
            fp = float(res.get("fake_prob", res.get("prob_fake", 0.0)))
            rp = float(res.get("real_prob", 1.0 - fp))
            res["fake_prob"] = fp
            res["real_prob"] = rp
            res.setdefault("label", "FAKE" if fp >= rp else "REAL")

        # [PATCH] Respect the UI toggle strictly
        if not explain:
            # Drop heavy fields when explanation is OFF
            for k in ("explain", "attn_weights", "window_probs", "windows"):
                res.pop(k, None)

        return res

    except Exception as e:
        # Always return an error dict instead of raising
        return {"type": "video", "error": f"{type(e).__name__}: {e}"}
