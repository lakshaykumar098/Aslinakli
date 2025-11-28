# dash_app/services/image_model.py
from __future__ import annotations
import base64, re
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np

from utils.images.infer_image import load_model  # warms singleton
from utils.images.infer_image import predict_image_array as _predict_array

_MODEL = load_model()  # pre-load model once

_ALLOWED_EXTS = {"jpg", "jpeg", "png", "webp", "bmp"}
_MIME_TO_EXT = {
    "image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png",
    "image/webp": "webp", "image/bmp": "bmp",
}

def _parse_data_url(contents: str) -> Tuple[Optional[str], Optional[bytes]]:
    """
    Accepts data URLs ('data:image/...;base64,AAAA') or raw base64.
    Returns (mime, rawbytes).
    """
    if not contents:
        return None, None
    if contents.startswith("data:"):
        m = re.match(r"^data:([^;]+);base64,(.*)$", contents)
        if not m:
            return None, None
        mime, b64 = m.group(1).strip(), m.group(2)
    else:
        mime, b64 = None, contents
    try:
        return mime, base64.b64decode(b64, validate=True)
    except Exception:
        return None, None

def _infer_ext(filename: str, mime: Optional[str]) -> Optional[str]:
    # best-effort validation for user feedback; we don't write files anymore
    ext = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "").replace("jpeg", "jpg")
    if mime and mime in _MIME_TO_EXT:
        ext = _MIME_TO_EXT[mime]
    return ext if ext in _ALLOWED_EXTS else None

def _decode_image_to_rgb(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def _normalize(res: Dict[str, Any], want_explain: bool, filename: str) -> Dict[str, Any]:
    out = {
        "type": res.get("type", "image"),
        # in-memory marker; keep filename for UI “File:” line if you show it
        "path": f"memory:{filename}" if filename else "",
        "label": res.get("label", ""),
        "fake_prob": float(res.get("fake_prob", 0.0) or 0.0),
        "real_prob": float(res.get("real_prob", 0.0) or 0.0),
        "low_confidence": bool(res.get("low_confidence", False)),
        "calibration_T": float(res.get("calibration_T", 1.0) or 1.0),
        "explain": res.get("explain", []) if want_explain else [],
    }
    if "error" in res:
        out["error"] = str(res["error"])
    if "loaded_from" in res:
        out["loaded_from"] = res["loaded_from"]
    return out

def predict_from_upload(contents: str, filename: str, explain: bool = False) -> Dict[str, Any]:
    if not contents or not filename:
        return {"type":"image","path": filename or "", "error":"No file data"}

    mime, raw = _parse_data_url(contents)
    if raw is None:
        return {"type":"image","path": filename, "error": "Invalid upload data (base64 parse failed)"}

    # Validate extension/MIME for nicer errors (we still run in-memory)
    ext = _infer_ext(filename, mime)
    if not ext:
        return {"type":"image","path": filename, "error": f"Unsupported file type. Allowed: {sorted(_ALLOWED_EXTS)}"}

    try:
        rgb = _decode_image_to_rgb(raw)               # in-memory decode
        res = _predict_array(rgb)                     # in-memory inference
        return _normalize(res, want_explain=explain, filename=filename)
    except Exception as e:
        return {"type":"image","path": filename, "error": f"Inference error: {e}"}
