#!/usr/bin/env python3
"""
ASLINAKLI • Image-only inference (CPU) for deepfake detection (file path or in-memory array).

What's new:
- predict_image_array(img_rgb: np.ndarray) -> dict  # in-memory, no disk writes
- predict_image(path) kept for compatibility
"""

import json, base64
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import timm
import torch
import mediapipe as mp
mp_face = mp.solutions.face_detection

# -----------------------
# Config / constants
# -----------------------
CONFIG_PATH  = Path("models/image_head_config.json")
CALIB_PATH   = Path("models/calibration.json")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BACKBONE = "xception"
IMG_SIZE = 299
DECISION_THRESHOLD = 0.5

TTA_FLIP = True
FACE_EXPAND = 0.20
QUALITY_LAPLACIAN_MIN = 90.0
LOW_CONF_EPS = 0.05

# Singleton
_model_singleton = None

# -----------------------
# Helpers
# -----------------------
def _to_tensor_norm(img_rgb: np.ndarray) -> torch.Tensor:
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # C,H,W
    return torch.from_numpy(x).float()

def _resize_center_square(img_rgb: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    if h < w:
        nh, nw = size, int(w * (size / h))
    else:
        nw, nh = size, int(h * (size / w))
    r = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = max(0, (nh - size) // 2), max(0, (nw - size) // 2)
    out = r[y0:y0 + size, x0:x0 + size]
    if out.shape[:2] != (size, size):
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return out

def _laplacian_var(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _face_crop(img_rgb: np.ndarray, detector, expand=FACE_EXPAND):
    h, w = img_rgb.shape[:2]
    res = detector.process(img_rgb)
    if not res or not res.detections:
        return None
    d = res.detections[0]
    bbox = d.location_data.relative_bounding_box
    x = int(bbox.xmin * w); y = int(bbox.ymin * h)
    bw = int(bbox.width * w); bh = int(bbox.height * h)
    ex = int(bw * expand); ey = int(bh * expand)
    x0 = max(0, x - ex); y0 = max(0, y - ey)
    x1 = min(w, x + bw + ex); y1 = min(h, y + bh + ey)
    if x1 <= x0 or y1 <= y0: return None
    crop = img_rgb[y0:y1, x0:x1]
    if crop is None or crop.size == 0: return None
    return crop

def _maybe_flip(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.flip(img_rgb, 1)

def _to_b64(img_rgb: np.ndarray, q: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

def _overlay_cam(img_rgb: np.ndarray, cam_map: np.ndarray) -> np.ndarray:
    cam = cam_map.copy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam = (cam * 255).astype(np.uint8)
    cam = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heat = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    out = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.6, heat, 0.4, 0)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def _gradcam_map_for_feats(feats: torch.Tensor) -> np.ndarray:
    grads = feats.grad
    w = grads.mean(dim=(2,3), keepdim=True)
    cam = (w * feats).sum(dim=1, keepdim=True)
    cam = cam.detach().cpu().numpy()[0,0]
    return np.maximum(cam, 0.0)

# -----------------------
# Model
# -----------------------
class ImageModel:
    def __init__(self):
        global BACKBONE, IMG_SIZE, DECISION_THRESHOLD

        # Read config
        if CONFIG_PATH.exists():
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            BACKBONE = cfg.get("backbone", BACKBONE)
            IMG_SIZE = int(cfg.get("input_size", IMG_SIZE))
            DECISION_THRESHOLD = float(cfg.get("threshold", DECISION_THRESHOLD))

        # Temperature
        self.T_image = 1.0
        if CALIB_PATH.exists():
            try:
                c = json.loads(CALIB_PATH.read_text())
                self.T_image = float(c.get("T_image", 1.0))
            except Exception:
                self.T_image = 1.0

        self.device = "cpu"
        self.model = timm.create_model(BACKBONE, pretrained=False, num_classes=1)
        self.model.eval()
        self._timm = self.model
        self._det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        # Prefer final weights; otherwise use latest checkpoint
        weights_path = Path(f"models/{BACKBONE}_finetuned_images.pt")
        state_obj = None
        if weights_path.exists():
            state_obj = torch.load(weights_path, map_location="cpu")
        else:
            latest = self._find_latest_ckpt(BACKBONE)
            if latest is None:
                raise FileNotFoundError(
                    f"No inference weights found. Expected {weights_path} or a checkpoint under checkpoints/images/{BACKBONE}/")
            state_obj = torch.load(latest, map_location="cpu")
        self._load_state_into_model(self.model, state_obj)

    @staticmethod
    def _find_latest_ckpt(backbone: str) -> Optional[Path]:
        ckpt_dir = Path(f"checkpoints/images/{backbone}")
        if not ckpt_dir.exists(): return None
        files = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None

    @staticmethod
    def _load_state_into_model(model: torch.nn.Module, state_obj):
        if isinstance(state_obj, dict) and "model_state" in state_obj:
            sd = state_obj["model_state"]
        else:
            sd = state_obj
        try:
            model.load_state_dict(sd, strict=True)
        except Exception:
            model.load_state_dict(sd, strict=False)

    def _forward_logits_and_feats(self, x_tensor: torch.Tensor):
        feats = self._timm.forward_features(x_tensor)   # [1,C,h,w]
        feats.retain_grad()
        pooled = self._timm.global_pool(feats)          # [1,C]
        logits = self._timm.get_classifier()(pooled)    # [1,1]
        return logits, feats

    def _prob_from_logit(self, logit: float) -> float:
        z = float(logit) / max(1e-6, self.T_image)
        return 1.0 / (1.0 + np.exp(-z))

    def _predict_single_rgb(self, img_rgb: np.ndarray, want_cam: bool = True) -> Dict[str, Any]:
        # Preprocess (face-first, quality gate)
        face = _face_crop(img_rgb, self._det, expand=FACE_EXPAND)
        crop = face if (face is not None and _laplacian_var(face) >= QUALITY_LAPLACIAN_MIN) else img_rgb
        crop = _resize_center_square(crop, IMG_SIZE)

        # TTA
        inputs = [crop]
        if TTA_FLIP:
            inputs.append(_maybe_flip(crop))

        probs: List[float] = []; cam_map = None
        for idx, im in enumerate(inputs):
            x = _to_tensor_norm(im)[None, ...]
            with torch.enable_grad():
                x = x.float()
                self._timm.zero_grad(set_to_none=True)
                logits, feats = self._forward_logits_and_feats(x)
                prob = self._prob_from_logit(float(logits.squeeze(1).item()))
                probs.append(prob)
                if want_cam and idx == 0:
                    logits.backward(retain_graph=True)
                    cam_map = _gradcam_map_for_feats(feats)

        p = float(np.mean(probs))
        label = "FAKE" if p >= DECISION_THRESHOLD else "REAL"
        low_conf = abs(p - DECISION_THRESHOLD) <= LOW_CONF_EPS

        explain = []
        if cam_map is not None:
            cam_overlay = _overlay_cam(crop, cam_map)
            explain.append({"thumb_b64": _to_b64(crop), "cam_b64": _to_b64(cam_overlay)})

        return {
            "type": "image",
            "fake_prob": p,
            "real_prob": 1.0 - p,
            "label": label,
            "low_confidence": bool(low_conf),
            "calibration_T": float(self.T_image),
            "explain": explain,
        }

    # ---------- Public APIs ----------
    def predict_path(self, image_path: Path) -> Dict[str, Any]:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return {"type":"image","path":str(image_path),"error":"Could not read image"}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = self._predict_single_rgb(rgb, want_cam=True)
        out["path"] = str(image_path)
        return out

    def predict_rgb(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        if not isinstance(img_rgb, np.ndarray) or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            return {"type": "image", "error": "predict_rgb expects an RGB uint8 ndarray (H,W,3)."}
        out = self._predict_single_rgb(img_rgb, want_cam=True)
        # in-memory: path is informational only
        out["path"] = ""
        return out

# -----------------------
# Public functions
# -----------------------
def load_model():
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = ImageModel()
    return _model_singleton

def predict_image(image_path: str) -> Dict[str, Any]:
    m = load_model()
    return m.predict_path(Path(image_path))

def predict_image_array(img_rgb: np.ndarray) -> Dict[str, Any]:
    m = load_model()
    return m.predict_rgb(img_rgb)
