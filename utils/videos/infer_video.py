#!/usr/bin/env python3
"""
ASLINAKLI • Video-only inference (CPU) for deepfake detection.

Features:
- EfficientNet-B0 backbone (fine-tuned on frames)
- Attention pooling head (per video)
- Multi-window voting over K frames
- Test-time augmentation: horizontal flip
- Face-quality gating (Laplacian variance)
- Temperature scaling for video probs (models/calibration.json: {"T_video": ...})
- Attention export + optional Grad-CAM thumbnails for explanations
"""

import json, math, base64, cv2, numpy as np
from pathlib import Path
import torch, torch.nn as nn
import torch.nn.functional as F
import timm
import mediapipe as mp

# --------- paths ----------
BACKBONE_WEIGHTS = Path("models/effb0_finetuned_frames.pt")
HEAD_WEIGHTS     = Path("models/video_attn_head.pt")
HEAD_CONFIG      = Path("models/video_head_config.json")
CALIB_PATH       = Path("models/calibration.json")  # {"T_video": 1.0}

# --------- preprocessing ----------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224
FACE_EXPAND = 0.20

# --------- decision / voting ----------
DECISION_THRESHOLD = 0.5        # fixed downstream rule on calibrated prob
NUM_WINDOWS = 3                  # phases over same video
WINDOW_PHASES = [0.0, 0.33, 0.66]
TTA_FLIP = True
QUALITY_LAPLACIAN_MIN = 90.0
LOW_CONF_EPS = 0.05

# --------- explainability ----------
EXPLAIN_TOPK = 3
ENABLE_EXPLAIN = True

_mp_face = mp.solutions.face_detection

# =================== MODELS ===================

class EffB0FeatureExtractor(nn.Module):
    def __init__(self, weights_path: Path):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=1)
        state = torch.load(str(weights_path), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)     # [B,C,H,W]
        pooled = self.model.global_pool(feats)     # [B,C]
        return pooled                              # [B,D]

class AttnPoolHead(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int = 256, mlp_dim: int = 768):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, 1)
        )
    def forward(self, E, return_weights: bool=False):  # E: [B,K,D]
        scores = self.attn(E)                      # [B,K,1]
        weights = torch.softmax(scores, dim=1)     # [B,K,1]
        pooled = torch.sum(weights * E, dim=1)     # [B,D]
        logit = self.mlp(pooled).squeeze(1)        # [B]
        if return_weights:
            return logit, weights.squeeze(-1)      # [B], [B,K]
        return logit

# --------- Grad-CAM (hookless; safe for timm) ----------
def _norm01(t: torch.Tensor) -> torch.Tensor:
    t = t - t.min()
    return t / (t.max() + 1e-8)

def _gradcam_map_for_feats(timm_model, head_mlp, x_tensor: torch.Tensor) -> torch.Tensor:
    """
    Build Grad-CAM map from conv features without persistent hooks.
    x_tensor: [1,3,H,W] with requires_grad=True
    Returns: [H,W] in [0,1]
    """
    timm_model.zero_grad(set_to_none=True)
    head_mlp.zero_grad(set_to_none=True)

    feats = timm_model.forward_features(x_tensor)   # [1,C,h,w]
    feats.retain_grad()

    pooled = timm_model.global_pool(feats)          # [1,C]
    score  = head_mlp(pooled).squeeze(1)            # [1]
    score.backward()

    grads = feats.grad                               # [1,C,h,w]
    weights = grads.mean(dim=(2,3), keepdim=True)    # [1,C,1,1]
    cam = (weights * feats).sum(dim=1, keepdim=True) # [1,1,h,w]
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=x_tensor.shape[-2:], mode="bilinear", align_corners=False)[0,0]
    return _norm01(cam)

# =================== PRE/POST ===================

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

def _face_crop(img_rgb: np.ndarray, detector, expand=FACE_EXPAND):
    h, w = img_rgb.shape[:2]
    res = detector.process(img_rgb)
    if not res.detections:
        return None
    d = res.detections[0]
    bbox = d.location_data.relative_bounding_box
    x = int(bbox.xmin * w); y = int(bbox.ymin * h)
    bw = int(bbox.width * w); bh = int(bbox.height * h)
    ex = int(bw * expand);    ey = int(bh * expand)
    x0 = max(0, x - ex); y0 = max(0, y - ey)
    x1 = min(w, x + bw + ex); y1 = min(h, y + bh + ey)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img_rgb[y0:y1, x0:x1]
    if crop is None or crop.size == 0:
        return None
    return crop

def _laplacian_var(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _read_video_frame_rgb(vpath: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(vpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def _sample_video_indices(total_frames: int, K: int, phase: float = 0.0):
    if total_frames <= 0:
        return []
    if K >= total_frames:
        return list(range(total_frames))
    step = total_frames / K
    return [int((i + 0.5 + phase) * step) for i in range(K)]

def _maybe_flip(img: np.ndarray):
    return np.ascontiguousarray(img[:, ::-1, :])

def _to_b64(img_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _overlay_cam(img_rgb: np.ndarray, cam01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    heat = (cam01 * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)       # BGR
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 1.0, heat, alpha, 0.0)
    return overlay

# =================== LOADER ===================

class VideoModel:
    def __init__(self):
        self.device = "cpu"
        cfg = json.loads(Path(HEAD_CONFIG).read_text(encoding="utf-8"))
        self.D = int(cfg["D"])
        self.K = int(cfg["K"])
        self.dim_hidden = int(cfg.get("dim_hidden", 256))
        self.mlp_dim = int(cfg.get("mlp_dim", 768))

        # temperature (video only)
        self.T_video = 1.0
        if CALIB_PATH.exists():
            try:
                c = json.loads(CALIB_PATH.read_text())
                self.T_video = float(c.get("T_video", 1.0))
            except Exception:
                pass

        # models
        self.backbone = EffB0FeatureExtractor(BACKBONE_WEIGHTS)
        self.head = AttnPoolHead(self.D, self.dim_hidden, self.mlp_dim)
        state = torch.load(str(HEAD_WEIGHTS), map_location="cpu")
        self.head.load_state_dict(state["state_dict"])
        self.backbone.eval(); self.head.eval()

        # face detector
        self.detector = _mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)

        # [PATCH] one-time debug so you see calibration parameters in console
        if not getattr(self, "_printed_cfg", False):
            print(f"[infer] T_video={self.T_video}  threshold={DECISION_THRESHOLD} (calibrated)")
            self._printed_cfg = True

    def _prob_from_logit(self, logit: float) -> float:
        # temperature scaling BEFORE sigmoid is already correct
        z = logit / float(self.T_video)
        return 1.0 / (1.0 + math.exp(-z))

    @torch.no_grad()
    def _embed_batch(self, imgs_rgb: list) -> np.ndarray:
        if len(imgs_rgb) == 0:
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            emb = self.backbone(dummy).cpu().numpy().astype(np.float32)
            return emb
        xs = []
        for im in imgs_rgb:
            if im is None or im.size == 0:
                im = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            face = _face_crop(im, self.detector)
            if face is None or face.size == 0:
                face = im
            if _laplacian_var(face) < QUALITY_LAPLACIAN_MIN and _laplacian_var(im) > QUALITY_LAPLACIAN_MIN:
                face = im
            face224 = _resize_center_square(face, IMG_SIZE)
            xs.append(_to_tensor_norm(face224))
        b = torch.stack(xs, 0)  # [B,3,224,224]
        E = self.backbone(b).cpu().numpy().astype(np.float32)  # [B,D]
        return E

    @torch.no_grad()
    def _predict_from_feats(self, E: np.ndarray, return_weights: bool=False):
        if E.shape[0] == 0:
            E = np.zeros((1, self.D), dtype=np.float32)
        Kcfg = self.K
        if E.shape[0] < Kcfg:
            last = E[-1:].copy()
            E = np.concatenate([E] + [last] * (Kcfg - E.shape[0]), axis=0)
        elif E.shape[0] > Kcfg:
            idxs = _sample_video_indices(E.shape[0], Kcfg, phase=0.0)
            E = E[idxs]
        Et = torch.from_numpy(E).unsqueeze(0).float()     # [1,K,D]
        if return_weights:
            logit, weights = self.head(Et, return_weights=True)
            return float(logit.item()), weights[0].cpu().numpy().tolist(), Kcfg
        else:
            logit = self.head(Et)
            return float(logit.item()), None, Kcfg

    @torch.no_grad()
    def predict_frames(self, frames_rgb: list, want_weights: bool=True):
        """
        Single-window prediction over given frames (with optional TTA).
        Returns dict with prob/logit/weights info.
        """
        batches = [frames_rgb]
        if TTA_FLIP:
            batches.append([_maybe_flip(f) if f is not None else None for f in frames_rgb])

        probs = []
        attn = None
        for bidx, frs in enumerate(batches):
            E = self._embed_batch(frs)
            logit, weights, Kcfg = self._predict_from_feats(E, return_weights=want_weights and (bidx == 0))
            p = self._prob_from_logit(logit)
            probs.append(p)
            if weights is not None:
                attn = weights

        prob = float(sum(probs) / len(probs))
        label_str = "FAKE" if prob >= DECISION_THRESHOLD else "REAL"
        label_int = 1 if label_str == "FAKE" else 0
        return {
            "fake_prob": prob,
            "real_prob": 1.0 - prob,
            "label": label_str,
            "label_int": label_int,
            "K": int(Kcfg),
            "attn_weights": attn,
        }

    # --------- explanation helpers ---------
    def _gradcam_for_face224(self, face224_rgb: np.ndarray) -> np.ndarray:
        x = _to_tensor_norm(face224_rgb).unsqueeze(0).requires_grad_(True)  # [1,3,224,224]
        try:
            with torch.enable_grad():
                cam = _gradcam_map_for_feats(self.backbone.model, self.head.mlp, x)
            return cam.detach().cpu().numpy().astype(np.float32)
        except Exception:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    def _explain_frames(self, frames_rgb: list, attn_weights: list, frame_indices: list, topk: int = EXPLAIN_TOPK):
        if not ENABLE_EXPLAIN or not attn_weights:
            return []
        K = min(len(attn_weights), len(frames_rgb))
        if K == 0:
            return []
        order = np.argsort(attn_weights)[::-1][:min(topk, K)]
        expl = []
        for idx in order:
            fr = frames_rgb[idx]
            if fr is None or fr.size == 0:
                continue
            face = _face_crop(fr, self.detector)
            if face is None or face.size == 0:
                face = fr
            face224 = _resize_center_square(face, IMG_SIZE)
            cam = self._gradcam_for_face224(face224)
            overlay = _overlay_cam(face224, cam, alpha=0.35)
            expl.append({
                "frame_idx": int(frame_indices[idx]) if frame_indices and idx < len(frame_indices) else idx,
                "attn": float(attn_weights[idx]),
                "thumb_b64": _to_b64(face224),
                "cam_b64": _to_b64(overlay),
            })
        return expl

# =================== PUBLIC API ===================

_model_singleton = None

def load_model():
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = VideoModel()
    return _model_singleton

def predict_video(video_path: str):
    m = load_model()
    vp = Path(video_path)
    cap = cv2.VideoCapture(str(vp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()

    windows = []
    for ph in WINDOW_PHASES[:NUM_WINDOWS]:
        idxs = _sample_video_indices(total, m.K if m.K > 0 else 16, phase=ph)
        frames = []
        for idx in idxs:
            fr = _read_video_frame_rgb(vp, idx)
            frames.append(fr if fr is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        out_w = m.predict_frames(frames, want_weights=True)
        out_w["phase"] = ph
        out_w["frame_indices"] = idxs
        out_w["frames_rgb"] = frames  # temp for thumbnails
        windows.append(out_w)

    window_probs = [w["fake_prob"] for w in windows]

    # [PATCH] robust aggregation: trimmed mean if >=5, else simple mean (no max fallback)
    if window_probs:
        if len(window_probs) >= 5:
            s = sorted(window_probs)[1:-1]  # drop min & max
            agg = float(np.mean(s)) if len(s) > 0 else float(np.mean(window_probs))
        else:
            agg = float(np.mean(window_probs))
    else:
        agg = 0.0

    best_ix = int(np.argmax(window_probs)) if window_probs else 0
    explain_list = []
    if windows:
        w = windows[best_ix]
        explain_list = m._explain_frames(
            frames_rgb=w.get("frames_rgb", []),
            attn_weights=w.get("attn_weights", []) or [],
            frame_indices=w.get("frame_indices", []) or [],
            topk=EXPLAIN_TOPK
        )
        for ww in windows:
            ww.pop("frames_rgb", None)

    label_str = "FAKE" if agg >= DECISION_THRESHOLD else "REAL"
    out = {
        "type": "video",
        "path": str(video_path),
        "fake_prob": agg,
        "real_prob": 1.0 - agg,
        "label": label_str,
        "K": m.K,
        "windows": windows,
        "window_probs": window_probs,
        "low_confidence": abs(agg - 0.5) < LOW_CONF_EPS,
        "calibration_T": m.T_video,
        "explain": explain_list,
    }
    return out
