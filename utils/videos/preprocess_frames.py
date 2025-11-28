#!/usr/bin/env python3
"""
Preprocess videos into face-centered frames for training.

- Uniformly samples K frames per video (default K=32)
- MediaPipe face detection with expanded bbox (default expand=0.20)
- Falls back to full frame when no face is detected
- Resizes to a square crop (default 224x224)
- Saves as PNGs: out_dir/<relative>/<video_stem>/frame_00001.png

Works with either:
  videos_root/fold1..5/real|fake/*.mp4
or
  videos_root/real|fake/*.mp4
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".3gp"}

def sample_indices(n_frames: int, k: int):
    if n_frames <= 0 or k <= 0:
        return []
    if k >= n_frames:
        return list(range(n_frames))
    step = n_frames / k
    # center of each segment
    return [int(i * step + step / 2) for i in range(k)]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_frame_bgr(cap: cv2.VideoCapture, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok or fr is None:
        return None
    return fr

def center_square_rgb(img_rgb: np.ndarray, size: int = 224) -> np.ndarray:
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
    if out.shape[0] != size or out.shape[1] != size:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return out

def face_crop_rgb(img_rgb: np.ndarray, detector, expand_ratio: float):
    """Return (crop_rgb or None)."""
    h, w = img_rgb.shape[:2]
    res = detector.process(img_rgb)
    if not res.detections:
        return None
    d0 = res.detections[0]
    bb = d0.location_data.relative_bounding_box
    x = int(bb.xmin * w); y = int(bb.ymin * h)
    bw = int(bb.width * w); bh = int(bb.height * h)
    ex = int(bw * expand_ratio); ey = int(bh * expand_ratio)
    x0 = max(0, x - ex); y0 = max(0, y - ey)
    x1 = min(w, x + bw + ex); y1 = min(h, y + bh + ey)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img_rgb[y0:y1, x0:x1]
    if crop is None or crop.size == 0:
        return None
    return crop

def collect_videos(root: Path):
    """
    Returns list of (video_path, rel_base) where rel_base is the subfolder structure
    under videos_root, excluding the filename. This preserves folds/labels.
    """
    vids = []
    root = root.resolve()
    # common layouts:
    #   videos_root/fold*/real|fake/*.mp4
    #   videos_root/real|fake/*.mp4
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            rel = p.parent.relative_to(root)  # keep foldX/real etc.
            vids.append((p, rel))
    return vids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_root", required=True, type=str,
                    help="Root containing fold*/real|fake or real|fake folders with videos.")
    ap.add_argument("--out_dir", required=True, type=str,
                    help="Where to save extracted frames.")
    ap.add_argument("--size", default=224, type=int, help="Output square size (default 224).")
    ap.add_argument("--frames", default=32, type=int, help="Frames per video (default 32).")
    ap.add_argument("--expand", default=0.20, type=float, help="Face bbox expansion ratio (default 0.20).")
    ap.add_argument("--min_det_conf", default=0.7, type=float, help="MediaPipe min detection confidence (default 0.7).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing frame folders.")
    args = ap.parse_args()

    videos_root = Path(args.videos_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=float(args.min_det_conf)
    )

    videos = collect_videos(videos_root)
    if not videos:
        print(f"[WARN] No videos found under: {videos_root}")
        return

    total = len(videos)
    print(f"[INFO] Found {total} videos. Sampling {args.frames} frames each → size {args.size}.")

    for i, (vpath, rel) in enumerate(videos, 1):
        stem = vpath.stem
        out_sub = out_dir / rel / stem
        if out_sub.exists() and not args.overwrite:
            # already processed; skip
            print(f"[{i}/{total}] SKIP (exists): {vpath}")
            continue
        # clear and recreate
        if out_sub.exists():
            for old in out_sub.glob("*"):
                try: old.unlink()
                except: pass
        ensure_dir(out_sub)

        cap = cv2.VideoCapture(str(vpath))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = sample_indices(n, args.frames)

        wrote = 0
        for j, fidx in enumerate(idxs, 1):
            fr_bgr = read_frame_bgr(cap, fidx)
            if fr_bgr is None:
                continue
            fr_rgb = cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB)

            # face crop with expansion; fallback to full frame
            face = face_crop_rgb(fr_rgb, detector, args.expand)
            crop = face if (face is not None and face.size > 0) else fr_rgb

            crop_sq = center_square_rgb(crop, size=args.size)
            # save PNG
            out_path = out_sub / f"frame_{j:05d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(crop_sq, cv2.COLOR_RGB2BGR))
            wrote += 1

        cap.release()
        print(f"[{i}/{total}] {vpath}  →  {wrote} frames @ {args.size}px -> {out_sub}")

    print("[DONE] Preprocessing complete.")

if __name__ == "__main__":
    main()
