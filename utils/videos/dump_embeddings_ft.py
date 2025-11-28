#!/usr/bin/env python3
"""
ASLINAKLI • Dump per-video embeddings using fine-tuned EfficientNet-B0.

Run from repo root:

  python utils/videos/dump_embeddings_ft.py \
      --csv-path splits/index.csv \
      --frames-root data_cache/frames224 \
      --out-root data_cache/embeds_ft \
      --ft-weights models/effb0_finetuned_frames.pt \
      --batch 64

Outputs:
  data_cache/embeds_ft/foldK/<video_id>.npy  with shape [K, D] (e.g., [32, 1280])
"""

import os, csv, argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import timm

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def parse_args():
    ap = argparse.ArgumentParser(description="Dump embeddings with fine-tuned EffNet-B0 (CPU)")
    ap.add_argument("--csv-path", default="splits/index.csv", type=str)
    ap.add_argument("--frames-root", default="data_cache/frames224", type=str)
    ap.add_argument("--out-root", default="data_cache/embeds_ft", type=str)
    ap.add_argument("--ft-weights", default="models/effb0_finetuned_frames.pt", type=str)
    ap.add_argument("--batch", default=64, type=int)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_index(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)  # video_path,label,fold,cache_dir,frames
    return rows

def read_frame_rgb(path: Path):
    img = cv2.imread(str(path))  # BGR
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

def to_tensor_norm(img_rgb):
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)

class EffB0FeatureExtractor(nn.Module):
    """Fine-tuned EfficientNet-B0 feature extractor (returns 1280-dim embeddings)."""
    def __init__(self, weights_path: str):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=1)
        state = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)   # [B,C,H,W]
        pooled = self.model.global_pool(feats)   # [B,C]
        return pooled

def dump_one_video(cache_dir: Path, k: int, model: EffB0FeatureExtractor, batch: int, out_path: Path):
    # Collect available frames (frame_*.png)
    frames = sorted(cache_dir.glob("frame_*.png"))
    if not frames:
        # fallback: all-zero embeddings
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
            emb = model(dummy).cpu().numpy().astype(np.float32)
        E = np.repeat(emb, k, axis=0)
        np.save(out_path, E)
        return E.shape

    # pad if fewer frames than expected
    if len(frames) < k:
        frames = frames + [frames[-1]] * (k - len(frames))
    elif len(frames) > k:
        frames = frames[:k]

    xs, embs = [], []
    with torch.no_grad():
        for i, fp in enumerate(frames, start=1):
            img = read_frame_rgb(fp)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            xs.append(to_tensor_norm(img))
            if len(xs) == batch or i == len(frames):
                b = torch.stack(xs, 0).float()
                xs = []
                e = model(b).cpu().numpy().astype(np.float32)
                embs.append(e)
    E = np.concatenate(embs, axis=0)  # [K,D]
    np.save(out_path, E)
    return E.shape

def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    rows = load_index(csv_path)
    if len(rows) == 0:
        print("[!] Empty index.csv. Run preprocess first.")
        return

    model = EffB0FeatureExtractor(args.ft_weights)

    saved, skipped = 0, 0
    first_shape = None
    print(f"[*] Dumping embeddings to {out_root} using {args.ft_weights}")
    for row in tqdm(rows):
        cache_dir = Path(row["cache_dir"])
        fold = int(row["fold"])
        k = int(row["frames"])  # from index.csv (e.g., 32)
        vid = cache_dir.name

        out_dir = out_root / f"fold{fold}"
        ensure_dir(out_dir)
        out_path = out_dir / f"{vid}.npy"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            shape = dump_one_video(cache_dir, k, model, args.batch, out_path)
            first_shape = first_shape or shape
            saved += 1
        except Exception as e:
            print(f"[!] Error: {cache_dir} -> {e}")

    print(f"[✓] Done. Saved: {saved}, Skipped: {skipped}")
    if first_shape:
        print(f"[i] Example embedding shape: {first_shape}")

if __name__ == "__main__":
    main()
