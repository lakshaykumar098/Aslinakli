#!/usr/bin/env python3
"""
Train EfficientNet-B0 backbone on preprocessed face frames (video-only).

- Expects frames produced by preprocess_frames.py at 224x224
  Layout examples (both supported):
    frames_dir/fold*/real|fake/<video_stem>/frame_*.png
    frames_dir/real|fake/<video_stem>/frame_*.png

- Trains a binary head (FAKE=1, REAL=0) with BCEWithLogitsLoss.
- Freezes most layers; unfreezes last 2 blocks + classifier for efficient finetune.
- Adds light aug (flip + color jitter) for robustness.
- Early stopping on val loss (patience=3).
- Saves best weights to models/effb0_finetuned_frames.pt
- NEW: Checkpoint resume (model + optimizer + LR + early-stop counters + RNG states).
"""

from pathlib import Path
import argparse, random, math, os, sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
import torchvision.transforms as T

# -------------------- Defaults / Paths --------------------
DEFAULT_OUT = Path("models/effb0_finetuned_frames.pt")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ----------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic-ish (keep cudnn fast)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def find_video_dirs(frames_root: Path):
    """
    Return a list of tuples: (video_dir, label)
      label: 0 for REAL, 1 for FAKE
    A video_dir contains frame_*.png files.
    """
    frames_root = frames_root.resolve()
    vids = []
    for sub in frames_root.rglob("*"):
        if sub.is_dir():
            parts = [p.name.lower() for p in sub.parents]
            if "real" in parts:
                label = 0
            elif "fake" in parts:
                label = 1
            else:
                continue
            # video frames folder?
            try:
                files = [f for f in os.listdir(sub) if os.path.isfile(sub / f)]
            except Exception:
                files = []
            if any((sub / f).suffix.lower() == ".png" and f.startswith("frame_") for f in files):
                vids.append((sub, label))
    return vids

class VideoFrameSampleDataset(Dataset):
    """
    Training dataset:
      - We want ~frames_per_video samples per video *per epoch*.
      - We build an index that repeats each video 'frames_per_video' times.
      - __getitem__ picks ONE random frame from that video folder.
    Validation dataset:
      - We build one sample per video by picking a deterministic frame (middle).
    """
    def __init__(self, video_dirs, frames_per_video: int, img_size: int, train: bool):
        super().__init__()
        self.video_dirs = video_dirs
        self.frames_per_video = frames_per_video
        self.img_size = img_size
        self.train = train

        if train:
            self.indices = []
            for i in range(len(video_dirs)):
                self.indices += [i] * frames_per_video
            random.shuffle(self.indices)
        else:
            self.indices = list(range(len(video_dirs)))

        # cache frames listing per video
        self.cache = {}
        for i, (vdir, _lab) in enumerate(self.video_dirs):
            try:
                files = [p for p in sorted(vdir.glob("frame_*.png"))]
            except Exception:
                files = []
            self.cache[i] = files

        # transforms
        if train:
            self.tx = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tx = T.Compose([
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.indices)

    def _pick_frame_for_train(self, files):
        if not files:
            return None
        return random.choice(files)

    def _pick_frame_for_val(self, files):
        if not files:
            return None
        return files[len(files)//2]

    def __getitem__(self, idx):
        vid_idx = self.indices[idx]
        vdir, label = self.video_dirs[vid_idx]
        files = self.cache.get(vid_idx, [])
        fp = self._pick_frame_for_train(files) if self.train else self._pick_frame_for_val(files)
        if fp is None or not fp.exists():
            x = torch.zeros(3, self.img_size, self.img_size)
            y = torch.tensor(float(label))
            return x, y

        img = Image.open(fp).convert("RGB")
        x = self.tx(img)
        y = torch.tensor(float(label))
        return x, y

def make_model(num_classes: int = 1, pretrained: bool = True):
    # Binary classifier: num_classes=1 for BCEWithLogitsLoss
    model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    return model

def unfreeze_last_blocks(model, n_blocks: int = 2):
    """
    EfficientNet-B0 in timm has model.blocks as a sequential of MBConv blocks.
    We'll unfreeze the last n_blocks plus classifier.
    """
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze last n blocks
    try:
        blocks = model.blocks
        for b in blocks[-n_blocks:]:
            for p in b.parameters():
                p.requires_grad = True
    except Exception:
        pass

    # unfreeze classifier
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, loader, opt, device):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)  # [B]
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y).sum().item()
    return total / max(n, 1), correct / max(n, 1)

# -------------------- Checkpoint utils --------------------

def default_ckpt_path(out_model: Path) -> Path:
    return out_model.with_suffix(".ckpt")

def save_checkpoint(path: Path, epoch: int, model: nn.Module, opt: torch.optim.Optimizer,
                    best_val: float, bad: int, seed: int):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "best_val": best_val,
        "bad": bad,
        "seed": seed,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(path))

def try_load_checkpoint(path: Path, model: nn.Module, opt: torch.optim.Optimizer):
    if not path.exists():
        return None
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    # restore RNG
    try:
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.random.set_rng_state(ckpt["rng"]["torch_cpu"])
        if torch.cuda.is_available() and ckpt["rng"]["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["torch_cuda"])
    except Exception:
        pass
    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "best_val": float(ckpt.get("best_val", float("inf"))),
        "bad": int(ckpt.get("bad", 0)),
        "seed": int(ckpt.get("seed", 1337)),
    }

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", type=str, required=True,
                    help="Root directory of extracted frames (from preprocess_frames.py).")
    ap.add_argument("--out_model", type=str, default=str(DEFAULT_OUT),
                    help="Where to save the best backbone weights (.pt).")
    ap.add_argument("--resume_ckpt", type=str, default="",
                    help="Path to resume checkpoint (.ckpt). If empty, will auto-use out_model with .ckpt suffix when present.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--frames_per_video", type=int, default=8,
                    help="How many samples per video per *epoch* for training.")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3, help="Early stopping patience on val loss.")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num_workers", type=int, default=0)  # keep 0 on Windows for safety
    args = ap.parse_args()

    set_seed(args.seed)

    frames_dir = Path(args.frames_dir)
    out_model = Path(args.out_model).resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.resume_ckpt).resolve() if args.resume_ckpt else default_ckpt_path(out_model)

    # collect video dirs
    vids = find_video_dirs(frames_dir)
    if not vids:
        raise SystemExit(f"No video frame folders found under: {frames_dir}")

    # split train/val at video level
    idx = np.arange(len(vids))
    np.random.shuffle(idx)
    split = int(len(idx) * (1.0 - args.val_ratio))
    tr_idx, va_idx = idx[:split], idx[split:]
    train_videos = [vids[i] for i in tr_idx]
    val_videos   = [vids[i] for i in va_idx]

    print(f"[INFO] Videos: total={len(vids)} train={len(train_videos)} val={len(val_videos)}")

    # datasets & loaders
    ds_tr = VideoFrameSampleDataset(train_videos, frames_per_video=args.frames_per_video,
                                    img_size=args.img_size, train=True)
    ds_va = VideoFrameSampleDataset(val_videos, frames_per_video=1,
                                    img_size=args.img_size, train=False)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=max(16, args.batch_size), shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # model
    model = make_model(num_classes=1, pretrained=True)
    unfreeze_last_blocks(model, n_blocks=2)
    trainable = count_trainable(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Params: total={total_params/1e6:.2f}M  trainable={trainable/1e6:.2f}M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimizer & simple cosine schedule
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    def lr_at_epoch(e):
        # cosine from lr -> lr*0.1 across epochs
        return args.lr * (0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * e / max(args.epochs,1)))))

    # ----- Resume if possible -----
    start_epoch = 1
    best_val = float("inf")
    best_state = None
    bad = 0

    if ckpt_path.exists():
        info = try_load_checkpoint(ckpt_path, model, opt)
        if info:
            start_epoch = info["epoch"] + 1           # continue AFTER last finished epoch
            best_val = info["best_val"]
            bad = info["bad"]
            # keep args.seed but RNG already restored
            print(f"[RESUME] Loaded checkpoint: {ckpt_path} (epoch={start_epoch-1}, best_val={best_val:.4f}, bad={bad})")
        else:
            print(f"[RESUME] No usable checkpoint at: {ckpt_path}")

    # Track current best (may be from resume)
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    try:
        for ep in range(start_epoch, args.epochs + 1):
            # update lr
            lr_now = lr_at_epoch(ep - 1)
            for g in opt.param_groups:
                g["lr"] = lr_now

            tr_loss = train_one_epoch(model, dl_tr, opt, device)
            va_loss, va_acc = eval_one_epoch(model, dl_va, device)

            print(f"[E{ep:02d}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_acc={va_acc*100:.2f}%  lr={lr_now:.2e}")

            # save checkpoint each epoch (AFTER validation finishes)
            save_checkpoint(ckpt_path, ep, model, opt, best_val, bad, args.seed)

            if va_loss < best_val - 1e-4:
                best_val = va_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"[EARLY STOP] No improvement for {args.patience} epochs.")
                    break

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving last checkpoint before exit...")
        # Save at the last fully completed epoch number (current ep-1)
        last_epoch = max(start_epoch, ep) if "ep" in locals() else 0
        save_checkpoint(ckpt_path, last_epoch, model, opt, best_val, bad, args.seed)
        raise
    finally:
        # always save best weights
        if best_state is None:
            best_state = model.state_dict()
        torch.save(best_state, str(out_model))
        print(f"[DONE] Saved best backbone to: {out_model}")
        print(f"[DONE] Last checkpoint at: {ckpt_path} (resume with --resume_ckpt \"{ckpt_path}\")")

if __name__ == "__main__":
    main()
