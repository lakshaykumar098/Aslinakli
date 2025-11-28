#!/usr/bin/env python3
"""
ASLINAKLI • Image training (audio-style) with autosave + resume.

Saves:
  checkpoints/images/<backbone>/epoch_<E>_valauc_<AUC>.pth
  checkpoints/images/<backbone>/best.pth
  models/<backbone>_finetuned_images.pt
  models/image_head_config.json
  models/calibration.json
  models/image_train_metrics.json
"""

import argparse, json, os, random, io, time, shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# -----------------------
# Defaults / config
# -----------------------
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BACKBONE = "xception"
INPUT_SIZE_MAP = {"xception": 299}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# -----------------------
# Utils
# -----------------------
def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class FileListDataset(Dataset):
    def __init__(self, files: List[str], labels: List[int], tfm):
        self.files = files; self.labels = labels; self.tfm = tfm
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        p = self.files[i]; y = int(self.labels[i])
        img = Image.open(p).convert("RGB")
        return self.tfm(img), y

class JPEGCompression:
    def __init__(self, quality=(50,95), p=0.35): self.q=quality; self.p=p
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p: return img
        q = random.randint(self.q[0], self.q[1])
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=q); buf.seek(0)
        return Image.open(buf).convert("RGB")

def build_transforms(input_size: int):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(input_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
        JPEGCompression(quality=(50,95), p=0.35),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(input_size * 1.05)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf

def scan_real_fake(root: Path):
    paths, labels = [], []
    for cls, lab in (("real",0),("fake",1)):
        pdir = root/cls
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            for p in pdir.glob(ext):
                paths.append(str(p)); labels.append(lab)
    return paths, labels

def split_stratified(paths, labels, val_ratio=0.1, test_ratio=0.1, seed=42):
    X = np.array(paths); y = np.array(labels)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(val_ratio+test_ratio), random_state=seed, stratify=y)
    rel = test_ratio/(val_ratio+test_ratio) if (val_ratio+test_ratio)>0 else 0.5
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel, random_state=seed, stratify=y_tmp)
    return list(X_train), list(y_train), list(X_val), list(y_val), list(X_test), list(y_test)

def build_model(backbone: str, num_classes=1):
    return timm.create_model(backbone, pretrained=True, num_classes=num_classes)

def bce_logits_loss(logits, targets, pos_weight=None, label_smoothing=0.05):
    if label_smoothing > 0:
        targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets)

def evaluate(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    total_loss = 0.0; n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.float().unsqueeze(1).to(device)
            logits = model(x)
            loss = bce_logits_loss(logits, y, pos_weight=None, label_smoothing=0.0)
            total_loss += float(loss.item()) * x.size(0); n += x.size(0)
            all_logits.append(logits.cpu().numpy()); all_targets.append(y.cpu().numpy())
    logits = np.concatenate(all_logits, 0).reshape(-1)
    targets = np.concatenate(all_targets,0).reshape(-1)
    probs = 1.0/(1.0+np.exp(-logits))
    auc = roc_auc_score(targets, probs) if len(np.unique(targets))==2 else float("nan")
    preds = (probs >= 0.5).astype(np.int64)
    f1 = f1_score(targets, preds, zero_division=0); acc = accuracy_score(targets, preds)
    return {"loss": total_loss/max(1,n), "auc": float(auc), "f1": float(f1), "acc": float(acc),
            "logits": logits, "targets": targets}

def calibrate_temperature(val_logits, val_targets):
    grid = np.linspace(0.5, 3.0, 26); best_T, best_nll = 1.0, float("inf")
    for T in grid:
        probs = 1.0/(1.0+np.exp(-val_logits / T))
        nll = log_loss(val_targets, probs, labels=[0,1])
        if nll < best_nll: best_nll, best_T = nll, float(T)
    return best_T, best_nll

def tune_threshold(val_logits, val_targets, T):
    probs = 1.0/(1.0+np.exp(-val_logits / T))
    grid = np.linspace(0.05, 0.95, 91); best_thr, best_f1 = 0.5, -1
    for thr in grid:
        preds = (probs >= thr).astype(np.int64)
        f1 = f1_score(val_targets, preds, zero_division=0)
        if f1 > best_f1: best_f1, best_thr = float(f1), float(thr)
    return best_thr, best_f1

def save_checkpoint(state: dict, ckpt_dir: Path, tag: str):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{tag}.pth"
    torch.save(state, path)
    return path

def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists(): return None
    files = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def prune_old_checkpoints(ckpt_dir: Path, keep_last: int = 5):
    # Only prune epoch_* checkpoints; never touch best.pth
    files = sorted(ckpt_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[keep_last:]:
        try:
            p.unlink()
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/images")
    ap.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.10)
    ap.add_argument("--test-split", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    # autosave/resume
    ap.add_argument("--ckpt-dir", type=str, default=None,
                    help="Defaults to checkpoints/images/<backbone>/")
    ap.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    ap.add_argument("--keep-last", type=int, default=5, help="Keep last N checkpoints")
    ap.add_argument("--resume", type=str, default="auto",
                    help="'auto' to use latest in ckpt dir, or provide path to .pth, or 'none'")

    args = ap.parse_args()
    seed_everything(args.seed)

    data_root = Path(args.data_root)
    assert (data_root/"real").exists() and (data_root/"fake").exists(), \
        f"{data_root}/real and {data_root}/fake must exist"

    input_size = INPUT_SIZE_MAP.get(args.backbone, 224)
    train_tf, eval_tf = build_transforms(input_size)

    paths, labels = scan_real_fake(data_root)
    assert len(paths) > 0, "No images found."

    # class imbalance -> pos_weight for fake=1
    pos_weight = None
    p = float(np.mean(labels))
    if p > 0 and p < 1:
        pos_weight = torch.tensor([(1 - p) / max(p, 1e-6)], dtype=torch.float32)

    Xtr, ytr, Xva, yva, Xte, yte = split_stratified(paths, labels,
                                                    val_ratio=args.val_split,
                                                    test_ratio=args.test_split,
                                                    seed=args.seed)
    ds_train = FileListDataset(Xtr, ytr, train_tf)
    ds_val   = FileListDataset(Xva, yva, eval_tf)
    ds_test  = FileListDataset(Xte, yte, eval_tf)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.backbone, num_classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best_auc = -1.0; best_state = None; patience = 6; bad = 0
    start_epoch = 1

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(f"checkpoints/images/{args.backbone}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -------- Resume ----------
    resume_path = None
    if args.resume and args.resume.lower() != "none":
        if args.resume.lower() == "auto":
            resume_path = latest_checkpoint(ckpt_dir)
        else:
            rp = Path(args.resume)
            resume_path = rp if rp.exists() else None

    if resume_path:
        print(f"[RESUME] Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        sched.load_state_dict(ckpt["scheduler_state"])
        try:
            scaler.load_state_dict(ckpt.get("scaler_state", {}))
        except Exception:
            pass
        best_auc = float(ckpt.get("best_auc", best_auc))
        bad = int(ckpt.get("bad_epochs", 0))
        start_epoch = int(ckpt.get("epoch", 1)) + 1
        print(f"[RESUME] epoch={start_epoch}  best_auc={best_auc:.4f}  bad={bad}")
    else:
        print("[RESUME] No checkpoint used.")

    # -------- Train ----------
    for epoch in range(start_epoch, args.epochs+1):
        model.train(); running_loss = 0.0; n = 0; t0 = time.time()
        for x, y in dl_train:
            x = x.to(device); y = y.float().unsqueeze(1).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(x)
                loss = bce_logits_loss(logits, y, pos_weight=pos_weight, label_smoothing=0.05)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            running_loss += float(loss.item()) * x.size(0); n += x.size(0)
        sched.step()
        tr_loss = running_loss / max(1,n)

        val = evaluate(model, dl_val, device)
        auc = val["auc"]; f1 = val["f1"]; acc = val["acc"]
        print(f"[{epoch:02d}/{args.epochs}] train_loss={tr_loss:.4f}  val_loss={val['loss']:.4f}  "
              f"AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}  time={time.time()-t0:.1f}s")

        # --- Save per-epoch ---
        if (epoch - 1) % args.save_every == 0:
            tag = f"epoch_{epoch:03d}_valauc_{auc:.4f}"
            save_checkpoint({
                "epoch": epoch,
                "model_state": {k: v.cpu() for k,v in model.state_dict().items()},
                "optimizer_state": opt.state_dict(),
                "scheduler_state": sched.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_auc": best_auc,
                "bad_epochs": bad,
                "backbone": args.backbone,
                "seed": args.seed,
            }, ckpt_dir, tag)
            prune_old_checkpoints(ckpt_dir, keep_last=args.keep_last)

        # --- Track best ---
        if auc > best_auc:
            best_auc = auc; bad = 0
            best_state = {k: v.cpu() for k,v in model.state_dict().items()}
            # save/update best checkpoint
            save_checkpoint({
                "epoch": epoch,
                "model_state": best_state,
                "optimizer_state": opt.state_dict(),
                "scheduler_state": sched.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_auc": best_auc,
                "bad_epochs": bad,
                "backbone": args.backbone,
                "seed": args.seed,
            }, ckpt_dir, "best")
        else:
            bad += 1
            if bad >= 6:
                print(f"Early stopping (no AUC improvement for {bad} epochs).")
                break

    # --- Load best for calibration + export ---
    if best_state is not None:
        model.load_state_dict(best_state)

    val = evaluate(model, dl_val, device)
    T_image, val_nll = calibrate_temperature(val["logits"], val["targets"])
    thr, best_f1 = tune_threshold(val["logits"], val["targets"], T_image)

    weights_path = MODELS_DIR / f"{args.backbone}_finetuned_images.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights -> {weights_path}")

    cfg = {"backbone": args.backbone, "input_size": int(INPUT_SIZE_MAP.get(args.backbone,224)),
           "threshold": float(thr)}
    (MODELS_DIR / "image_head_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Saved config  -> models/image_head_config.json")

    calib_path = MODELS_DIR / "calibration.json"
    calib = {}
    if calib_path.exists():
        try: calib = json.loads(calib_path.read_text())
        except Exception: calib = {}
    calib["T_image"] = float(T_image)
    calib_path.write_text(json.dumps(calib, indent=2))
    print(f"Saved calib   -> models/calibration.json (T_image={T_image:.3f}, val_NLL={val_nll:.4f})")

    test = evaluate(model, dl_test, device)
    probs_test = 1.0/(1.0+np.exp(-test["logits"]/T_image))
    preds_test = (probs_test >= thr).astype(np.int64)
    auc_test = roc_auc_score(test["targets"], probs_test) if len(np.unique(test["targets"]))==2 else float("nan")
    f1_test = f1_score(test["targets"], preds_test, zero_division=0)
    acc_test = accuracy_score(test["targets"], preds_test)
    print(f"[TEST] AUC={auc_test:.4f}  F1={f1_test:.4f}  Acc={acc_test:.4f}")

    summary = {
        "val": {"AUC": float(val["auc"]), "F1@0.5": float(val["f1"]), "Acc@0.5": float(val["acc"])},
        "calibration": {"T_image": float(T_image), "val_NLL": float(val_nll)},
        "threshold": float(thr),
        "test": {"AUC": float(auc_test), "F1": float(f1_test), "Acc": float(acc_test)}
    }
    (MODELS_DIR / "image_train_metrics.json").write_text(json.dumps(summary, indent=2))
    print("Saved metrics -> models/image_train_metrics.json")

if __name__ == "__main__":
    main()
