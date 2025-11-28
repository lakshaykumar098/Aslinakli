#!/usr/bin/env python3
"""
Export final image model artifacts from a training checkpoint (no training).

Inputs:
  - a checkpoint like: checkpoints/images/<backbone>/best.pth
    (supports both raw state_dict and dict with "model_state")

Outputs (written to models/):
  - <backbone>_finetuned_images.pt
  - image_head_config.json
  - calibration.json   (merges/updates T_image)
  - image_train_metrics.json (val/test metrics from this export run)

Calibration & threshold are computed from a fresh stratified split of data/images/{real|fake}.
"""

import argparse, json, io, random
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
# Data & transforms
# -----------------------
class FileListDataset(Dataset):
    def __init__(self, files: List[str], labels: List[int], tfm):
        self.files = files; self.labels = labels; self.tfm = tfm
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        p = self.files[i]; y = int(self.labels[i])
        img = Image.open(p).convert("RGB")
        return self.tfm(img), y

def build_transforms(input_size: int):
    eval_tf = transforms.Compose([
        transforms.Resize(int(input_size * 1.05)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return eval_tf

def scan_real_fake(root: Path) -> Tuple[List[str], List[int]]:
    paths, labels = [], []
    for cls, lab in (("real",0),("fake",1)):
        pdir = root / cls
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            for p in pdir.glob(ext):
                paths.append(str(p)); labels.append(lab)
    return paths, labels

def split_stratified(paths, labels, val_ratio=0.10, test_ratio=0.10, seed=42):
    X = np.array(paths); y = np.array(labels)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(val_ratio+test_ratio), random_state=seed, stratify=y)
    rel = test_ratio/(val_ratio+test_ratio) if (val_ratio+test_ratio)>0 else 0.5
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel, random_state=seed, stratify=y_tmp)
    return list(X_train), list(y_train), list(X_val), list(y_val), list(X_test), list(y_test)

# -----------------------
# Model & eval
# -----------------------
def build_model(backbone: str, num_classes=1):
    return timm.create_model(backbone, pretrained=False, num_classes=num_classes)

def load_state_into_model(model: torch.nn.Module, state_obj):
    if isinstance(state_obj, dict) and "model_state" in state_obj:
        sd = state_obj["model_state"]
    else:
        sd = state_obj
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        model.load_state_dict(sd, strict=False)

def evaluate(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    total_loss = 0.0; n = 0
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.float().unsqueeze(1).to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
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

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to checkpoint (e.g., checkpoints/images/xception/best.pth)")
    ap.add_argument("--data-root", type=str, default="data/images",
                    help="Folder with real/ and fake/ subfolders")
    ap.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    ap.add_argument("--val-split", type=float, default=0.10)
    ap.add_argument("--test-split", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    assert (data_root / "real").exists() and (data_root / "fake").exists(), \
        f"{data_root}/real and {data_root}/fake must exist"

    # Build model & load checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.backbone, num_classes=1).to(device)
    state_obj = torch.load(args.ckpt, map_location="cpu")
    load_state_into_model(model, state_obj)

    # Data split + loaders
    paths, labels = scan_real_fake(data_root)
    assert len(paths) > 0, "No images found under data/images/{real,fake}"
    Xtr, ytr, Xva, yva, Xte, yte = split_stratified(paths, labels,
                                                    val_ratio=args.val_split,
                                                    test_ratio=args.test_split,
                                                    seed=args.seed)
    input_size = INPUT_SIZE_MAP.get(args.backbone, 224)
    eval_tf = build_transforms(input_size)

    dl_val  = DataLoader(FileListDataset(Xva, yva, eval_tf),  batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    dl_test = DataLoader(FileListDataset(Xte, yte, eval_tf),  batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # ---- Calibration & threshold on VAL ----
    val = evaluate(model, dl_val, device)
    T_image, val_nll = calibrate_temperature(val["logits"], val["targets"])
    thr, best_f1 = tune_threshold(val["logits"], val["targets"], T_image)

    # ---- Save final model weights ----
    weights_path = MODELS_DIR / f"{args.backbone}_finetuned_images.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights -> {weights_path}")

    # ---- Save config ----
    cfg = {"backbone": args.backbone, "input_size": int(input_size), "threshold": float(thr)}
    (MODELS_DIR / "image_head_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Saved config  -> models/image_head_config.json")

    # ---- Save/merge calibration ----
    calib_path = MODELS_DIR / "calibration.json"
    calib = {}
    if calib_path.exists():
        try: calib = json.loads(calib_path.read_text())
        except Exception: calib = {}
    calib["T_image"] = float(T_image)
    calib_path.write_text(json.dumps(calib, indent=2))
    print(f"Saved calib   -> models/calibration.json (T_image={T_image:.3f}, val_NLL={val_nll:.4f})")

    # ---- Report & save metrics (TEST uses calibrated probs + tuned threshold) ----
    test = evaluate(model, dl_test, device)
    probs_test = 1.0/(1.0+np.exp(-test["logits"]/T_image))
    preds_test = (probs_test >= thr).astype(np.int64)
    auc_test = roc_auc_score(test["targets"], probs_test) if len(np.unique(test["targets"]))==2 else float("nan")
    f1_test = f1_score(test["targets"], preds_test, zero_division=0)
    acc_test = accuracy_score(test["targets"], preds_test)
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
