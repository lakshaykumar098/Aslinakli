#!/usr/bin/env python3
"""
ASLINAKLI • Train CPU Attention-Pooling head on video embeddings (video-only).

Upgrades:
- Larger hidden dim (256) and MLP (768)
- More epochs (80) with patience 8
- AdamW optimizer with weight decay
- Records hidden dims in config
"""

import os, json, csv, argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

# ---------------------------
# Model: Attention Pooling Head
# ---------------------------
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

    def forward(self, E):  # E: [B,K,D]
        scores = self.attn(E)                     # [B,K,1]
        weights = torch.softmax(scores, dim=1)    # [B,K,1]
        pooled = torch.sum(weights * E, dim=1)    # [B,D]
        logit = self.mlp(pooled).squeeze(1)       # [B]
        return logit

# ---------------------------
# Data
# ---------------------------
class EmbedDataset(Dataset):
    def __init__(self, items: List[Dict]):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        r = self.items[i]
        E = np.load(r["embed_path"]).astype(np.float32)   # [K,D]
        y = np.float32(r["label"])
        return torch.from_numpy(E), torch.from_numpy(np.array(y))

def collate(batch):
    Es, ys = zip(*batch)
    return torch.stack(Es,0).float(), torch.stack(ys,0).float()

# ---------------------------
# Utils
# ---------------------------
def load_index(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)  # video_path,label,fold,cache_dir,frames
    return rows

def build_items(index_rows, embeds_root: Path, folds: List[int]):
    items = []
    for row in index_rows:
        fold = int(row["fold"])
        if fold not in folds: 
            continue
        cache_dir = Path(row["cache_dir"])
        vid = cache_dir.name
        embed_path = embeds_root / f"fold{fold}" / f"{vid}.npy"
        if embed_path.exists():
            items.append({
                "embed_path": str(embed_path),
                "label": int(row["label"]),
                "fold": fold,
                "video_id": vid
            })
    return items

def class_weights(items):
    labels = np.array([it["label"] for it in items], dtype=np.int32)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    return torch.tensor([neg / max(pos,1)], dtype=torch.float32)

# ---------------------------
# Train / Eval
# ---------------------------
def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    losses = []
    for E,y in tqdm(loader, desc="train", leave=False):
        E = E.to(device); y = y.to(device)
        logit = model(E)
        loss = crit(logit, y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); logits=[]; labels=[]
    for E,y in tqdm(loader, desc="val  ", leave=False):
        logit = model(E.to(device))
        logits.append(logit.cpu()); labels.append(y)
    if len(logits)==0:
        return {"loss": float("nan"), "auc": float("nan"), "f1": float("nan"),
                "acc": float("nan"), "cm": [[0,0],[0,0]]}
    logits = torch.cat(logits).numpy(); labels = torch.cat(labels).numpy()
    probs = 1/(1+np.exp(-logits)); preds = (probs>=0.5).astype(np.int32)
    try: auc = roc_auc_score(labels, probs)
    except Exception: auc = float("nan")
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds).tolist()
    loss = float(np.mean(-(labels*np.log(probs+1e-8)+(1-labels)*np.log(1-probs+1e-8))))
    return {"loss": loss, "auc": float(auc), "f1": float(f1), "acc": float(acc), "cm": cm}

# ---------------------------
# Main
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train Attention head on embeddings (CPU)")
    ap.add_argument("--embeds-root", default="data_cache/embeds_ft", type=str)
    ap.add_argument("--index-csv", default="splits/index.csv", type=str)
    ap.add_argument("--train-folds", nargs="+", type=int, default=[1,2,3,4])
    ap.add_argument("--val-folds",   nargs="+", type=int, default=[5])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr",     type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out-dir", default="models", type=str)
    ap.add_argument("--backbone", default="efficientnet_b0_ft", type=str)
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    index_rows = load_index(Path(args.index_csv))
    embeds_root = Path(args.embeds_root)

    train_items = build_items(index_rows, embeds_root, args.train_folds)
    val_items   = build_items(index_rows, embeds_root, args.val_folds)
    if len(train_items)==0 or len(val_items)==0:
        print("[!] Empty train/val after filtering folds. Check paths/folds.")
        return

    # peek D and K
    sample_E = np.load(train_items[0]["embed_path"]).astype(np.float32)  # [K,D]
    K, D = sample_E.shape

    train_loader = DataLoader(EmbedDataset(train_items), batch_size=args.batch, shuffle=True,  num_workers=0, collate_fn=collate)
    val_loader   = DataLoader(EmbedDataset(val_items),   batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate)

    model = AttnPoolHead(D, dim_hidden=256, mlp_dim=768).to(device)
    pos_weight = class_weights(train_items)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)) if pos_weight is not None else nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -1.0
    bad = 0
    patience = args.patience

    save_path = Path(args.out_dir) / "video_attn_head.pt"
    cfg_path  = Path(args.out_dir) / "video_head_config.json"
    metrics_path = Path(args.out_dir) / "video_attn_metrics.json"

    best_epoch = -1
    best_snapshot = None

    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, crit, opt, device)
        val = evaluate(model, val_loader, device)
        print(f"Epoch {ep:02d} | train_loss {tr_loss:.4f} | val_auc {val['auc']:.4f} | val_f1 {val['f1']:.4f} | val_acc {val['acc']:.4f} | cm {val['cm']}")

        improved = (not np.isnan(val['auc'])) and (val['auc'] > best_auc)
        if improved:
            best_auc = val['auc']; bad = 0
            torch.save({"state_dict": model.state_dict(), "D": D, "K": K}, save_path)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({
                    "backbone": args.backbone,
                    "D": int(D),
                    "K": int(K),
                    "head": "attention",
                    "dim_hidden": 256,
                    "mlp_dim": 768
                }, f, indent=2)
            best_epoch = ep
            best_snapshot = {"epoch": ep, **val}
            print(f"  ✔ saved best to {save_path} (AUC {best_auc:.4f})")
        else:
            bad += 1
            if bad >= patience:
                print("  ⏹ early stopping")
                break

    # final eval
    state = torch.load(save_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    final = evaluate(model, val_loader, device)
    print(f"[FINAL] best_epoch {best_epoch} | val_auc {final['auc']:.4f} | val_f1 {final['f1']:.4f} | val_acc {final['acc']:.4f} | cm {final['cm']}")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": int(best_epoch),
            "val_auc": float(final["auc"]),
            "val_f1": float(final["f1"]),
            "val_acc": float(final["acc"]),
            "cm": final["cm"]
        }, f, indent=2)
    print(f"[i] Wrote metrics to {metrics_path}")
    print(f"[✓] Best val AUC: {best_auc:.4f} | Weights: {save_path}")

if __name__ == "__main__":
    main()
