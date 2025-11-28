#!/usr/bin/env python3
"""
Tune threshold OR fit a temperature scale for video-only attention head.

Usage (from project root):

# Find best threshold (accuracy or F1):
python utils/videos/tune_threshold.py \
  --embeds-root data_cache/embeds_ft \
  --index-csv splits/index.csv \
  --val-folds 5 \
  --head-path models/video_attn_head.pt \
  --config-path models/video_head_config.json \
  --optimize accuracy

# Or calibrate temperature scaling:
python utils/videos/tune_threshold.py \
  --embeds-root data_cache/embeds_ft \
  --index-csv splits/index.csv \
  --val-folds 5 \
  --head-path models/video_attn_head.pt \
  --config-path models/video_head_config.json \
  --calibrate
"""

import argparse, csv, json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

# ---------------- Heads ----------------
class StatsPoolHead(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*dim_in,512),
            nn.ReLU(True),
            nn.Linear(512,1)
        )
    def forward(self,E):
        mu = E.mean(1); sigma = E.std(1,unbiased=False)
        feat = torch.cat([mu,sigma],1)
        return self.mlp(feat).squeeze(1)

class AttnPoolHead(nn.Module):
    def __init__(self, dim_in, dim_hidden=256, mlp_dim=768):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, mlp_dim),
            nn.ReLU(True),
            nn.Linear(mlp_dim, 1)
        )
    def forward(self,E):
        w = torch.softmax(self.attn(E),1)
        pooled = (w*E).sum(1)
        return self.mlp(pooled).squeeze(1)

# ---------------- Data ----------------
class EmbedDS(Dataset):
    def __init__(self, items): self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        r=self.items[i]; E=np.load(r["embed_path"]).astype(np.float32); y=np.float32(r["label"])
        return torch.from_numpy(E), torch.from_numpy(np.array(y))

def load_index(csv_path):
    rows=[]
    with open(csv_path,"r",encoding="utf-8") as f:
        for row in csv.DictReader(f): rows.append(row)
    return rows

def build_items(index_rows, embeds_root, folds):
    items=[]
    for row in index_rows:
        fold=int(row["fold"])
        if fold not in folds: continue
        vid=Path(row["cache_dir"]).name
        p=Path(embeds_root)/f"fold{fold}"/f"{vid}.npy"
        if p.exists(): items.append({"embed_path":str(p),"label":int(row["label"])})
    return items

@torch.no_grad()
def collect_logits(model, loader):
    model.eval(); logits=[]; labels=[]
    for E,y in tqdm(loader, desc="collect", leave=False):
        logits.append(model(E).cpu()); labels.append(y)
    logits=torch.cat(logits).numpy(); labels=torch.cat(labels).numpy()
    return logits, labels

# ---------------- Main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--embeds-root", default="data_cache/embeds_ft")
    ap.add_argument("--index-csv", default="splits/index.csv")
    ap.add_argument("--val-folds", nargs="+", type=int, default=[5])
    ap.add_argument("--head-path", default="models/video_attn_head.pt")
    ap.add_argument("--config-path", default="models/video_head_config.json")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--optimize", choices=["accuracy","f1"], default="accuracy")
    ap.add_argument("--calibrate", action="store_true",
                    help="If set, fit a temperature T on logits instead of threshold grid.")
    ap.add_argument("--calib-path", default="models/calibration.json")
    args=ap.parse_args()

    cfg=json.load(open(args.config_path,"r"))
    D=int(cfg["D"]); head_type=cfg.get("head","attention")

    idx=load_index(args.index_csv)
    items=build_items(idx, args.embeds_root, args.val_folds)
    if not items: print("[!] No items to evaluate"); return
    dl=DataLoader(EmbedDS(items), batch_size=args.batch, shuffle=False)

    # build head & load weights
    if head_type=="attention":
        model=AttnPoolHead(D)
    else:
        model=StatsPoolHead(D)
    state=torch.load(args.head_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state)

    logits, labels = collect_logits(model, dl)
    labels = labels.astype(int)

    if args.calibrate:
        # ---------------- Temperature scaling ----------------
        print("[CALIB] Fitting temperature scaling (T_video)...")
        logits_t = torch.tensor(logits, dtype=torch.float32)
        y_t = torch.tensor(labels, dtype=torch.float32)

        T = torch.tensor(1.0, requires_grad=True)
        opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

        def closure():
            opt.zero_grad()
            scaled = logits_t / T
            loss = nn.BCEWithLogitsLoss()(scaled, y_t)
            loss.backward()
            return loss

        opt.step(closure)
        T_val = float(T.detach().item())
        print(f"[CALIB] Learned T_video = {T_val:.4f}")

        calib_path = Path(args.calib_path)
        calib = {}
        if calib_path.exists():
            try:
                calib = json.loads(calib_path.read_text())
            except: calib = {}
        calib["T_video"] = T_val
        with open(calib_path,"w",encoding="utf-8") as f:
            json.dump(calib,f,indent=2)
        print(f"[i] Wrote calibration to {calib_path}")
        return

    # ---------------- Threshold tuning ----------------
    probs = 1/(1+np.exp(-logits))

    best_thr=0.5; best_score=-1; best_cm=None
    for thr in np.linspace(0.0, 1.0, 1001):
        preds = (probs>=thr).astype(int)
        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, zero_division=0)
        score = acc if args.optimize=="accuracy" else f1
        if score>best_score:
            best_score=score; best_thr=float(thr)
            best_cm = confusion_matrix(labels, preds).tolist()

    try: auc=float(roc_auc_score(labels, probs))
    except Exception: auc=float("nan")

    print(f"[TUNE] head={head_type} | optimize={args.optimize}")
    print(f"[TUNE] best_threshold={best_thr:.4f} | best_{args.optimize}={best_score:.4f} | AUC={auc:.4f} | cm={best_cm}")

    # write threshold back
    cfg["threshold"]=best_thr
    with open(args.config_path,"w",encoding="utf-8") as f:
        json.dump(cfg,f,indent=2)
    print(f"[i] Wrote threshold={best_thr:.4f} to {args.config_path}")

if __name__=="__main__": main()
