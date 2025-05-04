
#!/usr/bin/env python
"""
eval.py - generate a visual grid from a checkpoint.
Optionally pass --list txtfile to evaluate a specific split
(e.g. test.txt). Otherwise uses the first batch of data_dir.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ColorizationDataset
from models import build_generator
from utils import save_triplet_grid

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(cfg.data_dir)
    if cfg.list:
        ds = ColorizationDataset.from_split_file(root, cfg.list)
    else:
        ds = ColorizationDataset(root)
    ld = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    gen = build_generator(cfg.encoder, pretrained=False).to(device)
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    gen.load_state_dict(ckpt["gen"])
    gen.eval()

    gray, color, _ = next(iter(ld))
    gray, color = gray.to(device), color.to(device)
    with torch.no_grad():
        pred = torch.tanh(gen(gray))

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_grid.png"
    save_triplet_grid(gray.cpu(), pred.cpu(), color.cpu(), out_path, n_vis=cfg.n_vis)
    print(f"Grid saved to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--list", help="txt file listing images to load")
    p.add_argument("--encoder", default="resnet18",
                   choices=["resnet18","resnet34","resnet50"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out_dir", default="eval_results")
    p.add_argument("--n_vis", type=int, default=8)
    cfg = p.parse_args()
    main(cfg)
