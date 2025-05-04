
"""
utils.py - normalisation helpers and grid saving.
"""
from pathlib import Path
import torch
from torchvision.utils import make_grid, save_image

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def denorm(x: torch.Tensor) -> torch.Tensor:
    out = (x * _IMAGENET_STD.to(x) + _IMAGENET_MEAN.to(x)).clamp(0,1)
    if out.ndim == 4 and out.size(0) == 1:
        out = out[0]
    return out

def from_tanh(x: torch.Tensor) -> torch.Tensor:
    out = ((x + 1) / 2).clamp(0,1)
    if out.ndim == 4 and out.size(0) == 1:
        out = out[0]
    return out


def to_01(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1]  →  [0, 1]   (handles 3-D and 4-D tensors)."""
    out = ((x + 1.0) / 2.0).clamp(0, 1)
    if out.ndim == 4 and out.size(0) == 1:   # drop singleton batch
        out = out[0]
    return out


def save_triplet_grid(gray, pred, color, out_path: str | Path, n_vis: int = 4):
    """Save a (gray | pred | color) × N grid PNG for quick eyeballing."""
    n = min(n_vis, gray.size(0))
    cells = []
    for i in range(n):
        cells += [
            to_01(gray[i].cpu()),
            to_01(pred[i].cpu()),
            to_01(color[i].cpu())
        ]
    grid = make_grid(cells, nrow=3, padding=2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
