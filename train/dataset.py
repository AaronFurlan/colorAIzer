
"""
dataset.py - Dataset for image colorization with optional split files.
"""

from pathlib import Path
from typing import List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# _IMAGENET_MEAN = [0.485, 0.456, 0.406]
# _IMAGENET_STD  = [0.229, 0.224, 0.225]
# _norm = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)


def make_transform(img_size: int, hflip: bool = False) -> transforms.Compose:
    tf = [
        transforms.Resize((img_size, img_size), InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0)   # (0,1) → (-1,1)
    ]
    if hflip:
        tf.insert(0, transforms.RandomHorizontalFlip())
    return transforms.Compose(tf)

_BASE_TF: transforms.Compose | None = None


class ColorizationDataset(Dataset):
    """Return (gray, color, stem) where both tensors are normalised."""
    def __init__(self,
             root_dir: str | Path,
             names_list: Optional[List[str]] = None,
             img_size: int = 256,
             hflip: bool = False):
        self.root_dir = Path(root_dir)
        if names_list:
            self.img_paths = [self.root_dir / n for n in names_list]
        else:
            self.img_paths = sorted(p for p in self.root_dir.iterdir()
                                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if not self.img_paths:
            raise RuntimeError(f'No images found in {root_dir}')

        global _BASE_TF               # create only once
        if _BASE_TF is None or _BASE_TF.transforms[0].size[0] != img_size:
            _BASE_TF = make_transform(img_size, hflip)
        self.tf = _BASE_TF
        
    @classmethod
    def from_split_file(cls, root_dir: str | Path, split_file: str | Path):
        """Instantiate using a txt file with one image filename per line."""
        lines = Path(split_file).read_text().splitlines()
        names = [l.strip() for l in lines if l.strip()]
        return cls(root_dir, names_list=names)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:        # unreadable / truncated
            print(f"⚠️  Skipping {path.name}: {e}")
            return self.__getitem__((idx + 1) % len(self))   # next image

        gray  = self.tf(img.convert("L")).expand(3, -1, -1)
        color = self.tf(img)
        #gray  = _norm(gray)
        #color = _norm(color)
        return gray, color, path.stem
