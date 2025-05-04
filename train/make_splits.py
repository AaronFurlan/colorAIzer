
"""
make_splits.py - Create train/val/test txt files for a folder of images.
"""
from pathlib import Path
import random, argparse, sys

def main(folder, val_pct, test_pct, seed):
    rng = random.Random(seed)
    folder = Path(folder)
    images = sorted(p.name for p in folder.iterdir()
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not images:
        print("No images found.", file=sys.stderr)
        sys.exit(1)
    rng.shuffle(images)
    n = len(images)
    n_val   = int(n * val_pct)
    n_test  = int(n * test_pct)
    n_train = n - n_val - n_test
    splits = {
        'train.txt': images[:n_train],
        'val.txt':   images[n_train:n_train+n_val],
        'test.txt':  images[n_train+n_val:]
    }
    for fname, lines in splits.items():
        (folder / fname).write_text("\n".join(lines))
    print({k: len(v) for k,v in splits.items()})

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--val_pct", type=float, default=0.05)
    p.add_argument("--test_pct", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.folder, args.val_pct, args.test_pct, args.seed)
