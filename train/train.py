
#!/usr/bin/env python
"""
train.py - GAN training with train/val split, W&B logging,
fixed validation grid, dataset cap and graceful Ctrl‑C.
"""
import argparse, time, sys, signal
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dataset import ColorizationDataset
from models import build_generator, PatchDiscriminator
from utils import save_triplet_grid
from tqdm.auto import tqdm

# ---- optional W&B ----
try:
    import wandb
except ImportError:
    class _Stub:            # type: ignore
        def init(self,*_,**__): pass
        def log(self,*_,**__): pass
        def finish(self,*_,**__): pass
        def Image(self,*_,**__): return None
    wandb = _Stub()         # type: ignore
# -----------------------

def maybe_freeze_encoder(gen, strategy: str, epoch: int, unfreeze_after: int):
    if strategy == "always":
        for p in gen.encoder.parameters(): p.requires_grad = False
    elif strategy == "never":
        for p in gen.encoder.parameters(): p.requires_grad = True
    else:
        freeze = epoch < unfreeze_after
        for p in gen.encoder.parameters(): p.requires_grad = not freeze

def main(cfg):
    use_wandb = not cfg.wandb_off
    if use_wandb:
        run_name = (
            f"{cfg.encoder}"
            f"_pre{cfg.pretrain_epochs}"
            f"_l1{cfg.l1_weight}"
            f"_{cfg.freeze_strategy}"
            f"_lr{cfg.lr_gen}"
            f"_bs{cfg.batch_size}"
        )
        wandb.init(project=cfg.project_name,
           config=vars(cfg),
           name=run_name,
           tags=["sweep"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(cfg.data_dir)
    train_txt = cfg.train_list if cfg.train_list else root / "train.txt"
    val_txt   = cfg.val_list   if cfg.val_list else root / "val.txt"

    train_ds = ColorizationDataset.from_split_file(root, train_txt)
    val_ds   = ColorizationDataset.from_split_file(root, val_txt)

    if cfg.max_images:
        train_ds = Subset(train_ds, range(min(cfg.max_images, len(train_ds))))
        val_target = int(len(train_ds) * cfg.val_pct)
        if val_target and len(val_ds) > val_target:
            val_ds = Subset(val_ds, range(val_target))

    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_ld   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)} imgs  Val: {len(val_ds)} imgs")

    gen  = build_generator(cfg.encoder).to(device)
    disc = PatchDiscriminator().to(device)

    mse = nn.MSELoss()
    l1  = nn.L1Loss()
    bce = nn.BCELoss()

    opt_pre  = optim.Adam(gen.parameters(),  lr=cfg.lr_pre,  betas=(0.5,0.999))
    opt_gen  = optim.Adam(gen.parameters(),  lr=cfg.lr_gen,  betas=(0.5,0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.lr_disc, betas=(0.5,0.999))

    run_stamp = wandb.run.id if use_wandb else datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(cfg.ckpt_dir) / run_stamp
    vis_dir  = Path(cfg.vis_dir)  / run_stamp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True,  exist_ok=True)

    # fixed validation batch
    fixed_val_gray, fixed_val_color, _ = next(iter(val_ld))
    fixed_val_gray, fixed_val_color = fixed_val_gray.to(device), fixed_val_color.to(device)

    def _dir_size(path: Path) -> int:
        return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())

    MAX_DIR_BYTES = 5 * 1024**3
    def save_ckpt(epoch: int, tag: str = ""):
        # 1) save the new checkpoint
        path = ckpt_dir / f"ckpt{tag}_ep{epoch}.pt"
        torch.save({"epoch": epoch,
                    "gen": gen.state_dict(),
                    "disc": disc.state_dict()},
                   path)
    
        # 2) if folder > 5 GB, prune oldest ckpt(s)
        #while _dir_size(ckpt_dir) > MAX_DIR_BYTES:
        #    ckpts = sorted(ckpt_dir.glob("ckpt*_ep*.pt"), key=lambda p: p.stat().st_mtime)
        #    if len(ckpts) <= 1:
        #        break  # never delete the only file
        #    oldest = ckpts[0]
        #    print(f"🗑️  {oldest.name} exceeds 5 GB cap — deleting")
        #    oldest.unlink()

    # graceful interrupt
    def _handler(_sig,_frm):
        print("\n⚠️  Interrupted. Saving checkpoint and exiting.")
        save_ckpt(current_epoch, "_interrupt")
        if use_wandb: wandb.finish()
        sys.exit(0)
    signal.signal(signal.SIGINT, _handler)

    # 1) optional pre‑train
    current_epoch = 0
    for ep in range(1, cfg.pretrain_epochs + 1):
        current_epoch = ep
        gen.train(); running = 0.; t0 = time.time()
        pbar = tqdm(train_ld, desc=f"Pre {ep}/{cfg.pretrain_epochs}", leave=False)
        for gray, color, _ in pbar:
            gray, color = gray.to(device), color.to(device)
            pred = torch.tanh(gen(gray))
            loss = mse(pred, color)
            opt_pre.zero_grad(); loss.backward(); opt_pre.step()
            running += loss.item()
            pbar.set_postfix(mse=f"{loss.item():.3f}")
        avg = running / len(train_ld)
        print(f"[Pre] {ep}/{cfg.pretrain_epochs}  MSE={avg:.4f}  {time.time()-t0:.1f}s")
        if use_wandb: 
            wandb.log({"pre/mse": avg, "epoch": current_epoch})
            # grids every N epochs
            if cfg.grid_every and ep % cfg.grid_every == 0:
                with torch.no_grad():
                    val_pred = torch.tanh(gen(fixed_val_gray))
                grid_path = vis_dir / f"pre_grid_ep{ep}.png"
                save_triplet_grid(fixed_val_gray.cpu(), val_pred.cpu(), fixed_val_color.cpu(),
                                  grid_path, n_vis=cfg.n_vis)
                wandb.log({"pre_grid": wandb.Image(str(grid_path))}, step=current_epoch)



    # freeze schedule
    if cfg.freeze_strategy.startswith("after_"):
        unfreeze_after = int(cfg.freeze_strategy.split("_")[1])
        strategy_tag = "after"
    else:
        unfreeze_after = 0
        strategy_tag = cfg.freeze_strategy

    # 2) GAN
    for ep in tqdm(range(1, cfg.gan_epochs + 1), desc="GAN epochs"):
        current_epoch = cfg.pretrain_epochs + ep
        gen.train(); disc.train()
        maybe_freeze_encoder(gen, strategy_tag, ep, unfreeze_after)

        run_d = run_g = 0.; t0 = time.time()
        pbar = tqdm(train_ld, desc=f"Epoch {ep}/{cfg.gan_epochs}", leave=False)
        for gray, color, _ in pbar:
            gray, color = gray.to(device), color.to(device)
            fake = torch.tanh(gen(gray))

            real_p = disc(gray, color)
            fake_p = disc(gray, fake.detach())
            real_lbl = torch.ones_like(real_p)
            fake_lbl = torch.zeros_like(fake_p)
            loss_d = 0.5*(bce(real_p, real_lbl) + bce(fake_p, fake_lbl))
            opt_disc.zero_grad(); loss_d.backward(); opt_disc.step()

            pred_p = disc(gray, fake)
            loss_g = bce(pred_p, real_lbl) + cfg.l1_weight * l1(fake, color)
            opt_gen.zero_grad(); loss_g.backward(); opt_gen.step()

            run_d += loss_d.item(); run_g += loss_g.item()
            pbar.set_postfix(
                d_loss=f"{loss_d.item():.3f}",
                g_loss=f"{loss_g.item():.3f}"
            )

        avg_d = run_d/len(train_ld); avg_g = run_g/len(train_ld)
        print(f"[GAN] {ep}/{cfg.gan_epochs}  D={avg_d:.4f}  G={avg_g:.4f}  {time.time()-t0:.1f}s")
        if use_wandb:
            wandb.log({"gan/loss_d": avg_d, "gan/loss_g": avg_g, "epoch": current_epoch})

        # grids every N epochs
        if cfg.grid_every and ep % cfg.grid_every == 0:
            with torch.no_grad():
                val_pred = torch.tanh(gen(fixed_val_gray))
            grid_path = vis_dir / f"val_grid_ep{ep}.png"
            save_triplet_grid(fixed_val_gray.cpu(), val_pred.cpu(), fixed_val_color.cpu(),
                              grid_path, n_vis=cfg.n_vis)
            if use_wandb:
                wandb.log({"val_grid": wandb.Image(str(grid_path))}, step=current_epoch)

        # checkpoint
        if cfg.ckpt_every and ep % cfg.ckpt_every == 0:
            save_ckpt(current_epoch)

    if use_wandb: wandb.finish()
    print("✅ training complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--train_list")
    p.add_argument("--val_list")
    p.add_argument("--encoder", default="resnet18",
                   choices=["resnet18","resnet34","resnet50"])
    p.add_argument("--pretrain_epochs", type=int, default=0)
    p.add_argument("--gan_epochs", type=int, default=100)
    p.add_argument("--freeze_strategy", default="always",
                   help="always | never | after_<k>")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--l1_weight", type=float, default=100)
    p.add_argument("--lr_pre",  type=float, default=1e-3)
    p.add_argument("--lr_gen",  type=float, default=2e-4)
    p.add_argument("--lr_disc", type=float, default=1e-4)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--ckpt_every", type=int, default=10)
    p.add_argument("--vis_dir", default="grids")
    p.add_argument("--grid_every", type=int, default=5)
    p.add_argument("--n_vis", type=int, default=4)
    p.add_argument("--wandb_off", action="store_true")
    p.add_argument("--project_name", default="colorizer_v3")
    p.add_argument("--val_pct", type=float, default=0.05,
               help="Validation split expressed as fraction of *effective* "
                    "train size (applied after --max_images cap).")
    cfg = p.parse_args()
    main(cfg)
