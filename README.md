# colorAIzer
CVAI project to colorize black and white images

# Setup Repository
This project uses Python 3.12, so make sure this is installed on your device. Afterwards, the virtual environment creation and dependency installation can both be done by running the script setup_repository.sh as shown below

Linux/macOS: 
Create .venv, upgrades pip and installs requirements.txt
````shell
source setup_repository.sh 
````

Windows: 
````powershell
.\setup_repository.ps1
````


## Training
````
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
````

python train.py \
  --data_dir /exchange/cvai/colorAIzer/data/coco2017 \
  --pretrain_epochs 40 \
  --gan_epochs 50 \
  --freeze_strategy after_5 \
  --lr_disc 5e-5 \
  --lr_gen 5e-4 \
  --grid_every 1 \
  --ckpt_every 1 \
  --max_images 5000 \
  --val_pct 0.05 \
  --l1_weight 50
