# %%

# %%

# %%

# %%
import argparse
import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import nibabel as nib
from monai.networks.nets import DynUNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, ToTensord, EnsureTyped,
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Train DynUNet on FLARE volumes")
    parser.add_argument("--config", type=str, default="configs/train_dyunet.yaml", help="YAML config path")
    parser.add_argument("--img-path", type=str, default=None)
    parser.add_argument("--lbl-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("img_path", "data/raw/flare/images")
    cfg.setdefault("lbl_path", "data/raw/flare/labels")
    cfg.setdefault("checkpoint_dir", "checkpoints/dyunet")
    cfg.setdefault("num_classes", 14)
    cfg.setdefault("patch_size", [96, 96, 96])
    cfg.setdefault("batch_size", 2)
    cfg.setdefault("num_epochs", 300)
    cfg.setdefault("val_interval", 5)
    cfg.setdefault("num_workers", 2)
    cfg.setdefault("hu_min", -125)
    cfg.setdefault("hu_max", 275)
    cfg.setdefault("seed", 42)
    cfg.setdefault("train_count", 30)
    cfg.setdefault("val_count", 10)
    cfg.setdefault("test_count", 10)
    if args.img_path is not None:
        cfg["img_path"] = args.img_path
    if args.lbl_path is not None:
        cfg["lbl_path"] = args.lbl_path
    if args.checkpoint_dir is not None:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    return cfg


_args = parse_args()
_cfg = load_config(_args)

IMG_PATH = Path(_cfg["img_path"])
LBL_PATH = Path(_cfg["lbl_path"])
CKPT_DIR = Path(_cfg["checkpoint_dir"])
CKPT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = _cfg["num_classes"]
PATCH_SIZE = tuple(_cfg["patch_size"])
BATCH_SIZE = _cfg["batch_size"]
NUM_EPOCHS = _cfg["num_epochs"]
VAL_INTERVAL = _cfg["val_interval"]
NUM_WORKERS = _cfg["num_workers"]
HU_MIN, HU_MAX = _cfg["hu_min"], _cfg["hu_max"]

LABEL_MAP = {
    1: "liver",     2: "r_kidney",   3: "spleen",
    4: "pancreas",  5: "aorta",      6: "ivc",
    7: "r_adrenal", 8: "l_adrenal",  9: "gallbladder",
    10: "esophagus", 11: "stomach",  12: "duodenum",
    13: "l_kidney"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(_cfg["seed"])
labeled_vols = sorted(list(IMG_PATH.glob("*.nii*")))
random.shuffle(labeled_vols)

train_end = _cfg["train_count"]
val_end = train_end + _cfg["val_count"]
test_end = val_end + _cfg["test_count"]
train_vols = labeled_vols[:train_end]
val_vols = labeled_vols[train_end:val_end]
test_vols = labeled_vols[val_end:test_end]

print(f"Train: {len(train_vols)} | Val: {len(val_vols)} | Test: {len(test_vols)}")

def make_data_dicts(vol_list, lbl_path, include_label=True):
    """Build list of {image, label} dicts for MONAI Dataset."""
    data = []
    for vol_path in vol_list:
        vol_id = vol_path.name
        for suffix in ["_0000.nii.gz", "_0000.nii", ".nii.gz", ".nii"]:
            vol_id = vol_id.replace(suffix, "")

        lbl_file = lbl_path / f"{vol_id}.nii"
        if not lbl_file.exists():
            lbl_file = lbl_path / f"{vol_id}.nii.gz"

        entry = {"image": str(vol_path)}
        if include_label and lbl_file.exists():
            entry["label"] = str(lbl_file)
        elif include_label:
            print(f"⚠️  Label not found: {vol_id}")
            continue
        data.append(entry)
    return data

train_dicts = make_data_dicts(train_vols, LBL_PATH)
val_dicts   = make_data_dicts(val_vols,   LBL_PATH)
test_dicts  = make_data_dicts(test_vols,  LBL_PATH)

print(f"Train dicts: {len(train_dicts)}")
print(f"Val dicts  : {len(val_dicts)}")
print(f"Test dicts : {len(test_dicts)}")

base_transforms = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
 
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.5, 1.5, 1.5),
        mode=("bilinear", "nearest"),
    ),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=HU_MIN, a_max=HU_MAX,
        b_min=0.0,    b_max=1.0,
        clip=True,
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"]),
]

train_transforms = Compose(base_transforms + [
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=PATCH_SIZE,
        pos=1, neg=1,
        num_samples=2,      
        image_key="image",
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose(base_transforms + [
    ToTensord(keys=["image", "label"]),
])

train_ds = Dataset(data=train_dicts, transform=train_transforms)
val_ds   = Dataset(data=val_dicts,   transform=val_transforms)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True)

val_loader   = DataLoader(
    val_ds, batch_size=1, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True)


kernels  = [[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]]
strides  = [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]

model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=NUM_CLASSES,
    kernel_size=kernels,
    strides=strides,
    upsample_kernel_size=strides[1:],   
    norm_name="instance",              
    deep_supervision=True,             
    deep_supr_num=2,                    
).to(device)

print(f"DynUNet parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
loss_fn = DiceCELoss(
    to_onehot_y=True,      
    softmax=True,            
    include_background=False, 
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,           
    weight_decay=1e-5,
)

scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=1, eta_min=1e-6)

dice_metric = DiceMetric(
    include_background=False,
    reduction="mean_batch",  
    get_not_nans=True,
)

post_pred  = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
post_label = AsDiscrete(to_onehot=NUM_CLASSES)

scaler = torch.amp.GradScaler('cuda')

def save_checkpoint(epoch, model, optimizer, scheduler, scaler,
                    train_losses, val_dices, per_organ_dices, filename):
    torch.save({
        'epoch':            epoch,
        'model_state_dict': model.state_dict(),
        'optimizer':        optimizer.state_dict(),
        'scheduler':        scheduler.state_dict(),
        'scaler':           scaler.state_dict(),
        'train_losses':     train_losses,
        'val_dices':        val_dices,
        'per_organ_dices':  per_organ_dices,
    }, CKPT_DIR / filename)
    print(f"✅ Saved: {CKPT_DIR / filename}")

def model_inference(x):
    out = model(x)
    return out[:, 0] if out.ndim == 6 else out  

train_losses     = []
val_dices        = []
per_organ_dices  = []
best_val_dice    = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in pbar:
        images = batch["image"].to(device)   
        labels = batch["label"].to(device)  

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(images)

            if outputs.ndim == 6:
                weights = [0.5333, 0.2667, 0.2]
                loss = sum(
                    w * loss_fn(outputs[:, idx], labels)
                    for idx, w in enumerate(weights)
                    if idx < outputs.shape[1]
                )
            else:
                loss = loss_fn(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  NaN/Inf loss at epoch {epoch+1}, skipping batch.")
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr":   f"{optimizer.param_groups[0]['lr']:.1e}"
        })

    scheduler.step()
    avg_loss = total_loss / max(n_batches, 1)
    train_losses.append(avg_loss)

    if (epoch + 1) % VAL_INTERVAL == 0:
        model.eval()
        dice_metric.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader,
                              desc=f"Val Epoch {epoch+1}", leave=False):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=2,
                    predictor=model_inference,
                    overlap=0.5,
                )

                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]

                preds_list  = [post_pred(i)  for i in decollate_batch(outputs)]
                labels_list = [post_label(i) for i in decollate_batch(labels)]
                dice_metric(y_pred=preds_list, y=labels_list)

   
        per_class_dice, not_nans = dice_metric.aggregate()
        per_class_dice = per_class_dice.cpu().numpy()

        epoch_organ_dice = {}
        for label_idx, organ_name in LABEL_MAP.items():
            class_idx = label_idx - 1  
            if class_idx < len(per_class_dice):
                epoch_organ_dice[organ_name] = float(per_class_dice[class_idx])

        avg_val_dice = np.nanmean(per_class_dice)
        val_dices.append(avg_val_dice)
        per_organ_dices.append(epoch_organ_dice)

        print(f"\nEpoch {epoch+1:>3} | Train Loss: {avg_loss:.4f} "
              f"| Val Dice: {avg_val_dice:.4f}")
        print("  Per-organ Val Dice:")
        for organ, d in sorted(epoch_organ_dice.items(), key=lambda x: x[1]):
            print(f"    {organ:<14} {d:.3f}  {'█' * int(d * 20)}")

        is_best = avg_val_dice > best_val_dice
        if is_best:
            best_val_dice = avg_val_dice
            save_checkpoint(epoch+1, model, optimizer, scheduler, scaler,
                            train_losses, val_dices, per_organ_dices,
                            "dynunet_BEST.pth")
            print(f"  🏆 New best val dice: {best_val_dice:.4f}")

        save_checkpoint(epoch+1, model, optimizer, scheduler, scaler,
                        train_losses, val_dices, per_organ_dices,
                        f"dynunet_epoch_{epoch+1}.pth")
    else:
        print(f"Epoch {epoch+1:>3} | Train Loss: {avg_loss:.4f}")

print(f"\n✅ Training complete. Best val dice: {best_val_dice:.4f}")

# %%

# %%

# %%

# %%
