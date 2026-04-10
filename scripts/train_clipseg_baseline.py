import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPSegProcessor

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tgated.constants import NEGATIVE_RATIO
from tgated.data import CLIPSegBaselineDataset, make_baseline_collate_fn, make_weighted_sampler
from tgated.models import CLIPSegBaseline

def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIPSeg baseline")
    parser.add_argument("--config", type=str, default="configs/train_clipseg_baseline.yaml", help="YAML config path")
    parser.add_argument("--train-manifest", type=str, default=None, help="Optional explicit train manifest path")
    parser.add_argument("--val-manifest", type=str, default=None, help="Optional explicit val manifest path")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("batch_size", 64)
    cfg.setdefault("num_epochs", 25)
    cfg.setdefault("freeze_vision_layers", 4)
    cfg.setdefault("num_workers", 2)
    cfg.setdefault("negative_ratio", NEGATIVE_RATIO)
    cfg.setdefault("checkpoint_dir", "checkpoints/clipseg_baseline")

    if args.train_manifest is not None:
        cfg["train_manifest"] = args.train_manifest
    if args.val_manifest is not None:
        cfg["val_manifest"] = args.val_manifest

    if args.checkpoint_dir is not None:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    return cfg


def save_checkpoint(
    epoch,
    model,
    optimizer,
    scaler,
    train_loss_values,
    train_dice_values,
    val_loss_values,
    val_dice_values,
    per_organ_val_dice,
    filename,
    checkpoint_dir,
):
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_loss_values": train_loss_values,
        "train_dice_values": train_dice_values,
        "val_loss_values": val_loss_values,
        "val_dice_values": val_dice_values,
        "per_organ_val_dice": per_organ_val_dice,
    }
    save_path = ckpt_dir / filename
    torch.save(checkpoint, save_path)
    print(f"Saved: {save_path}")


def main():
    args = parse_args()
    cfg = load_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=True)

    train_ds = CLIPSegBaselineDataset(cfg["train_manifest"], strict_mask=True, neg_ratio=cfg["negative_ratio"])
    val_ds = CLIPSegBaselineDataset(cfg["val_manifest"], strict_mask=False, neg_ratio=0.0)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=make_weighted_sampler(train_ds),
        num_workers=cfg["num_workers"],
        pin_memory=True,
        collate_fn=make_baseline_collate_fn(processor),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        collate_fn=make_baseline_collate_fn(processor),
    )

    model = CLIPSegBaseline(freeze_vision_layers=cfg["freeze_vision_layers"]).to(device)
    optimizer = AdamW(
        [
            {"params": model.base_model.clip.vision_model.parameters(), "lr": 1e-6},
            {"params": model.base_model.clip.text_model.parameters(), "lr": 1e-6},
            {"params": model.base_model.decoder.parameters(), "lr": 5e-5},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-7)

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(sigmoid=True, batch=False, include_background=True)
    train_dice_metric = DiceMetric(include_background=True, reduction="mean")
    val_dice_metric = DiceMetric(include_background=True, reduction="mean")

    train_loss_values, train_dice_values = [], []
    val_loss_values, val_dice_values = [], []
    per_organ_val_dice = []
    best_val_dice = 0.0

    checkpoint_dir = cfg["checkpoint_dir"]
    for epoch in range(cfg["num_epochs"]):
        model.train()
        train_dice_metric.reset()
        total_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{cfg['num_epochs']}")
        for i, batch in enumerate(pbar):
            images = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            masks = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(images, input_ids, attention_mask)
                loss = bce_loss(logits, masks) + dice_loss(logits, masks)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_dice_metric(y_pred=preds, y=masks)
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{optimizer.param_groups[2]['lr']:.1e}"})

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_dice = train_dice_metric.aggregate().item()
        train_loss_values.append(avg_train_loss)
        train_dice_values.append(avg_train_dice)

        model.eval()
        val_dice_metric.reset()
        total_val_loss = 0.0
        organ_dice_sum = defaultdict(float)
        organ_dice_count = defaultdict(int)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                for batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{cfg['num_epochs']}"):
                    images = batch["pixel_values"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    masks = batch["labels"].to(device)
                    organ_keys = batch["organ_key"]

                    logits = model(images, input_ids, attention_mask)
                    loss = bce_loss(logits, masks) + dice_loss(logits, masks)
                    total_val_loss += loss.item()

                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_dice_metric(y_pred=preds, y=masks)

                    for b in range(preds.shape[0]):
                        p = preds[b]
                        t = masks[b]
                        intersection = (p * t).sum()
                        denom = p.sum() + t.sum()
                        sample_dice = (2.0 * intersection / denom).item() if denom > 0 else 1.0
                        organ_dice_sum[organ_keys[b]] += sample_dice
                        organ_dice_count[organ_keys[b]] += 1

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_dice = val_dice_metric.aggregate().item()
        val_loss_values.append(avg_val_loss)
        val_dice_values.append(avg_val_dice)

        epoch_organ_dice = {o: organ_dice_sum[o] / organ_dice_count[o] for o in organ_dice_sum}
        per_organ_val_dice.append(epoch_organ_dice)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            save_checkpoint(
                epoch + 1,
                model,
                optimizer,
                scaler,
                train_loss_values,
                train_dice_values,
                val_loss_values,
                val_dice_values,
                per_organ_val_dice,
                "clipseg_baseline_BEST.pth",
                checkpoint_dir,
            )

        save_checkpoint(
            epoch + 1,
            model,
            optimizer,
            scaler,
            train_loss_values,
            train_dice_values,
            val_loss_values,
            val_dice_values,
            per_organ_val_dice,
            f"clipseg_baseline_epoch_{epoch+1}.pth",
            checkpoint_dir,
        )

    print(f"Training complete. Best val dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
