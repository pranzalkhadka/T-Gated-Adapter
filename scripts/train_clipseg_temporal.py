"""Train temporal CLIPSeg adapter using modular src implementation."""

import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPSegProcessor

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tgated.constants import NEGATIVE_RATIO
from tgated.data import TemporalDataset, make_weighted_sampler
from tgated.models import CLIPSegTemporalAdapter, gate_sparsity_loss, get_raw_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train temporal CLIPSeg adapter")
    parser.add_argument("--config", type=str, default="configs/train_clipseg_temporal.yaml", help="YAML config path")
    parser.add_argument("--data-root", type=str, default=None, help="Path to prepared 2D data root containing train_labeled/val_labeled")
    parser.add_argument("--train-manifest", type=str, default=None, help="Optional explicit train manifest path")
    parser.add_argument("--val-manifest", type=str, default=None, help="Optional explicit val manifest path")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("batch_size", 6)
    cfg.setdefault("context_size", 5)
    cfg.setdefault("num_epochs", 5)
    cfg.setdefault("lambda_gate", 0.001)
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("negative_ratio", NEGATIVE_RATIO)
    cfg.setdefault("checkpoint_dir", "checkpoints/clipseg_temporal")

    if args.train_manifest is not None:
        cfg["train_manifest"] = args.train_manifest
    if args.val_manifest is not None:
        cfg["val_manifest"] = args.val_manifest

    if args.data_root is not None and (args.train_manifest is None or args.val_manifest is None):
        data_root = Path(args.data_root)
        default_train = data_root / "train_labeled" / "manifest.jsonl"
        default_val = data_root / "val_labeled" / "manifest.jsonl"
        direct_train = data_root / "manifest.jsonl"
        sibling_val = data_root.parent / "val_labeled" / "manifest.jsonl"

        if args.train_manifest is None:
            if default_train.exists():
                cfg["train_manifest"] = str(default_train)
            elif direct_train.exists():
                cfg["train_manifest"] = str(direct_train)
        if args.val_manifest is None:
            if default_val.exists():
                cfg["val_manifest"] = str(default_val)
            elif sibling_val.exists():
                cfg["val_manifest"] = str(sibling_val)
    if args.checkpoint_dir is not None:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    return cfg


def save_checkpoint(
    epoch,
    model,
    optimizer,
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
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss_values": train_loss_values,
            "train_dice_values": train_dice_values,
            "val_loss_values": val_loss_values,
            "val_dice_values": val_dice_values,
            "per_organ_val_dice": per_organ_val_dice,
            "current_lrs": [pg["lr"] for pg in optimizer.param_groups],
        },
        ckpt_dir / filename,
    )


def main():
    args = parse_args()
    cfg = load_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=True)

    train_ds = TemporalDataset(
        cfg["train_manifest"], processor, context_size=cfg["context_size"], strict_mask=True, neg_ratio=cfg["negative_ratio"]
    )
    val_ds = TemporalDataset(cfg["val_manifest"], processor, context_size=cfg["context_size"], strict_mask=False, neg_ratio=0.0)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=make_weighted_sampler(train_ds),
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True
    )

    model = CLIPSegTemporalAdapter(context_size=cfg["context_size"]).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.base_model.clip.vision_model.parameters(), "lr": 1e-6},
            {"params": model.base_model.clip.text_model.parameters(), "lr": 1e-6},
            {"params": model.temporal_layers.parameters(), "lr": 5e-5},
            {"params": model.transformer_norm.parameters(), "lr": 5e-5},
            {"params": model.spatial_context.parameters(), "lr": 5e-5},
            {"params": model.gate_proj.parameters(), "lr": 5e-5},
            {"params": model.proj_in.parameters(), "lr": 5e-5},
            {"params": model.proj_out.parameters(), "lr": 5e-5},
            {"params": model.mlp_head.parameters(), "lr": 5e-5},
            {"params": model.base_model.decoder.parameters(), "lr": 1e-5},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-7)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(sigmoid=True, batch=False, include_background=True)
    val_dice_metric = DiceMetric(include_background=True, reduction="mean")

    train_loss_values, train_dice_values = [], []
    val_loss_values, val_dice_values = [], []
    per_organ_val_dice = []
    best_val_dice = 0.0

    checkpoint_dir = cfg["checkpoint_dir"]
    for epoch in range(cfg["num_epochs"]):
        model.train()
        total_train_loss = 0.0
        trained_steps = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{cfg['num_epochs']}")
        for i, batch in enumerate(pbar):
            images = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            masks = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(images, input_ids, attention_mask).float()
            gate_reg = gate_sparsity_loss(model, cfg["lambda_gate"])
            loss = bce_loss(logits, masks) + dice_loss(logits, masks) + gate_reg

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

            optimizer.step()
            total_train_loss += loss.item()
            trained_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{optimizer.param_groups[2]['lr']:.1e}"})

            if i % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        scheduler.step()
        avg_train_loss = total_train_loss / max(trained_steps, 1)
        train_loss_values.append(avg_train_loss)
        train_dice_values.append(0.0)

        model.eval()
        val_dice_metric.reset()
        total_val_loss = 0.0
        organ_dice_sum = defaultdict(float)
        organ_dice_count = defaultdict(int)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in val_loader:
                images = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                masks = batch["labels"].to(device)
                organ_keys_b = batch["organ_key"]

                raw = get_raw_model(model)
                logits = raw(images, input_ids, attention_mask).float()
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
                    organ_dice_sum[organ_keys_b[b]] += sample_dice
                    organ_dice_count[organ_keys_b[b]] += 1

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
                train_loss_values,
                train_dice_values,
                val_loss_values,
                val_dice_values,
                per_organ_val_dice,
                "temporal_adapter_no_unlabeled_BEST.pth",
                checkpoint_dir,
            )

        save_checkpoint(
            epoch + 1,
            model,
            optimizer,
            train_loss_values,
            train_dice_values,
            val_loss_values,
            val_dice_values,
            per_organ_val_dice,
            f"temporal_adapter_no_unlabeled_epoch_{epoch+1}.pth",
            checkpoint_dir,
        )

    print(f"Done. Best val dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
