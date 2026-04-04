"""Run 3D evaluation for temporal CLIPSeg checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.metrics.meandice import compute_dice
from PIL import Image
from tqdm import tqdm
from transformers import CLIPSegProcessor

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tgated.constants import ORGAN_TEXT_PROMPTS
from tgated.models import CLIPSegTemporalAdapter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

organ_keys = list(ORGAN_TEXT_PROMPTS.keys())
organ_texts = list(ORGAN_TEXT_PROMPTS.values())


def parse_args():
    parser = argparse.ArgumentParser(description="Run temporal model inference")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="YAML config path")
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("test_dir", "data/processed/flare_2d/test_labeled")
    cfg.setdefault("checkpoint_path", "checkpoints/clipseg_temporal/temporal_adapter_no_unlabeled_BEST.pth")
    cfg.setdefault("context_size", 5)
    if args.test_dir is not None:
        cfg["test_dir"] = args.test_dir
    if args.checkpoint_path is not None:
        cfg["checkpoint_path"] = args.checkpoint_path
    return cfg


def load_model(checkpoint_path, context_size):
    model = CLIPSegTemporalAdapter(context_size=context_size).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model


def run_3d_evaluation(model, processor, test_dir):
    with open(test_dir / "manifest.jsonl", "r") as f:
        entries = [json.loads(line) for line in f]

    text_inputs = processor.tokenizer(organ_texts, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = text_inputs.input_ids.to(DEVICE)
    attention_mask = text_inputs.attention_mask.to(DEVICE)

    vol_entries = {}
    for e in entries:
        vol_entries.setdefault(e["vol_id"], []).append(e)

    vol_ids = sorted(vol_entries.keys())
    final_results = {k: [] for k in organ_keys}

    for vid in vol_ids:
        v_entries = vol_entries[vid]
        slice_map, mask_map = {}, {}
        for e in v_entries:
            z = int(e["slice_idx"])
            slice_map[z] = test_dir / e["image"]
            if e.get("mask") and e.get("prompt"):
                mask_map[(z, e["prompt"])] = test_dir / e["mask"]

        unique_z = sorted(slice_map.keys())
        depth = len(unique_z)
        n_organs = len(organ_keys)
        all_preds = torch.zeros(n_organs, depth, 352, 352)
        all_gts = torch.zeros(n_organs, depth, 352, 352)

        for zi, _ in enumerate(tqdm(unique_z, desc=f"{vid}", leave=False)):
            ctx = []
            for off in range(-2, 3):
                zi_neighbor = max(0, min(depth - 1, zi + off))
                ctx.append(Image.open(slice_map[unique_z[zi_neighbor]]).convert("RGB"))

            img_in = processor.image_processor(images=ctx, do_resize=False, do_center_crop=False, return_tensors="pt")
            pix = img_in.pixel_values.unsqueeze(0).to(DEVICE).repeat(n_organs, 1, 1, 1, 1)

            with torch.no_grad():
                logits = model(pix, input_ids, attention_mask)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu()
                all_preds[:, zi, :, :] = preds[:, 0, :, :]

            z_val = unique_z[zi]
            for oi, organ_key in enumerate(organ_keys):
                key = (z_val, organ_key)
                if key in mask_map:
                    gt_arr = np.array(Image.open(mask_map[key])) > 0
                    all_gts[oi, zi, :, :] = torch.from_numpy(gt_arr).float()

        for oi, organ_key in enumerate(organ_keys):
            y_pred = all_preds[oi].unsqueeze(0).unsqueeze(0)
            y_gt = all_gts[oi].unsqueeze(0).unsqueeze(0)
            gt_sum = y_gt.sum().item()
            pred_sum = y_pred.sum().item()

            if gt_sum > 0:
                dice_val = compute_dice(y_pred, y_gt, ignore_empty=False).item()
                final_results[organ_key].append(dice_val)
            elif pred_sum > 0:
                final_results[organ_key].append(0.0)

    total_scores = []
    print("\n3D TEST SET RESULTS")
    for organ_key in organ_keys:
        scores = final_results[organ_key]
        if scores:
            m = np.mean(scores)
            total_scores.append(m)
            print(f"{organ_key:<14} {m:.4f} (N={len(scores)})")
        else:
            print(f"{organ_key:<14} N/A")

    if total_scores:
        print(f"Mean Dice (all organs): {np.mean(total_scores):.4f}")
    return final_results


def main():
    args = parse_args()
    cfg = load_config(args)
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=True)
    model = load_model(cfg["checkpoint_path"], cfg["context_size"])
    run_3d_evaluation(model, processor, Path(cfg["test_dir"]))


if __name__ == "__main__":
    main()
