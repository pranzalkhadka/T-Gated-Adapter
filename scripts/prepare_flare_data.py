"""Prepare FLARE 3D volumes into 2D axial slices/manifests for CLIPSeg training."""

import argparse
import json
import random
from pathlib import Path

import imageio
import nibabel as nib
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

ORGAN_MAP = {
    1: "liver",
    2: "r_kidney",
    3: "spleen",
    4: "pancreas",
    5: "aorta",
    6: "ivc",
    7: "r_adrenal",
    8: "l_adrenal",
    9: "gallbladder",
    10: "esophagus",
    11: "stomach",
    12: "duodenum",
    13: "l_kidney",
}


def window_and_normalize(slice_img, clip_window):
    lo, hi = clip_window
    arr = np.clip(slice_img.astype(np.float32), lo, hi)
    return (((arr - lo) / (hi - lo)) * 255).astype(np.uint8)


def process_set(volumes, split_name, out_root, lbl_path, clip_window, target_size, is_labeled=True):
    print(f"\nProcessing {split_name} ({len(volumes)} volumes)...")
    split_dir = out_root / split_name
    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    if is_labeled:
        mask_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for vol_path in tqdm(volumes):
        vol_id = vol_path.name.replace("_0000.nii.gz", "").replace("_0000.nii", "")
        vol_img_out = img_dir / vol_id
        vol_img_out.mkdir(parents=True, exist_ok=True)

        img_vol = nib.load(str(vol_path)).get_fdata()

        mask_vol = None
        if is_labeled:
            m_path = lbl_path / f"{vol_id}.nii"
            if m_path.exists():
                mask_vol = nib.load(str(m_path)).get_fdata()

        depth = img_vol.shape[-1]

        for z in range(depth):
            slice_img = window_and_normalize(img_vol[:, :, z], clip_window)
            img_pil = Image.fromarray(slice_img).resize(target_size, Image.LANCZOS)

            slice_name = f"z{z:03d}.png"
            imageio.imwrite(vol_img_out / slice_name, np.array(img_pil))
            relative_img_path = f"images/{vol_id}/{slice_name}"

            if is_labeled and mask_vol is not None:
                slice_mask = mask_vol[:, :, z].astype(np.int32)
                present_labels = np.unique(slice_mask)
                present_labels = present_labels[present_labels != 0]

                vol_mask_out = mask_dir / vol_id
                vol_mask_out.mkdir(parents=True, exist_ok=True)

                if len(present_labels) == 0:
                    manifest.append(
                        {
                            "vol_id": vol_id,
                            "slice_idx": z,
                            "image": relative_img_path,
                            "mask": None,
                            "prompt": None,
                            "labeled": True,
                        }
                    )
                else:
                    for lab in present_labels:
                        organ_name = ORGAN_MAP.get(int(lab), f"label{lab}")
                        binary_mask = (slice_mask == lab).astype(np.uint8) * 255
                        mask_pil = Image.fromarray(binary_mask).resize(target_size, Image.NEAREST)

                        mask_name = f"z{z:03d}_{organ_name}.png"
                        imageio.imwrite(vol_mask_out / mask_name, np.array(mask_pil))

                        manifest.append(
                            {
                                "vol_id": vol_id,
                                "slice_idx": z,
                                "image": relative_img_path,
                                "mask": f"masks/{vol_id}/{mask_name}",
                                "prompt": organ_name,
                                "labeled": True,
                            }
                        )
            else:
                manifest.append(
                    {
                        "vol_id": vol_id,
                        "slice_idx": z,
                        "image": relative_img_path,
                        "mask": None,
                        "prompt": None,
                        "labeled": False,
                    }
                )

    with open(split_dir / "manifest.jsonl", "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare FLARE axial 2D dataset")
    parser.add_argument("--config", type=str, default="configs/data_prep_flare.yaml", help="YAML config path")
    parser.add_argument("--img-path", type=str, default=None)
    parser.add_argument("--lbl-path", type=str, default=None)
    parser.add_argument("--out-root", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("img_path", "data/raw/flare/images")
    cfg.setdefault("lbl_path", "data/raw/flare/labels")
    cfg.setdefault("out_root", "data/processed/flare_2d")
    cfg.setdefault("seed", 42)
    cfg.setdefault("clip_min", -125)
    cfg.setdefault("clip_max", 275)
    cfg.setdefault("target_size", [352, 352])
    cfg.setdefault("train_count", 30)
    cfg.setdefault("val_count", 10)
    cfg.setdefault("test_count", 10)

    if args.img_path is not None:
        cfg["img_path"] = args.img_path
    if args.lbl_path is not None:
        cfg["lbl_path"] = args.lbl_path
    if args.out_root is not None:
        cfg["out_root"] = args.out_root

    img_path = Path(cfg["img_path"])
    lbl_path = Path(cfg["lbl_path"])
    out_root = Path(cfg["out_root"])

    random.seed(cfg["seed"])
    labeled_vols = sorted(list(img_path.glob("*.nii*")))
    random.shuffle(labeled_vols)

    process_set(
        labeled_vols[: cfg["train_count"]],
        "train_labeled",
        out_root,
        lbl_path,
        (cfg["clip_min"], cfg["clip_max"]),
        tuple(cfg["target_size"]),
        is_labeled=True,
    )
    process_set(
        labeled_vols[cfg["train_count"] : cfg["train_count"] + cfg["val_count"]],
        "val_labeled",
        out_root,
        lbl_path,
        (cfg["clip_min"], cfg["clip_max"]),
        tuple(cfg["target_size"]),
        is_labeled=True,
    )
    process_set(
        labeled_vols[cfg["train_count"] + cfg["val_count"] : cfg["train_count"] + cfg["val_count"] + cfg["test_count"]],
        "test_labeled",
        out_root,
        lbl_path,
        (cfg["clip_min"], cfg["clip_max"]),
        tuple(cfg["target_size"]),
        is_labeled=True,
    )

    print("\nData Preparation Complete at 352x352 resolution!")


if __name__ == "__main__":
    main()
