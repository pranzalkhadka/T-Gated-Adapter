"""Prepare FLARE 3D volumes into triplanar 2D slices and manifests.

This is the script equivalent of `notebooks/data_prep_flare.ipynb`.
"""

import argparse
import json
import random
from pathlib import Path

import imageio
import nibabel as nib
import numpy as np
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

PLANES = [
    ("axial", lambda vol, i: vol[:, :, i], lambda vol: vol.shape[2]),
    ("coronal", lambda vol, i: vol[:, i, :], lambda vol: vol.shape[1]),
    ("sagittal", lambda vol, i: vol[i, :, :], lambda vol: vol.shape[0]),
]


def window_and_normalize(slice_img: np.ndarray, clip_window: tuple[int, int]) -> np.ndarray:
    lo, hi = clip_window
    arr = np.clip(slice_img.astype(np.float32), lo, hi)
    return (((arr - lo) / (hi - lo)) * 255).astype(np.uint8)


def process_volume_plane(
    vol_id: str,
    img_vol: np.ndarray,
    mask_vol: np.ndarray | None,
    plane_name: str,
    slicer,
    depth_fn,
    img_dir: Path,
    mask_dir: Path,
    is_labeled: bool,
    clip_window: tuple[int, int],
    target_size: tuple[int, int],
) -> list[dict]:
    entries = []
    n_slices = depth_fn(img_vol)

    vol_img_out = img_dir / plane_name / vol_id
    vol_img_out.mkdir(parents=True, exist_ok=True)

    if is_labeled and mask_vol is not None:
        vol_mask_out = mask_dir / plane_name / vol_id
        vol_mask_out.mkdir(parents=True, exist_ok=True)
    else:
        vol_mask_out = None

    for idx in range(n_slices):
        raw_slice = slicer(img_vol, idx)
        normed = window_and_normalize(raw_slice, clip_window)
        img_pil = Image.fromarray(normed).resize(target_size, Image.LANCZOS)

        slice_name = f"{idx:03d}.png"
        img_save_path = vol_img_out / slice_name
        imageio.imwrite(img_save_path, np.array(img_pil))
        relative_img = f"images/{plane_name}/{vol_id}/{slice_name}"

        if is_labeled and mask_vol is not None:
            slice_mask = slicer(mask_vol, idx).astype(np.int32)
            present_labels = np.unique(slice_mask)
            present_labels = present_labels[present_labels != 0]

            if len(present_labels) == 0:
                entries.append(
                    {
                        "vol_id": vol_id,
                        "plane": plane_name,
                        "slice_idx": idx,
                        "image": relative_img,
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

                    mask_name = f"{idx:03d}_{organ_name}.png"
                    imageio.imwrite(vol_mask_out / mask_name, np.array(mask_pil))

                    entries.append(
                        {
                            "vol_id": vol_id,
                            "plane": plane_name,
                            "slice_idx": idx,
                            "image": relative_img,
                            "mask": f"masks/{plane_name}/{vol_id}/{mask_name}",
                            "prompt": organ_name,
                            "labeled": True,
                        }
                    )
        else:
            entries.append(
                {
                    "vol_id": vol_id,
                    "plane": plane_name,
                    "slice_idx": idx,
                    "image": relative_img,
                    "mask": None,
                    "prompt": None,
                    "labeled": False,
                }
            )

    return entries


def process_set(
    volumes: list[Path],
    split_name: str,
    out_root: Path,
    lbl_path: Path,
    clip_window: tuple[int, int],
    target_size: tuple[int, int],
    is_labeled: bool = True,
) -> None:
    print(f"\nProcessing {split_name} ({len(volumes)} volumes)...")
    split_dir = out_root / split_name
    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    if is_labeled:
        mask_dir.mkdir(parents=True, exist_ok=True)

    all_entries = []

    for vol_path in tqdm(volumes):
        vol_id = vol_path.name.replace("_0000.nii.gz", "").replace("_0000.nii", "")
        img_vol = nib.load(str(vol_path)).get_fdata()

        mask_vol = None
        if is_labeled:
            m_path = lbl_path / (vol_id + ".nii")
            if m_path.exists():
                mask_vol = nib.load(str(m_path)).get_fdata()

        for plane_name, slicer, depth_fn in PLANES:
            entries = process_volume_plane(
                vol_id,
                img_vol,
                mask_vol,
                plane_name,
                slicer,
                depth_fn,
                img_dir,
                mask_dir,
                is_labeled,
                clip_window,
                target_size,
            )
            all_entries.extend(entries)

    manifest_path = split_dir / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    for plane_name, _, _ in PLANES:
        plane_entries = [e for e in all_entries if e["plane"] == plane_name]
        plane_manifest = split_dir / f"manifest_{plane_name}.jsonl"
        with plane_manifest.open("w") as f:
            for entry in plane_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"  {plane_name:>8}: {len(plane_entries):>7} entries")

    print(f"  {'TOTAL':>8}: {len(all_entries):>7} entries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FLARE triplanar 2D dataset")
    parser.add_argument("--img-path", type=Path, default=Path("data/raw/flare/images"))
    parser.add_argument("--lbl-path", type=Path, default=Path("data/raw/flare/labels"))
    parser.add_argument("--out-root", type=Path, default=Path("data/processed/flare_2d"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-min", type=int, default=-125)
    parser.add_argument("--clip-max", type=int, default=275)
    parser.add_argument("--target-size", type=int, nargs=2, default=[352, 352])
    parser.add_argument("--train-count", type=int, default=30)
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--test-count", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    volumes = sorted(list(args.img_path.glob("*.nii*")))
    random.shuffle(volumes)

    n_train = args.train_count
    n_val = args.val_count
    n_test = args.test_count

    clip_window = (args.clip_min, args.clip_max)
    target_size = tuple(args.target_size)

    process_set(
        volumes[:n_train],
        "train_labeled",
        args.out_root,
        args.lbl_path,
        clip_window,
        target_size,
        is_labeled=True,
    )
    process_set(
        volumes[n_train : n_train + n_val],
        "val_labeled",
        args.out_root,
        args.lbl_path,
        clip_window,
        target_size,
        is_labeled=True,
    )
    process_set(
        volumes[n_train + n_val : n_train + n_val + n_test],
        "test_labeled",
        args.out_root,
        args.lbl_path,
        clip_window,
        target_size,
        is_labeled=True,
    )

    print("\nTriplanar data preparation complete.")


if __name__ == "__main__":
    main()
