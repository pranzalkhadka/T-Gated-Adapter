"""Dataset utilities for CLIPSeg baseline and temporal training."""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm

from tgated.constants import LATERALIZED_ORGANS, ORGAN_TEXT_PROMPTS, ORGAN_WEIGHTS


class CLIPSegBaselineDataset(Dataset):
    def __init__(self, manifest_path, strict_mask=True, neg_ratio=0.10, seed=42):
        self.root_dir = Path(manifest_path).parent
        with open(manifest_path, "r") as f:
            self.entries = [json.loads(line) for line in f]

        self.valid_indices = []
        self.sample_weights = []

        if not strict_mask:
            for i, x in enumerate(self.entries):
                if x["prompt"] is not None and x["mask"] is not None:
                    self.valid_indices.append(i)
                    self.sample_weights.append(ORGAN_WEIGHTS.get(x["prompt"], 1.0))
            return

        rng = random.Random(seed)
        null_slice_indices = []
        pos_count_per_organ = defaultdict(int)

        for i, x in enumerate(tqdm(self.entries, desc="Checking masks")):
            if x["prompt"] is None:
                null_slice_indices.append(i)
                continue

            mask_arr = np.array(Image.open(self.root_dir / x["mask"]))
            if mask_arr.max() == 0:
                continue

            self.valid_indices.append(i)
            self.sample_weights.append(ORGAN_WEIGHTS.get(x["prompt"], 1.0))
            pos_count_per_organ[x["prompt"]] += 1

        if null_slice_indices:
            for organ in pos_count_per_organ:
                n_neg = max(1, int(pos_count_per_organ[organ] * neg_ratio))
                sampled = rng.choices(null_slice_indices, k=n_neg)
                for null_idx in sampled:
                    synthetic = {
                        "type": "negative",
                        "src_idx": null_idx,
                        "prompt": organ,
                        "vol_id": self.entries[null_idx]["vol_id"],
                        "slice_idx": self.entries[null_idx]["slice_idx"],
                        "image": self.entries[null_idx]["image"],
                    }
                    self.valid_indices.append(synthetic)
                    self.sample_weights.append(ORGAN_WEIGHTS.get(organ, 1.0) * 0.4)

    def __len__(self):
        return len(self.valid_indices)

    def _apply_augmentation(self, image_tensor, labels, prompt):
        if prompt not in LATERALIZED_ORGANS and random.random() > 0.5:
            image_tensor = TF.hflip(image_tensor)
            if labels is not None:
                labels = TF.hflip(labels)

        angle = random.uniform(-5.0, 5.0)
        image_tensor = TF.rotate(image_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
        if labels is not None:
            labels = TF.rotate(labels, angle, interpolation=TF.InterpolationMode.NEAREST)
        return image_tensor, labels

    def __getitem__(self, i):
        entry = self.valid_indices[i]

        if isinstance(entry, dict):
            src_entry = self.entries[entry["src_idx"]]
            prompt = entry["prompt"]
            image = Image.open(self.root_dir / src_entry["image"]).convert("RGB")
            text = ORGAN_TEXT_PROMPTS.get(prompt, prompt)
            labels = torch.zeros(1, 352, 352, dtype=torch.float32)
            img_tensor = TF.to_tensor(image)
            img_tensor, labels = self._apply_augmentation(img_tensor, labels, prompt)
            return {
                "image": img_tensor,
                "text": text,
                "organ_key": prompt,
                "labels": labels,
                "is_negative": True,
                "vol_id": entry["vol_id"],
                "slice_idx": torch.tensor(entry["slice_idx"], dtype=torch.long),
            }

        center_entry = self.entries[entry]
        prompt = center_entry["prompt"]
        text = ORGAN_TEXT_PROMPTS.get(prompt, prompt) if prompt else "abdominal organ"
        image = Image.open(self.root_dir / center_entry["image"]).convert("RGB")
        mask_arr = np.array(Image.open(self.root_dir / center_entry["mask"]))
        labels = torch.tensor((mask_arr > 0).astype(np.float32)).unsqueeze(0)
        img_tensor = TF.to_tensor(image)
        img_tensor, labels = self._apply_augmentation(img_tensor, labels, prompt if prompt else "")
        return {
            "image": img_tensor,
            "text": text,
            "organ_key": prompt if prompt else "unknown",
            "labels": labels,
            "is_negative": False,
            "vol_id": center_entry["vol_id"],
            "slice_idx": torch.tensor(center_entry["slice_idx"], dtype=torch.long),
        }


class TemporalDataset(Dataset):
    def __init__(self, manifest_path, processor, context_size=5, strict_mask=True, neg_ratio=0.10, seed=42):
        self.processor = processor
        self.context_size = context_size
        self.half_ctx = context_size // 2
        self.root_dir = Path(manifest_path).parent

        self.precomputed_tokens = {}
        all_prompts = list(ORGAN_TEXT_PROMPTS.values()) + ["abdominal organ"]
        for text in all_prompts:
            inputs = processor.tokenizer([text] * context_size, padding="max_length", truncation=True, return_tensors="pt")
            self.precomputed_tokens[text] = {
                "input_ids": inputs.input_ids[0],
                "attention_mask": inputs.attention_mask[0],
            }

        with open(manifest_path, "r") as f:
            self.entries = [json.loads(line) for line in f]

        self.vol_indices = {}
        for idx, entry in enumerate(self.entries):
            self.vol_indices.setdefault(entry["vol_id"], []).append(idx)
        for vid in self.vol_indices:
            self.vol_indices[vid].sort(key=lambda x: self.entries[x]["slice_idx"])

        self.global_to_local = {}
        for vid, gidx_list in self.vol_indices.items():
            for local_pos, gidx in enumerate(gidx_list):
                self.global_to_local[gidx] = local_pos

        self.valid_indices = []
        self.sample_weights = []

        rng = random.Random(seed)
        null_slice_indices = []
        pos_count_per_organ = defaultdict(int)

        for i, x in enumerate(tqdm(self.entries, desc="Checking Masks")):
            if x["prompt"] is None:
                null_slice_indices.append(i)
                continue
            mask_arr = np.array(Image.open(self.root_dir / x["mask"]))
            if mask_arr.max() == 0 and strict_mask:
                continue
            self.valid_indices.append(i)
            self.sample_weights.append(ORGAN_WEIGHTS.get(x["prompt"], 1.0))
            if mask_arr.max() > 0:
                pos_count_per_organ[x["prompt"]] += 1

        if strict_mask and null_slice_indices:
            for organ in pos_count_per_organ:
                n_neg = max(1, int(pos_count_per_organ[organ] * neg_ratio))
                sampled = rng.choices(null_slice_indices, k=n_neg)
                for null_idx in sampled:
                    synthetic = {
                        "type": "negative",
                        "src_idx": null_idx,
                        "prompt": organ,
                        "vol_id": self.entries[null_idx]["vol_id"],
                        "slice_idx": self.entries[null_idx]["slice_idx"],
                        "image": self.entries[null_idx]["image"],
                    }
                    self.valid_indices.append(synthetic)
                    self.sample_weights.append(ORGAN_WEIGHTS.get(organ, 1.0) * 0.4)

    def _get_tokens(self, text_prompt):
        if text_prompt not in self.precomputed_tokens:
            inputs = self.processor.tokenizer([text_prompt] * self.context_size, padding="max_length", truncation=True, return_tensors="pt")
            self.precomputed_tokens[text_prompt] = {
                "input_ids": inputs.input_ids[0],
                "attention_mask": inputs.attention_mask[0],
            }
        return self.precomputed_tokens[text_prompt]

    def __len__(self):
        return len(self.valid_indices)

    def load_image(self, rel_path):
        return Image.open(self.root_dir / rel_path).convert("RGB")

    def _build_context_stack(self, vol_id, center_global_idx):
        vol_list = self.vol_indices[vol_id]
        local_pos = self.global_to_local[center_global_idx]
        stack = []
        for offset in range(-self.half_ctx, self.half_ctx + 1):
            neighbor_pos = max(0, min(local_pos + offset, len(vol_list) - 1))
            stack.append(self.load_image(self.entries[vol_list[neighbor_pos]]["image"]))
        return stack

    def _apply_augmentation(self, pixel_values, labels, prompt):
        if prompt not in LATERALIZED_ORGANS and random.random() > 0.5:
            pixel_values = TF.hflip(pixel_values)
            if labels is not None:
                labels = TF.hflip(labels)

        angle = random.uniform(-5.0, 5.0)
        pixel_values = TF.rotate(pixel_values, angle, interpolation=TF.InterpolationMode.BILINEAR)
        if labels is not None:
            labels = TF.rotate(labels, angle, interpolation=TF.InterpolationMode.NEAREST)
        return pixel_values, labels

    def __getitem__(self, i):
        entry = self.valid_indices[i]

        if isinstance(entry, dict):
            src_idx = entry["src_idx"]
            prompt = entry["prompt"]
            vol_id = entry["vol_id"]
            stack = self._build_context_stack(vol_id, src_idx)
            img_inputs = self.processor.image_processor(images=stack, do_resize=False, do_center_crop=False, return_tensors="pt")
            text_prompt = ORGAN_TEXT_PROMPTS.get(prompt, prompt)
            cached = self._get_tokens(text_prompt)
            pixel_values = img_inputs.pixel_values
            labels = torch.zeros(1, 352, 352, dtype=torch.float32)
            pixel_values, labels = self._apply_augmentation(pixel_values, labels, prompt)
            return {
                "pixel_values": pixel_values,
                "input_ids": cached["input_ids"],
                "attention_mask": cached["attention_mask"],
                "prompt": text_prompt,
                "organ_key": prompt,
                "labels": labels,
                "is_negative": True,
                "vol_id": vol_id,
                "slice_idx": torch.tensor(entry["slice_idx"], dtype=torch.long),
            }

        center_entry = self.entries[entry]
        vol_id = center_entry["vol_id"]
        prompt = center_entry["prompt"]
        stack = self._build_context_stack(vol_id, entry)
        img_inputs = self.processor.image_processor(images=stack, do_resize=False, do_center_crop=False, return_tensors="pt")
        text_prompt = ORGAN_TEXT_PROMPTS.get(prompt, prompt) if prompt else "abdominal organ"
        cached = self._get_tokens(text_prompt)
        pixel_values = img_inputs.pixel_values
        mask_arr = np.array(Image.open(self.root_dir / center_entry["mask"]))
        labels = torch.tensor((mask_arr > 0).astype(np.float32)).unsqueeze(0)
        pixel_values, labels = self._apply_augmentation(pixel_values, labels, prompt if prompt else "")
        return {
            "pixel_values": pixel_values,
            "input_ids": cached["input_ids"],
            "attention_mask": cached["attention_mask"],
            "prompt": text_prompt,
            "organ_key": prompt if prompt else "unknown",
            "is_negative": False,
            "vol_id": vol_id,
            "slice_idx": torch.tensor(center_entry["slice_idx"], dtype=torch.long),
            "labels": labels,
        }


def make_weighted_sampler(dataset):
    weights = torch.tensor(dataset.sample_weights, dtype=torch.float)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_baseline_collate_fn(processor):
    def collate_fn(batch):
        images = [TF.to_pil_image(b["image"]) for b in batch]
        texts = [b["text"] for b in batch]
        labels = torch.stack([b["labels"] for b in batch])
        organ_keys = [b["organ_key"] for b in batch]
        inputs = processor(images=images, text=texts, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "pixel_values": inputs.pixel_values,
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels,
            "organ_key": organ_keys,
        }

    return collate_fn
