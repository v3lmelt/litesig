"""Writer-disjoint pair dataset backed by NPZ arrays."""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class NpzSample:
    idx_a: int
    idx_b: int
    same_writer: bool
    writer_label: int
    forgery_label: int
    is_forged_a: bool = False
    is_forged_b: bool = False
    path_a: Optional[str] = None
    path_b: Optional[str] = None


class NpzWriterDisjointDataset(Dataset):
    """Creates signature pairs with writer-disjoint train/val splits from NPZ arrays."""

    def __init__(
        self,
        npz_path: str,
        split: str,
        transform,
        writer_split_ratio: float = 0.8,
        split_seed: int = 42,
        partitions: Optional[Dict[str, List[str]]] = None,
        positive_pairs_per_writer: Optional[int] = None,
        negative_pairs_per_writer: Optional[int] = None,
        pair_sampling_seed: Optional[int] = None,
        balance_pairs: bool = True,
        oversample: bool = False,
        match_by_sig: bool = False,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        if transform is None:
            raise ValueError("NpzWriterDisjointDataset requires a torchvision-like transform")
        if match_by_sig:
            raise ValueError("match_by_sig is not supported for NPZ datasets (no sig_id metadata)")

        self.npz_path = Path(npz_path)
        self.split = split
        self.transform = transform
        self.writer_split_ratio = writer_split_ratio
        self.split_seed = split_seed
        self.positive_pairs_per_writer = positive_pairs_per_writer
        self.negative_pairs_per_writer = negative_pairs_per_writer
        self._pair_seed = pair_sampling_seed if pair_sampling_seed is not None else split_seed
        self.balance_pairs = balance_pairs
        self.oversample = oversample
        self._warned_limits: set[str] = set()
        if input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3")
        self.input_channels = input_channels

        arrays = np.load(self.npz_path, allow_pickle=False)
        if "images" not in arrays or "writer_ids" not in arrays or "is_forged" not in arrays:
            raise ValueError("NPZ must contain images, writer_ids, and is_forged arrays")
        self.images = arrays["images"]
        self.writer_ids = arrays["writer_ids"].astype(str)
        self.is_forged = arrays["is_forged"].astype(np.uint8)
        self.paths = arrays["paths"].astype(str) if "paths" in arrays else None

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.npz_path}")

        self._writers = self._index_writers()
        if not self._writers:
            raise RuntimeError(f"No signature metadata found in {self.npz_path}")

        if partitions is None:
            partitions = self._split_writers()
        self._partitions = partitions
        if split not in self._partitions:
            raise ValueError(f"Split '{split}' not present in provided partitions {list(self._partitions)}")

        self.writer_label_map = {writer: idx for idx, writer in enumerate(self._partitions.get("train", []))}
        self.samples = self._build_pairs(self._partitions[split])
        if not self.samples:
            raise RuntimeError("No pairs could be formed; check NPZ metadata")

    @property
    def partitions(self) -> Dict[str, List[str]]:
        return self._partitions

    @property
    def num_writers(self) -> int:
        return len(self.writer_label_map)

    def _index_writers(self) -> Dict[str, Dict[str, List[int]]]:
        writers: Dict[str, Dict[str, List[int]]] = {}
        for idx, (writer_id, is_forged) in enumerate(zip(self.writer_ids, self.is_forged)):
            bucket = writers.setdefault(str(writer_id), {"Genuine": [], "Forged": []})
            subset = "Forged" if int(is_forged) == 1 else "Genuine"
            bucket[subset].append(idx)
        return writers

    def _split_writers(self) -> Dict[str, List[str]]:
        writer_ids = sorted(self._writers.keys())
        if len(writer_ids) < 2:
            raise RuntimeError("Need at least two writers for disjoint split")
        rng = random.Random(self.split_seed)
        rng.shuffle(writer_ids)
        split_idx = max(1, int(len(writer_ids) * self.writer_split_ratio))
        split_idx = min(len(writer_ids) - 1, split_idx)
        return {
            "train": writer_ids[:split_idx],
            "val": writer_ids[split_idx:],
        }

    def _build_pairs(self, target_writers: List[str]) -> List[NpzSample]:
        samples: List[NpzSample] = []
        rng = random.Random(self._pair_seed)

        for writer in target_writers:
            pools = self._writers[writer]
            genuine = sorted(pools["Genuine"])
            forged = sorted(pools["Forged"])
            if len(genuine) < 2:
                continue
            writer_label = self.writer_label_map.get(writer, -1)
            pos_pairs = self._positive_pairs(genuine, writer_label, writer, rng)
            neg_pairs = self._negative_pairs(genuine, forged, writer_label, writer, rng)
            if self.balance_pairs:
                pos_pairs, neg_pairs = self._match_pair_counts(pos_pairs, neg_pairs, rng)
            samples.extend(pos_pairs)
            samples.extend(neg_pairs)

        return samples

    @staticmethod
    def _sample_pairs(pairs: List[NpzSample], target: int, rng: random.Random) -> List[NpzSample]:
        if not pairs:
            return []
        if len(pairs) == target:
            return pairs[:]

        if len(pairs) > target:
            shuffled = pairs[:]
            rng.shuffle(shuffled)
            return shuffled[:target]

        full_repeats = target // len(pairs)
        remainder = target % len(pairs)
        result = pairs * full_repeats
        if remainder > 0:
            shuffled = pairs[:]
            rng.shuffle(shuffled)
            result.extend(shuffled[:remainder])
        return result

    def _match_pair_counts(
        self, positives: List[NpzSample], negatives: List[NpzSample], rng: random.Random
    ) -> tuple[List[NpzSample], List[NpzSample]]:
        if not positives or not negatives:
            return positives, negatives

        target = max(len(positives), len(negatives)) if self.oversample else min(len(positives), len(negatives))
        if target == 0:
            return [], []
        if len(positives) != target:
            positives = self._sample_pairs(positives, target, rng)
        if len(negatives) != target:
            negatives = self._sample_pairs(negatives, target, rng)
        return positives, negatives

    def _positive_pairs(
        self,
        genuine: List[int],
        writer_label: int,
        writer_id: str,
        rng: random.Random,
    ) -> List[NpzSample]:
        pairs: List[NpzSample] = []
        if len(genuine) < 2:
            return pairs
        if self.positive_pairs_per_writer is None:
            for i in range(len(genuine) - 1):
                idx_a = genuine[i]
                idx_b = genuine[i + 1]
                pairs.append(
                    NpzSample(
                        idx_a=idx_a,
                        idx_b=idx_b,
                        same_writer=True,
                        writer_label=writer_label,
                        forgery_label=0,
                        is_forged_a=False,
                        is_forged_b=False,
                        path_a=self._get_path(idx_a),
                        path_b=self._get_path(idx_b),
                    )
                )
            return pairs

        max_unique = len(genuine) * (len(genuine) - 1) // 2
        if max_unique == 0:
            return pairs
        target = min(self.positive_pairs_per_writer, max_unique)
        if self.positive_pairs_per_writer > max_unique:
            self._warn_if_limited("positive", writer_id, self.positive_pairs_per_writer, max_unique)
        combo_indices = list(itertools.combinations(range(len(genuine)), 2))
        selected = rng.sample(combo_indices, target) if target < len(combo_indices) else combo_indices
        for i, j in selected:
            idx_a = genuine[i]
            idx_b = genuine[j]
            pairs.append(
                NpzSample(
                    idx_a=idx_a,
                    idx_b=idx_b,
                    same_writer=True,
                    writer_label=writer_label,
                    forgery_label=0,
                    is_forged_a=False,
                    is_forged_b=False,
                    path_a=self._get_path(idx_a),
                    path_b=self._get_path(idx_b),
                )
            )
        return pairs

    def _negative_pairs(
        self,
        genuine: List[int],
        forged: List[int],
        writer_label: int,
        writer_id: str,
        rng: random.Random,
    ) -> List[NpzSample]:
        pairs: List[NpzSample] = []
        if not forged or not genuine:
            return pairs
        if self.negative_pairs_per_writer is None:
            for idx, fake in enumerate(forged):
                ref = genuine[idx % len(genuine)]
                pairs.append(
                    NpzSample(
                        idx_a=ref,
                        idx_b=fake,
                        same_writer=False,
                        writer_label=writer_label,
                        forgery_label=1,
                        is_forged_a=False,
                        is_forged_b=True,
                        path_a=self._get_path(ref),
                        path_b=self._get_path(fake),
                    )
                )
            return pairs

        max_unique = len(genuine) * len(forged)
        if max_unique == 0:
            return pairs
        target = min(self.negative_pairs_per_writer, max_unique)
        if self.negative_pairs_per_writer > max_unique:
            self._warn_if_limited("negative", writer_id, self.negative_pairs_per_writer, max_unique)
        combo_indices = [(i, j) for i in range(len(genuine)) for j in range(len(forged))]
        selected = rng.sample(combo_indices, target) if target < len(combo_indices) else combo_indices
        for gi, fj in selected:
            idx_a = genuine[gi]
            idx_b = forged[fj]
            pairs.append(
                NpzSample(
                    idx_a=idx_a,
                    idx_b=idx_b,
                    same_writer=False,
                    writer_label=writer_label,
                    forgery_label=1,
                    is_forged_a=False,
                    is_forged_b=True,
                    path_a=self._get_path(idx_a),
                    path_b=self._get_path(idx_b),
                )
            )
        return pairs

    def _warn_if_limited(self, pair_type: str, writer_id: str, requested: int, available: int) -> None:
        key = f"{pair_type}:{writer_id}"
        if key in self._warned_limits:
            return
        self._warned_limits.add(key)
        print(
            f"[NpzWriterDisjointDataset] Requested {pair_type} pairs ({requested}) exceed available {available} for writer '{writer_id}'. Using {available} instead."
        )

    def _get_path(self, idx: int) -> Optional[str]:
        if self.paths is None:
            return None
        if idx < 0 or idx >= len(self.paths):
            return None
        return str(self.paths[idx])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        mode = "RGB" if self.input_channels == 3 else "L"
        img_a = Image.fromarray(self.images[sample.idx_a]).convert(mode)
        img_b = Image.fromarray(self.images[sample.idx_b]).convert(mode)
        img_a = self.transform(img_a)
        img_b = self.transform(img_b)
        return {
            "img_a": img_a,
            "img_b": img_b,
            "same_writer": torch.tensor(1.0 if sample.same_writer else 0.0, dtype=torch.float32),
            "writer_id": torch.tensor(sample.writer_label, dtype=torch.long),
            "forgery_label": torch.tensor(sample.forgery_label, dtype=torch.long),
            "forg_label_a": torch.tensor(1.0 if sample.is_forged_a else 0.0, dtype=torch.float32),
            "forg_label_b": torch.tensor(1.0 if sample.is_forged_b else 0.0, dtype=torch.float32),
            "path_a": sample.path_a,
            "path_b": sample.path_b,
        }
