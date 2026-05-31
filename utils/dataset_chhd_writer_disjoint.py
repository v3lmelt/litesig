"""Writer-disjoint pair dataset for CHHD genuine-only flat-folder data."""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class ChhdSample:
    path_a: Path
    path_b: Path
    same_writer: bool
    writer_label: int
    forgery_label: int
    is_forged_a: bool = False
    is_forged_b: bool = False


class ChhdWriterDisjointDataset(Dataset):
    """Creates writer-disjoint pairs for flat-folder genuine-only CHHD data.

    Expected filename pattern is CHHD-style, e.g. ``0001-01-20210607-S-N-p1.png``.
    All files are treated as genuine handwriting samples.
    """

    def __init__(
        self,
        root: str,
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
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        if transform is None:
            raise ValueError("ChhdWriterDisjointDataset requires a torchvision-like transform")
        if input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3")

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.writer_split_ratio = writer_split_ratio
        self.split_seed = split_seed
        self.positive_pairs_per_writer = positive_pairs_per_writer
        self.negative_pairs_per_writer = negative_pairs_per_writer
        self._pair_seed = pair_sampling_seed if pair_sampling_seed is not None else split_seed
        self.balance_pairs = balance_pairs
        self.oversample = oversample
        self.input_channels = input_channels
        self._warned_limits: set[str] = set()

        self._writers = self._scan_writers()
        if not self._writers:
            raise RuntimeError(f"No CHHD files found under {self.root}")
        if partitions is None:
            partitions = self._split_writers()
        self._partitions = partitions
        if split not in self._partitions:
            raise ValueError(f"Split '{split}' not present in provided partitions {list(self._partitions)}")
        self.writer_label_map = {writer: idx for idx, writer in enumerate(self._partitions.get("train", []))}
        self.samples = self._build_pairs(self._partitions[split])

    @property
    def partitions(self) -> Dict[str, List[str]]:
        return self._partitions

    @property
    def num_writers(self) -> int:
        return len(self.writer_label_map)

    @staticmethod
    def _extract_writer_id(stem: str) -> str:
        parts = stem.split("-")
        if parts:
            writer_id = parts[0].strip()
            if writer_id:
                return writer_id
        return stem.strip()

    def _scan_writers(self) -> Dict[str, List[Path]]:
        writers: Dict[str, List[Path]] = {}
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            writer_id = self._extract_writer_id(path.stem)
            writers.setdefault(writer_id, []).append(path)
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

    def _build_pairs(self, target_writers: List[str]) -> List[ChhdSample]:
        samples: List[ChhdSample] = []
        rng = random.Random(self._pair_seed)
        for writer in target_writers:
            genuine = sorted(self._writers[writer])
            if len(genuine) < 2:
                continue
            writer_label = self.writer_label_map.get(writer, -1)
            pos_pairs = self._positive_pairs(genuine, writer_label, writer, rng)
            neg_pairs = self._negative_pairs(writer, genuine, writer_label, target_writers, rng)
            if self.balance_pairs:
                pos_pairs, neg_pairs = self._match_pair_counts(pos_pairs, neg_pairs, rng)
            samples.extend(pos_pairs)
            samples.extend(neg_pairs)

        if not samples:
            raise RuntimeError("No pairs could be formed; check CHHD dataset structure")
        return samples

    @staticmethod
    def _sample_pairs(pairs: List[ChhdSample], target: int, rng: random.Random) -> List[ChhdSample]:
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
        self,
        positives: List[ChhdSample],
        negatives: List[ChhdSample],
        rng: random.Random,
    ) -> tuple[List[ChhdSample], List[ChhdSample]]:
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
        genuine: List[Path],
        writer_label: int,
        writer_id: str,
        rng: random.Random,
    ) -> List[ChhdSample]:
        pairs: List[ChhdSample] = []
        if len(genuine) < 2:
            return pairs
        if self.positive_pairs_per_writer is None:
            for i in range(len(genuine) - 1):
                pairs.append(
                    ChhdSample(
                        path_a=genuine[i],
                        path_b=genuine[i + 1],
                        same_writer=True,
                        writer_label=writer_label,
                        forgery_label=0,
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
            pairs.append(
                ChhdSample(
                    path_a=genuine[i],
                    path_b=genuine[j],
                    same_writer=True,
                    writer_label=writer_label,
                    forgery_label=0,
                )
            )
        return pairs

    def _negative_pairs(
        self,
        writer: str,
        genuine: List[Path],
        writer_label: int,
        target_writers: List[str],
        rng: random.Random,
    ) -> List[ChhdSample]:
        pairs: List[ChhdSample] = []
        other_writers = [wid for wid in target_writers if wid != writer and self._writers.get(wid)]
        if not genuine or not other_writers:
            return pairs

        other_weights = [len(self._writers[wid]) for wid in other_writers]
        max_unique = len(genuine) * sum(other_weights)
        if max_unique == 0:
            return pairs

        if self.negative_pairs_per_writer is None:
            for anchor in genuine:
                imp_writer = rng.choices(other_writers, weights=other_weights, k=1)[0]
                imp_path = rng.choice(self._writers[imp_writer])
                pairs.append(
                    ChhdSample(
                        path_a=anchor,
                        path_b=imp_path,
                        same_writer=False,
                        writer_label=writer_label,
                        forgery_label=1,
                    )
                )
            return pairs

        target = min(self.negative_pairs_per_writer, max_unique)
        if self.negative_pairs_per_writer > max_unique:
            self._warn_if_limited("negative", writer, self.negative_pairs_per_writer, max_unique)

        selected: set[tuple[str, str]] = set()
        attempts = 0
        max_attempts = max(100, target * 50)
        while len(pairs) < target and attempts < max_attempts:
            attempts += 1
            anchor = rng.choice(genuine)
            imp_writer = rng.choices(other_writers, weights=other_weights, k=1)[0]
            imp_path = rng.choice(self._writers[imp_writer])
            key = (str(anchor), str(imp_path))
            if key in selected:
                continue
            selected.add(key)
            pairs.append(
                ChhdSample(
                    path_a=anchor,
                    path_b=imp_path,
                    same_writer=False,
                    writer_label=writer_label,
                    forgery_label=1,
                )
            )

        if len(pairs) == target:
            return pairs

        anchors = genuine[:]
        rng.shuffle(anchors)
        shuffled_writers = other_writers[:]
        rng.shuffle(shuffled_writers)
        for anchor in anchors:
            if len(pairs) >= target:
                break
            for imp_writer in shuffled_writers:
                if len(pairs) >= target:
                    break
                imp_candidates = self._writers[imp_writer][:]
                rng.shuffle(imp_candidates)
                for imp_path in imp_candidates:
                    key = (str(anchor), str(imp_path))
                    if key in selected:
                        continue
                    selected.add(key)
                    pairs.append(
                        ChhdSample(
                            path_a=anchor,
                            path_b=imp_path,
                            same_writer=False,
                            writer_label=writer_label,
                            forgery_label=1,
                        )
                    )
                    if len(pairs) >= target:
                        break
        return pairs

    def _warn_if_limited(
        self,
        pair_type: str,
        writer_id: str,
        requested: int,
        available: int,
    ) -> None:
        key = f"{pair_type}:{writer_id}"
        if key in self._warned_limits:
            return
        self._warned_limits.add(key)
        print(
            f"[ChhdWriterDisjointDataset] Requested {pair_type} pairs ({requested}) exceed available {available} for writer '{writer_id}'. Using {available} instead."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        mode = "RGB" if self.input_channels == 3 else "L"
        img_a = Image.open(sample.path_a).convert(mode)
        img_b = Image.open(sample.path_b).convert(mode)
        img_a = self.transform(img_a)
        img_b = self.transform(img_b)
        return {
            "img_a": img_a,
            "img_b": img_b,
            "same_writer": torch.tensor(1.0 if sample.same_writer else 0.0, dtype=torch.float32),
            "writer_id": torch.tensor(sample.writer_label, dtype=torch.long),
            "forgery_label": torch.tensor(sample.forgery_label, dtype=torch.long),
            "forg_label_a": torch.tensor(0.0, dtype=torch.float32),
            "forg_label_b": torch.tensor(0.0, dtype=torch.float32),
            "path_a": str(sample.path_a),
            "path_b": str(sample.path_b),
        }
