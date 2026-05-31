"""Dataset helper ensuring writer-disjoint splits for signature verification."""
from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class Sample:
    path_a: Path
    path_b: Path
    same_writer: bool
    writer_label: int
    forgery_label: int
    is_forged_a: bool = False                                            
    is_forged_b: bool = False                                            


class WriterDisjointDataset(Dataset):
    """Creates signature pairs with writer-disjoint train/val splits."""

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
        match_by_sig: bool = False,
        input_channels: int = 1,
        forged_subsets: Optional[Sequence[str]] = None,
    ) -> None:
        """Creates signature pairs with writer-disjoint train/val splits.
        
        Args:
            root: Path to dataset root directory.
            split: "train" or "val".
            transform: Torchvision-like transform to apply to images.
            writer_split_ratio: Ratio of writers for training (default 0.8).
            split_seed: Random seed for writer split.
            partitions: Optional pre-defined writer partitions.
            positive_pairs_per_writer: Max positive pairs per writer (or per sig if match_by_sig=True).
            negative_pairs_per_writer: Max negative pairs per writer (or per sig if match_by_sig=True).
            pair_sampling_seed: Random seed for pair sampling.
            balance_pairs: Whether to balance positive/negative pairs.
            oversample: If balancing, use oversampling instead of downsampling.
            match_by_sig: If True, only pair samples with matching signature id (Hansig style).
                          When True, positive_pairs_per_writer/negative_pairs_per_writer become per-sig limits.
            input_channels: Expected input channels (1=grayscale, 3=RGB).
            forged_subsets: Optional top-level subfolders under Forged/ to include.
                            Useful for datasets like UTSig with multiple forgery protocols.
        """
        super().__init__()
        if transform is None:
            raise ValueError("WriterDisjointDataset requires a torchvision-like transform")
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
        self.match_by_sig = match_by_sig
        if input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3")
        self.input_channels = input_channels
        self.forged_subsets = (
            {name.strip().lower() for name in forged_subsets if str(name).strip()}
            if forged_subsets
            else None
        )
        self._warned_limits: set[str] = set()
        self._writers = self._scan_writers()
        if not self._writers:
            raise RuntimeError(f"No signature files found under {self.root}")
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
        """Best-effort writer id parsing for varying filename conventions.
        
        Supports multiple naming conventions:
        - Hansig style: original_w1_2_3 / forgery_w238_885_20 -> extracts "w1" / "w238"
        - BHSig style: B-S-<writer>-<type>-<idx> -> extracts "B-S-<writer>"
        - Generic underscore style: <writer>_... -> extracts first segment
        """
                                                                                               
        match = re.search(r"(?:^|[_-])(w\d+)(?:[_-]|$)", stem, re.IGNORECASE)
        if match:
            return match.group(1).lower()                                              
        
                                                           
        if "_" in stem:
            return stem.split("_")[0]
        
                                              
        if "-" in stem:
            parts = stem.split("-")
                                                    
            if len(parts) >= 3:
                return "-".join(parts[:-2])
        
        return stem

    @staticmethod
    def _extract_sig_id(stem: str) -> Optional[str]:
        """Extract signature id from filename.
        
        Hansig style: original_w1_2_3 -> sig="2"
                      forgery_w238_885_20 -> sig="885"
        
        Returns None if sig cannot be extracted (will fallback to old behavior).
        """
                                                
        match = re.match(r"(?:original|forgery)_w\d+_(\d+)_\d+", stem, re.IGNORECASE)
        if match:
            return match.group(1)
        
                                                                     
        parts = stem.split("_")
        if len(parts) >= 4:
            if re.match(r"w\d+", parts[1], re.IGNORECASE):
                return parts[2]
        
        return None

    def _scan_writers(self):
        """Scan root directory for genuine/forged signature images.
        
        Supports multiple directory naming conventions:
        - Genuine: "Genuine", "genuine"
        - Forged: "Forged", "forged"
        
        Internal keys are always normalized to "Genuine" / "Forged".
        
        Returns:
            When match_by_sig=False:
                Dict[writer_id, {"Genuine": [paths...], "Forged": [paths...]}]
            When match_by_sig=True:
                Dict[writer_id, {"Genuine": {sig_id: [paths...]}, "Forged": {sig_id: [paths...]}}]
        """
                                                                                      
        subset_candidates = {
            "Genuine": ["Genuine", "genuine"],
            "Forged": ["Forged", "forged"],
        }
        matched_forged = False
        
        if self.match_by_sig:
                                                                            
            writers: Dict[str, Dict[str, Dict[str, List[Path]]]] = {}
            
            for subset_key, candidates in subset_candidates.items():
                subset_dir = None
                for candidate in candidates:
                    candidate_dir = self.root / candidate
                    if candidate_dir.exists() and candidate_dir.is_dir():
                        subset_dir = candidate_dir
                        break
                
                if subset_dir is None:
                    continue
                
                for path in subset_dir.rglob("*"):
                    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                        continue
                    if subset_key == "Forged" and not self._is_allowed_forged_path(subset_dir, path):
                        continue
                    writer_id = self._extract_writer_id(path.stem)
                    sig_id = self._extract_sig_id(path.stem)
                    if subset_key == "Forged":
                        matched_forged = True
                    
                                                                       
                    if sig_id is None:
                        sig_id = "__default__"
                    
                    writer_bucket = writers.setdefault(writer_id, {"Genuine": {}, "Forged": {}})
                    sig_bucket = writer_bucket[subset_key].setdefault(sig_id, [])
                    sig_bucket.append(path)
            
            if self.forged_subsets is not None and not matched_forged:
                raise RuntimeError(
                    f"No forged files matched subsets {sorted(self.forged_subsets)} under {self.root / 'Forged'}"
                )
            return writers
        else:
                                                                             
            writers: Dict[str, Dict[str, List[Path]]] = {}
            
            for subset_key, candidates in subset_candidates.items():
                subset_dir = None
                for candidate in candidates:
                    candidate_dir = self.root / candidate
                    if candidate_dir.exists() and candidate_dir.is_dir():
                        subset_dir = candidate_dir
                        break
                
                if subset_dir is None:
                    continue
                
                for path in subset_dir.rglob("*"):
                    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                        continue
                    if subset_key == "Forged" and not self._is_allowed_forged_path(subset_dir, path):
                        continue
                    writer_id = self._extract_writer_id(path.stem)
                    if subset_key == "Forged":
                        matched_forged = True
                    writer_bucket = writers.setdefault(writer_id, {"Genuine": [], "Forged": []})
                    writer_bucket[subset_key].append(path)
            
            if self.forged_subsets is not None and not matched_forged:
                raise RuntimeError(
                    f"No forged files matched subsets {sorted(self.forged_subsets)} under {self.root / 'Forged'}"
                )
            return writers

    def _is_allowed_forged_path(self, subset_dir: Path, path: Path) -> bool:
        """Filter forged files by top-level folder name when requested.

        For flat Forged/ layouts (no subdirectories), all files remain eligible.
        """
        if self.forged_subsets is None:
            return True
        rel_parts = path.relative_to(subset_dir).parts
        if len(rel_parts) <= 1:
            return True
        return rel_parts[0].lower() in self.forged_subsets

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

    def _build_pairs(self, target_writers: List[str]) -> List[Sample]:
        """Build signature pairs for the given writers.
        
        When match_by_sig=False (default):
            - Positive pairs: any two genuine samples from same writer
            - Negative pairs: any genuine vs any forged from same writer
            
        When match_by_sig=True:
            - Positive pairs: genuine samples with same (writer, sig)
            - Negative pairs: genuine vs forged with same (writer, sig)
        """
        samples: List[Sample] = []
        rng = random.Random(self._pair_seed)
        
        if self.match_by_sig:
                                                          
            for writer in target_writers:
                pools = self._writers[writer]
                genuine_by_sig = pools["Genuine"]                            
                forged_by_sig = pools["Forged"]                              
                
                all_sigs = set(genuine_by_sig.keys()) | set(forged_by_sig.keys())
                writer_label = self.writer_label_map.get(writer, -1)
                
                for sig in sorted(all_sigs):
                    genuine = sorted(genuine_by_sig.get(sig, []))
                    forged = sorted(forged_by_sig.get(sig, []))
                    
                                                                          
                    pos_pairs: List[Sample] = []
                    if len(genuine) >= 2:
                        pos_pairs = self._positive_pairs(genuine, writer_label, f"{writer}:{sig}", rng)
                    
                                                                
                    neg_pairs: List[Sample] = []
                    if genuine and forged:
                        neg_pairs = self._negative_pairs(genuine, forged, writer_label, f"{writer}:{sig}", rng)
                    
                    if self.balance_pairs:
                        pos_pairs, neg_pairs = self._match_pair_counts(pos_pairs, neg_pairs, rng)
                    
                    samples.extend(pos_pairs)
                    samples.extend(neg_pairs)
        else:
                                                                  
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
        
        if not samples:
            raise RuntimeError("No pairs could be formed; check dataset structure")
        return samples

    @staticmethod
    def _sample_pairs(pairs: List[Sample], target: int, rng: random.Random) -> List[Sample]:
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
        self, positives: List[Sample], negatives: List[Sample], rng: random.Random
    ) -> tuple[List[Sample], List[Sample]]:
        if not positives or not negatives:
            return positives, negatives
        
        if self.oversample:
            target = max(len(positives), len(negatives))
        else:
            target = min(len(positives), len(negatives))
            
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
    ) -> List[Sample]:
        pairs: List[Sample] = []
        if len(genuine) < 2:
            return pairs
        if self.positive_pairs_per_writer is None:
            for i in range(len(genuine) - 1):
                pairs.append(
                    Sample(
                        path_a=genuine[i],
                        path_b=genuine[i + 1],
                        same_writer=True,
                        writer_label=writer_label,
                        forgery_label=0,
                        is_forged_a=False,
                        is_forged_b=False,
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
        if target < len(combo_indices):
            selected = rng.sample(combo_indices, target)
        else:
            selected = combo_indices
        for i, j in selected:
            pairs.append(
                Sample(
                    path_a=genuine[i],
                    path_b=genuine[j],
                    same_writer=True,
                    writer_label=writer_label,
                    forgery_label=0,
                    is_forged_a=False,
                    is_forged_b=False,
                )
            )
        return pairs

    def _negative_pairs(
        self,
        genuine: List[Path],
        forged: List[Path],
        writer_label: int,
        writer_id: str,
        rng: random.Random,
    ) -> List[Sample]:
        pairs: List[Sample] = []
        if not forged or not genuine:
            return pairs
        if self.negative_pairs_per_writer is None:
            for idx, fake in enumerate(forged):
                ref = genuine[idx % len(genuine)]
                pairs.append(
                    Sample(
                        path_a=ref,
                        path_b=fake,
                        same_writer=False,
                        writer_label=writer_label,
                        forgery_label=1,
                        is_forged_a=False,
                        is_forged_b=True,
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
        if target < len(combo_indices):
            selected = rng.sample(combo_indices, target)
        else:
            selected = combo_indices
        for gi, fj in selected:
            pairs.append(
                Sample(
                    path_a=genuine[gi],
                    path_b=forged[fj],
                    same_writer=False,
                    writer_label=writer_label,
                    forgery_label=1,
                    is_forged_a=False,
                    is_forged_b=True,
                )
            )
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
            f"[WriterDisjointDataset] Requested {pair_type} pairs ({requested}) exceed available {available} for writer '{writer_id}'. Using {available} instead."
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
            "forg_label_a": torch.tensor(1.0 if sample.is_forged_a else 0.0, dtype=torch.float32),
            "forg_label_b": torch.tensor(1.0 if sample.is_forged_b else 0.0, dtype=torch.float32),
                                                                                           
            "path_a": str(sample.path_a),
            "path_b": str(sample.path_b),
        }
