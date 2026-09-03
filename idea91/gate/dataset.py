"""Datasets for the K1 gate (design P5, P6).

Two things here are not incidental:

* **The 80/20 split is by source image**, not by edit.  Both edits of an image --
  the REMOVE and its matched CONTROL_OBJ -- carry the same background pixels, so
  splitting by edit would leak the background across the split and inflate every
  AUROC.
* **Full-image rows keep native pixel sizes.**  The gate must see what the policy
  sees, so the loader batches by exact shape rather than resizing to a common
  size; a bucket with a single member simply yields a batch of one.

Edited images are composed at load time from the original plus the stored window
(P2), so the bank costs one window PNG per edit rather than a second full copy.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from idea91.edits.build import load_edited, read_image
from idea91.gate import inputs as I
from idea91.masks import Box

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class GateSample:
    """One edited image, its label, and everything needed to compose it."""

    sample_id: str
    image_id: str  # the source image: the split and the per-image AUROC key
    label: int  # 1 = REMOVE, 0 = CONTROL_OBJ (or whatever the row contrasts)
    original_path: Path
    window_path: Path
    window_xyxy: tuple[int, int, int, int]
    hole_box: Box | None = None
    control_source: str | None = None
    hole_area_bin: str | None = None

    def compose(self, edits_root: Path) -> np.ndarray:
        original = read_image(self.original_path)
        return load_edited(
            original,
            {"window_xyxy_px": list(self.window_xyxy), "window_path": str(self.window_path)},
            edits_root,
        )


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB -> CHW float, ImageNet-normalised."""
    t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


class GateDataset(Dataset):
    """Full-image and crop rows: one tensor per edit."""

    def __init__(
        self,
        samples: list[GateSample],
        row: I.GateRow,
        edits_root: Path,
        *,
        cache: dict | None = None,
    ) -> None:
        self.samples = samples
        self.row = row
        self.edits_root = Path(edits_root)
        self.cache = cache  # optional {sample_id: np.ndarray} for small banks

    def __len__(self) -> int:
        return len(self.samples)

    def render(self, sample: GateSample) -> np.ndarray:
        if self.cache is not None and sample.sample_id in self.cache:
            return self.cache[sample.sample_id]
        image = sample.compose(self.edits_root)
        out = I.render(image, self.row, hole_box=sample.hole_box)
        if self.cache is not None:
            self.cache[sample.sample_id] = out
        return out

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return to_tensor(self.render(sample)), sample.label, idx

    def shape_of(self, idx: int) -> tuple[int, int]:
        arr = self.render(self.samples[idx])
        return arr.shape[:2]


class TileDataset(Dataset):
    """Row (ii): native-resolution 512 px tiles, each inheriting the image label.

    ``mode='train'`` draws one tile per image per epoch (so an epoch is one pass
    over images, not over tiles); ``mode='eval'`` returns every tile, and the
    caller aggregates per image by the maximum tile score.
    """

    def __init__(
        self,
        samples: list[GateSample],
        edits_root: Path,
        *,
        mode: str = "train",
        tile: int = I.TILE_PX,
        seed: int = 0,
        cache: dict | None = None,
    ) -> None:
        self.samples = samples
        self.edits_root = Path(edits_root)
        self.mode = mode
        self.tile = tile
        self.seed = seed
        self.epoch = 0
        self.cache = cache
        self._index: list[tuple[int, int]] = []
        if mode == "eval":
            for i, s in enumerate(samples):
                self._index += [(i, t) for t in range(len(self._tiles(s)))]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _tiles(self, sample: GateSample) -> list[np.ndarray]:
        key = sample.sample_id
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        out = [self._pad(t) for t in I.tiles(sample.compose(self.edits_root), self.tile)]
        if self.cache is not None:
            self.cache[key] = out
        return out

    def _pad(self, tile: np.ndarray) -> np.ndarray:
        """Pad a short tile up to the tile size so a batch has one shape.

        Only images smaller than 512 px produce a short tile (interior tiles are
        always full), so this touches the edge case, not the common path.
        """
        h, w = tile.shape[:2]
        if (h, w) == (self.tile, self.tile):
            return tile
        return np.pad(
            tile, ((0, max(0, self.tile - h)), (0, max(0, self.tile - w)), (0, 0)), mode="reflect"
        )

    def __len__(self) -> int:
        return len(self.samples) if self.mode == "train" else len(self._index)

    def __getitem__(self, idx: int):
        if self.mode == "train":
            sample = self.samples[idx]
            tiles = self._tiles(sample)
            rng = np.random.default_rng((self.seed, self.epoch, idx))
            pick = int(rng.integers(0, len(tiles)))
            return to_tensor(tiles[pick]), sample.label, idx
        i, t = self._index[idx]
        sample = self.samples[i]
        return to_tensor(self._tiles(sample)[t]), sample.label, i


class ShapeBucketSampler(Sampler[list[int]]):
    """Batches of items that share an exact pixel shape (native rows)."""

    def __init__(
        self,
        shapes: list[tuple[int, int]],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.buckets: dict[tuple[int, int], list[int]] = {}
        for i, shape in enumerate(shapes):
            self.buckets.setdefault(shape, []).append(i)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng((self.seed, self.epoch))
        batches: list[list[int]] = []
        for members in self.buckets.values():
            order = list(members)
            if self.shuffle:
                order = [order[i] for i in rng.permutation(len(order))]
            batches += [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return iter(batches)

    def __len__(self) -> int:
        return sum(
            (len(m) + self.batch_size - 1) // self.batch_size for m in self.buckets.values()
        )


def split_by_image(
    samples: list[GateSample], *, train_frac: float = 0.8, seed: int = 0
) -> tuple[list[GateSample], list[GateSample]]:
    """P6's 80/20 split by source image, so no image straddles the split."""
    image_ids = sorted({s.image_id for s in samples})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(image_ids))
    n_train = int(round(train_frac * len(image_ids)))
    train_ids = {image_ids[int(i)] for i in order[:n_train]}
    train = [s for s in samples if s.image_id in train_ids]
    test = [s for s in samples if s.image_id not in train_ids]
    return train, test
