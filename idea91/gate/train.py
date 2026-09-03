"""Classifier training and evaluation for the K1 gate (design P6).

    ResNet-18 and ViT-S/16 (dynamic image size) from timm, ImageNet-pretrained,
    5 epochs, AdamW lr 1e-4, batch 8 to 16 with mixed precision at the native
    rows and 64 at 512 px, 80/20 split by source image, AUROC per image on the
    20%, 3 seeds, mean and bootstrap CI over images; AUROC also split by
    ``control_source``.

Scores are per *image*: a full-image row scores the image directly, and the tile
row takes the maximum over the image's tiles (P5).  The AUROC and its bootstrap
CI are then over images, which is also the resampling unit in
:mod:`shared.stats`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from idea91.gate import inputs as I
from idea91.gate.dataset import (
    GateDataset,
    GateSample,
    ShapeBucketSampler,
    TileDataset,
    split_by_image,
)
from shared.stats import Interval, auroc, auroc_ci

EPOCHS = 5  # P6
LR = 1e-4  # P6
BATCH_NATIVE = 8  # P6: "batch 8 to 16 ... at the native rows"
BATCH_512 = 64  # P6: "and 64 at 512 px"
SEEDS = (0, 1, 2)  # P6: 3 seeds
ADVERSARY_SEEDS = (0,)  # P6: "one seed each" for the strong adversaries

TIMM_NAMES = {
    I.RESNET18: "resnet18.a1_in1k",
    I.VIT_S16: "vit_small_patch16_224.augreg_in21k_ft_in1k",
    I.DINOV2_B: "vit_base_patch14_dinov2.lvd142m",
    # the forensic detector is a file, not a timm name (P6); PINS.md records it
}


def batch_size_for(row: I.GateRow) -> int:
    return BATCH_512 if row.input_kind in ("full_512", "shown_crop", "tiles_512") else BATCH_NATIVE


def build_model(classifier: str, device: str = "cuda") -> torch.nn.Module:
    """A timm binary head; ViT-family models get dynamic image size (P6)."""
    import timm

    if classifier not in TIMM_NAMES:
        raise ValueError(
            f"{classifier!r} has no timm name; the forensic detector loads from the file "
            "pinned in shared/env/PINS.md"
        )
    kwargs = {"pretrained": True, "num_classes": 1}
    if classifier != I.RESNET18:
        # Native-resolution rows hand a ViT arbitrary sizes (a 1,024 px longer
        # side gives 1024x683), and patch embedding needs a multiple of the patch
        # size, so pad inside the model rather than resizing the input and losing
        # the native pixels the gate is meant to see.
        kwargs["dynamic_img_size"] = True
        kwargs["dynamic_img_pad"] = True
    model = timm.create_model(TIMM_NAMES[classifier], **kwargs)
    return model.to(device)


@dataclass
class RowResult:
    """One gate row, one hole type, averaged over seeds."""

    row: str
    classifier: str
    hole_type: str
    contrast: str  # 'REMOVE_vs_CONTROL_OBJ' or 'REMOVE_vs_CONTROL_BG'
    auroc_mean: float
    auroc_by_seed: list[float]
    ci: Interval
    n_images: int
    by_control_source: dict[str, float] = field(default_factory=dict)

    def line(self) -> str:
        return (
            f"{self.row:28s} {self.hole_type:5s} {self.contrast:24s} "
            f"AUROC {self.auroc_mean:.3f} [{self.ci.lo:.3f}, {self.ci.hi:.3f}] "
            f"n={self.n_images} seeds={['%.3f' % a for a in self.auroc_by_seed]}"
        )


def _loader(dataset, row: I.GateRow, *, shuffle: bool, seed: int, workers: int = 4):
    batch = batch_size_for(row)
    if row.input_kind in ("full_cap", "full_1024") and isinstance(dataset, GateDataset):
        shapes = [dataset.shape_of(i) for i in range(len(dataset))]
        sampler = ShapeBucketSampler(shapes, batch, shuffle=shuffle, seed=seed)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=workers), sampler
    return (
        DataLoader(
            dataset,
            batch_size=batch,
            shuffle=shuffle,
            num_workers=workers,
            drop_last=False,
        ),
        None,
    )


def train_one(
    train_samples: list[GateSample],
    test_samples: list[GateSample],
    row: I.GateRow,
    edits_root: Path,
    *,
    seed: int = 0,
    epochs: int = EPOCHS,
    device: str = "cuda",
    workers: int = 4,
    cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Train one (row, seed) and score the held-out images.

    Returns ``(labels, scores, image_ids)``, one entry per held-out *image*.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if row.input_kind == "tiles_512":
        train_ds = TileDataset(train_samples, edits_root, mode="train", seed=seed, cache=cache)
        test_ds = TileDataset(test_samples, edits_root, mode="eval", seed=seed, cache=cache)
    else:
        train_ds = GateDataset(train_samples, row, edits_root, cache=cache)
        test_ds = GateDataset(test_samples, row, edits_root, cache=cache)

    model = build_model(row.classifier, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_loader, sampler = _loader(train_ds, row, shuffle=True, seed=seed, workers=workers)

    model.train()
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if isinstance(train_ds, TileDataset):
            train_ds.set_epoch(epoch)
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device).float().unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    per_sample: dict[int, list[float]] = defaultdict(list)
    test_loader, _ = _loader(test_ds, row, shuffle=False, seed=seed, workers=workers)
    with torch.inference_mode():
        for x, _, idx in test_loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x).float().squeeze(1)
            for i, s in zip(idx.tolist(), logits.tolist()):
                per_sample[int(i)].append(float(s))

    # P5: aggregate the tile row per image by the maximum tile score
    labels, scores, image_ids = [], [], []
    for i, sample in enumerate(test_samples):
        vals = per_sample.get(i)
        if not vals:
            continue
        labels.append(sample.label)
        scores.append(max(vals) if row.aggregate == "max_tile" else float(np.mean(vals)))
        image_ids.append(sample.image_id)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.array(labels), np.array(scores), image_ids


def run_row(
    samples: list[GateSample],
    row: I.GateRow,
    edits_root: Path,
    *,
    hole_type: str,
    contrast: str,
    seeds: tuple[int, ...] = SEEDS,
    epochs: int = EPOCHS,
    device: str = "cuda",
    workers: int = 4,
    cache: dict | None = None,
) -> RowResult:
    """P6 for one row: 3 seeds, mean AUROC, bootstrap CI over images."""
    per_seed: list[float] = []
    pooled_labels: list[np.ndarray] = []
    pooled_scores: list[np.ndarray] = []
    pooled_images: list[str] = []
    sources: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))

    for seed in seeds:
        train_s, test_s = split_by_image(samples, seed=seed)
        labels, scores, image_ids = train_one(
            train_s,
            test_s,
            row,
            edits_root,
            seed=seed,
            epochs=epochs,
            device=device,
            workers=workers,
            cache=cache,
        )
        per_seed.append(auroc(labels, scores))
        pooled_labels.append(labels)
        pooled_scores.append(scores)
        pooled_images += image_ids
        by_id = {s.image_id: s for s in test_s}
        for label, score, image_id in zip(labels, scores, image_ids):
            src = by_id[image_id].control_source or "unknown"
            sources[src][0].append(int(label))
            sources[src][1].append(float(score))

    labels = np.concatenate(pooled_labels) if pooled_labels else np.array([])
    scores = np.concatenate(pooled_scores) if pooled_scores else np.array([])
    ci = auroc_ci(labels, scores, np.array(pooled_images))
    return RowResult(
        row=row.name,
        classifier=row.classifier,
        hole_type=hole_type,
        contrast=contrast,
        auroc_mean=float(np.nanmean(per_seed)) if per_seed else float("nan"),
        auroc_by_seed=per_seed,
        ci=ci,
        n_images=len(set(pooled_images)),
        by_control_source={
            src: auroc(np.array(ls), np.array(ss)) for src, (ls, ss) in sources.items()
        },
    )
