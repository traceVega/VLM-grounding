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
    normalise,
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
    #: ``sample_id -> held-out score``, averaged over the seeds that tested it.
    #: P10's verified-removed-pairs AUROC restricts this row to the pairs the
    #: verifier confirmed, so the per-sample scores have to survive the run.
    scores_by_sample: dict[str, float] = field(default_factory=dict)

    def line(self) -> str:
        return (
            f"{self.row:28s} {self.hole_type:5s} {self.contrast:24s} "
            f"AUROC {self.auroc_mean:.3f} [{self.ci.lo:.3f}, {self.ci.hi:.3f}] "
            f"n={self.n_images} seeds={['%.3f' % a for a in self.auroc_by_seed]}"
        )


def _loader(dataset, row: I.GateRow, *, shuffle: bool, seed: int, workers: int = 0):
    """A DataLoader, single-process by default.

    ``workers=0`` is not a tuning choice.  On this host the checks stage died
    twice on its first row with ``CUDA error: unknown error`` out of
    ``cuMemcpyHtoDAsync``, while the same copy of the same shape ran thirty
    times in a row from a plain script -- the difference being four forked
    DataLoader workers, forked after the model had already initialised CUDA.
    With ``workers=0`` the identical run completed thirteen rows without a
    fault (DEVIATIONS D-32).

    The cost is small now that the render cache turned loading into an ``.npy``
    read and normalising moved to the device: about 17% on a warm row, against
    a stage that otherwise does not finish at all.
    """
    batch = batch_size_for(row)
    if row.input_kind in ("full_cap", "full_1024") and isinstance(dataset, GateDataset):
        shapes = [dataset.shape_of(i) for i in range(len(dataset))]
        sampler = ShapeBucketSampler(shapes, batch, shuffle=shuffle, seed=seed)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers)
        return (loader if workers else ThreadPrefetcher(loader)), sampler
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
    )
    return (loader if workers else ThreadPrefetcher(loader)), None


class ThreadPrefetcher:
    """Overlap the loader's reads with the model's compute, using threads.

    The gate is I/O bound, not compute bound: one training is five epochs plus
    an eval pass over about eleven thousand samples, and each sample reads its
    original JPEG and its window PNG -- roughly 90 GB per training, which at the
    114 MB/s this volume sustains is about thirteen minutes of pure reading.
    Measured, the GPU sat between 0 and 19% throughout.

    DataLoader workers would hide that, but they are processes, and forking them
    after the model has initialised CUDA crashes this host (D-32). Threads do
    not fork, so they inherit no CUDA context; and the work being overlapped --
    file reads, JPEG and PNG decode in OpenCV, the numpy composite -- releases
    the GIL, so a single background thread can keep the queue full while the
    main thread runs the model.

    Deliberately one thread and a shallow queue: the point is to overlap, not to
    parallelise, and a deep queue would just hold more decoded images in a
    process that has 23 GB to work with.
    """

    def __init__(self, loader, depth: int = 3) -> None:
        self.loader = loader
        self.depth = depth

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        import queue
        import threading

        done = object()
        batches: queue.Queue = queue.Queue(maxsize=self.depth)

        def fill():
            try:
                for batch in self.loader:
                    batches.put(batch)
            except Exception as exc:  # surface it on the consuming side
                batches.put(exc)
                return
            batches.put(done)

        thread = threading.Thread(target=fill, daemon=True)
        thread.start()
        while True:
            item = batches.get()
            if item is done:
                return
            if isinstance(item, Exception):
                raise item
            yield item


def train_one(
    train_samples: list[GateSample],
    test_samples: list[GateSample],
    row: I.GateRow,
    edits_root: Path,
    *,
    seed: int = 0,
    epochs: int = EPOCHS,
    device: str = "cuda",
    workers: int = 0,
    cache: dict | None = None,
    render_cache=None,
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
        train_ds = GateDataset(train_samples, row, edits_root, cache=cache,
                               render_cache=render_cache)
        test_ds = GateDataset(test_samples, row, edits_root, cache=cache,
                              render_cache=render_cache)

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
            # uint8 across the bus, normalised on the device (dataset.to_tensor)
            x = normalise(x.to(device, non_blocking=True))
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
            x = normalise(x.to(device, non_blocking=True))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x).float().squeeze(1)
            for i, s in zip(idx.tolist(), logits.tolist()):
                per_sample[int(i)].append(float(s))

    # P5: aggregate the tile row per image by the maximum tile score
    labels, scores, image_ids, sample_ids = [], [], [], []
    for i, sample in enumerate(test_samples):
        vals = per_sample.get(i)
        if not vals:
            continue
        labels.append(sample.label)
        scores.append(max(vals) if row.aggregate == "max_tile" else float(np.mean(vals)))
        image_ids.append(sample.image_id)
        sample_ids.append(sample.sample_id)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.array(labels), np.array(scores), image_ids, sample_ids


#: Contrasts whose positive class is a real referent removal.  P20's freeze sits
#: before any classifier sees one.  Written as a positive list rather than an
#: exclusion so a contrast added later is guarded by default and has to be
#: deliberately named safe.
REAL_REMOVAL_PREFIX = "REMOVE_vs_"


def _contrast_has_real_removals(contrast: str) -> bool:
    """The ladders ('global_q75', 'local_q50_1-2%') and the 1b null
    ('CONTROL_OBJ_vs_CONTROL_OBJ_2') contain no REMOVE and may run pre-freeze."""
    return contrast.startswith(REAL_REMOVAL_PREFIX)


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
    workers: int = 0,
    cache: dict | None = None,
    render_cache=None,
    allow_unfrozen: bool = False,
) -> RowResult:
    """P6 for one row: 3 seeds, mean AUROC, bootstrap CI over images.

    Refuses to train on a contrast containing real removals unless the B0a
    freeze is in force.  P20 puts the freeze "before the first classifier trains
    on real removals", and until now the only thing enforcing that was the CLI
    stage -- so a script calling this function directly walked straight past it.
    One did, on 2026-09-03, while timing throughput (DEVIATIONS D-31).  The
    guard belongs here, next to the training, not at the entry point that
    happened to be used.

    ``allow_unfrozen`` exists for the check-1a ladders and the 1b nulls, which
    P20 explicitly permits before the freeze because they run on their own pairs
    and see no real removal.
    """
    if not allow_unfrozen and _contrast_has_real_removals(contrast):
        from idea91 import freeze as F

        F.require(F.B0A, what=f"training on {contrast}")
    per_seed: list[float] = []
    pooled_labels: list[np.ndarray] = []
    pooled_scores: list[np.ndarray] = []
    pooled_images: list[str] = []
    sources: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))

    sample_scores: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        train_s, test_s = split_by_image(samples, seed=seed)
        labels, scores, image_ids, sample_ids = train_one(
            train_s,
            test_s,
            row,
            edits_root,
            seed=seed,
            epochs=epochs,
            device=device,
            workers=workers,
            cache=cache,
            render_cache=render_cache,
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
        for sample_id, score in zip(sample_ids, scores):
            sample_scores[sample_id].append(float(score))

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
        scores_by_sample={k: float(np.mean(v)) for k, v in sample_scores.items()},
    )
