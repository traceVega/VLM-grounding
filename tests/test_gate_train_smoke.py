"""A real training pass on a toy bank (design B4a: "per-image AUROC CI on a toy set").

Marked ``gpu``: it builds a small edit bank on disk, downloads timm's ImageNet
weights and trains for one epoch on the card.  Deselect with ``-m 'not gpu'``.

The bank is built so the *answer is known*: the REMOVE class is filled by a
heavy blur and the CONTROL class by the same operator, so the two classes are
matched and a competent classifier should land near chance -- while the
JPEG ladder on the same images, which is a real difference, should be well above
it.  A run where the ladder cannot separate q30 is a broken training path, not a
clean editor, which is exactly the distinction acceptance check 1a exists to
make.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from idea91.edits.build import materialize  # noqa: E402
from idea91.edits.inpaint import TeleaInpainter  # noqa: E402
from idea91.edits.sampler import Instance, plan_k1_image  # noqa: E402
from idea91.gate import inputs as I  # noqa: E402
from idea91.gate import ladders as L  # noqa: E402
from idea91.gate.dataset import GateSample, split_by_image  # noqa: E402
from idea91.gate.train import run_row  # noqa: E402

from .conftest import disc, textured_image  # noqa: E402

pytestmark = pytest.mark.gpu

N_IMAGES = 24
SHAPE = (300, 400)


def build_bank(root: Path) -> list[GateSample]:
    inpainter = TeleaInpainter()
    samples: list[GateSample] = []
    for i in range(N_IMAGES):
        rng = np.random.default_rng(100 + i)
        image = textured_image(SHAPE, seed=i)
        original_path = root / "originals" / f"img{i:03d}.png"
        original_path.parent.mkdir(parents=True, exist_ok=True)
        import cv2

        cv2.imwrite(str(original_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # two matched objects, placed at mirrored positions so centrality matches
        cx = int(rng.integers(90, 130))
        cy = int(rng.integers(120, 180))
        r = int(rng.integers(16, 22))
        instances = [
            Instance("ref", disc(SHAPE, cx, cy, r), "referent", "cat"),
            Instance("ctl", disc(SHAPE, 400 - cx, cy, r), "labelled_other_class", "dog"),
        ]
        plans = plan_k1_image(
            instances[0], instances, rng=np.random.default_rng(i), with_background=False
        )
        assert plans is not None, "the toy scene must yield a matched control"
        for plan in plans:
            if plan.operator not in ("REMOVE", "CONTROL_OBJ"):
                continue
            built = materialize(
                image,
                plan,
                inpainter,
                image_id=f"img{i:03d}",
                instance_id="ref",
                set_or_pool="synthetic",
                edits_root=root / "edits",
                freeze_version="toy",
            )
            samples.append(
                GateSample(
                    sample_id=f"img{i:03d}_{plan.operator}",
                    image_id=f"img{i:03d}",
                    label=1 if plan.operator == "REMOVE" else 0,
                    original_path=original_path,
                    window_path=Path(built.index_row["window_path"]),
                    window_xyxy=tuple(built.index_row["window_xyxy_px"]),
                    hole_box=None,
                    control_source=plan.control_source or "referent",
                )
            )
    return samples


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy_bank")
    return {"root": root, "samples": build_bank(root)}


def test_the_split_never_puts_one_image_on_both_sides(bank):
    train, test = split_by_image(bank["samples"], seed=0)
    assert train and test
    assert not ({s.image_id for s in train} & {s.image_id for s in test})
    # both edits of an image travel together
    for s in test:
        assert sum(1 for t in bank["samples"] if t.image_id == s.image_id) == 2


@pytest.mark.parametrize("row_name", ["rep_resnet18_512", "ii_vit_s16_tiles_native"])
def test_a_gate_row_trains_and_scores_every_held_out_image(bank, row_name):
    row = I.ROWS_BY_NAME[row_name]
    result = run_row(
        bank["samples"],
        row,
        bank["root"] / "edits",
        hole_type="mask",
        contrast="REMOVE_vs_CONTROL_OBJ",
        seeds=(0,),
        epochs=1,
        workers=0,
    )
    assert np.isfinite(result.auroc_mean)
    assert 0.0 <= result.auroc_mean <= 1.0
    assert result.n_images == len(split_by_image(bank["samples"], seed=0)[1]) // 2
    assert np.isfinite(result.ci.lo) and np.isfinite(result.ci.hi)
    assert "labelled_other_class" in result.by_control_source
    assert row.name in result.line()


def test_the_local_jpeg_ladder_produces_a_separable_pair(bank):
    """The ladder must be able to damage an image the gate can then see.

    Not a full check-1a run (that needs the 1b pairs and a trained classifier);
    this asserts the ladder's operator does something measurable at q30, so a
    later AUROC near 0.5 is about the classifier, not about a no-op ladder.
    """
    sample = bank["samples"][0]
    image = sample.compose(bank["root"] / "edits")
    hole = disc(SHAPE, 110, 150, 30)
    damaged = L.local_jpeg(image, hole, 30)
    reference = L.local_jpeg(image, hole, 95)
    assert np.abs(damaged.astype(int) - image.astype(int)).sum() > np.abs(
        reference.astype(int) - image.astype(int)
    ).sum()
