"""big-LaMa: the config resolver and the pickle stubs without weights, and the
real fill with them (design B1: "LaMa fills a window").

The weight-bearing tests are marked ``gpu`` and skip when ``VLMG_LAMA_DIR`` does
not point at an unpacked ``big-lama`` directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from idea91.edits import lama

from .conftest import disc, textured_image


def lama_dir() -> Path | None:
    raw = os.environ.get("VLMG_LAMA_DIR") or (Path.home() / "vlmg-data/raw/big-lama/big-lama")
    path = Path(raw).expanduser()
    return path if (path / "models" / "best.ckpt").is_file() else None


# --- no weights needed -------------------------------------------------------


def test_interpolations_resolve_like_omegaconf():
    """big-LaMa's generator block chains three ``${a.b.c}`` references."""
    config = {
        "generator": {
            "init_conv_kwargs": {"ratio_gout": 0},
            "downsample_conv_kwargs": {
                "ratio_gin": "${generator.init_conv_kwargs.ratio_gout}",
                "ratio_gout": "${generator.downsample_conv_kwargs.ratio_gin}",
            },
            "resnet_conv_kwargs": {
                "ratio_gin": 0.75,
                "ratio_gout": "${generator.resnet_conv_kwargs.ratio_gin}",
            },
        }
    }
    out = lama.resolve_interpolations(config)["generator"]
    assert out["downsample_conv_kwargs"] == {"ratio_gin": 0, "ratio_gout": 0}
    assert out["resnet_conv_kwargs"] == {"ratio_gin": 0.75, "ratio_gout": 0.75}


def test_interpolation_cycles_are_reported_not_hung():
    config = {"a": {"x": "${a.y}", "y": "${a.x}"}}
    with pytest.raises(ValueError, match="cycle"):
        lama.resolve_interpolations(config)


def test_values_that_are_not_interpolations_pass_through():
    config = {"g": {"kind": "ffc_resnet", "ngf": 64, "act": None, "layers": [1, 2]}}
    assert lama.resolve_interpolations(config) == config


def test_the_stub_finder_satisfies_the_lightning_pickle():
    """The checkpoint references training packages we deliberately do not install."""
    lama._install_stubs()
    import omegaconf.listconfig  # noqa: F401  -- resolves via the stub
    import pytorch_lightning.callbacks  # noqa: F401

    from pytorch_lightning import LightningModule  # type: ignore

    assert LightningModule is lama._StubAny
    import numpy  # a real package must still import normally

    assert numpy.__name__ == "numpy"


def test_a_missing_checkpoint_says_where_to_put_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="VLMG_LAMA_DIR"):
        lama.read_checkpoint(tmp_path)


def test_the_inpainter_refuses_a_directory_without_weights(tmp_path):
    from idea91.edits.inpaint import LamaInpainter

    with pytest.raises(FileNotFoundError, match="big-lama directory"):
        LamaInpainter(model_dir=tmp_path)


# --- with the real weights ---------------------------------------------------

pytestmark_gpu = pytest.mark.gpu


@pytest.mark.gpu
@pytest.mark.skipif(lama_dir() is None, reason="big-lama checkpoint not on this machine")
def test_the_released_checkpoint_reads_as_an_ffc_resnet():
    ckpt = lama.read_checkpoint(lama_dir())
    assert ckpt.generator_kwargs["input_nc"] == 4  # RGB + mask
    assert ckpt.generator_kwargs["output_nc"] == 3
    assert ckpt.generator_kwargs["n_blocks"] == 18
    assert ckpt.generator_kwargs["resnet_conv_kwargs"]["ratio_gout"] == 0.75  # resolved
    assert 50e6 < ckpt.n_params < 55e6
    assert all(not k.startswith("generator.") for k in ckpt.state_dict)


@pytest.mark.gpu
@pytest.mark.skipif(lama_dir() is None, reason="big-lama checkpoint not on this machine")
def test_lama_fills_a_window_and_touches_nothing_outside_it():
    from idea91.edits import composite as C
    from idea91.edits.inpaint import LamaInpainter

    shape = (512, 512)
    image = textured_image(shape, seed=3)
    hole = disc(shape, 256, 256, 60)
    image = image.copy()
    image[hole] = (220, 30, 30)  # a flat red blob, so removal is measurable

    inpainter = LamaInpainter(model_dir=lama_dir())
    assert inpainter.kill_grade is True
    assert len(inpainter.weights_sha256) == 64  # goes into every index row

    filled = inpainter.fill(image, hole)
    assert filled.shape == image.shape and filled.dtype == np.uint8

    out = C.composite_in_mask(image, filled, hole)
    C.assert_in_mask(image, out, hole)  # check 1c holds on a real edit
    assert out[hole][:, 0].mean() < 0.7 * image[hole][:, 0].mean()  # the blob went


@pytest.mark.gpu
@pytest.mark.skipif(lama_dir() is None, reason="big-lama checkpoint not on this machine")
def test_a_non_multiple_of_eight_window_round_trips():
    """Windows are clipped to the image edge, so odd sizes are the common case."""
    from idea91.edits.inpaint import LamaInpainter

    shape = (411, 677)
    image = textured_image(shape, seed=4)
    hole = disc(shape, 300, 200, 40)
    out = LamaInpainter(model_dir=lama_dir()).fill(image, hole)
    assert out.shape == image.shape
