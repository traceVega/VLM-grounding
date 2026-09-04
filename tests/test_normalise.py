"""Normalisation moved from the loader to the device: same numbers, less work."""

from __future__ import annotations

import numpy as np
import torch

from idea91.gate.dataset import IMAGENET_MEAN, IMAGENET_STD, normalise, to_tensor


def reference(image: np.ndarray) -> torch.Tensor:
    """What the loader used to produce, kept here as the thing to match."""
    t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def test_the_loader_now_hands_over_bytes():
    image = np.random.default_rng(0).integers(0, 255, (7, 5, 3), dtype=np.uint8)
    t = to_tensor(image)
    assert t.dtype == torch.uint8, "four bytes per channel across the bus was the cost"
    assert t.shape == (3, 7, 5), "CHW, as the model wants"


def test_the_result_is_numerically_what_it_was_before():
    image = np.random.default_rng(1).integers(0, 255, (9, 11, 3), dtype=np.uint8)
    moved = normalise(to_tensor(image).unsqueeze(0))[0]
    assert torch.allclose(moved, reference(image), atol=1e-6)


def test_channel_order_survives_the_move():
    """A silent CHW/HWC or RGB/BGR flip here would change every AUROC and
    nothing would raise."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0] = 255  # pure red
    out = normalise(to_tensor(image).unsqueeze(0))[0]
    expected_r = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    expected_g = (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    assert torch.allclose(out[0], torch.full((4, 4), expected_r), atol=1e-6)
    assert torch.allclose(out[1], torch.full((4, 4), expected_g), atol=1e-6)


def test_a_read_only_render_from_the_cache_is_accepted():
    """np.load(mmap_mode='r') returns a read-only array; torch refuses to share
    memory with one, and the old ascontiguousarray path only warned."""
    image = np.random.default_rng(2).integers(0, 255, (6, 6, 3), dtype=np.uint8)
    image.setflags(write=False)
    t = to_tensor(image)
    assert t.shape == (3, 6, 6)
    t[0, 0, 0] = 7  # writable, so no undefined behaviour downstream


def test_a_batch_normalises_per_image_not_across_the_batch():
    a = np.zeros((3, 3, 3), dtype=np.uint8)
    b = np.full((3, 3, 3), 255, dtype=np.uint8)
    batch = torch.stack([to_tensor(a), to_tensor(b)])
    out = normalise(batch)
    assert torch.allclose(out[0], reference(a), atol=1e-6)
    assert torch.allclose(out[1], reference(b), atol=1e-6)


def test_normalise_does_not_corrupt_the_input_batch():
    """div_ is in place; it must land on the float copy, not the uint8 source."""
    image = np.full((2, 2, 3), 128, dtype=np.uint8)
    batch = to_tensor(image).unsqueeze(0)
    before = batch.clone()
    normalise(batch)
    assert torch.equal(batch, before)
