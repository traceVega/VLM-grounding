"""Synthetic scenes, so the edit stack's assertions are testable without SAM 3."""

from __future__ import annotations

import numpy as np
import pytest

from idea91.edits.sampler import Instance


def disc(shape: tuple[int, int], cx: int, cy: int, r: int) -> np.ndarray:
    h, w = shape
    ys, xs = np.ogrid[:h, :w]
    return ((xs - cx) ** 2 + (ys - cy) ** 2) <= r * r


def rect(shape: tuple[int, int], x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def textured_image(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    """A deterministic RGB image with structure at several frequencies.

    Not noise: a smooth gradient plus a checker plus a little noise, so an
    inpainter's smear is visible and a JPEG ladder has something to destroy.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (
        60
        + 80 * np.sin(xx / 37.0)
        + 60 * np.cos(yy / 23.0)
        + 30 * (((xx // 16).astype(int) + (yy // 16).astype(int)) % 2)
    )
    img = np.stack([base, base * 0.8 + 40, base * 0.6 + 80], axis=-1)
    img = img + rng.normal(0, 4, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture
def scene():
    """A 1,000 x 800 scene: a referent and three candidate objects.

    Geometry is chosen so the P3 filters have something to accept and something
    to reject:

    * ``ref``      r=40 at (300, 400) -- the referent
    * ``ok``       r=44 at (700, 400) -- matched area, matched centre distance
    * ``too_big``  r=80 at (700, 200) -- area ratio 4, fails the area filter
    * ``off_axis`` r=40 at (960, 760) -- far corner, fails the centrality filter
    * ``crowded``  r=40 at (505, 400) -- touches ``neighbour``, fails exclusion
    * ``neighbour``r=20 at (548, 400) -- unremovable context next to ``crowded``
    """
    shape = (800, 1000)
    instances = [
        Instance("ref", disc(shape, 300, 400, 40), "referent", "cat"),
        Instance("ok", disc(shape, 700, 400, 44), "labelled_other_class", "dog"),
        Instance("too_big", disc(shape, 700, 200, 80), "labelled_other_class", "bus"),
        Instance("off_axis", disc(shape, 960, 760, 40), "labelled_other_class", "cup"),
        Instance("crowded", disc(shape, 505, 400, 40), "labelled_other_class", "bag"),
        Instance("neighbour", disc(shape, 548, 400, 20), "labelled_other_class", "strap"),
    ]
    return {
        "shape": shape,
        "image": textured_image(shape),
        "referent": instances[0],
        "instances": instances,
        "by_id": {i.instance_id: i for i in instances},
    }
