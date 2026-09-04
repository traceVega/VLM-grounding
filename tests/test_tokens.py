"""P21's token histogram and the cap that has to be in force for it to hold."""

from __future__ import annotations

import json

import pytest

from shared.harness import tokens as T


def hist(sizes, prompt_tokens=258, **kw):
    return T.histogram_for_sizes(
        sizes, model_id="qwen3vl-8b-instruct", set_name="groundingme",
        prompt_tokens=prompt_tokens, **kw
    )


# --- the arithmetic -----------------------------------------------------------


def test_an_image_under_the_cap_is_not_upscaled():
    """P21 sets max_pixels and leaves min_pixels at the processor default, so
    this differs from IDEA-11's D4, which pins min = max and upscales."""
    small = T.qwen3vl_visual_tokens(320, 240)
    # ceiling, not floor: a side is padded up to the 32 px merged-patch grid
    assert small == 10 * 8
    assert small < T.qwen3vl_visual_tokens(3200, 2400)


def test_an_image_over_the_cap_lands_near_the_cap_cost():
    ceiling = T.MAX_PIXELS // (16 * 16 * 2 * 2)
    for size in ((7680, 7680), (4000, 3000), (2643, 1500)):
        assert T.qwen3vl_visual_tokens(*size) == pytest.approx(ceiling, rel=0.05)


def test_aspect_ratio_survives_the_cap():
    wide = T.qwen3vl_visual_tokens(8000, 2000)
    square = T.qwen3vl_visual_tokens(4000, 4000)
    assert wide == pytest.approx(square, rel=0.05), "both are capped to the same area"


# --- the histogram ------------------------------------------------------------


def test_the_verdict_is_about_the_worst_image_not_the_median():
    """'No request may be rejected' is a maximum, not an average."""
    sizes = [(640, 480)] * 999 + [(7680, 7680)]
    h = hist(sizes, max_model_len=2000)
    assert h.any_rejected, "one oversized image is enough to breach P21"


def test_headroom_is_reported_when_everything_fits():
    h = hist([(2643, 1500)] * 10)
    assert not h.any_rejected
    assert "PASS" in h.report() and "headroom" in h.report()


def test_the_failure_says_what_to_change():
    h = hist([(7680, 7680)], max_model_len=100)
    text = h.report()
    assert "REJECTED" in text
    assert "max_model_len" in text and "max_pixels" in text


def test_the_over_cap_share_is_reported_because_p21_asks_for_it():
    h = hist([(640, 480), (7680, 7680), (4000, 4000)])
    assert h.n_over_cap == 2


def test_the_histogram_round_trips_through_json(tmp_path):
    h = hist([(2643, 1500)] * 5)
    path = h.write(tmp_path / "tokens.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_images"] == 5
    assert payload["any_rejected"] is False
    assert "p100" in payload["percentiles"]


def test_an_empty_set_does_not_divide_by_zero():
    h = hist([])
    assert h.percentiles() == {}
    assert h.total_max == h.prompt_tokens


# --- the cap guard ------------------------------------------------------------


class FakeProcessor:
    """Stands in for the real processor: capped or not, as told."""

    def __init__(self, capped: bool) -> None:
        self.capped = capped

    def __call__(self, images, text, return_tensors):  # noqa: ARG002
        import torch

        width, height = images[0].size
        if self.capped and width * height > T.MAX_PIXELS:
            scale = (T.MAX_PIXELS / (width * height)) ** 0.5
            width, height = int(width * scale), int(height * scale)
        return {"image_grid_thw": torch.tensor([[1, height // 16, width // 16]])}

    class image_processor:  # noqa: N801
        merge_size = 2


def test_an_uncapped_processor_is_refused_before_it_costs_a_run():
    """The failure is silent by construction: an ineffective cap makes inputs
    the server then refuses one at a time, and those items leave P15's
    denominators without anything raising."""
    with pytest.raises(RuntimeError, match="not in force"):
        T.assert_cap_is_in_force(FakeProcessor(capped=False))


def test_the_refusal_names_the_actual_remedy():
    with pytest.raises(RuntimeError, match="from_pretrained"):
        T.assert_cap_is_in_force(FakeProcessor(capped=False))


def test_a_capped_processor_passes():
    T.assert_cap_is_in_force(FakeProcessor(capped=True))  # must not raise


def test_the_formula_is_documented_as_approximate_not_authoritative():
    """Measured: the formula says 2,400 where the processor says 2,340, because
    the processor rounds inside its own resize. Anything deciding whether a
    request fits must use the processor."""
    assert "not** authoritative" in T.qwen3vl_visual_tokens.__doc__
