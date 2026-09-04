"""P21's token-count histogram, written before the run (design 4.6, B8).

    a token-count histogram is written before the run and no request may be
    rejected

The two clauses are one job: count what each image would cost in visual tokens
under the run's resolution policy, and check that prompt plus image stays inside
the server's context.  A request that exceeds it is not a bad row -- vLLM
refuses it outright, so the item silently leaves the denominator of every rate
in P15.  Finding that at B11, after the ORIGINAL condition is scored and the K2
freeze has landed, would be expensive; finding it now costs a CPU pass over the
processor.

The count is exact rather than estimated: the real processor at the pinned
revision decides how an image is tiled, and a formula reimplementing that would
be a second source of truth that could drift.  Only the processor is loaded, not
the weights, so this runs on the CPU in minutes.

Qwen3-VL's cost follows from its patch grid: an image is resized so its area
fits ``max_pixels``, then cut into 16 px patches which are merged 2x2, so the
visual token count is about ``pixels / (16 * 16 * 2 * 2)``.  Molmo2's follows
from its crop tiling instead (P19), which is why its policy is recorded as crops
rather than pixels and why the two need separate passes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

#: P21: the run-level override for Qwen3-VL.
MAX_PIXELS = 2_457_600
#: P21: vLLM is served with this, and nothing may exceed it.
MAX_MODEL_LEN = 4096


@dataclass
class TokenHistogram:
    """What the run would send, and whether the server would take it."""

    model_id: str
    set_name: str
    n_images: int
    max_pixels: int
    max_model_len: int
    visual_tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    #: images whose visual tokens alone exceed the context
    n_over_context: int = 0
    #: images the resolution policy had to downscale (P21 reports this share)
    n_over_cap: int = 0

    @property
    def total_max(self) -> int:
        return (max(self.visual_tokens) if self.visual_tokens else 0) + self.prompt_tokens

    @property
    def any_rejected(self) -> bool:
        """P21: 'no request may be rejected'."""
        return self.total_max > self.max_model_len

    def percentiles(self) -> dict[str, float]:
        if not self.visual_tokens:
            return {}
        values = np.array(self.visual_tokens)
        return {f"p{p}": float(np.percentile(values, p)) for p in (50, 90, 99, 100)}

    def report(self) -> str:
        lines = [
            f"P21 token histogram: {self.model_id} on {self.set_name}",
            f"  {self.n_images} images, max_pixels {self.max_pixels:,}, "
            f"max_model_len {self.max_model_len:,}",
            f"  over the pixel cap (downscaled): {self.n_over_cap} "
            f"({self.n_over_cap / max(self.n_images, 1):.1%})",
            f"  prompt tokens: {self.prompt_tokens}",
        ]
        for name, value in self.percentiles().items():
            lines.append(f"  visual tokens {name}: {value:,.0f}")
        lines.append(f"  worst case prompt+image: {self.total_max:,}")
        if self.any_rejected:
            lines.append(
                f"  REJECTED: {self.total_max:,} exceeds max_model_len "
                f"{self.max_model_len:,} -- P21 forbids this. Raise max_model_len or "
                "lower max_pixels before the run; do not discover it per request."
            )
        else:
            headroom = self.max_model_len - self.total_max
            lines.append(f"  PASS: {headroom:,} tokens of headroom on the worst image")
        return "\n".join(lines)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self) | {
            "percentiles": self.percentiles(),
            "total_max": self.total_max,
            "any_rejected": self.any_rejected,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def load_capped_processor(hf_path: str, revision: str, max_pixels: int = MAX_PIXELS):
    """The processor with P21's cap actually in force.

    **The cap must be passed at construction.**  Qwen3-VL ships a
    ``Qwen2VLImageProcessor`` whose size lives in a ``SizeDict`` of
    ``longest_edge``/``shortest_edge`` *areas*; it has no ``max_pixels``
    attribute.  Assigning ``processor.image_processor.max_pixels = ...`` after
    the fact therefore creates an unused attribute and silently changes nothing:
    measured on a 2643x1500 image, 3,901 visual tokens before and after.  Passed
    to ``from_pretrained`` the same number becomes 2,340, because transformers
    maps the keyword onto ``size.longest_edge`` while building it.

    That matters more than a factor of 1.7.  Uncapped, the largest GroundingME
    image (59 Mpx) is about 57,600 visual tokens against ``max_model_len`` 4,096,
    so vLLM would refuse the request outright -- and P21 says no request may be
    rejected.  A silently ineffective cap would not fail loudly; it would drop
    items out of every denominator in P15.
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(hf_path, revision=revision, max_pixels=max_pixels)


def qwen3vl_visual_tokens(
    width: int, height: int, *, max_pixels: int = MAX_PIXELS, patch: int = 16, merge: int = 2
) -> int:
    """Approximate visual tokens for one image under P21's cap.

    Convenient for a size distribution, and **not** authoritative: the processor
    rounds to the patch grid inside its own resize, so this runs a little high
    (2,400 against the processor's 2,340 on a 2643x1500 image).  Use
    :func:`histogram_from_processor` for anything that decides whether a request
    fits; this exists to sketch the shape without loading transformers.

    The resize preserves aspect ratio and is downscale-only, matching
    ``full_at_cap``: P21 sets ``max_pixels`` and leaves ``min_pixels`` at the
    processor default, so an image below the cap is passed through rather than
    upscaled.  This differs from IDEA-11's D4, which pins min = max.
    """
    pixels = width * height
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
    grid = patch * merge
    # The processor rounds each side up to the merged patch grid.
    cols = -(-width // grid)
    rows = -(-height // grid)
    return int(cols * rows)


def processor_visual_tokens(processor, width: int, height: int) -> int:
    """Exact visual tokens, from the processor itself.

    A blank image of the right size is enough: the token count is a function of
    the dimensions, so nothing has to be read off disk.
    """
    from PIL import Image

    inputs = processor(
        images=[Image.new("RGB", (int(width), int(height)))],
        text="<|image_pad|>",
        return_tensors="pt",
    )
    grid = inputs["image_grid_thw"]
    merge = getattr(processor.image_processor, "merge_size", 2)
    return int(grid.prod(dim=-1).sum()) // (merge * merge)


def histogram_for_sizes(
    sizes: list[tuple[int, int]],
    *,
    model_id: str,
    set_name: str,
    prompt_tokens: int,
    max_pixels: int = MAX_PIXELS,
    max_model_len: int = MAX_MODEL_LEN,
) -> TokenHistogram:
    """The histogram from image sizes alone -- no images opened, no model loaded."""
    counts = [qwen3vl_visual_tokens(w, h, max_pixels=max_pixels) for w, h in sizes]
    return TokenHistogram(
        model_id=model_id,
        set_name=set_name,
        n_images=len(sizes),
        max_pixels=max_pixels,
        max_model_len=max_model_len,
        visual_tokens=counts,
        prompt_tokens=prompt_tokens,
        n_over_context=sum(1 for c in counts if c + prompt_tokens > max_model_len),
        n_over_cap=sum(1 for w, h in sizes if w * h > max_pixels),
    )


def histogram_from_processor(
    sizes: list[tuple[int, int]],
    *,
    hf_path: str,
    revision: str,
    model_id: str,
    set_name: str,
    prompt_tokens: int,
    max_pixels: int = MAX_PIXELS,
    max_model_len: int = MAX_MODEL_LEN,
) -> TokenHistogram:
    """P21's histogram, counted by the processor rather than estimated.

    Distinct sizes are counted once and reused: a benchmark's images repeat
    sizes heavily, and the processor call is the expensive part.
    """
    processor = load_capped_processor(hf_path, revision, max_pixels)
    seen: dict[tuple[int, int], int] = {}
    counts: list[int] = []
    for size in sizes:
        key = (int(size[0]), int(size[1]))
        if key not in seen:
            seen[key] = processor_visual_tokens(processor, *key)
        counts.append(seen[key])

    return TokenHistogram(
        model_id=model_id,
        set_name=set_name,
        n_images=len(sizes),
        max_pixels=max_pixels,
        max_model_len=max_model_len,
        visual_tokens=counts,
        prompt_tokens=prompt_tokens,
        n_over_context=sum(1 for c in counts if c + prompt_tokens > max_model_len),
        n_over_cap=sum(1 for w, h in sizes if w * h > max_pixels),
    )


def assert_cap_is_in_force(processor, max_pixels: int = MAX_PIXELS) -> None:
    """Refuse a processor whose cap did not take, before it costs a whole run.

    The failure this guards is silent by construction: an ineffective cap
    produces valid-looking inputs that the server then refuses, one item at a
    time, and those items leave P15's denominators without anything raising.
    """
    huge = processor_visual_tokens(processor, 8000, 8000)
    ceiling = max_pixels // (16 * 16 * 2 * 2)
    if huge > ceiling * 1.1:
        raise RuntimeError(
            f"the {max_pixels:,} px cap is not in force: an 8000x8000 image still "
            f"costs {huge:,} visual tokens, against about {ceiling:,} capped. "
            "Pass max_pixels to from_pretrained -- assigning "
            "image_processor.max_pixels afterwards does nothing on this processor "
            "(it keeps its size in a SizeDict and has no such attribute)."
        )
