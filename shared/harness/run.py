"""The K2 run loop: items and conditions in, ``outputs.parquet`` out (design 4.6).

One row per (item, condition, protocol).  The loop itself is deliberately dull
-- resolve, compose, ask, parse, score, record -- because everything interesting
has been pushed into pieces that can be tested without a served model:
:mod:`conditions` decides what to send, :mod:`parsers` decides what came back,
and :mod:`tokens` decides whether the request will be accepted at all.

Three things it enforces rather than assumes.

**P20's K2 freeze.** "K2 freeze: P8 to P21 are frozen ... before the first
ORIGINAL item is scored."  So scoring ORIGINAL requires B0b, exactly as training
on a real removal requires B0a.  The K1 guard was added after a probe walked
past the CLI check (DEVIATIONS D-31); this one is here from the start.

**P21's rejection rule.** "no request may be rejected."  A request that exceeds
the server's context is not a bad row -- it never becomes a row at all, and the
item silently leaves the denominator of every rate in P15.  The loop checks the
budget before sending and refuses to start a run that cannot fit.

**``image_px_sent`` on every row.**  SPEC requires it, and it is the only record
of what the model actually saw once the P21 cap has resized 99.3% of the set.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from shared.harness import conditions as C
from shared.harness import parsers as P
from shared.harness import prompts as PR
from shared.harness import schema as S


class Client(Protocol):
    """What the loop needs of a served model.  vLLM and a fake both satisfy it."""

    def generate(self, prompt: str, image: np.ndarray) -> "Answer": ...


@dataclass
class Answer:
    text: str
    latency_ms: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass
class Item:
    """One row of ``items.parquet``, with what the conditions need."""

    item_id: str
    image_path: Path
    expr: str
    gt_boxes_xyxy_px: list[tuple[float, float, float, float]] = field(default_factory=list)
    pair_id: str | None = None
    head_noun: str | None = None
    absent_category: str | None = None
    attr_swapped_expr: str | None = None
    hole_type: str = "mask"

    @property
    def gt_box(self) -> tuple[float, float, float, float] | None:
        return tuple(self.gt_boxes_xyxy_px[0]) if self.gt_boxes_xyxy_px else None


@dataclass
class RunConfig:
    run_id: str
    model_id: str
    coordinate_convention: str
    parser: str = "qwen3vl_box"
    protocols: tuple[str, ...] = ("primary",)
    conditions: tuple[str, ...] = (C.ORIGINAL,)
    non_kill: bool = False
    max_pixels: int = 2_457_600
    max_model_len: int = 4096
    none_patterns: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: protocol -> prompt file.  The secondary is its own file rather than the
    #: primary plus an appended suffix: two ways to build the same string is how
    #: T_NULL's template drifted from the instruction it controls for, and a
    #: test already asserts the secondary file is the primary plus P12's
    #: sentence.  It also means ``prompt_version`` is the hash of what was
    #: actually sent, which is what SPEC asks for.
    prompt_names: dict[str, str] = field(
        default_factory=lambda: {
            "primary": "grounding_qwen3vl_primary",
            "secondary": "grounding_qwen3vl_secondary",
        }
    )


def cap_image(image: np.ndarray, max_pixels: int) -> np.ndarray:
    """P21's downscale-only cap, and the size that goes in ``image_px_sent``."""
    from idea91.gate.inputs import full_at_cap

    return full_at_cap(image, max_pixels)


def require_k2_freeze(conditions: Iterable[str], *, non_kill: bool) -> None:
    """P20: no ORIGINAL item is scored before B0b.

    A ``--non-kill`` run is exempt because that is what P12's dev slices and the
    parity run of check 4 are, and they must be able to run before the freeze;
    the manifest records which kind a run was, and a non-kill id cannot enter a
    kill table.
    """
    if non_kill or C.ORIGINAL not in set(conditions):
        return
    from idea91 import freeze as F

    F.require(F.B0B, what="scoring the ORIGINAL condition")


def check_budget(
    sizes: list[tuple[int, int]], *, prompt_tokens: int, cfg: RunConfig
) -> None:
    """P21: refuse a run whose largest request the server would reject."""
    from shared.harness import tokens as T

    histogram = T.histogram_for_sizes(
        sizes,
        model_id=cfg.model_id,
        set_name="run",
        prompt_tokens=prompt_tokens,
        max_pixels=cfg.max_pixels,
        max_model_len=cfg.max_model_len,
    )
    if histogram.any_rejected:
        raise SystemExit(
            "P21 says no request may be rejected, and this run has one that would be:\n"
            + histogram.report()
        )


def row_for(
    item: Item,
    resolved: C.Resolved,
    parsed: P.Parsed,
    answer: Answer,
    *,
    cfg: RunConfig,
    protocol: str,
    prompt_version: str,
    image_px_sent: tuple[int, int],
) -> dict:
    """One ``outputs.parquet`` row (SPEC Section 4).

    ``iou_gt``, ``point_in_gt`` and ``is_failure`` are ORIGINAL-only by schema:
    the other conditions have no ground-truth box to score against, which is the
    whole reason P14 compares their boxes against the ORIGINAL box instead.
    """
    is_original = resolved.condition == C.ORIGINAL
    box = [float(v) for v in parsed.box_xyxy_px] if parsed.box_xyxy_px else None
    # exactly two floats: the schema checks it, and padding to four would read
    # as a degenerate box to anything that did not look at output_type
    point = [float(v) for v in parsed.point_xy_px] if parsed.point_xy_px else None

    iou_gt = point_in_gt = is_failure = None
    if is_original and item.gt_box is not None:
        if parsed.box_xyxy_px is not None:
            iou_gt = P.iou(parsed.box_xyxy_px, item.gt_box)
            is_failure = iou_gt < 0.5
        elif parsed.point_xy_px is not None:
            point_in_gt = P.point_in_box(parsed.point_xy_px, item.gt_box)
            is_failure = not point_in_gt
        else:
            is_failure = True  # no box and no point on ORIGINAL is a failure

    return {
        "run_id": cfg.run_id,
        "model_id": cfg.model_id,
        "item_id": item.item_id,
        "condition": resolved.condition,
        "pair_id": item.pair_id,
        "abstain_protocol": protocol,
        "prompt_version": prompt_version,
        "raw_text": answer.text,
        "parse_ok": parsed.parse_ok,
        "output_type": parsed.output_type,
        "n_boxes": parsed.n_boxes,
        "box_xyxy_px": box,
        "point_xy_px": point,
        "iou_gt": iou_gt,
        "point_in_gt": point_in_gt,
        "is_failure": is_failure,
        "image_px_sent": [int(image_px_sent[0]), int(image_px_sent[1])],
        "latency_ms": int(answer.latency_ms),
        "tokens_in": answer.tokens_in,
        "tokens_out": answer.tokens_out,
    }


def run(
    items: list[Item],
    cfg: RunConfig,
    client: Client,
    *,
    compose=None,
    progress_every: int = 100,
) -> list[dict]:
    """Every (item, condition, protocol), as ``outputs.parquet`` rows.

    ``compose(item, operator)`` returns the edited image for an image condition;
    it is injected because composing depends on the edit bank, which the harness
    should not have to know about.
    """
    require_k2_freeze(cfg.conditions, non_kill=cfg.non_kill)

    from idea91.edits.build import read_image

    missing = [p for p in cfg.protocols if p not in cfg.prompt_names]
    if missing:
        raise ValueError(f"no prompt file named for protocol(s) {missing}")
    templates = {
        protocol: PR.load(cfg.prompt_names[protocol], non_kill=cfg.non_kill)
        for protocol in cfg.protocols
    }
    parse = P.PARSERS[cfg.parser]
    rows: list[dict] = []

    for n, item in enumerate(items, 1):
        original = read_image(item.image_path)
        for condition in cfg.conditions:
            try:
                resolved = C.resolve(
                    condition,
                    expr=item.expr,
                    head_noun=item.head_noun,
                    absent_category=item.absent_category,
                    attr_swapped_expr=item.attr_swapped_expr,
                    hole_type=item.hole_type,
                )
            except ValueError as exc:
                # A condition this item cannot carry -- no absent category, no
                # attribute rewrite. Skipped and counted, never faked.
                print(f"  {item.item_id} {condition}: skipped, {exc}")
                continue

            image = original
            if resolved.uses_edited_image:
                if compose is None:
                    raise ValueError(f"{condition} needs an edited image; pass compose=")
                image = compose(item, resolved.operator)
            sent = cap_image(image, cfg.max_pixels)
            height, width = sent.shape[:2]

            for protocol in cfg.protocols:
                template = templates[protocol]
                text = template.render(expr=resolved.expr)
                started = time.perf_counter()
                answer = client.generate(text, sent)
                if not answer.latency_ms:
                    answer.latency_ms = int((time.perf_counter() - started) * 1000)

                parsed = parse(
                    answer.text,
                    convention=cfg.coordinate_convention,
                    original_wh=(original.shape[1], original.shape[0]),
                    sent_wh=(width, height),
                    none_patterns=cfg.none_patterns.get(protocol, ()),
                )
                rows.append(
                    row_for(
                        item, resolved, parsed, answer,
                        cfg=cfg, protocol=protocol,
                        prompt_version=template.version,
                        image_px_sent=(width, height),
                    )
                )
        if progress_every and n % progress_every == 0:
            print(f"  [{n}/{len(items)}] {len(rows)} rows")
    return rows


def write(rows: list[dict], path: Path):
    """Validated on the way out, so a bad row fails here and not in analysis."""
    return S.write_table(rows, path, "outputs")
