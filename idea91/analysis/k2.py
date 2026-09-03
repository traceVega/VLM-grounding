"""P15 rates and the P16 stop rules, as computable objects.

P16 verbatim, with the design's own annotations:

    (a) Both Qwen3-VL-8B and Molmo2-8B output ``none`` on more than 60% of clean
        REMOVE items: no rejection headroom, stop the training half; expected to
        be unfireable under the primary protocol and reported as such.
    (b) Secondary-protocol block only: both models emit a box on more than 80% of
        clean REMOVE items and abstain on more than 30% of CONTROL_OBJ items.
    (b') Primary-protocol form of (b): both models emit a box on more than 80% of
        clean REMOVE items, their box shifts on more than 30% of ORIGINAL-correct
        CONTROL_OBJ items, and that shift exceeds the CONTROL_BG shift on the same
        items by more than 10 points; a CONTROL_BG shift within 10 points of the
        CONTROL_OBJ shift reads as instability of the model, not confusion by the
        edit, and does not fire.
    (c) Same-box below 20% for both models: keep only the rejection half.
    Point estimates decide; CIs reported.

The decisive clean column is verifier-clean when acceptance check 3 passes and
human-clean otherwise; :func:`decisive_clean_column` is the one place that
choice is made.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from shared.stats import Interval, rate_ci

# P16 thresholds
NONE_ON_REMOVE_MAX = 0.60  # (a)
BOX_ON_REMOVE_MAX = 0.80  # (b), (b')
ABSTAIN_ON_CONTROL_MAX = 0.30  # (b)
SHIFT_ON_CONTROL_MAX = 0.30  # (b')
SHIFT_EXCESS_OVER_BG = 0.10  # (b') instability guard
SAME_BOX_MIN = 0.20  # (c)
FRONTIER_SAME_BOX_READING = 0.05  # "added reading, not a stop"

KAPPA_MIN = 0.60  # acceptance check 3


def decisive_clean_column(kappa: float | None) -> str:
    """Check 3: verifier-clean is decisive only if human-vs-V1 kappa >= 0.6."""
    if kappa is not None and np.isfinite(kappa) and kappa >= KAPPA_MIN:
        return "remove_clean_verifier"
    return "remove_clean_human"


def _rate(df: pd.DataFrame, column: str, seed: int = 0) -> Interval:
    """A P15 rate over the rows already restricted to its denominator."""
    if df.empty or column not in df:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    sub = df[df[column].notna()]
    if sub.empty:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    clusters = sub["image_id"].to_numpy() if "image_id" in sub else None
    return rate_ci(sub[column].to_numpy().astype(bool), clusters, seed=seed)


@dataclass
class ModelRates:
    """Every P15 rate for one model, protocol and clean column."""

    model_id: str
    abstain_protocol: str
    clean_column: str
    none_on_remove: Interval
    box_on_remove: Interval
    same_box: Interval
    same_box_r30: Interval
    shift_control_obj: Interval
    shift_control_bg: Interval
    abstain_control_obj: Interval
    t_null_pass: Interval
    t_null_prior_pass: Interval
    t_head_pass: Interval
    unresolved_share: Interval
    redundant_share: Interval
    undetermined_share: Interval
    n_pairs: int = 0

    @property
    def shift_excess(self) -> float:
        """CONTROL_OBJ shift minus CONTROL_BG shift, the (b') instability guard."""
        return self.shift_control_obj.point - self.shift_control_bg.point


def compute_rates(
    relations: pd.DataFrame,
    *,
    model_id: str,
    abstain_protocol: str,
    clean_column: str,
    seed: int = 0,
) -> ModelRates:
    """P15 for one model and protocol.

    Denominators, spelled out because they differ per statistic:

    * ``none``/``box`` on clean REMOVE: clean items only.
    * same-box: clean *and* ORIGINAL-correct *and* not UNRESOLVED (P15 excludes
      UNRESOLVED items from same-box).
    * box shift on the controls: ORIGINAL-correct items.
    * the shares: every pair.
    """
    df = relations[
        (relations["model_id"] == model_id)
        & (relations["abstain_protocol"] == abstain_protocol)
    ]
    clean = df[df[clean_column] == True]  # noqa: E712 -- None must not count as clean
    same_box_frame = clean[(clean["orig_correct"] == True) & (clean["unresolved"] != True)]  # noqa: E712
    orig_correct = df[df["orig_correct"] == True]  # noqa: E712

    return ModelRates(
        model_id=model_id,
        abstain_protocol=abstain_protocol,
        clean_column=clean_column,
        none_on_remove=_rate(clean, "none_on_remove", seed),
        box_on_remove=_rate(clean, "box_on_remove", seed),
        same_box=_rate(same_box_frame, "same_box_50", seed),
        same_box_r30=_rate(same_box_frame, "same_box_r30", seed),
        shift_control_obj=_rate(orig_correct, "shift_on_control_obj", seed),
        shift_control_bg=_rate(orig_correct, "shift_on_control_bg", seed),
        abstain_control_obj=_rate(df, "abstain_on_control_obj", seed),
        t_null_pass=_rate(df, "t_null_pass", seed),
        t_null_prior_pass=_rate(df, "t_null_prior_pass", seed),
        t_head_pass=_rate(df, "t_head_pass", seed),
        unresolved_share=_rate(df, "unresolved", seed),
        redundant_share=_rate(df, "redundant", seed),
        undetermined_share=_rate(df, "undetermined", seed),
        n_pairs=int(len(df)),
    )


# --- P16 -------------------------------------------------------------------


@dataclass
class StopRule:
    rule: str
    fires: bool
    consequence: str
    detail: str = ""
    unfireable: bool = False

    def line(self) -> str:
        if self.unfireable:
            return f"({self.rule}) not fireable under this protocol: {self.detail}"
        verdict = "FIRES" if self.fires else "does not fire"
        return f"({self.rule}) {verdict}: {self.detail} -> {self.consequence if self.fires else 'continue'}"


def _both(rates: list[ModelRates], predicate) -> bool:
    return len(rates) >= 2 and all(predicate(r) for r in rates)


def rule_a(rates: list[ModelRates]) -> StopRule:
    fires = _both(rates, lambda r: r.none_on_remove.point > NONE_ON_REMOVE_MAX)
    detail = "; ".join(f"{r.model_id} none={r.none_on_remove.point:.3f}" for r in rates)
    primary = all(r.abstain_protocol == "primary" for r in rates)
    return StopRule(
        rule="a",
        fires=fires,
        consequence="no rejection headroom; stop the training half",
        detail=detail
        + (
            "  [expected to be unfireable under the benchmark's primary protocol, "
            "where the rejection rate is near zero]"
            if primary
            else ""
        ),
    )


def rule_b(rates: list[ModelRates]) -> StopRule:
    """Secondary-protocol block only."""
    if any(r.abstain_protocol != "secondary" for r in rates):
        return StopRule(
            rule="b",
            fires=False,
            consequence="",
            detail="defined on the secondary-protocol block only; see (b') for the primary",
            unfireable=True,
        )
    fires = _both(
        rates,
        lambda r: r.box_on_remove.point > BOX_ON_REMOVE_MAX
        and r.abstain_control_obj.point > ABSTAIN_ON_CONTROL_MAX,
    )
    detail = "; ".join(
        f"{r.model_id} box={r.box_on_remove.point:.3f} abstain_ctl={r.abstain_control_obj.point:.3f}"
        for r in rates
    )
    return StopRule("b", fires, "the edits confuse; fix the editor before any other use", detail)


def rule_b_prime(rates: list[ModelRates]) -> StopRule:
    """Primary-protocol form of (b), with the CONTROL_BG instability guard."""
    if any(r.abstain_protocol != "primary" for r in rates):
        return StopRule(
            rule="b'", fires=False, consequence="", detail="primary-protocol form only",
            unfireable=True,
        )
    fires = _both(
        rates,
        lambda r: r.box_on_remove.point > BOX_ON_REMOVE_MAX
        and r.shift_control_obj.point > SHIFT_ON_CONTROL_MAX
        and r.shift_excess > SHIFT_EXCESS_OVER_BG,
    )
    detail = "; ".join(
        f"{r.model_id} box={r.box_on_remove.point:.3f} "
        f"shift_obj={r.shift_control_obj.point:.3f} shift_bg={r.shift_control_bg.point:.3f} "
        f"excess={r.shift_excess:+.3f}"
        for r in rates
    )
    guard = ""
    if not fires and _both(
        rates,
        lambda r: r.box_on_remove.point > BOX_ON_REMOVE_MAX
        and r.shift_control_obj.point > SHIFT_ON_CONTROL_MAX,
    ):
        guard = (
            "  [box and shift thresholds met, but the CONTROL_BG shift is within "
            f"{SHIFT_EXCESS_OVER_BG:.2f} of the CONTROL_OBJ shift: that reads as model "
            "instability, not confusion by the edit, and does not fire]"
        )
    return StopRule(
        "b'", fires, "the edits confuse; fix the editor before any other use", detail + guard
    )


def rule_c(rates: list[ModelRates]) -> StopRule:
    fires = _both(rates, lambda r: r.same_box.point < SAME_BOX_MIN)
    detail = "; ".join(
        f"{r.model_id} same_box={r.same_box.point:.3f} (n={r.same_box.n})" for r in rates
    )
    return StopRule("c", fires, "keep only the rejection half", detail)


def frontier_reading(same_box_point: float) -> str:
    """"Added reading, not a stop" (P16)."""
    if np.isfinite(same_box_point) and same_box_point < FRONTIER_SAME_BOX_READING:
        return (
            f"frontier same-box {same_box_point:.3f} < {FRONTIER_SAME_BOX_READING:.2f}: the audit "
            "headline will be 'boxes are load-bearing' and the metric path is planned "
            "(added reading, not a stop)"
        )
    return f"frontier same-box {same_box_point:.3f}: no added reading fires"


@dataclass
class K2Verdict:
    rates: list[ModelRates]
    kappa_human_vs_v1: float | None = None
    frontier_same_box: float | None = None
    calibration_line: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def clean_column(self) -> str:
        return decisive_clean_column(self.kappa_human_vs_v1)

    def stop_rules(self) -> list[StopRule]:
        primary = [r for r in self.rates if r.abstain_protocol == "primary"]
        secondary = [r for r in self.rates if r.abstain_protocol == "secondary"]
        out = [rule_a(primary), rule_b_prime(primary), rule_c(primary)]
        if secondary:
            out.append(rule_b(secondary))
        return out

    def report(self) -> str:
        lines = [
            f"K2 stop rules (primary protocol; decisive clean column: {self.clean_column}"
            + (
                f", human-vs-V1 kappa {self.kappa_human_vs_v1:.3f}"
                if self.kappa_human_vs_v1 is not None
                else ", kappa not yet measured"
            )
            + ")",
        ]
        lines += [f"  - {r.line()}" for r in self.stop_rules()]
        if self.frontier_same_box is not None:
            lines.append(f"  - {frontier_reading(self.frontier_same_box)}")
        if self.calibration_line:
            lines.append(f"  - {self.calibration_line}")
        lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)
