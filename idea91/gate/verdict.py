"""The K1 verdict of design P7.

    Pass if the maximum over the gate rows (i), (ii) and (iii) of the AUROC of
    ResNet-18 and of ViT-S/16 on REMOVE versus CONTROL_OBJ is at most 0.6, on
    both hole types.  REMOVE versus CONTROL_BG is reported on the same rows.  A
    DINOv2-B AUROC above 0.7 on any gate row is reported in the first table of
    every downstream use.  The sensitivity requirement of check 1a must hold at
    the frozen gate resolution, and a PASS is always reported together with its
    local floor per hole-area bin from the local ladder.

A PASS is therefore never a bare number: it is conditional on the ladders, and
:meth:`K1Verdict.headline` refuses to print one without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from idea91.gate.inputs import GATE_ROWS
from idea91.gate.ladders import GlobalLadderVerdict, NullVerdict, local_floor_line

GATE_THRESHOLD = 0.60  # P7
DINOV2_REPORT_THRESHOLD = 0.70  # P7
HOLE_TYPES = ("mask", "rect")

GATE_ROW_NAMES = tuple(r.name for r in GATE_ROWS)


@dataclass
class K1Verdict:
    """Everything P7 needs, and nothing it does not."""

    #: (hole_type, gate_row_name) -> AUROC on REMOVE vs CONTROL_OBJ
    auroc_remove_vs_control_obj: dict[tuple[str, str], float]
    #: same keys, REMOVE vs CONTROL_BG -- reported, never gating
    auroc_remove_vs_control_bg: dict[tuple[str, str], float] = field(default_factory=dict)
    #: (hole_type, row) -> DINOv2-B AUROC, reported
    auroc_dinov2: dict[tuple[str, str], float] = field(default_factory=dict)
    global_ladder: GlobalLadderVerdict | None = None
    #: hole-area bin -> the smallest detected local step, or None
    local_floors: dict[str, int | None] = field(default_factory=dict)
    null_1b: NullVerdict | None = None
    #: check 1c is an assert, not a measurement: True once it has been run
    null_1c_asserted: bool = False
    gate_resolution_note: str = ""
    freeze_version: str = "unfrozen"

    # --- the P7 maximum ------------------------------------------------------

    def max_auroc(self, hole_type: str) -> float:
        vals = [
            v
            for (ht, row), v in self.auroc_remove_vs_control_obj.items()
            if ht == hole_type and row in GATE_ROW_NAMES and np.isfinite(v)
        ]
        return max(vals) if vals else float("nan")

    def argmax_row(self, hole_type: str) -> str:
        best, name = -np.inf, ""
        for (ht, row), v in self.auroc_remove_vs_control_obj.items():
            if ht == hole_type and row in GATE_ROW_NAMES and np.isfinite(v) and v > best:
                best, name = v, row
        return name

    @property
    def missing_rows(self) -> list[tuple[str, str]]:
        return [
            (ht, row)
            for ht in HOLE_TYPES
            for row in GATE_ROW_NAMES
            if (ht, row) not in self.auroc_remove_vs_control_obj
        ]

    # --- the verdict ---------------------------------------------------------

    @property
    def gate_passes(self) -> bool:
        """The P7 criterion alone; :attr:`decisive` adds the validity conditions."""
        if self.missing_rows:
            return False
        return all(self.max_auroc(ht) <= GATE_THRESHOLD for ht in HOLE_TYPES)

    @property
    def sensitivity_holds(self) -> bool:
        return bool(self.global_ladder and self.global_ladder.passes)

    @property
    def nulls_hold(self) -> bool:
        return bool(self.null_1b and self.null_1b.passes) and self.null_1c_asserted

    @property
    def decisive(self) -> bool:
        """A PASS only means something when the ladders and nulls hold."""
        return self.gate_passes and self.sensitivity_holds and self.nulls_hold

    @property
    def dinov2_flag(self) -> dict[tuple[str, str], float]:
        return {
            k: v
            for k, v in self.auroc_dinov2.items()
            if np.isfinite(v) and v > DINOV2_REPORT_THRESHOLD
        }

    def headline(self) -> str:
        if self.missing_rows:
            return f"K1: INCOMPLETE -- gate rows not run: {sorted(self.missing_rows)}"
        per_hole = ", ".join(
            f"{ht}: max AUROC {self.max_auroc(ht):.3f} ({self.argmax_row(ht)})"
            for ht in HOLE_TYPES
        )
        if not self.gate_passes:
            return (
                f"K1: FAIL -- {per_hole}; above the P7 threshold of {GATE_THRESHOLD:.2f}. "
                "Change the editor or the control design; no downstream consumer may "
                "touch the bank (E91-1 stop rule)."
            )
        if not self.sensitivity_holds:
            return (
                f"K1: NOT DECISIVE -- {per_hole}, at or below {GATE_THRESHOLD:.2f}, but the "
                "check 1a sensitivity requirement does not hold at this gate resolution, so "
                "the low AUROC may be blindness rather than a clean editor."
            )
        if not self.nulls_hold:
            return (
                f"K1: NOT DECISIVE -- {per_hole}, but the check 1b/1c nulls have not both "
                "held; the AUROC does not isolate the class difference."
            )
        return f"K1: PASS -- {per_hole}; at or below the P7 threshold of {GATE_THRESHOLD:.2f}."

    def conditions(self) -> list[str]:
        """The lines a PASS is always reported with (P7, check 1a)."""
        out: list[str] = []
        if self.global_ladder:
            out.append(self.global_ladder.verdict_line())
        if self.null_1b:
            out.append(self.null_1b.verdict_line())
        out.append(
            "check 1c paired-crop null: "
            + ("asserted (compositing is in-mask)" if self.null_1c_asserted else "NOT RUN")
        )
        for bin_name, floor in self.local_floors.items():
            out.append(local_floor_line(bin_name, floor))
        flagged = self.dinov2_flag
        if flagged:
            worst = max(flagged.items(), key=lambda kv: kv[1])
            out.append(
                f"DINOv2-B adversary above {DINOV2_REPORT_THRESHOLD:.2f} on "
                f"{worst[0][1]} ({worst[0][0]} holes): AUROC {worst[1]:.3f} -- carry this line "
                "into the first table of every downstream use (P7)"
            )
        if self.gate_resolution_note:
            out.append(self.gate_resolution_note)
        return out

    def report(self) -> str:
        return "\n".join([self.headline(), *(f"  - {c}" for c in self.conditions())])
