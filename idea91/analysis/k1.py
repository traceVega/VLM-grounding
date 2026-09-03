"""``tables/k1.md``: the K1 gate table and the P7 verdict (design 4.4, 4.10).

    tables/k1.md with rows per control type (CONTROL_OBJ gate rows first,
    CONTROL_BG comparison rows), per hole type, per classifier, per input row,
    per ``control_source``, the removal-success rate per class and per
    ``control_source`` with class-agnostic controls marked unverified, the gate
    AUROC on verified-removed pairs per ``control_source``, and the local floor
    per hole-area bin.

The verdict block comes first and always carries its conditions, so no reader
sees a PASS without the ladder that makes it mean something (P7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from idea91.gate.train import RowResult
from idea91.gate.verdict import K1Verdict
from shared.stats import Interval

CONTRAST_OBJ = "REMOVE_vs_CONTROL_OBJ"
CONTRAST_BG = "REMOVE_vs_CONTROL_BG"


@dataclass
class K1Tables:
    results: list[RowResult]
    verdict: K1Verdict
    #: ('REMOVE'|'CONTROL_OBJ', control_source) -> removal-success rate
    removal_success: dict[tuple[str, str], Interval] = field(default_factory=dict)
    #: control_source -> gate AUROC restricted to verified-removed pairs
    auroc_verified_pairs: dict[str, float] = field(default_factory=dict)
    #: editor -> (inside-hole, outside-hole) high-pass energy, design 4.4
    frequency_check: dict[str, tuple[float, float]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    run_ids: list[str] = field(default_factory=list)
    editor: str = ""
    editor_weights_sha256: str = ""

    # --- rendering -----------------------------------------------------------

    def _auroc_table(self, contrast: str) -> list[str]:
        rows = [r for r in self.results if r.contrast == contrast]
        if not rows:
            return ["_no rows_", ""]
        out = [
            "| gate row | classifier | hole type | gating | AUROC | 95% CI | images | per seed |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(rows, key=lambda r: (r.hole_type, r.row)):
            gating = "yes" if r.row.startswith(("i_", "ii_", "iii_")) else "no"
            seeds = ", ".join(f"{a:.3f}" for a in r.auroc_by_seed)
            out.append(
                f"| {r.row} | {r.classifier} | {r.hole_type} | {gating} | "
                f"{r.auroc_mean:.3f} | [{r.ci.lo:.3f}, {r.ci.hi:.3f}] | {r.n_images} | {seeds} |"
            )
        out.append("")
        return out

    def _by_source_table(self) -> list[str]:
        rows = [r for r in self.results if r.contrast == CONTRAST_OBJ and r.by_control_source]
        if not rows:
            return ["_no per-source rows_", ""]
        sources = sorted({s for r in rows for s in r.by_control_source})
        out = [
            "| gate row | hole type | " + " | ".join(sources) + " |",
            "|---" * (2 + len(sources)) + "|",
        ]
        for r in sorted(rows, key=lambda r: (r.hole_type, r.row)):
            cells = [
                f"{r.by_control_source[s]:.3f}" if s in r.by_control_source else "-"
                for s in sources
            ]
            out.append(f"| {r.row} | {r.hole_type} | " + " | ".join(cells) + " |")
        out.append("")
        return out

    def _removal_success_table(self) -> list[str]:
        if not self.removal_success:
            return ["_removal success not measured_", ""]
        out = [
            "| class | control_source | removal success | 95% CI | n | verifiable |",
            "|---|---|---|---|---|---|",
        ]
        from idea91.relations.verifier_rules import removal_success_verifiable

        for (klass, source), interval in sorted(self.removal_success.items()):
            verifiable = removal_success_verifiable(source)
            mark = "yes" if verifiable else "**no -- unverified (V2 only)**"
            out.append(
                f"| {klass} | {source} | {interval.point:.3f} | "
                f"[{interval.lo:.3f}, {interval.hi:.3f}] | {interval.n} | {mark} |"
            )
        out.append("")
        out.append(
            "Controls with `control_source = class_agnostic` have no label to ask about "
            "(P10) and are never mixed into the verified-pairs row."
        )
        out.append("")
        return out

    def render(self) -> str:
        v = self.verdict
        lines = [
            "# K1: artifact gate",
            "",
            f"Freeze version: `{v.freeze_version}` | editor: `{self.editor}` "
            f"(weights sha256 `{self.editor_weights_sha256[:16] or 'unpinned'}`) | "
            f"runs: {', '.join(self.run_ids) or 'none recorded'}",
            "",
            "## Verdict (P7)",
            "",
            f"**{v.headline()}**",
            "",
        ]
        lines += [f"- {c}" for c in v.conditions()]
        lines += [
            "",
            "## Gate rows: REMOVE versus CONTROL_OBJ",
            "",
            "The P7 maximum is taken over the gating rows only; the reported rows and the "
            "adversaries are shown for reading, not for the verdict.",
            "",
        ]
        lines += self._auroc_table(CONTRAST_OBJ)
        lines += ["## Comparison rows: REMOVE versus CONTROL_BG", ""]
        lines += self._auroc_table(CONTRAST_BG)
        lines += ["## Gate AUROC by `control_source` (CONTROL_OBJ rows)", ""]
        lines += self._by_source_table()
        lines += ["## Removal success (P10)", ""]
        lines += self._removal_success_table()

        if self.auroc_verified_pairs:
            lines += ["## Gate AUROC on verified-removed pairs", "", "| control_source | AUROC |", "|---|---|"]
            for source, value in sorted(self.auroc_verified_pairs.items()):
                lines.append(f"| {source} | {value:.3f} |")
            lines.append("")

        if self.frequency_check:
            lines += [
                "## Frequency-domain check (design 4.4, once per editor)",
                "",
                "| editor | high-pass energy inside the hole | outside | ratio |",
                "|---|---|---|---|",
            ]
            for editor, (inside, outside) in sorted(self.frequency_check.items()):
                ratio = inside / outside if outside else float("nan")
                lines.append(f"| {editor} | {inside:.1f} | {outside:.1f} | {ratio:.3f} |")
            lines.append("")

        if self.counts:
            lines += ["## Counts", "", "| quantity | n |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in sorted(self.counts.items())]
            lines.append("")
        return "\n".join(lines)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        return path


def verdict_from_results(
    results: list[RowResult],
    *,
    global_ladder=None,
    null_1b=None,
    local_floors: dict[str, int | None] | None = None,
    null_1c_asserted: bool = False,
    freeze_version: str = "unfrozen",
    gate_resolution_note: str = "",
) -> K1Verdict:
    """Assemble the P7 verdict from the trained rows."""
    obj = {
        (r.hole_type, r.row): r.auroc_mean
        for r in results
        if r.contrast == CONTRAST_OBJ and r.classifier in ("resnet18", "vit_s16")
    }
    bg = {
        (r.hole_type, r.row): r.auroc_mean for r in results if r.contrast == CONTRAST_BG
    }
    dino = {
        (r.hole_type, r.row): r.auroc_mean
        for r in results
        if r.classifier == "dinov2_b" and np.isfinite(r.auroc_mean)
    }
    return K1Verdict(
        auroc_remove_vs_control_obj=obj,
        auroc_remove_vs_control_bg=bg,
        auroc_dinov2=dino,
        global_ladder=global_ladder,
        local_floors=local_floors or {},
        null_1b=null_1b,
        null_1c_asserted=null_1c_asserted,
        freeze_version=freeze_version,
        gate_resolution_note=gate_resolution_note,
    )
