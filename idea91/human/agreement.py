"""P11's agreement statistics, and acceptance check 3's verdict.

    human-human Fleiss kappa (3-way and binarised) on the random 200 and on all
    300; human-versus-V1 Cohen's kappa on the random 200 (check 3)

Check 3 is the one that decides which column carries the K2 headline: at kappa
0.6 or above the verifier-clean column is decisive, below it the human-clean
column is, and the check escalates to 500 items.  That switch lives in
:func:`idea91.analysis.k2.decisive_clean_column`; this module supplies the number
it reads.

The binarised view maps ``clean`` against ``remnant`` or ``failed``, because that
is the distinction the verifier makes and the only one the two can be compared
on.  ``skip`` is not a label: unsure items leave every kappa and are reported as
their own count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from idea91.human.tasks import LABELS, RANDOM_STRATUM
from shared.stats import cohens_kappa, fleiss_kappa

KAPPA_MIN = 0.60  # acceptance check 3
SKIP = "skip"


def majority_label(labels: list[str]) -> str | None:
    """P11's majority rule.  ``None`` when no label has a strict majority."""
    usable = [x for x in labels if x in LABELS]
    if not usable:
        return None
    counts = {x: usable.count(x) for x in set(usable)}
    best = max(counts.values())
    winners = [x for x, c in counts.items() if c == best]
    if len(winners) != 1 or best * 2 <= len(usable):
        return None
    return winners[0]


def binarise(label: str | None) -> bool | None:
    """clean versus not clean -- the only axis the verifier also scores."""
    if label is None or label == SKIP:
        return None
    return label == "clean"


def label_matrix(labels: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """items x annotators, from the long-form CSV."""
    wide = labels.pivot_table(
        index="window_sha256", columns="annotator_id", values="label", aggfunc="first"
    )
    return wide, list(wide.columns)


def counts_for_fleiss(wide: pd.DataFrame, categories: tuple[str, ...]) -> np.ndarray:
    """items x categories, dropping items that are not fully rated."""
    rows = []
    for _, row in wide.iterrows():
        values = [v for v in row.tolist() if isinstance(v, str) and v in categories]
        if len(values) != len(row):
            continue  # Fleiss needs the same number of raters on every item
        rows.append([values.count(c) for c in categories])
    return np.array(rows, dtype=float) if rows else np.zeros((0, len(categories)))


@dataclass
class AgreementReport:
    n_items: int
    n_annotators: int
    fleiss_3way: float
    fleiss_binary: float
    fleiss_3way_random_only: float
    cohen_human_vs_v1: float
    n_kappa_items: int
    majority: pd.Series = field(default_factory=pd.Series)
    skipped: int = 0
    no_majority: int = 0
    fleiss_by_stratum: dict = field(default_factory=dict)
    strata_present: tuple[str, ...] = ()

    @property
    def check_3_passes(self) -> bool:
        return np.isfinite(self.cohen_human_vs_v1) and self.cohen_human_vs_v1 >= KAPPA_MIN

    @property
    def check_3_applies(self) -> bool:
        """Check 3 is defined on P11's random 200, not on a pilot or an escalation."""
        return RANDOM_STRATUM in self.strata_present

    def verdict_line(self) -> str:
        if not self.check_3_applies:
            return (
                "acceptance check 3: not applicable -- it is defined on P11's random stratum "
                f"and this set has none (strata present: {', '.join(self.strata_present) or 'none'}). "
                "Human-human agreement above is what B9's pilot reports."
            )
        if not np.isfinite(self.cohen_human_vs_v1):
            return (
                "acceptance check 3: NOT MEASURABLE -- random-stratum items exist, but none "
                "carries both a human majority and a verifier answer"
            )
        if self.check_3_passes:
            return (
                f"acceptance check 3: PASS (human-vs-V1 kappa {self.cohen_human_vs_v1:.3f} "
                f">= {KAPPA_MIN:.2f} on {self.n_kappa_items} items); verifier-clean is the "
                "decisive column"
            )
        return (
            f"acceptance check 3: FAIL (human-vs-V1 kappa {self.cohen_human_vs_v1:.3f} "
            f"< {KAPPA_MIN:.2f} on {self.n_kappa_items} items); human-clean carries the "
            "headline and the check escalates to 500 items (P11)"
        )

    def report(self) -> str:
        lines = [
            f"P11 agreement: {self.n_items} items, {self.n_annotators} annotators",
            f"  human-human Fleiss kappa, 3-way:      {self.fleiss_3way:.3f}",
            f"  human-human Fleiss kappa, binarised:  {self.fleiss_binary:.3f}",
        ]
        for stratum, value in sorted(self.fleiss_by_stratum.items()):
            lines.append(f"  human-human Fleiss, {stratum:<16s} {value:.3f}")
        lines += [
            f"  unsure labels: {self.skipped}   items without a majority: {self.no_majority}",
            self.verdict_line(),
        ]
        return "\n".join(lines)


def analyse(
    labels: pd.DataFrame,
    verifier_clean: pd.Series | None = None,
    strata: pd.Series | None = None,
) -> AgreementReport:
    """Every statistic P11 asks for, from the long-form label CSV.

    ``verifier_clean`` is indexed by ``window_sha256`` and is V1's boolean; it is
    what check 3 compares the human majority against.
    """
    wide, annotators = label_matrix(labels)
    three = counts_for_fleiss(wide, LABELS)

    binary_wide = wide.map(lambda v: "clean" if v == "clean" else ("not_clean" if v in LABELS else v))
    binary = counts_for_fleiss(binary_wide, ("clean", "not_clean"))

    majority = wide.apply(lambda row: majority_label(row.tolist()), axis=1)

    random_only = np.zeros((0, len(LABELS)))
    by_stratum: dict[str, float] = {}
    present: tuple[str, ...] = ()
    if strata is not None and len(wide):
        aligned = strata.reindex(wide.index)
        present = tuple(sorted({s for s in aligned.dropna().unique()}))
        for stratum in present:
            subset = counts_for_fleiss(wide[(aligned == stratum).fillna(False)], LABELS)
            if len(subset):
                by_stratum[stratum] = fleiss_kappa(subset)
        keep = aligned == RANDOM_STRATUM
        if keep.any():
            random_only = counts_for_fleiss(wide[keep.fillna(False)], LABELS)

    cohen, n_kappa = float("nan"), 0
    if verifier_clean is not None and len(majority):
        human = majority.map(binarise)
        frame = pd.DataFrame({"human": human, "v1": verifier_clean.reindex(majority.index)})
        if strata is not None:
            frame = frame[strata.reindex(frame.index) == RANDOM_STRATUM]  # check 3: the random 200
        frame = frame.dropna()
        if len(frame):
            cohen = cohens_kappa(frame["human"].to_numpy(), frame["v1"].to_numpy())
            n_kappa = len(frame)

    return AgreementReport(
        n_items=len(wide),
        n_annotators=len(annotators),
        fleiss_3way=fleiss_kappa(three) if len(three) else float("nan"),
        fleiss_binary=fleiss_kappa(binary) if len(binary) else float("nan"),
        fleiss_3way_random_only=fleiss_kappa(random_only) if len(random_only) else float("nan"),
        cohen_human_vs_v1=cohen,
        n_kappa_items=n_kappa,
        majority=majority,
        skipped=int((labels["label"] == SKIP).sum()),
        no_majority=int(majority.isna().sum()),
        fleiss_by_stratum=by_stratum,
        strata_present=present,
    )
