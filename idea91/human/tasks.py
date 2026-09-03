"""Build the P11 annotation draw and render what annotators see.

P11's frame, in full:

    after the ORIGINAL condition has been scored (a pre-specified, data-dependent
    rule), 200 REMOVE items drawn with seed 0, stratified by size bin, from
    determined items (P10) whose ORIGINAL box is correct for at least one of the
    two models, plus 100 items drawn from determined items that V1 calls not
    clean [...] shown the edit window at 2 times the hole and the full image,
    with the head noun and the expression, never the original; labels clean,
    remnant (traces of the object remain), failed (object still present)

Two things here are load-bearing and are enforced in code rather than left to
the interface:

* **Never the original.**  Only the *edited* image is rendered, at two scales.
  An annotator who saw the before shot would be scoring their memory, not the
  edit.
* **Blind to condition.**  The task file carries no operator, no condition and
  no verifier answer.  Those live in a separate key file that the app never
  reads, so a leak would have to be deliberate.

Item order is shuffled per annotator, so position carries no signal either.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STRATUM = "random"
V1_NEGATIVE_STRATUM = "v1_negative"
ESCALATION_STRATUM = "escalation"
PILOT_STRATUM = "pilot"

N_RANDOM = 200  # P11
N_V1_NEGATIVE = 100  # P11
N_ESCALATION = 500  # P11, if acceptance check 3 fails

LABELS = ("clean", "remnant", "failed")
WINDOW_ZOOM = 2.0  # P11: "the edit window at 2 times the hole"


@dataclass
class Task:
    """One item as the annotator sees it.  Deliberately free of condition."""

    task_id: str
    window_sha256: str
    head_noun: str
    expr: str
    window_image: str  # path relative to the task directory
    full_image: str
    stratum: str

    def to_json(self) -> dict:
        return asdict(self)


def stratified_draw(
    frame: pd.DataFrame,
    n: int,
    *,
    seed: int = 0,
    stratify_by: str = "size_bin",
) -> pd.DataFrame:
    """P11's seeded, size-stratified draw.

    Proportional allocation, with the remainder given to the largest strata so
    the total is exactly ``n`` whenever the frame is big enough.
    """
    if frame.empty or n <= 0:
        return frame.iloc[:0]
    if len(frame) <= n:
        return frame.copy()
    if stratify_by not in frame.columns:
        return frame.sample(n=n, random_state=seed)

    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in frame.groupby(stratify_by, dropna=False)}
    sizes = {k: len(g) for k, g in groups.items()}
    total = sum(sizes.values())
    quota = {k: int(np.floor(n * s / total)) for k, s in sizes.items()}
    # hand out the remainder to the biggest strata first, capped by availability
    remainder = n - sum(quota.values())
    for k in sorted(sizes, key=lambda k: -sizes[k]):
        if remainder <= 0:
            break
        if quota[k] < sizes[k]:
            quota[k] += 1
            remainder -= 1

    picked = []
    for k, g in groups.items():
        take = min(quota.get(k, 0), len(g))
        if take:
            idx = rng.permutation(len(g))[:take]
            picked.append(g.iloc[sorted(idx)])
    out = pd.concat(picked) if picked else frame.iloc[:0]
    return out.sort_values("window_sha256") if "window_sha256" in out else out


def build_frame(
    relations: pd.DataFrame,
    verifier_clean: pd.Series,
    determined: pd.Series,
) -> pd.DataFrame:
    """P11's eligible frame: determined items whose ORIGINAL box is correct.

    "correct for at least one of the two models" -- so the frame is built after
    ORIGINAL has been scored, which is why P11 calls it a data-dependent rule and
    pre-specifies it anyway.
    """
    frame = relations.copy()
    frame["determined"] = determined
    frame["verifier_clean"] = verifier_clean
    correct = frame.groupby("pair_id")["orig_correct"].transform("max").astype(bool)
    return frame[frame["determined"] & correct].drop_duplicates("pair_id")


def draw_p11(
    frame: pd.DataFrame,
    *,
    seed: int = 0,
    n_random: int = N_RANDOM,
    n_v1_negative: int = N_V1_NEGATIVE,
) -> pd.DataFrame:
    """The 200 + 100 of P11, tagged by stratum."""
    random_part = stratified_draw(frame, n_random, seed=seed)
    remaining = frame[~frame.index.isin(random_part.index)]
    negatives = remaining[remaining["verifier_clean"] == False]  # noqa: E712
    negative_part = stratified_draw(negatives, n_v1_negative, seed=seed)
    return pd.concat(
        [random_part.assign(sample_stratum=RANDOM_STRATUM),
         negative_part.assign(sample_stratum=V1_NEGATIVE_STRATUM)]
    ).reset_index(drop=True)


# --- rendering ---------------------------------------------------------------


def render_task_images(
    edited_full: np.ndarray,
    hole: np.ndarray,
    out_dir: Path,
    task_id: str,
    *,
    zoom: float = WINDOW_ZOOM,
    max_side: int = 900,
) -> tuple[str, str]:
    """Write the two views P11 specifies.  Both are of the *edited* image.

    The window view is the hole's box grown to ``zoom`` times its longer side.
    Neither view is annotated: a drawn outline would tell the annotator where to
    look, and "is the object still there" is exactly the question they are being
    asked to answer unaided.
    """
    import cv2

    from idea91.masks import bbox_xyxy

    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = edited_full.shape[:2]

    x0, y0, x1, y1 = bbox_xyxy(hole)
    side = max(x1 - x0, y1 - y0) * zoom
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(side, 64)
    a = int(max(0, min(cx - side / 2, width - side)))
    b = int(max(0, min(cy - side / 2, height - side)))
    crop = edited_full[b : int(min(height, b + side)), a : int(min(width, a + side))]

    def save(image: np.ndarray, name: str) -> str:
        scale = min(1.0, max_side / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imwrite(str(out_dir / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        return name

    return save(crop, f"{task_id}_window.jpg"), save(edited_full, f"{task_id}_full.jpg")


def write_tasks(tasks: list[Task], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tasks.json"
    path.write_text(json.dumps([t.to_json() for t in tasks], indent=2), encoding="utf-8")
    return path


def read_tasks(out_dir: Path) -> list[Task]:
    path = Path(out_dir) / "tasks.json"
    return [Task(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


def write_key(rows: list[dict], out_dir: Path) -> Path:
    """The condition and verifier answer per task -- never served to the app.

    Kept beside the tasks so analysis can join, and named so that anyone reading
    the directory can see it is not for annotators.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "KEY_do_not_show_annotators.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def order_for(tasks: list[Task], annotator_id: str, seed: int = 0) -> list[Task]:
    """A per-annotator shuffle, so item position carries no signal."""
    rng = np.random.default_rng(abs(hash((annotator_id, seed))) % (2**32))
    return [tasks[i] for i in rng.permutation(len(tasks))]
