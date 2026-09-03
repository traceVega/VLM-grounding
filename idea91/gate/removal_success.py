"""P10's K1 removal success, and the gate AUROC on verified-removed pairs.

    K1 removal success: the same class-label question on the window for both K1
    classes (REMOVE and CONTROL_OBJ) where a class label exists; removal success
    and the gate AUROC on verified-removed pairs are reported per
    ``control_source``, and controls with ``control_source`` equal to
    class_agnostic, which have no label to ask about, are marked unverified
    (V2 only) and never mixed into the verified-pairs row.

Two numbers come out of this pass, and they answer different questions.

*Removal success* asks whether the editor did its job: after removing the thing,
does the verifier still see one?  It is a property of the edit bank, reported per
class and per ``control_source`` so a source whose controls are poorly removed is
visible rather than averaged away.

*The gate AUROC on verified-removed pairs* asks whether the gate's headline
survives restriction to the pairs where both sides actually worked.  A gate that
passes only because half its controls still contain their object has not been
tested on the contrast P7 describes.  Restricting to pairs where the REMOVE
verified clean *and* its matched CONTROL_OBJ verified clean is the honest read,
and it is why this is computed per ``control_source``: class_agnostic controls
have no label to ask about, so they cannot enter the row at all.

The verifier is served separately (``shared/judges/serve.sh``), so this pass is
CPU-cheap and network-bound; every answer is cached by (model, prompt, image,
question), so a re-run costs nothing and a partial run resumes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from idea91.gate.inputs import full_at_longer_side
from idea91.relations.verifier_rules import (
    WINDOW_DISPLAY_MAX_PX,
    parse_yes_no,
    removal_success_verifiable,
)
from shared.stats import Interval, auroc, rate_ci

PROMPT_FILE = "verifier_classlabel.txt"

#: The two K1 classes P10 asks the class-label question of.
CLASSES = ("REMOVE", "CONTROL_OBJ")


def png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def window_view(edited: np.ndarray, window_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """The edit window, shown at up to 1,536 px on the longer side (P10)."""
    x0, y0, x1, y1 = (int(v) for v in window_xyxy)
    crop = edited[y0:y1, x0:x1]
    return full_at_longer_side(crop, WINDOW_DISPLAY_MAX_PX)


@dataclass
class VerifierRow:
    """One class-label question and its answer."""

    sample_id: str
    image_id: str
    operator: str  # REMOVE | CONTROL_OBJ
    hole_type: str
    control_source: str | None
    class_label: str
    answer: str | None
    #: P10: removal succeeded when the verifier no longer sees the class
    removed: bool | None = None

    def __post_init__(self) -> None:
        yes = parse_yes_no(self.answer)
        # Unparseable counts as not clean (P10), so an unreadable answer is a
        # failed removal rather than a missing one -- it must not be dropped,
        # which would bias the rate upward.
        self.removed = False if yes is None else not yes


@dataclass
class RemovalSuccess:
    rows: list[VerifierRow] = field(default_factory=list)

    def by_class_and_source(self) -> dict[tuple[str, str], Interval]:
        """P10's table: removal success per class and per ``control_source``."""
        buckets: dict[tuple[str, str], list[bool]] = {}
        for row in self.rows:
            source = row.control_source or ("-" if row.operator == "REMOVE" else "unknown")
            buckets.setdefault((row.operator, source), []).append(bool(row.removed))
        return {key: rate_ci(np.array(values, dtype=bool)) for key, values in buckets.items()}

    def verified(self) -> dict[str, set[str]]:
        """``operator -> the sample_ids whose removal the verifier confirmed``."""
        out: dict[str, set[str]] = {}
        for row in self.rows:
            if row.removed:
                out.setdefault(row.operator, set()).add(row.sample_id)
        return out


def pair_key(image_id: str, hole_type: str) -> str:
    """A REMOVE and its matched CONTROL_OBJ share an image and a hole type."""
    return f"{image_id}::{hole_type}"


def auroc_on_verified_pairs(
    scores: dict[str, float],
    rows: list[VerifierRow],
) -> dict[str, float]:
    """Gate AUROC restricted to pairs where both sides verified removed.

    ``scores`` maps ``sample_id`` to the gate classifier's score.  The result is
    keyed by the control's ``control_source``: P10 forbids mixing class_agnostic
    controls in, because they were never asked the question.
    """
    by_pair: dict[str, dict[str, VerifierRow]] = {}
    for row in rows:
        by_pair.setdefault(pair_key(row.image_id, row.hole_type), {})[row.operator] = row

    per_source: dict[str, tuple[list[float], list[int]]] = {}
    for members in by_pair.values():
        remove = members.get("REMOVE")
        control = members.get("CONTROL_OBJ")
        if remove is None or control is None:
            continue
        if not (remove.removed and control.removed):
            continue  # the pair is not a verified-removed pair
        source = control.control_source or "unknown"
        if not removal_success_verifiable(source):
            continue  # class_agnostic: never mixed into the verified-pairs row
        if remove.sample_id not in scores or control.sample_id not in scores:
            continue
        values, labels = per_source.setdefault(source, ([], []))
        values.append(scores[remove.sample_id])
        labels.append(1)
        values.append(scores[control.sample_id])
        labels.append(0)

    return {
        source: auroc(np.array(labels, dtype=int), np.array(values, dtype=float))
        for source, (values, labels) in per_source.items()
        if len(set(labels)) == 2
    }


def base_operator(operator: str) -> str | None:
    """``RECT_CONTROL_OBJ`` -> ``CONTROL_OBJ``; ``None`` for operators P10 skips.

    Exact comparison after stripping the prefix, not ``endswith``: ``CONTROL_OBJ_2``
    is the check-1b twin, not a class P10 asks about.
    """
    name = operator[len("RECT_") :] if operator.startswith("RECT_") else operator
    return name if name in CLASSES else None


def instance_labels(set_name: str) -> dict[str, str]:
    """``instance_id -> class label``, from the instance store.

    The edit index carries instance ids, not labels, so the question P10 asks --
    "is there a {class_label} in this image" -- needs this join.  A REMOVE asks
    about its referent, a CONTROL_OBJ about the instance that was removed in its
    place, which is why both ``instance_id`` and ``control_instance_id`` are
    looked up here rather than a single column being assumed.
    """
    import pyarrow.parquet as pq

    from shared import paths

    directory = paths.PREPARED / set_name / "instances"
    out: dict[str, str] = {}
    for shard in sorted(Path(directory).glob("shard-*.parquet")):
        table = pq.read_table(shard, columns=["instance_id", "label"])
        for instance_id, label in zip(
            table.column("instance_id").to_pylist(), table.column("label").to_pylist()
        ):
            if label:
                out.setdefault(instance_id, label)
    return out


def run(
    index,
    image_dir: Path,
    edits_root: Path,
    *,
    client=None,
    labels: dict[str, str] | None = None,
    set_name: str = "openimages_pool",
    limit: int | None = None,
    hole_types: tuple[str, ...] = ("mask", "rect"),
    progress: bool = True,
) -> RemovalSuccess:
    """Ask the class-label question of every edit that carries a class label.

    ``client`` defaults to the pinned edit verifier of P10.  Edits whose class
    label is missing are skipped and counted, not guessed at: P10 scopes the
    question to "where a class label exists", and a class_agnostic control has
    no label by construction.
    """
    from idea91.edits.build import load_edited, read_image
    from shared.harness import prompts

    if client is None:
        from shared.judges.client import JudgeClient

        client = JudgeClient(role="edit_verifier")

    prompt = prompts.load(PROMPT_FILE)
    if labels is None:
        labels = instance_labels(set_name)
    columns = {n: index.column(n).to_pylist() for n in index.column_names}

    out = RemovalSuccess()
    skipped_no_label = 0
    n = index.num_rows if limit is None else min(limit, index.num_rows)
    for i in range(n):
        operator = base_operator(columns["operator"][i])
        if operator is None or columns["hole_type"][i] not in hole_types:
            continue
        instance_id = (
            columns["instance_id"][i]
            if operator == "REMOVE"
            else columns["control_instance_id"][i]
        )
        label = labels.get(instance_id) if instance_id else None
        if not label:
            skipped_no_label += 1
            continue

        image_id = columns["image_id"][i]
        original = read_image(image_dir / f"{image_id}.jpg")
        edited = load_edited(
            original,
            {
                "window_xyxy_px": columns["window_xyxy_px"][i],
                "window_path": columns["window_path"][i],
            },
            edits_root,
        )
        view = window_view(edited, columns["window_xyxy_px"][i])
        answer = client.ask(
            prompt.render(class_label=label),
            png_bytes(view),
            prompt_version=prompt.version,
            question_key=f"k1_removal_success::{columns['window_sha256'][i]}",
        )
        out.rows.append(
            VerifierRow(
                sample_id=columns["window_sha256"][i],
                image_id=image_id,
                operator=operator,
                hole_type=columns["hole_type"][i],
                control_source=columns["control_source"][i],
                class_label=str(label),
                answer=answer.text,
            )
        )
        if progress and len(out.rows) % 200 == 0:
            print(f"  [{len(out.rows)}] verified ({client.cache_hits} cached)")

    if skipped_no_label:
        print(f"  {skipped_no_label} edits skipped: no class label to ask about (P10)")
    return out
