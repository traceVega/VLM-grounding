"""The two-stage freeze of P20, made self-verifying.

    Two freeze points.  K1 freeze: P1 to P7 are frozen after the pool, instances,
    edits and count tables exist and before the first classifier trains on real
    removals (the JPEG ladder and the nulls of check 1 may run before).  K2
    freeze: P8 to P21 are frozen after the K2 instances, edits and count tables
    exist and before the first ORIGINAL item is scored.  Each manifest carries
    the version in force.  After a freeze no value changes except through the
    declared contingencies; any other change forks the document and both
    versions' results are reported.

A freeze written as prose is a promise.  This module makes it a check: the
registry below names every constant the pre-registration binds, :func:`collect`
reads them out of the live code, and the record stores their digest.  After the
tag, :func:`verify` re-reads the same constants and reports any that moved, and
:func:`require` refuses to let a stage run against a drifted freeze.  So a
constant edited after B0a cannot quietly reach the gate rows -- the run stops
and names the value, the frozen number and the current one.

The registry is deliberately explicit rather than "every uppercase name in the
package": a freeze must say what it covers, and a wildcard would silently take
in whatever a later edit happened to add.

Two kinds of value are frozen.  Code constants come from the registry.
Data-dependent values -- the gate resolution check 1a settles, the digest of the
count table -- have no home in the source and are passed to :func:`create` by
the stage that establishes them; they are frozen by being recorded, and drift in
them is caught by the count-table digest rather than by re-reading.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

from shared import paths

B0A = "B0a"
B0B = "B0b"
FREEZE_POINTS = (B0A, B0B)

#: The declared contingencies of P20 -- the only post-freeze changes that do not
#: fork the document.  A drifted value matching one of these is reported as a
#: contingency rather than a violation, and still has to be acknowledged.
CONTINGENCIES = {
    B0A: ("the check 1a resolution raise before the K1 freeze",),
    B0B: ("O1: OpenRef unavailable, K2 becomes every eligible GroundingME item",),
}

#: ``(pre-registered definition, module, attribute)``, in the order they are
#: reported.  Every entry is a value P1 to P7 (B0a) or P8 to P21 (B0b) binds.
B0A_REGISTRY: tuple[tuple[str, str, str], ...] = (
    # P1: the pool and the referent choice
    ("P1", "idea91.instances.build", "CLASS_AGNOSTIC_MIN_AREA"),
    ("P1", "idea91.instances.automask", "MIN_AREA_FRAC"),
    ("P1", "idea91.instances.automask", "MAX_AREA_FRAC"),
    ("P1", "idea91.instances.automask", "POINTS_PER_SIDE"),
    ("P1", "idea91.instances.automask", "PRED_IOU_THRESH"),
    ("P1", "idea91.instances.automask", "STABILITY_SCORE_THRESH"),
    ("P1", "idea91.instances.automask", "NMS_IOU_THRESH"),
    # P2: the window operator and the hole
    ("P2", "idea91.edits.window", "WINDOW_FACTOR"),
    ("P2", "idea91.edits.window", "WINDOW_MIN_SIDE"),
    ("P2", "idea91.edits.window", "WINDOW_MAX_SIDE"),
    ("P2", "idea91.edits.window", "LAMA_MAX_SIDE"),
    ("P2", "idea91.edits.window", "LAMA_PAD_MULTIPLE"),
    ("P2", "idea91.edits.sampler", "HOLE_DILATION_PX"),
    ("P2", "idea91.edits.sampler", "HOLE_TYPES"),
    # P3: the control sampler and its exclusion set
    ("P3", "idea91.edits.sampler", "EXCLUSION_DILATION_PX"),
    ("P3", "idea91.edits.sampler", "CONTROL_BITE_TOLERANCE"),
    ("P3", "idea91.edits.sampler", "CLASS_AGNOSTIC_MIN_AREA_FRAC"),
    ("P3", "idea91.edits.sampler", "AREA_RATIO_RANGE"),
    ("P3", "idea91.edits.sampler", "CENTRALITY_TOLERANCE"),
    ("P3", "idea91.edits.sampler", "MAX_TRIES"),
    ("P3", "idea91.edits.sampler", "CONTROL_SOURCES"),
    # P5: the gate rows and the resolutions they see
    ("P5", "idea91.gate.inputs", "MPX_CAP"),
    ("P5", "idea91.gate.inputs", "TILE_PX"),
    ("P5", "idea91.gate.inputs", "SHOWN_CROP_FACTOR"),
    ("P5", "idea91.gate.inputs", "SHOWN_CROP_PX"),
    ("P5", "idea91.gate.inputs", "GATE_ROWS"),
    ("P5", "idea91.gate.inputs", "REPORTED_ROWS"),
    ("P5", "idea91.gate.inputs", "ADVERSARY_ROWS"),
    # P6: the classifiers and their training
    ("P6", "idea91.gate.train", "EPOCHS"),
    ("P6", "idea91.gate.train", "LR"),
    ("P6", "idea91.gate.train", "BATCH_NATIVE"),
    ("P6", "idea91.gate.train", "SEEDS"),
    ("P6", "idea91.gate.train", "ADVERSARY_SEEDS"),
    # P7: the gate itself, and the ladders a PASS is conditional on
    ("P7", "idea91.gate.verdict", "GATE_THRESHOLD"),
    ("P7", "idea91.gate.verdict", "DINOV2_REPORT_THRESHOLD"),
    ("P7", "idea91.gate.ladders", "GLOBAL_LADDER_QUALITIES"),
    ("P7", "idea91.gate.ladders", "LOCAL_LADDER_QUALITIES"),
    ("P7", "idea91.gate.ladders", "GLOBAL_REQUIRED_Q75_AUROC"),
    ("P7", "idea91.gate.ladders", "GLOBAL_REQUIRED_RUNG_AUROC"),
    ("P7", "idea91.gate.ladders", "SENSITIVITY_RUNG"),
    ("P7", "idea91.gate.ladders", "LOCAL_DETECTION_AUROC"),
    ("P7", "idea91.gate.ladders", "HOLE_AREA_BIN_EDGES"),
    ("P7", "idea91.gate.ladders", "LADDER_IMAGES"),
)

REGISTRY: dict[str, tuple[tuple[str, str, str], ...]] = {
    B0A: B0A_REGISTRY,
    B0B: (),  # filled when P8 to P21 land; K2 is not yet buildable
}

#: Some pre-registered decisions are not a number but a behaviour, so no
#: constant can hold them.  The local ladder's reference class (Q-5) and the
#: tile sampler of gate row (ii) (Q-7) are both of that kind: each was settled
#: before the freeze and each sets a reported floor, so each is frozen by the
#: source text of the function that implements it.  Editing the function after
#: the freeze breaks it, exactly as editing a constant would.
#:
#: ``(pre-registered definition, module, qualified name, why it is frozen)``
B0A_BEHAVIOURS: tuple[tuple[str, str, str, str], ...] = (
    ("P7", "idea91.gate.ladders", "local_ladder_pair",
     "Q-5: the local ladder's reference class sets the local floor"),
    ("P5", "idea91.gate.dataset", "TileDataset",
     "Q-7: how row (ii) draws training tiles decides what it can detect at all"),
    ("P6", "idea91.gate.dataset", "ShapeBucketSampler",
     "batching by exact shape, so native-resolution rows keep their pixels"),
    ("P3", "idea91.edits.sampler", "exclusion_violation",
     "Q-12/Q-13: reading B is the exclusion rule both controls are sampled under"),
)


def _source_digest(module_name: str, qualified_name: str) -> str:
    """sha256 of a function or class's source text, whitespace-normalised.

    Normalising trailing whitespace keeps a reformat from reading as a change of
    substance, while any edit to the logic still breaks the digest.
    """
    import inspect

    module = import_module(module_name)
    obj = module
    for part in qualified_name.split("."):
        obj = getattr(obj, part)
    text = "\n".join(line.rstrip() for line in inspect.getsource(obj).splitlines()).strip()
    return hashlib.sha256(text.encode()).hexdigest()


def _canonical(value: object) -> object:
    """A JSON-safe, order-stable form -- tuples become lists, dataclasses dicts."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if hasattr(value, "__dataclass_fields__"):
        return _canonical(asdict(value))
    return repr(value)


def collect(point: str = B0A) -> dict[str, object]:
    """Read every registered constant out of the live code."""
    if point not in REGISTRY:
        raise ValueError(f"unknown freeze point {point!r}; expected {list(FREEZE_POINTS)}")
    out: dict[str, object] = {}
    for definition, module_name, attribute in REGISTRY[point]:
        module = import_module(module_name)
        if not hasattr(module, attribute):
            raise AttributeError(
                f"{point} freezes {module_name}.{attribute}, which no longer exists. "
                "Either the constant was renamed (update the registry in the same commit "
                "that renames it) or it was deleted (which forks the document under P20)."
            )
        out[f"{definition}:{module_name}.{attribute}"] = _canonical(getattr(module, attribute))
    if point == B0A:
        for definition, module_name, qualified_name, why in B0A_BEHAVIOURS:
            key = f"{definition}:{module_name}.{qualified_name}()"
            out[key] = {"source_sha256": _source_digest(module_name, qualified_name), "why": why}
    return out


def digest(values: dict[str, object]) -> str:
    blob = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=paths.REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def working_tree_clean() -> bool:
    return _git("status", "--porcelain") == ""


@dataclass
class FreezeRecord:
    """What a freeze point actually froze, and enough to prove it later."""

    point: str
    created_utc: str
    git_commit: str
    git_tag: str
    values: dict[str, object]
    values_sha256: str
    #: data-dependent values with no home in the source: the gate resolution
    #: check 1a settled, the digest of the count table committed with the freeze
    data_values: dict[str, object] = field(default_factory=dict)
    #: P8 and P20 say the count table is *committed* with the freeze.  Generated
    #: tables under ``tables/`` are gitignored, so the text lives here, in the
    #: one artifact that is committed -- the record is self-contained.
    count_table: str = ""
    count_table_sha256: str = ""
    signed_off_by: str = ""
    note: str = ""

    @property
    def version(self) -> str:
        """What goes in every manifest's ``freeze_version`` (P20)."""
        return f"{self.point}@{self.values_sha256[:12]}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def record_path(point: str = B0A) -> Path:
    return paths.REPO_ROOT / "freeze" / f"{point}.json"


def read(point: str = B0A) -> FreezeRecord | None:
    path = record_path(point)
    if not path.is_file():
        return None
    return FreezeRecord(**json.loads(path.read_text(encoding="utf-8")))


@dataclass
class Drift:
    key: str
    frozen: object
    current: object

    def line(self) -> str:
        return f"  {self.key}\n      frozen  {self.frozen!r}\n      current {self.current!r}"


def verify(point: str = B0A) -> tuple[FreezeRecord | None, list[Drift]]:
    """Re-read the registry and report what moved since the freeze."""
    record = read(point)
    if record is None:
        return None, []
    current = collect(point)
    drifts = [
        Drift(key, record.values.get(key), current.get(key))
        for key in sorted(set(record.values) | set(current))
        if record.values.get(key) != current.get(key)
    ]
    return record, drifts


def require(point: str = B0A, *, what: str = "this stage") -> FreezeRecord:
    """Refuse to run a post-freeze stage without an intact freeze.

    This is the guard that makes P20 mean something operationally: the gate rows
    are the first classifiers to see a real removal, so they may not run before
    the freeze exists, nor after one of its values has moved.
    """
    record, drifts = verify(point)
    if record is None:
        raise SystemExit(
            f"{what} runs after the {point} freeze (P20), but no freeze record exists at "
            f"{record_path(point)}.\n"
            f"Run:  python -m idea91.run_k1 freeze --sign-off '<your name>'"
        )
    if drifts:
        lines = "\n".join(d.line() for d in drifts)
        contingencies = "\n".join(f"  - {c}" for c in CONTINGENCIES.get(point, ()))
        raise SystemExit(
            f"{what} refuses to run: {len(drifts)} value(s) frozen at {point} have changed.\n"
            f"{lines}\n\n"
            "Under P20 the only permitted post-freeze changes are the declared "
            f"contingencies:\n{contingencies}\n"
            "Any other change forks the document and both versions' results are reported. "
            "Revert the value, or fork deliberately and re-freeze."
        )
    return record


def create(
    point: str = B0A,
    *,
    sign_off: str,
    data_values: dict[str, object] | None = None,
    count_table: str = "",
    note: str = "",
    tag: bool = True,
    allow_dirty: bool = False,
) -> FreezeRecord:
    """Write the freeze record and, unless told not to, tag the commit.

    ``sign_off`` is required and recorded: P20 binds a human decision, and an
    unattended run should not be able to freeze the pre-registration on its own.
    """
    if not sign_off.strip():
        raise SystemExit("a freeze needs a sign-off name; P20 binds a human decision")
    if not allow_dirty and not working_tree_clean():
        raise SystemExit(
            "the working tree is dirty; a freeze tag must point at committed code.\n"
            "Commit first, or pass --allow-dirty and accept that the tag is not reproducible."
        )
    existing = read(point)
    if existing is not None:
        raise SystemExit(
            f"{point} is already frozen ({existing.version}, {existing.created_utc}, "
            f"signed off by {existing.signed_off_by or 'nobody'}).\n"
            f"Re-freezing forks the document under P20; delete {record_path(point)} "
            "deliberately if that is what you mean."
        )

    values = collect(point)
    record = FreezeRecord(
        point=point,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=_git("rev-parse", "HEAD"),
        git_tag=f"freeze-{point}" if tag else "",
        values=values,
        values_sha256=digest(values),
        data_values=data_values or {},
        count_table=count_table,
        count_table_sha256=hashlib.sha256(count_table.encode()).hexdigest() if count_table else "",
        signed_off_by=sign_off.strip(),
        note=note,
    )
    path = record_path(point)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.to_json(), encoding="utf-8")
    if tag:
        _git("tag", "-a", record.git_tag, "-m", f"{point} freeze {record.version} ({sign_off})")
        # P20's freeze points *are* git tags, so a record naming one that does
        # not exist is a broken guarantee rather than a cosmetic slip.  This
        # happened on the real B0a: `git tag -a` needs a committer identity, WSL
        # had none, and `_git` swallowed the failure -- leaving a record that
        # claimed `freeze-B0a` against a repository with no tags at all.
        if record.git_tag not in _git("tag", "-l").splitlines():
            record.git_tag = ""
            path.write_text(record.to_json(), encoding="utf-8")
            print(
                f"WARNING: could not create the tag {point} claims. The record is written "
                f"and its git_tag is now empty rather than a false claim.\n"
                f"  Create it yourself against the recorded commit:\n"
                f"    git tag -a freeze-{point} {record.git_commit[:12]} -m '{point} freeze "
                f"{record.version}'\n"
                "  (a likely cause is no committer identity: git config user.name / user.email)"
            )
    return record


# --- the sign-off sheet ------------------------------------------------------


def sheet(point: str = B0A, *, data_values: dict[str, object] | None = None) -> str:
    """Every value the freeze would bind, for a human to read before signing.

    Grouped by pre-registered definition, because that is the unit P20 freezes
    and the unit a reader checks against the design.
    """
    values = collect(point)
    by_definition: dict[str, list[tuple[str, object]]] = {}
    for key, value in values.items():
        definition, name = key.split(":", 1)
        by_definition.setdefault(definition, []).append((name, value))

    lines = [
        f"# {point} freeze: the values to sign off",
        "",
        f"Digest `{digest(values)[:16]}` | commit `{_git('rev-parse', '--short', 'HEAD') or '?'}` "
        f"| working tree {'clean' if working_tree_clean() else '**dirty**'}",
        "",
        "P20 freezes these before the first classifier trains on a real removal. "
        "After the freeze no value changes except through the declared contingencies "
        f"({'; '.join(CONTINGENCIES.get(point, ())) or 'none'}); any other change forks the "
        "document and both versions' results are reported.",
        "",
    ]
    for definition in sorted(by_definition):
        lines += [f"## {definition}", "", "| value | frozen at |", "|---|---|"]
        for name, value in by_definition[definition]:
            rendered = json.dumps(value)
            if len(rendered) > 90:
                rendered = rendered[:87] + "..."
            lines.append(f"| `{name}` | `{rendered}` |")
        lines.append("")
    if data_values:
        lines += [
            "## Data-dependent (established by the run, not by the source)",
            "",
            "| value | frozen at |",
            "|---|---|",
        ]
        for name, value in sorted(data_values.items()):
            lines.append(f"| `{name}` | `{json.dumps(value)}` |")
        lines.append("")
    return "\n".join(lines)
