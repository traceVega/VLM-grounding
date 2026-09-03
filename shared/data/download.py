"""The download plan (design 4.2, B2).

Prints what would be fetched, where it lands, how big it is and under which
licence, and fetches nothing unless ``--yes`` is passed.  A source whose locator
or revision is still ``PIN_REQUIRED`` is refused rather than guessed at, and a
``manual`` source (a signed link, a form) prints its instructions instead.

    python -m shared.data.download                 # the plan, nothing fetched
    python -m shared.data.download --core          # only the non-optional rows
    python -m shared.data.download --only groundingme --yes
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from shared import paths
from shared.data.sources import HF, HTTP, MANUAL, PIN, SOURCES, Source


@dataclass
class Step:
    source: Source
    destination: Path
    command: list[str] | None
    blocker: str | None = None

    @property
    def ready(self) -> bool:
        return self.command is not None and self.blocker is None


def plan_for(source: Source, root: Path | None = None) -> Step:
    root = root or paths.RAW
    destination = root / source.key
    unpinned_note = (
        ""
        if source.pinned
        else f" Locator/revision is still {PIN}: pin it in shared/data/sources.py and "
        "shared/env/PINS.md."
    )
    if source.kind == MANUAL:
        # said first even when unpinned: "no script will fetch this" is the
        # actionable fact, and the pin follows from doing it by hand
        return Step(
            source,
            destination,
            None,
            "manual download (signed link or form); fetch by hand into the destination, "
            "then record the file sha256 in shared/env/PINS.md." + unpinned_note,
        )
    if not source.pinned:
        return Step(
            source,
            destination,
            None,
            f"locator/revision still {PIN}: read the source and pin it in "
            "shared/data/sources.py and shared/env/PINS.md before fetching",
        )
    if source.kind == HTTP:
        return Step(
            source,
            destination,
            ["curl", "-L", "--fail", "--output", str(destination / Path(source.locator).name),
             source.locator],
        )
    if source.kind == HF:
        return Step(
            source,
            destination,
            ["huggingface-cli", "download", source.locator, "--revision", source.revision,
             "--repo-type", "dataset", "--local-dir", str(destination)],
        )
    return Step(source, destination, None, f"unknown kind {source.kind!r}")


def build_plan(sources: list[Source], root: Path | None = None) -> list[Step]:
    return [plan_for(s, root) for s in sources]


def render(steps: list[Step]) -> str:
    lines = [
        f"Download plan -- data root {paths.DATA_ROOT}",
        "",
        f"{'source':22s} {'GB':>6s}  {'state':9s} destination",
        "-" * 100,
    ]
    ready = blocked = 0.0
    for step in steps:
        state = "ready" if step.ready else "blocked"
        ready += step.source.approx_gb if step.ready else 0
        blocked += 0 if step.ready else step.source.approx_gb
        lines.append(
            f"{step.source.key:22s} {step.source.approx_gb:6.1f}  {state:9s} {step.destination}"
        )
        lines.append(f"{'':22s} {'':6s}  licence: {step.source.licence}")
        if step.blocker:
            lines.append(f"{'':22s} {'':6s}  BLOCKED: {step.blocker}")
        elif step.command:
            lines.append(f"{'':22s} {'':6s}  $ {' '.join(step.command)}")
        lines.append("")
    lines += [
        f"fetchable now: {ready:.1f} GB;  blocked: {blocked:.1f} GB",
        "",
        "Nothing was fetched. Re-run with --yes to execute the ready steps.",
    ]
    return "\n".join(lines)


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path if path.exists() else path.anchor or "/")
    return usage.free / 1024**3


def execute(steps: list[Step]) -> int:
    failures = 0
    for step in steps:
        if not step.ready:
            print(f"skip {step.source.key}: {step.blocker}")
            continue
        step.destination.mkdir(parents=True, exist_ok=True)
        print(f"fetching {step.source.key} -> {step.destination}")
        result = subprocess.run(step.command)
        if result.returncode != 0:
            failures += 1
            print(f"  FAILED ({result.returncode})")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", action="store_true", help="skip the optional sources")
    ap.add_argument("--only", nargs="*", default=None, help="source keys to include")
    ap.add_argument("--yes", action="store_true", help="actually fetch the ready steps")
    args = ap.parse_args()

    chosen = [
        s
        for s in SOURCES
        if (args.only is None or s.key in args.only) and (not args.core or not s.optional)
    ]
    steps = build_plan(chosen)
    print(render(steps))
    needed = sum(s.source.approx_gb for s in steps if s.ready)
    available = free_gb(paths.DATA_ROOT.parent)
    print(f"free on the data volume: {available:.0f} GB")
    if needed > available:
        raise SystemExit(f"not enough space: {needed:.1f} GB needed, {available:.0f} GB free")
    if args.yes:
        raise SystemExit(execute(steps))


if __name__ == "__main__":
    main()
