"""Stage driver.

    python -m datagen.run select  --run test50 --n 50
    python -m datagen.run write   --run test50
    python -m datagen.run policy  --run test50 --which target,neg
    python -m datagen.run sibling --run test50
    python -m datagen.run policy  --run test50 --which sibling
    python -m datagen.run listen  --run test50
    python -m datagen.run check   --run test50
    python -m datagen.run blind   --run test50
    python -m datagen.run report  --run test50

Every model stage appends one JSONL row per finished item and skips rows that are
already there, so a killed stage is re-run with the same command and loses at
most one item.  One model is resident per process.
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["select", "write", "policy", "sibling", "listen", "check", "blind", "report", "reflip", "export"])
    ap.add_argument("--out", default=None, help="export: output JSONL path")
    ap.add_argument("--model", default="8b", choices=["8b", "4b"], help="policy: which base model answers")
    ap.add_argument("--run", default="test50")
    ap.add_argument("--n", type=int, default=50, help="select: how many scenes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-label", type=int, default=5, help="select: cap per OpenImages label (0 = no cap)")
    ap.add_argument("--which", default="target,neg", help="policy: target,neg or sibling")
    ap.add_argument("--limit", type=int, default=None, help="model stages: stop after N items")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"], help="listen: HF remote code or a vLLM endpoint")
    ap.add_argument("--checker", default="gemma4", choices=["gemma4", "qwen3vl", "gemini", "gemini-flash", "gemini-b", "gemini-flash-b"], help="check: which verifier model (gemini* = API; -b = one call per candidate)")
    ap.add_argument("--primary-checker", default="gemma4", choices=["gemma4", "gemini", "gemini-flash", "gemini-b", "gemini-flash-b"], help="report: whose verdict matrix drives the gates")
    ap.add_argument("--exclude-run", default="", help="select: comma-separated runs whose scenes are skipped")
    ap.add_argument("--target-only", action="store_true", help="check: verify the target instance only (second opinion)")
    ap.add_argument("--no-closeup", action="store_true", help="write/sibling/check: v1 single-image prompts")
    args = ap.parse_args()

    t0 = time.time()
    if args.no_closeup:
        from datagen import checker as _ck, writer as _wr
        _ck.CLOSEUP = False
        _wr.CLOSEUP = False
    if args.stage == "select":
        from datagen import select
        select.run(args.run, args.n, args.seed, args.per_label, [r for r in args.exclude_run.split(",") if r])
    elif args.stage == "write":
        from datagen import writer
        writer.run_write(args.run, args.limit)
    elif args.stage == "sibling":
        from datagen import writer
        writer.run_sibling(args.run, args.limit)
    elif args.stage == "blind":
        from datagen import writer
        writer.run_blind(args.run, args.limit)
    elif args.stage == "policy":
        from datagen import policy
        policy.run(args.run, [w.strip() for w in args.which.split(",")], args.limit, args.model)
    elif args.stage == "listen":
        if args.backend == "vllm":
            from datagen import listener_vllm as listener
        else:
            from datagen import listener
        listener.run(args.run, args.limit)
    elif args.stage == "check":
        if args.checker.startswith("gemini"):
            from datagen import checker_api as checker
        else:
            from datagen import checker
        checker.run(args.run, args.limit, args.checker, args.target_only)
    elif args.stage == "report":
        from datagen import report
        report.run(args.run, args.primary_checker)
    elif args.stage == "reflip":
        from datagen import reflip
        reflip.run(args.run, args.limit, args.primary_checker,
                   args.checker if args.checker.startswith("gemini") else "gemini-b")
    elif args.stage == "export":
        from datagen import export
        export.run(args.run, args.primary_checker, args.out)
    print(f"[{args.stage}] done in {(time.time() - t0) / 60:.1f} min", file=sys.stderr)


if __name__ == "__main__":
    main()
