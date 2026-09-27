"""Training curves from the queue logs (backfill) and from train_log.jsonl (new runs).

    python -m train.plot_curves --logs ~/vlmg-results/*.log --names sft_hint_v3,grpo_v3d_hint,grpo_v3e_long

For every run named in a log ("=== hh:mm:ss [k] ... --name NAME ..."), the SFT lines
("epoch E step S/T loss L lr R") and the GRPO window lines ("epoch E step S/T | neg: r .. null ..
| cross: .. | pos: r .. null .. fa .. | bad .. inc ..") are parsed into
$VLMG_DATA_ROOT/train/NAME/train_log.jsonl (one record per logged step) and plotted to
$VLMG_RESULTS/curves/NAME.png; --names draws one combined figure curves/combined.png.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

from train import data as D

HDR = re.compile(r"^=== \d\d:\d\d:\d\d \[\d\] .*--name (\S+)")
SFT = re.compile(r"epoch (\d+) step (\d+)/(\d+) loss ([\d.]+) lr ([\d.e+-]+) ([\d.]+) min")
GRPO = re.compile(r"epoch (\d+) step (\d+)/(\d+) \| neg: r ([-\d.na]+) null ([\d.na]+) ev ([\d.na]+) used ([\d.na]+) \| cross: r ([-\d.na]+) null ([\d.na]+) \| "
                  r"pos: r ([-\d.na]+) null ([\d.na]+) fa ([\d.na]+) used ([\d.na]+) \| bad ([\d.na]+) inc ([\d.na]+) \| ([\d.]+) min")


def f(x: str):
    return None if x == "nan" else float(x)


def parse_logs(paths: list[str]) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for p in paths:
        name = None
        for line in open(os.path.expanduser(p), encoding="utf-8", errors="ignore"):
            m = HDR.match(line)
            if m:
                name = m.group(1)
                runs.setdefault(name, [])
                continue
            if name is None:
                continue
            m = SFT.search(line)
            if m:
                runs[name].append({"kind": "sft", "epoch": int(m.group(1)), "step": int(m.group(2)), "total": int(m.group(3)),
                                   "loss": float(m.group(4)), "lr": float(m.group(5)), "min": float(m.group(6))})
                continue
            m = GRPO.search(line)
            if m:
                g = m.groups()
                runs[name].append({"kind": "grpo", "epoch": int(g[0]), "step": int(g[1]), "total": int(g[2]),
                                   "neg_r": f(g[3]), "neg_null": f(g[4]), "neg_ev": f(g[5]), "neg_used": f(g[6]),
                                   "cross_r": f(g[7]), "cross_null": f(g[8]), "pos_r": f(g[9]), "pos_null": f(g[10]),
                                   "pos_fa": f(g[11]), "pos_used": f(g[12]), "bad": f(g[13]), "inc": f(g[14]), "min": float(g[15])})
    return {k: v for k, v in runs.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", default=[str(Path.home() / "vlmg-results" / "*.log")])
    ap.add_argument("--names", default=None, help="comma-separated run names for a combined figure")
    ap.add_argument("--no-backfill", action="store_true")
    args = ap.parse_args()
    paths = sorted(set(sum((glob.glob(os.path.expanduser(p)) for p in args.logs), [])))
    runs = parse_logs(paths)
    out_dir = Path.home() / "vlmg-results" / "curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in runs.items():
        d = D.TRAIN_ROOT / name
        if d.is_dir() and not args.no_backfill and not (d / "train_log.jsonl").is_file():
            with open(d / "train_log.jsonl", "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in (args.names.split(",") if args.names else runs) if n in runs]
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 2.8 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        recs = runs[name]
        if recs[0]["kind"] == "sft":
            ax.plot([r["min"] for r in recs], [r["loss"] for r in recs], label="SFT loss", marker=".", ms=3)
            ax.set_ylabel("loss")
            ax.set_xlabel("minutes")
        else:
            steps = [r["step"] + (r["epoch"] - 1) * r["total"] // max(1, max(x["epoch"] for x in recs)) for r in recs]
            steps = [(r["epoch"] - 1) * (recs[-1]["total"] // max(1, recs[-1]["epoch"])) + r["step"] if False else i for i, r in enumerate(recs)]
            xs = [r["min"] for r in recs]
            for key, lab in (("pos_r", "pos mean reward"), ("neg_r", "neg mean reward"), ("neg_null", "neg null frac"), ("pos_null", "pos null frac"), ("pos_fa", "pos false-accusation"), ("bad", "unparsable")):
                ys = [r[key] for r in recs]
                if any(y is not None for y in ys):
                    ax.plot(xs, [y if y is not None else float("nan") for y in ys], label=lab, marker=".", ms=3)
            ax.set_xlabel("minutes")
            ax.set_ylim(-1.1, 1.6)
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3, loc="lower right")
    fig.tight_layout()
    out = out_dir / ("combined.png" if args.names else "all.png")
    fig.savefig(out, dpi=110)
    print(f"{len(runs)} runs parsed: {', '.join(runs)}; figure -> {out}")
    for name in names:
        recs = runs[name]
        if recs[0]["kind"] == "grpo":
            k = max(1, len(recs) // 4)
            head = recs[:k]
            tail = recs[-k:]
            avg = lambda rs, key: sum(r[key] for r in rs if r[key] is not None) / max(1, sum(r[key] is not None for r in rs))
            print(f"  {name}: {len(recs)} windows | pos reward {avg(head, 'pos_r'):.2f} -> {avg(tail, 'pos_r'):.2f} | neg reward {avg(head, 'neg_r'):.2f} -> {avg(tail, 'neg_r'):.2f} | "
                  f"pos null {avg(head, 'pos_null'):.2f} -> {avg(tail, 'pos_null'):.2f} | neg null {avg(head, 'neg_null'):.2f} -> {avg(tail, 'neg_null'):.2f} | bad {avg(head, 'bad'):.2f} -> {avg(tail, 'bad'):.2f}")
        else:
            print(f"  {name}: {len(recs)} points | loss {recs[0]['loss']:.3f} -> {recs[-1]['loss']:.3f} ({recs[-1]['min']:.1f} min)")


if __name__ == "__main__":
    main()
