"""Answer-first diagnostics: how good is candidate 1 and what do the audits do to it.

    python -m train.c1_stats <tag>[+<tag>...] [...] [--base base4b,prdev_base4b]

Candidate 1 is the model's own first proposal, or the injected answer in runs evaluated with eval_suite --first-box-from.
Per benchmark group (GME Rejection / positives, RefCOCO / + / g, PR-Bench dev Reject / positive tasks, own):
  base        the base model's direct answer on the same items (--base), the level candidate 1 has to hold
  c1 ok       candidate 1 is correct (IoU >= 0.5 with a ground-truth box)
  any         some candidate is correct
  acc         official accuracy of the rule-derived answer
  veto|ok     candidate 1 correct but vetoed by its audit (a named mismatch): the false veto rate
  line FA     named mismatches per audit line on a correct candidate 1 (lines per item in brackets)
  pass|wrong  candidate 1 wrong (or a no-target item) but its audit passes: the sycophancy side
  rescue      candidate 1 vetoed and the final answer correct (positives)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from train import coa as COA
from train import data as D

GROUPS = ("gme_rej", "gme_pos", "RefCOCO", "RefCOCOplus", "RefCOCOg", "pr_reject", "pr_attribute", "pr_position", "pr_interaction", "pr_relation",
          "pr_commonsense", "own_neg", "own_pos")


def group_of(name: str, r: dict) -> str | None:
    if name == "gme":
        return "gme_rej" if r.get("dimension") == "Rejection" else "gme_pos"
    if name == "refcoco":
        return r.get("set")
    if name == "prbench_dev":
        return "pr_" + str(r.get("set"))
    if name == "own":
        return "own_neg" if r.get("kind") == "negative" else "own_pos"
    return None


def load(tags: str, name: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tag in [t for t in tags.replace(",", "+").split("+") if t]:
        f = D.TRAIN_ROOT / "eval" / tag / f"{name}.jsonl"
        if f.is_file():
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    out.setdefault(r["id"], r)
    return out


def stats(tags: str, base_tags: str | None) -> None:
    s = defaultdict(lambda: defaultdict(float))
    for name in ("gme", "refcoco", "prbench_dev", "own"):
        base = load(base_tags, name) if base_tags else {}
        for i, r in load(tags, name).items():
            g = group_of(name, r)
            if g is None or r.get("coa") is None:
                continue
            d = s[g]
            d["n"] += 1
            d["acc"] += r["correct"]
            if i in base:
                d["nb"] += 1
                d["base"] += base[i]["correct"]
            pos = r["n_gt"] > 0
            audits = [COA.parse_audit(t) for t in r["coa"]["audits"]]
            if not audits:  # no candidate proposed
                continue
            d["k"] += 1
            a1 = audits[0]
            vetoed = not (a1["format_ok"] and not COA.named_mismatches(a1["lines"]))
            c1_ok = pos and (r.get("hint_iou") or 0.0) >= 0.5  # hint_iou = IoU of candidate 1 with the ground truth (eval_suite.score_item)
            d["any"] += pos and (r.get("cand_recall_iou") or 0.0) >= 0.5
            if c1_ok:
                d["c1_ok"] += 1
                d["veto_ok"] += vetoed
                if a1["format_ok"]:
                    d["lines_ok"] += len(a1["lines"])
                    d["named_ok"] += len(COA.named_mismatches(a1["lines"]))
                    d["items_lines"] += 1
            else:
                d["c1_wrong"] += 1
                d["pass_wrong"] += not vetoed
            if vetoed and pos:
                d["vetoed_pos"] += 1
                d["rescue"] += r["correct"]
    print(f"[{tags}]")
    print(f"  {'group':15s} {'n':>5s} {'base':>6s} {'c1 ok':>6s} {'any':>6s} {'acc':>6s} {'veto|ok':>8s} {'line FA':>13s} {'pass|wrong':>10s} {'rescue':>7s}")
    for g in GROUPS:
        d = s.get(g)
        if not d or not d["n"]:
            continue
        pct = lambda a, b: f"{100 * d[a] / d[b]:5.1f}" if d[b] else "    -"
        neg = g in ("gme_rej", "pr_reject", "own_neg")
        lfa = f"{100 * d['named_ok'] / d['lines_ok']:5.1f} ({d['lines_ok'] / d['items_lines']:.1f})" if d["lines_ok"] else "     -"
        print(f"  {g:15s} {int(d['n']):5d} {pct('base', 'nb'):>6s} {'    -' if neg else pct('c1_ok', 'n'):>6s} {'    -' if neg else pct('any', 'n'):>6s} {pct('acc', 'n'):>6s} "
              f"{pct('veto_ok', 'c1_ok'):>8s} {lfa:>13s} {pct('pass_wrong', 'c1_wrong'):>10s} {pct('rescue', 'vetoed_pos'):>7s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+", help="eval tags; join the tags of one run with + (e.g. full_grpo_coa10b+prdev_grpo_coa10b)")
    ap.add_argument("--base", default=None, help="comma-separated eval tags of the base model's direct answers")
    a = ap.parse_args()
    for t in a.tags:
        stats(t, a.base)
