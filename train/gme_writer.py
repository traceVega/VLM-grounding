"""Data recipe A (2026-09-27): GME-style long descriptions of our own verified instances, three one-detail falsifications each,
and a per-instance check that also names the observed value of every detail.  Gemini through the API; OpenImages only.

    python -m train.gme_writer write [--limit N] [--workers 4] [--cap-usd 5]
    python -m train.gme_writer check [--limit N] [--workers 4] [--cap-usd 5]
    python -m train.gme_export                      # gates + items (train/gme_export.py)

Outputs under $VLMG_DATA_ROOT/train/gme_style/: write.jsonl (per scene), check.jsonl (per scene, per instance), usage.json.
Resumable per scene.  Cost is estimated from token usage (prompt 1.25 $/M, output+thought 10 $/M, the fit of the September
rationale pass) and the run stops at --cap-usd.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from datagen import common as C
from shared.harness import prompts
from train import data as D
from train.long_items import SOURCES, clause_bank

ROOT = D.TRAIN_ROOT / "gme_style"
USD_PROMPT, USD_OUT = 1.25 / 1e6, 10.0 / 1e6
MAX_SIDE = 1024
RETRIES = 5


def cost_usd(u: dict) -> float:
    return u["prompt_tokens"] * USD_PROMPT + (u["output_tokens"] + u["thought_tokens"]) * USD_OUT


def load_usage() -> dict:
    f = ROOT / "usage.json"
    return json.load(open(f, encoding="utf-8")) if f.is_file() else {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0}


class Gemini:
    def __init__(self, model: str, cap_usd: float, max_out: int):
        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit("GEMINI_API_KEY is not set (WSL: ~/.vlmg-secrets)")
        from google import genai
        from google.genai import types

        self.client = genai.Client()
        self.model = model
        self.cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=max_out, thinking_config=types.ThinkingConfig(thinking_level="low"))
        self.cap = cap_usd
        self.usage = load_usage()
        self.lock = threading.Lock()

    def over_cap(self) -> bool:
        return cost_usd(self.usage) >= self.cap

    def ask(self, views: list, text: str) -> str:
        last = None
        for attempt in range(RETRIES):
            if self.over_cap():
                return "ERROR: cap reached"
            try:
                resp = self.client.models.generate_content(model=self.model, contents=[*views, text], config=self.cfg)
                u = resp.usage_metadata
                with self.lock:
                    self.usage["calls"] += 1
                    self.usage["prompt_tokens"] += (u.prompt_token_count or 0) if u else 0
                    self.usage["output_tokens"] += (u.candidates_token_count or 0) if u else 0
                    self.usage["thought_tokens"] += (getattr(u, "thoughts_token_count", 0) or 0) if u else 0
                    json.dump(self.usage, open(ROOT / "usage.json", "w", encoding="utf-8"))
                return (resp.text or "").strip()
            except Exception as e:  # rate limit / transient
                last = e
                time.sleep(min(60, (2 ** attempt) + random.random()))
        return f"ERROR: {last}"


def shrink(image: Image.Image) -> tuple[Image.Image, float]:
    w, h = image.size
    s = min(1.0, MAX_SIDE / max(w, h))
    return (image.resize((round(w * s), round(h * s)), Image.LANCZOS), s) if s < 1.0 else (image, 1.0)


def scenes() -> list[dict]:
    """One job per scene: the original target (else the first instance with >= 4 verified facts) and its true clauses."""
    items = []
    for f in SOURCES:
        if Path(f).is_file():
            items.extend(D.load_items(Path(f)))
    out = []
    for (run, group), b in sorted(clause_bank(items).items()):
        base, clauses, boxes = b["base"], b["clauses"], b["boxes"]
        iids = sorted(boxes)
        if len(iids) < 2:
            continue
        cands = sorted(b["targets"]) + [i for i in iids if i not in b["targets"]]
        for t in cands:
            facts = [c for c, vs in clauses.items() if vs.get(t) == "yes"]
            if len(facts) >= 4:
                out.append({"run": run, "group": group, "image": base["image"], "image_wh": base["image_wh"], "category": base["category"],
                            "label": base.get("label"), "target": t, "boxes": {str(i): boxes[i] for i in iids}, "facts": facts})
                break
    return out


_DESC = re.compile(r"<description>\s*(.*?)\s*</description>", re.S)
_DET = re.compile(r"<details>\s*(.*?)\s*</details>", re.S)
_LINE = re.compile(r"^\s*(\d+)[.:)]\s*(.+?)\s*$", re.M)
_VER = re.compile(r'<version\s+kind="([ABC])"\s+detail="(\d+)"\s*>\s*(.*?)\s*</version>', re.S)


def parse_describe(text: str) -> tuple[str | None, list[str]]:
    d, l = _DESC.search(text or ""), _DET.search(text or "")
    desc = d.group(1).strip() if d else None
    details = [m.group(2).rstrip(".") for m in _LINE.finditer(l.group(1))] if l else []
    return desc, details


def parse_flips(text: str) -> list[dict]:
    out = []
    for kind, idx, body in _VER.findall(text or ""):
        f = {}
        for key in ("old", "new", "detail", "description"):
            m = re.search(rf"^{key}:\s*(.+?)\s*$", body, re.M | re.S if key == "description" else re.M)
            f[key] = m.group(1).strip() if m else None
        if all(f.values()):
            f["detail"] = re.sub(r"^\s*\d+[.:)]\s*", "", f["detail"]).rstrip(".")  # "4. has a yellow frame" -> "has a yellow frame"
            if int(idx) - 1 not in {o["idx"] for o in out}:  # one falsification per detail
                out.append({"kind": kind, "idx": int(idx) - 1, **f})
    return out


def stage_write(args) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / "write.jsonl"
    done = {(r["run"], r["group"]) for r in C.read_jsonl(out)} if out.is_file() else set()
    jobs = [s for s in scenes() if (s["run"], s["group"]) not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"write: {len(done)} done, {len(jobs)} to run, cost so far {cost_usd(load_usage()):.2f} $", flush=True)
    g = Gemini(args.model, args.cap_usd, 4096)  # thinking tokens count against this budget: 1400 truncated the three versions
    t_a = prompts.load("gme_writer_describe", non_kill=True)
    t_b = prompts.load("gme_writer_flip", non_kill=True)

    def one(s: dict) -> dict:
        full = Image.open(s["image"]).convert("RGB")
        small, sc = shrink(full)
        box = s["boxes"][str(s["target"])]
        views_a = [C.outline(small, [v * sc for v in box], "red"), C.closeup(full, box)]
        facts = "\n".join(f"- The {s['category']} {c}." for c in s["facts"])
        raw_a = g.ask(views_a, t_a.render(category=s["category"], facts=facts))
        desc, details = parse_describe(raw_a)
        rec = {**s, "raw_describe": raw_a, "description": desc, "details": details, "versions": [], "raw_flip": None}
        if desc and len(details) >= 5:
            det_text = "\n".join(f"{k + 1}. {d}" for k, d in enumerate(details))
            raw_b = g.ask([small], t_b.render(category=s["category"], description=desc, details=det_text))
            rec["raw_flip"] = raw_b
            rec["versions"] = [v for v in parse_flips(raw_b) if 0 <= v["idx"] < len(details)]
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, rec in enumerate(ex.map(one, jobs), 1):
            C.append_jsonl(out, rec)
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} scenes | {len(rec['details'])} details, {len(rec['versions'])} versions | cost {cost_usd(g.usage):.2f} $", flush=True)
            if g.over_cap():
                print("cap reached, stopping", flush=True)
                break


_CHK = re.compile(r"^\s*\**\s*(\d+)\s*[.:)\-]+\s*\**\s*(yes|no|unclear)\b\s*\|?\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_check(text: str, n: int) -> list[tuple[str, str]]:
    out = [("unparsed", "")] * n
    for m in _CHK.finditer(text or ""):
        k = int(m.group(1)) - 1
        if 0 <= k < n and out[k][0] == "unparsed":
            out[k] = (m.group(2).lower(), m.group(3).strip().rstrip("."))
    return out


def stage_check(args) -> None:
    """Every instance of every written scene: the description's details + every falsified detail, verdict + observed value."""
    src, out = ROOT / "write.jsonl", ROOT / "check.jsonl"
    written = [r for r in C.read_jsonl(src) if r.get("description") and r.get("versions")]
    done = {(r["run"], r["group"]) for r in C.read_jsonl(out)} if out.is_file() else set()
    jobs = [r for r in written if (r["run"], r["group"]) not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"check: {len(written)} written scenes, {len(done)} done, {len(jobs)} to run, cost so far {cost_usd(load_usage()):.2f} $", flush=True)
    g = Gemini(args.model, args.cap_usd, 2048)
    tmpl = prompts.load("verifier_clauses_seen", non_kill=True)

    def one(r: dict) -> dict:
        full = Image.open(r["image"]).convert("RGB")
        small, sc = shrink(full)
        stmts = list(r["details"]) + [v["detail"] for v in r["versions"]]
        items = "\n".join(f"{k + 1}. The {r['category']} {c}." for k, c in enumerate(stmts))
        text = tmpl.render(category=r["category"], items=items)
        rows = []
        for iid, box in r["boxes"].items():
            views = [C.outline(small, [v * sc for v in box], "red"), C.closeup(full, box)]
            raw = g.ask(views, text)
            parsed = parse_check(raw, len(stmts))
            if sum(p[0] == "unparsed" for p in parsed) > 0 and not raw.startswith("ERROR"):
                raw = g.ask(views, text + f"\nYou must answer all {len(stmts)} statements, one line each.")
                parsed = parse_check(raw, len(stmts))
            rows.append({"iid": int(iid), "box": box, "is_target": int(iid) == r["target"], "verdicts": [p[0] for p in parsed], "seen": [p[1] for p in parsed], "raw": raw})
        return {"run": r["run"], "group": r["group"], "n_details": len(r["details"]), "n_versions": len(r["versions"]), "rows": rows}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, rec in enumerate(ex.map(one, jobs), 1):
            C.append_jsonl(out, rec)
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} scenes | cost {cost_usd(g.usage):.2f} $", flush=True)
            if g.over_cap():
                print("cap reached, stopping", flush=True)
                break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["write", "check"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cap-usd", type=float, default=5.0)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    args = ap.parse_args()
    (stage_write if args.stage == "write" else stage_check)(args)
    print(f"usage {json.dumps(load_usage())} = {cost_usd(load_usage()):.2f} $", flush=True)


if __name__ == "__main__":
    main()
