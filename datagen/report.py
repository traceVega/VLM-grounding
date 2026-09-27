"""Stage `report` (CPU): join every stage into master records, print the gate
table, and write a human review page (one card per scene, instances numbered on
the image, every model output and every gate flag beside it).

Outputs (run root on the ext4 volume, plus a Windows-side copy of the review):
    records.jsonl   one master record per scene
    gates.md        counts and yields per gate
    review/         index.html + img/*.jpg   (also copied to <repo>/review/<run>/)
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from datagen import common as C
from shared import paths

COLOURS = {"target": (0, 200, 0), "sibling": (255, 140, 0), "pol_target": (255, 0, 0),
           "pol_neg": (200, 0, 200), "pol_sibling": (0, 120, 255), "other": (160, 160, 160)}


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render(scene: dict, rec: dict, out_path: Path, max_w: int = 960) -> None:
    img = Image.open(C.image_path(scene["image_id"])).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_w / w)
    if scale < 1.0:
        img = img.resize((round(w * scale), round(h * scale)))
    draw = ImageDraw.Draw(img)
    font = _font(max(14, round(18 * scale)))

    def box(b, colour, width, label=None):
        x0, y0, x1, y1 = [v * scale for v in b]
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=width)
        if label:
            tw = draw.textlength(label, font=font)
            draw.rectangle([x0, y0 - 20, x0 + tw + 6, y0], fill=colour)
            draw.text((x0 + 3, y0 - 19), label, fill=(255, 255, 255), font=font)

    s_iid = rec.get("sibling_iid")
    for inst in scene["instances"]:
        if inst["is_target"]:
            box(inst["box"], COLOURS["target"], 4, f"{inst['iid']} T")
        elif inst["iid"] == s_iid:
            box(inst["box"], COLOURS["sibling"], 4, f"{inst['iid']} S")
        else:
            box(inst["box"], COLOURS["other"], 2, str(inst["iid"]))
    for which, key in (("target", "pol_target"), ("neg", "pol_neg"), ("sibling", "pol_sibling")):
        b = (rec.get("policy") or {}).get(which, {}).get("box")
        if b:
            box(b, COLOURS[key], 2)

    def dot(p, colour):
        if p:
            x, y = p[0] * scale, p[1] * scale
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=colour, outline=(0, 0, 0))

    lst = rec.get("listener") or {}
    dot((lst.get("target") or {}).get("point"), COLOURS["target"])
    dot((lst.get("sibling") or {}).get("point"), COLOURS["sibling"])
    img.save(out_path, quality=88)


PRIMARY_FILES = {"gemma4": "checker.jsonl", "gemini": "checker_gemini.jsonl", "gemini-flash": "checker_gemini_flash.jsonl",
                 "gemini-b": "checker_gemini_b.jsonl", "gemini-flash-b": "checker_gemini_flash_b.jsonl"}


def _sentence_satisfiers(c: dict | None, flipped_idx) -> list[int] | None:
    """Candidates that may satisfy the WHOLE negative expression: the flipped detail is
    yes/unclear on them AND every other detail is yes/unclear.  The per-clause
    `zero_satisfier` gate only looks at the flipped detail and kills scenes where another
    candidate shares that one detail but fails the rest (test50c review, 2026-09-22)."""
    if not c or flipped_idx is None:
        return None
    out = []
    for r in c["instances"]:
        if r.get("neg_verdict") not in ("yes", "unclear"):
            continue
        others = [v for k, v in enumerate(r["verdicts"]) if k != flipped_idx]
        if all(v in ("yes", "unclear") for v in others):
            out.append(r["iid"])
    return out


def render_sheet(scene: dict, rec: dict, out_path: Path, cell: int = 380, cols: int = 4) -> None:
    """Contact sheet of every candidate's close-up (the crop the writer and checker saw),
    labelled with its id and T / S, so a reviewer can verify details at native resolution."""
    img = Image.open(C.image_path(scene["image_id"])).convert("RGB")
    insts = scene["instances"]
    n = len(insts)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (24, 27, 31))
    draw = ImageDraw.Draw(sheet)
    font = _font(20)
    s_iid = rec.get("sibling_iid")
    for k, inst in enumerate(insts):
        crop = C.closeup(img, inst["box"])
        crop.thumbnail((cell - 8, cell - 8))
        x0, y0 = (k % cols) * cell + 4, (k // cols) * cell + 4
        sheet.paste(crop, (x0, y0))
        if inst["is_target"]:
            label, colour = f"{inst['iid']} T", COLOURS["target"]
        elif inst["iid"] == s_iid:
            label, colour = f"{inst['iid']} S", COLOURS["sibling"]
        else:
            label, colour = str(inst["iid"]), COLOURS["other"]
        tw = draw.textlength(label, font=font)
        draw.rectangle([x0, y0, x0 + tw + 10, y0 + 26], fill=colour)
        draw.text((x0 + 5, y0 + 2), label, fill=(255, 255, 255), font=font)
        draw.rectangle([x0 - 2, y0 - 2, x0 + crop.size[0] + 1, y0 + crop.size[1] + 1], outline=colour, width=3)
    sheet.save(out_path, quality=85)


def build_records(root: Path, primary: str = "gemma4") -> list[dict]:
    scenes = C.read_jsonl(root / "scenes.jsonl")
    written = C.by_image(C.read_jsonl(root / "write.jsonl"))
    sib = C.by_image(C.read_jsonl(root / "sibling.jsonl"))
    lst = C.by_image(C.read_jsonl(root / "listener.jsonl"))
    chk = C.by_image(C.read_jsonl(root / PRIMARY_FILES[primary]))
    chk2 = C.by_image(C.read_jsonl(root / "checker2.jsonl"))
    bl = C.by_image(C.read_jsonl(root / "blind.jsonl"))
    reflips = C.by_image(C.read_jsonl(root / "reflip.jsonl"))
    pol: dict[str, dict] = {}
    for r in C.read_jsonl(root / "policy.jsonl"):
        pol.setdefault(r["image_id"], {})[r["which"]] = r
    records = []
    for sc in scenes:
        iid = sc["image_id"]
        w = written.get(iid)
        if not w:
            continue
        p = pol.get(iid, {})
        pt, pn, ps = p.get("target"), p.get("neg"), p.get("sibling")
        l, c, b, s = lst.get(iid), chk.get(iid), bl.get(iid), sib.get(iid)
        c2 = chk2.get(iid)
        rf = reflips.get(iid)
        reflipped = None
        if rf and rf.get("accepted"):  # overlay the second-chance negative
            reflipped = rf["reason"]
            w = dict(w, expr_neg=rf["expr_neg"], changed_from=rf["changed_from"], changed_to=rf["changed_to"],
                     flipped_idx=rf["flipped_idx"], clause_neg=rf["clause_neg"], flip_failed=False)
            if c:
                c = json.loads(json.dumps(c))
                for r_ in c["instances"]:
                    r_["neg_verdict"] = (rf.get("neg_clause_verdicts") or {}).get(str(r_["iid"]))
                c["zero_satisfier"] = all(r_["neg_verdict"] == "no" for r_ in c["instances"])
            b = rf.get("blind") or b
            pn = p.get("neg2")
        t_row = next((r for r in c["instances"] if r["is_target"]), None) if c else None
        t_row2 = next((r for r in c2["instances"] if r["is_target"]), None) if c2 else None
        n_no = sum(1 for v in t_row["verdicts"] if v == "no") if t_row else None
        agree = None
        if t_row and t_row2 and len(t_row["verdicts"]) == len(t_row2["verdicts"]):
            pairs = [(a, b2) for a, b2 in zip(t_row["verdicts"], t_row2["verdicts"])
                     if a != "unclear" and b2 != "unclear"]
            agree = (sum(1 for a, b2 in pairs if a == b2) / len(pairs)) if pairs else None
        gates = {
            "writer_target_all_yes": c["target_all_yes"] if c else None,
            "writer_target_at_most_one_no": (n_no <= 1) if n_no is not None else None,
            "checkers_agree_on_target": (agree >= 0.8) if agree is not None else None,
            "siblings_distinct": c["siblings_distinct"] if c else None,
            "listener_unique_hit": l["unique_hit"] if l else None,
            "zero_satisfier": c["zero_satisfier"] if c else None,
            "zero_satisfier_sentence": (len(_sentence_satisfiers(c, w["flipped_idx"])) == 0) if (c and w.get("clause_neg")) else None,
            "base_boxes_on_neg": (pn["output_type"] == "box" and (pn.get("p_box") or 0) >= 0.99
                                  and pn["boxed_iid"] is not None) if pn else None,
            "base_wrong_on_target": (pt["boxed_iid"] is not None and pt["boxed_iid"] != sc["target_iid"]) if pt else None,
            "base_correct_on_target": ((pt.get("iou_target") or 0) >= 0.5) if pt else None,
            "blind_judge_fooled": (b["judge_found_flip"] is False) if b and b["judge_found_flip"] is not None else None,
            "sibling_unique_hit": l.get("sibling_unique_hit") if l else None,
            "flip_identified": (w["flipped_idx"] is not None and bool(w["expr_neg"])
                                and w["expr_neg"].strip().lower() != w["expr_target"].strip().lower()),
        }
        core = [gates[k] for k in ("writer_target_all_yes", "listener_unique_hit", "zero_satisfier",
                                   "base_boxes_on_neg", "flip_identified")]
        gates["negative_usable"] = all(core) if all(v is not None for v in core) else None
        core2 = [gates[k] for k in ("writer_target_all_yes", "listener_unique_hit", "zero_satisfier_sentence",
                                    "base_boxes_on_neg", "flip_identified")]
        gates["negative_usable_sentence"] = all(core2) if all(v is not None for v in core2) else None
        gates["positive_hard"] = (gates["base_wrong_on_target"] and gates["listener_unique_hit"]
                                  and gates["writer_target_all_yes"]) if pt and l and c else None
        rec = {
            "image_id": iid, "label": sc["label"], "category": sc["category"], "image_wh": sc["image_wh"],
            "instances": [{k: i[k] for k in ("iid", "box", "area", "is_target")} for i in sc["instances"]],
            "target_iid": sc["target_iid"], "sibling_iid": s["sibling_iid"] if s else None,
            "sibling_reason": s["sibling_reason"] if s else None,
            "expressions": {"e_T": w["expr_target"], "e_S": s["expr_sibling"] if s else None,
                            "e_T_neg": w["expr_neg"]},
            "head": w["head"], "clauses": w["clauses"], "clauses_neg": w["clauses_neg"],
            "flipped": {"idx": w["flipped_idx"], "from": w["changed_from"], "to": w["changed_to"],
                        "clause_neg": w["clause_neg"]},
            "load_bearing": l["load_bearing"] if l else None,
            "clause_verdicts": {str(r["iid"]): r["verdicts"] for r in c["instances"]} if c else None,
            "clause_verdicts_checker2_target": t_row2["verdicts"] if t_row2 else None,
            "checker_agreement_target": agree,
            "checker_name": (c or {}).get("checker"),
            "sentence_satisfiers": _sentence_satisfiers(c, w["flipped_idx"]),
            "neg_clause_verdicts": {str(r["iid"]): r["neg_verdict"] for r in c["instances"]} if c else None,
            "policy": {k: {f: v.get(f) for f in ("output_type", "box", "boxed_iid", "boxed_iou", "iou_target",
                                                  "p_null", "p_box", "p_null_norm", "raw")}
                       for k, v in p.items()},
            "listener": {k: (l.get(k) if l else None) for k in ("target", "sibling")},
            "blind": {k: (b.get(k) if b else None) for k in ("judge_letter", "judge_found_flip", "neg_is_a")},
            "gates": gates,
            "reflipped": reflipped,
        }
        records.append(rec)
    return records


def gate_table(records: list[dict]) -> str:
    keys = ["flip_identified", "writer_target_all_yes", "writer_target_at_most_one_no",
            "checkers_agree_on_target", "siblings_distinct", "listener_unique_hit",
            "zero_satisfier", "zero_satisfier_sentence", "base_correct_on_target", "base_wrong_on_target", "base_boxes_on_neg",
            "blind_judge_fooled", "sibling_unique_hit", "negative_usable", "negative_usable_sentence", "positive_hard"]
    lines = ["| gate | pass | measured | rate |", "|---|---:|---:|---:|"]
    for k in keys:
        vals = [r["gates"][k] for r in records if r["gates"][k] is not None]
        n_pass = sum(1 for v in vals if v)
        rate = f"{n_pass / len(vals):.0%}" if vals else "-"
        lines.append(f"| {k} | {n_pass} | {len(vals)} | {rate} |")
    lb = [x for r in records if r["load_bearing"] for x in r["load_bearing"]]
    if lb:
        lines.append(f"| load_bearing clauses (listener fails when dropped) | {sum(lb)} | {len(lb)} | {sum(lb) / len(lb):.0%} |")
    pn = [r["policy"].get("neg", {}).get("p_null") for r in records if r["policy"].get("neg")]
    pn = [v for v in pn if v is not None]
    if pn:
        import math
        med = sorted(pn)[len(pn) // 2]
        lines.append(f"| median log10 p(null) on negatives (base) | {math.log10(max(med, 1e-30)):.1f} | {len(pn)} | - |")
    return "\n".join(lines)


def _tag(v) -> str:
    if v is None:
        return "<span class='na'>无</span>"
    return "<span class='ok'>通过</span>" if v else "<span class='bad'>未过</span>"


GATE_ZH = {
    "flip_identified": "改假被唯一识别",
    "writer_target_all_yes": "写手细节全部被核对者确认",
    "writer_target_at_most_one_no": "写手细节最多一处不符",
    "checkers_agree_on_target": "两个核对者在目标上一致（≥80%）",
    "siblings_distinct": "每个兄弟至少一条细节不符",
    "listener_unique_hit": "听者只看 e_T 唯一指到目标",
    "sibling_unique_hit": "听者只看 e_S 唯一指到兄弟",
    "zero_satisfier": "改假细节没有任何候选满足（单条）",
    "zero_satisfier_sentence": "改假后整句没有任何候选满足",
    "negative_usable_sentence": "负样本全门通过（整句版零满足）",
    "base_correct_on_target": "基座在 e_T 下框对（IoU≥0.5）",
    "base_wrong_on_target": "基座在 e_T 下框错",
    "base_boxes_on_neg": "基座在 e_T⁻ 下仍自信框物体",
    "blind_judge_fooled": "纯文本判官没猜出改句",
    "negative_usable": "负样本全门通过",
    "positive_hard": "困难正样本",
}
POLICY_ZH = {"target": "e_T", "neg": "e_T⁻", "sibling": "e_S"}
OUTPUT_ZH = {"box": "框", "null": "拒答", None: "无输出"}


def load_notes(run: str) -> dict | None:
    """Reviewer notes for a run: datagen/reviews/<run>.json (buckets + per-scene verdicts)."""
    f = Path(__file__).resolve().parent / "reviews" / f"{run}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


STYLE = """
<style>
:root{--bg:#f6f7f9;--card:#ffffff;--ink:#1b1f24;--muted:#5b6470;--line:#d9dee5;--code:#eef1f4;
 --target:#1e9e5a;--sibling:#e08a1e;--pol-t:#d64545;--pol-n:#b03ab0;--pol-s:#2f7fe0;--other:#8a939e;
 --yes:#dcf3e4;--no:#f8dada;--unclear:#fff1c2;--na:#edeff2;--ok:#177a45;--bad:#b42323;--warn:#8a6100;--flip:#8a2e8a;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#14171b;--card:#1c2127;--ink:#e6e9ee;--muted:#9aa3ae;--line:#2c333b;--code:#232a32;
 --yes:#1f4a31;--no:#4d2323;--unclear:#4a3e14;--na:#2a3138;--ok:#5fd28f;--bad:#f08080;--warn:#f0c65a;--flip:#d78ad7;}}
:root[data-theme="dark"]{--bg:#14171b;--card:#1c2127;--ink:#e6e9ee;--muted:#9aa3ae;--line:#2c333b;--code:#232a32;
 --yes:#1f4a31;--no:#4d2323;--unclear:#4a3e14;--na:#2a3138;--ok:#5fd28f;--bad:#f08080;--warn:#f0c65a;--flip:#d78ad7;}
body{background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",Roboto,sans-serif;font-size:14px;line-height:1.55;margin:0;padding-block:20px;padding-inline:16px}
.wrap{max-width:1400px;margin:0 auto}
h1{font-size:22px;margin:0 0 6px 0;text-wrap:balance} .sub{color:var(--muted);margin:0 0 14px 0}
h2{font-size:18px;margin:30px 0 4px 0;padding-top:12px;border-top:2px solid var(--line);text-wrap:balance;scroll-margin-top:12px}
.bdesc{color:var(--muted);margin:0 0 12px 0;max-width:90ch}
.overview{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:0 0 18px 0}
.overview h2{border:0;margin:0 0 6px 0;padding:0;font-size:16px} .overview p{max-width:90ch;margin:6px 0}
.legend{font-size:13px;color:var(--muted);margin:0 0 18px 0;line-height:1.7;max-width:110ch}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:-1px;margin-right:4px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin:0 0 20px 0}
.tile{display:block;background:var(--card);border:1px solid var(--line);border-left-width:5px;border-radius:8px;padding:10px 12px;text-decoration:none;color:inherit}
.tile b{display:block;font-size:22px;font-variant-numeric:tabular-nums;line-height:1.2} .tile span{font-size:12.5px;color:var(--muted)}
.tile.k-ok{border-left-color:var(--ok)} .tile.k-unsure{border-left-color:var(--warn)} .tile.k-bad{border-left-color:var(--bad)}
.headline{display:flex;flex-wrap:wrap;gap:8px 22px;margin:0 0 16px 0;font-size:15px} .headline b{font-variant-numeric:tabular-nums}
.card{display:grid;grid-template-columns:minmax(0,560px) minmax(0,1fr);gap:16px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:0 0 18px 0;scroll-margin-top:12px}
@media (max-width:980px){.card{grid-template-columns:minmax(0,1fr)}}
.card img{max-width:100%;height:auto;border-radius:6px;display:block}
h3{margin:0 0 6px 0;font-size:15px;text-wrap:balance} .sr{color:var(--muted);margin:0 0 8px 0;font-size:12px}
.chip{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;vertical-align:1px;margin-left:6px}
.chip.k-ok{background:var(--yes);color:var(--ok)} .chip.k-unsure{background:var(--unclear);color:var(--warn)} .chip.k-bad{background:var(--no);color:var(--bad)}
.verdict{border-left:4px solid var(--ok);background:var(--code);padding:8px 12px;border-radius:0 6px 6px 0;margin:8px 0 10px 0}
.verdict.k-unsure{border-left-color:var(--warn)} .verdict.k-bad{border-left-color:var(--bad)}
.verdict b{display:block;font-size:12px;letter-spacing:.04em;color:var(--muted);margin-bottom:2px}
p{margin:6px 0} .tw{overflow-x:auto}
table.m{border-collapse:collapse;font-size:12px;margin:6px 0;min-width:100%} table.m th,table.m td{border:1px solid var(--line);padding:3px 7px;text-align:center;white-space:nowrap}
td.cl{text-align:left;white-space:normal;min-width:220px}
td.v-yes{background:var(--yes)} td.v-no{background:var(--no)} td.v-unclear{background:var(--unclear)} td.v-unparsed,td.v-na{background:var(--na)} tr.neg td{font-weight:600}
.ok{color:var(--ok);font-weight:600} .bad{color:var(--bad);font-weight:600} .na{color:var(--muted)} .flip{color:var(--flip)}
ul{margin:2px 0 6px 18px;padding:0} ul.g{columns:2;column-gap:18px} @media (max-width:600px){ul.g{columns:1}}
code{background:var(--code);padding:1px 4px;border-radius:3px;font-size:12px}
a{color:var(--pol-s)}
</style>
"""


def _card(idx: int, r: dict, img_rel: str, note: dict | None, bucket: dict | None) -> str:
    e, g, f = r["expressions"], r["gates"], r["flipped"]
    iids = [i["iid"] for i in r["instances"]]
    cv = r["clause_verdicts"] or {}
    nv = r["neg_clause_verdicts"] or {}
    head_cells = "".join(f"<th>{'T' if i == r['target_iid'] else ('S' if i == r['sibling_iid'] else '')}{i}</th>" for i in iids) + "<th>T·核对者2</th>"
    rows = []
    for k, cl in enumerate(r["clauses"]):
        lb = "" if not r["load_bearing"] or k >= len(r["load_bearing"]) else (" <b>[承重]</b>" if r["load_bearing"][k] else "")
        flag = " <span class='flip'>[被改假]</span>" if f["idx"] == k else ""
        cells = ""
        for i in iids:
            vs = cv.get(str(i)) or []
            v = vs[k] if k < len(vs) else "na"
            cells += f"<td class='v-{v}'>{v if v != 'na' else '-'}</td>"
        v2 = r.get("clause_verdicts_checker2_target") or []
        v2k = v2[k] if k < len(v2) else "na"
        rows.append(f"<tr><td class='cl'>{html.escape(cl)}{flag}{lb}</td>{cells}<td class='v-{v2k}'>{v2k if v2k != 'na' else '-'}</td></tr>")
    if f["clause_neg"]:
        cells = "".join(f"<td class='v-{nv.get(str(i)) or 'na'}'>{nv.get(str(i)) or '-'}</td>" for i in iids)
        rows.append(f"<tr class='neg'><td class='cl'><b>改假后：</b> {html.escape(f['clause_neg'])}</td>{cells}<td class='v-na'>-</td></tr>")
    pol = ""
    for w, v in r["policy"].items():
        pn = v.get("p_null")
        pol += (f"<li><b>{POLICY_ZH.get(w, w)}</b>：{OUTPUT_ZH.get(v.get('output_type'), v.get('output_type'))} → 候选 {v.get('boxed_iid')}（IoU {v.get('boxed_iou')}），"
                f"与目标 IoU {v.get('iou_target')}，p(框) {round(v['p_box'], 4) if v.get('p_box') is not None else '-'}，"
                f"p(拒答) {f'{pn:.1e}' if pn is not None else '-'}</li>")
    lt, ls = r["listener"]["target"], r["listener"]["sibling"]
    listen = (f"e_T → 候选 {lt.get('hit_iid')}（{lt.get('output_type')}，{lt.get('n_points')} 个点）" if lt else "无")
    if ls:
        listen += f"；e_S → 候选 {ls.get('hit_iid')}（{ls.get('output_type')}，{ls.get('n_points')} 个点）"
    gates = "".join(f"<li>{GATE_ZH.get(k, k)}：{_tag(v)}</li>" for k, v in g.items())
    bj = r["blind"]["judge_found_flip"]
    bj_zh = "是" if bj else ("否" if bj is False else "无")
    kind = bucket["kind"] if bucket else "na"
    chip = f"<span class='chip k-{kind}'>{html.escape(bucket['title'].split(' · ')[0])}</span>" if bucket else ""
    if r.get("reflipped"):
        chip += f"<span class='chip k-unsure'>重试改假（{'原改假失败' if r['reflipped'] == 'flip_failed' else '原改假被满足'}）</span>"
    verdict = (f"<div class='verdict k-{kind}'><b>我的看法</b>{html.escape(note['note'])}</div>" if note else "")
    return f"""
<section class='card' id='{r['image_id']}'>
  <div><img src='{img_rel}/{r['image_id']}.jpg' alt='场景 {r['image_id']}' loading='lazy'>
       <p class='sr'>每个候选的放大裁剪（写手和核对者看到的图）：</p>
       <img src='{img_rel}/{r['image_id']}_cands.jpg' alt='候选放大裁剪 {r['image_id']}' loading='lazy'></div>
  <div>
    <h3>#{idx} · {r['image_id']} · {html.escape(r['label'])}（{html.escape(r['category'])}）· {len(iids)} 个候选 · 目标 T={r['target_iid']} · 兄弟 S={r['sibling_iid']}{chip}</h3>
    <p class='sr'>兄弟的选法：{html.escape(r['sibling_reason'] or '')}</p>
    {verdict}
    <p><b>e_T</b>（目标描述）：{html.escape(e['e_T'] or '')}</p>
    <p><b>e_S</b>（兄弟描述）：{html.escape(e['e_S'] or '')}</p>
    <p><b>e_T⁻</b>（改假一条细节后的负样本）：{html.escape(e['e_T_neg'] or '（改假失败，为空）')}<br>
       <span class='flip'>改动：{html.escape(f['from'] or '?')} → {html.escape(f['to'] or '?')}</span></p>
    <div class='tw'><table class='m'><tr><th>细节（核对者 {html.escape(str(r.get('checker_name') or 'Gemma4').split('/')[-1].split('@')[0])}：对该候选成立吗？）</th>{head_cells}</tr>{''.join(rows)}</table></div>
    <p><b>基座 Qwen3-VL-8B（GroundingME 原 prompt）</b></p><ul>{pol}</ul>
    <p><b>听者 Molmo2（只看描述指点）</b>：{listen}</p>
    <p><b>纯文本判官</b>：选了 {r['blind']['judge_letter']}，猜中改句：{bj_zh}</p>
    <p><b>自动门</b></p><ul class='g'>{gates}</ul>
  </div>
</section>"""


def review_body(records: list[dict], img_rel: str, run: str, notes: dict | None = None) -> str:
    """The page content (no document wrapper): what the Artifact tool wraps, and what
    `index.html` wraps for a local browser. `notes` (datagen/reviews/<run>.json) adds the
    reviewer's per-scene verdict and groups the cards by bucket."""
    n = len(records)
    idx_of = {r["image_id"]: i for i, r in enumerate(records)}
    buckets = (notes or {}).get("buckets") or []
    cases = (notes or {}).get("cases") or {}
    by_bucket: dict[str, list[dict]] = {b["id"]: [] for b in buckets}
    rest = []
    for r in records:
        c = cases.get(r["image_id"])
        if c and c.get("bucket") in by_bucket:
            by_bucket[c["bucket"]].append(r)
        else:
            rest.append(r)
    binfo = {b["id"]: b for b in buckets}

    def cnt(k):
        return sum(1 for r in records if r["gates"].get(k))

    # headline numbers
    n_ok = sum(len(by_bucket[b["id"]]) for b in buckets if b["kind"] == "ok")
    n_unsure = sum(len(by_bucket[b["id"]]) for b in buckets if b["kind"] == "unsure")
    n_bad = sum(len(by_bucket[b["id"]]) for b in buckets if b["kind"] == "bad")
    n_neg_measured = sum(1 for r in records if r["gates"].get("negative_usable") is not None)
    headline = ""
    if notes:
        headline = (f"<div class='headline'><span>我判定可用的负样本 <b>{n_ok} / {n}</b></span>"
                    f"<span>存疑 <b>{n_unsure}</b></span><span>不可用 <b>{n_bad}</b></span>"
                    f"<span>自动门全部通过 <b>{cnt('negative_usable')} / {n_neg_measured}</b></span>"
                    f"<span>基座在 e_T 下框错 <b>{cnt('base_wrong_on_target')} / {n}</b></span></div>")
    tiles = "".join(
        f"<a class='tile k-{b['kind']}' href='#b-{b['id']}'><b>{len(by_bucket[b['id']])}</b><span>{html.escape(b['title'])}</span></a>"
        for b in buckets)
    if rest and notes:
        tiles += f"<a class='tile' href='#b-rest'><b>{len(rest)}</b><span>未评</span></a>"
    overview = ""
    if notes:
        paras = "".join(f"<p>{html.escape(t)}</p>" for t in notes.get("summary", []))
        overview = f"<div class='overview'><h2>总评</h2>{paras}<p class='sr'>{html.escape(notes.get('reviewer', ''))}</p></div>"
    legend = ("<p class='legend'>图例：<span class='sw' style='background:var(--target)'></span>粗绿框 = 目标 T · "
              "<span class='sw' style='background:var(--sibling)'></span>粗橙框 = 兄弟 S · <span class='sw' style='background:var(--other)'></span>灰框 = 其他同类候选 · "
              "细框是基座 Qwen3-VL-8B 的输出：<span class='sw' style='background:var(--pol-t)'></span>红 = 看 e_T，<span class='sw' style='background:var(--pol-n)'></span>紫 = 看 e_T⁻，"
              "<span class='sw' style='background:var(--pol-s)'></span>蓝 = 看 e_S · 圆点是听者 Molmo2 的指点（绿 = e_T，橙 = e_S）。"
              "表格每格是核对者对\"这条细节对这个候选成立吗\"的回答；最后一列是第二核对者（Qwen3-VL-8B）对目标的回答。"
              "\"承重\"= 删掉这条细节后听者就指错了。</p>")

    sections = []
    if notes:
        for b in buckets:
            rs = by_bucket[b["id"]]
            cards = "".join(_card(idx_of[r["image_id"]], r, img_rel, cases.get(r["image_id"]), b) for r in rs)
            sections.append(f"<h2 id='b-{b['id']}'>{html.escape(b['title'])}（{len(rs)}）</h2><p class='bdesc'>{html.escape(b['desc'])}</p>{cards or '<p class=sr>无</p>'}")
        if rest:
            cards = "".join(_card(idx_of[r["image_id"]], r, img_rel, None, None) for r in rest)
            sections.append(f"<h2 id='b-rest'>未评（{len(rest)}）</h2>{cards}")
    else:
        sections.append("".join(_card(i, r, img_rel, None, None) for i, r in enumerate(records)))
    sub = (f"{n} 个场景，全部本地模型：写手 Qwen3.5-9B，核对者 Gemma4-12B（第二核对者 Qwen3-VL-8B），"
           f"基座 Qwen3-VL-8B，听者 Molmo2-8B。卡片按我的判断分组；每张卡片顶部是我逐张看图后的看法，下面是流水线的原始输出。")
    return (f"<title>Hard-Sample Review {html.escape(run)}</title>{STYLE}<div class='wrap'>"
            f"<h1>难样本数据审核 · {html.escape(run)}</h1><p class='sub'>{sub}</p>{headline}{overview}"
            f"<div class='tiles'>{tiles}</div>{legend}{''.join(sections)}</div>")


def review_html(records: list[dict], img_rel: str, run: str = "", notes: dict | None = None) -> str:
    """A standalone document for a local browser."""
    return ("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'></head><body>"
            + review_body(records, img_rel, run, notes) + "</body></html>")


def run(run: str, primary: str = "gemma4") -> None:
    root = C.run_root(run)
    records = build_records(root, primary)
    sfx = "" if primary == "gemma4" else f"-{primary}"
    with open(root / f"records{sfx}.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    table = gate_table(records)
    (root / f"gates{sfx}.md").write_text(f"# datagen gates · run {run} · primary checker {primary} · {len(records)} scenes\n\n{table}\n", encoding="utf-8")
    print(table)

    scenes = C.by_image(C.read_jsonl(root / "scenes.jsonl"))
    rev = root / f"review{sfx}"
    (rev / "img").mkdir(parents=True, exist_ok=True)
    for r in records:
        render(scenes[r["image_id"]], r, rev / "img" / f"{r['image_id']}.jpg")
        render_sheet(scenes[r["image_id"]], r, rev / "img" / f"{r['image_id']}_cands.jpg")
    notes = load_notes(run)
    page_run = run + sfx
    (rev / "index.html").write_text(review_html(records, "img", page_run, notes), encoding="utf-8")
    (rev / "artifact.html").write_text(review_body(records, "img", page_run, notes), encoding="utf-8")
    win = paths.REPO_ROOT / "review" / page_run
    # replace only what this stage generates; never wipe reviewer files kept next to them
    # (2026-09-22: a wholesale rmtree deleted 129 reviewer notes stored under review/<run>/packets)
    win.mkdir(parents=True, exist_ok=True)
    if (win / "img").exists():
        shutil.rmtree(win / "img")
    shutil.copytree(rev / "img", win / "img")
    for name in ("index.html", "artifact.html"):
        shutil.copy(rev / name, win / name)
    shutil.copy(root / f"gates{sfx}.md", win / "gates.md")
    print(f"\nrecords: {root / ('records' + sfx + '.jsonl')}\nreview:  {win / 'index.html'}")
