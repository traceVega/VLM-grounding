"""COA — commit / observe / accuse: the per-candidate audit head of the propose -> audit protocol.

Turn 1 (unchanged): conditions + one image_zoom_in tool call with <= 6 candidate boxes.
Turn 2, once per candidate (isolated): the full scene with that candidate outlined in red + its
close-up; the model writes one typed line per clause  "j. [type] claimed | seen: value | verdict"
and a <verdict>.  The harness decides: a candidate FITS iff it has no named mismatch on a
non-relation clause (seen non-empty and different from the claim; text clauses by string
containment); among fitting candidates the first proposed wins; none -> null.  Relation and
unsure lines never reject.  Bare "no" does not exist in this head: a false accusation has to
name a concrete counter-value, which labels can penalise.
"""

from __future__ import annotations

import json
import re

TYPES = ("color", "text", "part", "state", "count", "material", "shape", "relation")
_COLORS = r"\b(red|blue|green|yellow|orange|purple|pink|brown|black|white|gr[ae]y|silver|gold(?:en)?|beige|tan|navy|teal|maroon|dark|light|pale|bright|striped|checkered|plaid|colou?r)\b"
_REL = r"\b(left|right|front|behind|back|next to|beside|between|near|above|below|under|over|on top of|center|centre|middle|foreground|background|closest|farthest|nearest|leftmost|rightmost|second from|first from|row|edge of the (?:frame|image)|side of the (?:frame|image))\b"
_TEXT = r"\"|'|\b(text|reads?|says?|written|letter(?:s|ing)?|word|number|digit|logo|label|sign|printed|writes?)\b|\d"
_COUNT = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|single|pair|several|multiple|many|no|only|both|\d+)\b\s+\w+"
_MATERIAL = r"\b(wooden|wood|metal(?:lic)?|plastic|glass|leather|fabric|cloth|stone|brick|concrete|paper|cardboard|rubber|steel|iron|fur|wool|denim|ceramic)\b"
_STATE = r"\b(open|closed|sitting|standing|lying|walking|running|holding|carrying|wearing|looking|facing|parked|moving|turned|raised|folded|bent|leaning|resting|hanging|tilted|broken|wet|dry|lit|on|off)\b"
_SHAPE = r"\b(round|square|rectangular|oval|circular|triangular|curved|straight|pointed|flat|tall|short|long|wide|narrow|thin|thick|small|large|big|tiny|huge)\b"


def clause_type(text: str) -> str:
    t = text.lower()
    if re.search(_REL, t):
        return "relation"
    if re.search(_TEXT, t) and not re.search(r"\b(one|two|three|four)\b", t):
        return "text"
    if re.search(_COLORS, t):
        return "color"
    if re.search(_COUNT, t):
        return "count"
    if re.search(_MATERIAL, t):
        return "material"
    if re.search(_STATE, t):
        return "state"
    if re.search(_SHAPE, t):
        return "shape"
    return "part"


_LINE = re.compile(r"^\s*(\d+)\.\s*(?:\[(\w+)\]\s*)?(.*?)\s*\|\s*seen:\s*(.*?)\s*\|\s*(match|mismatch|unsure)\b", re.I | re.M)
EXEMPTIONS = False  # True = the GME-motivated patches (relation never rejects, text by containment); off in the general head
_AUDIT = re.compile(r"<audit>(.*?)</audit>", re.S | re.I)
_VERDICT = re.compile(r"<verdict>(.*?)</verdict>", re.S | re.I)


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    s = re.sub(r"\b(a|an|the|is|are|it|its|with|of|in|on|and|has|have|this|that|there|very|some|slightly)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_audit(text: str) -> dict:
    """{'lines': [{idx, type, claimed, seen, verdict}], 'verdict': str|None, 'format_ok': bool}"""
    m = _AUDIT.search(text or "")
    body = m.group(1) if m else (text or "")
    lines = []
    for lm in _LINE.finditer(body):
        typ = (lm.group(2) or "").lower()
        lines.append({"idx": int(lm.group(1)), "type": typ if typ in TYPES else None, "claimed": lm.group(3).strip(),
                      "seen": lm.group(4).strip(), "verdict": lm.group(5).lower()})
    vm = _VERDICT.search(text or "")
    return {"lines": lines, "verdict": vm.group(1).strip() if vm else None, "format_ok": bool(m) and len(lines) > 0}


NEG = {"not", "no", "none", "without", "instead", "rather", "than", "different", "missing", "absent", "visible", "unclear", "cannot", "can", "see", "seen", "tell"}


def named_value(seen: str, claimed: str) -> bool:
    """The seen value names something beyond the claim: its content words (negations and stop words removed) are
    not a subset of the claim's content words.  'not visible', 'different' and copies of the claim are not named."""
    ws = set(norm(seen).split()) - NEG
    wc = set(norm(claimed).split())
    return bool(ws) and not ws <= wc


def named_mismatches(lines: list[dict]) -> list[dict]:
    """Lines that reject the candidate: non-relation, verdict mismatch, a seen value that is non-empty and
    differs from the claim; text lines by containment of the quoted claim in the transcription."""
    out = []
    for ln in lines:
        if ln["verdict"] != "mismatch" or (EXEMPTIONS and ln["type"] == "relation"):
            continue
        seen, claimed = norm(ln["seen"]), norm(ln["claimed"])
        if not named_value(ln["seen"], ln["claimed"]):
            continue
        if EXEMPTIONS and ln["type"] == "text":
            q = re.findall(r"[\"']([^\"']{1,60})[\"']", ln["claimed"]) or re.findall(r"(?:[A-Z][A-Z0-9]{1,}|\d+)", ln["claimed"])
            if q and all(norm(x) in seen for x in q):
                continue  # the transcription contains the claimed string / number: not a mismatch
        out.append(ln)
    return out


def decide(audits: list[dict], boxes: list) -> tuple[list | None, int | None, list[bool]]:
    """(answer box, chosen candidate index, fits per candidate) for candidates in proposal order."""
    fits = []
    for a in audits:
        fits.append(bool(a.get("format_ok")) and not named_mismatches(a["lines"]))
    for k, f in enumerate(fits):
        if f:
            return boxes[k], k, fits
    return None, None, fits


def render_audit(m: dict, row: dict, rationale_of) -> str:
    """Label audit block for one label row: verdict yes -> match, no -> mismatch, unclear -> unsure; the
    seen value is the flipped cell's exact original value, else the kept rationale, else the claim."""
    lines = []
    first_mis = None
    for j, (cond, v) in enumerate(zip(m["conditions"], row["verdicts"])):
        typ = clause_type(cond)
        pre = (m.get("seen_pre") or {}).get(str(j)) or (row.get("seen") or {}).get(str(j))  # matrix-level, then per-row observed values (GME-style data)
        if pre:
            seen = pre
        elif m.get("flipped_idx") == j and row["iid"] == m.get("first_iid") and m.get("flip_from") and v == "no":
            seen = m["flip_from"] + (f", not {m['flip_to']}" if m.get("flip_to") else "")
        else:
            seen = rationale_of(row, j) or cond
        verdict = {"yes": "match", "no": "mismatch", "unclear": "unsure"}.get(v, "unsure")
        if verdict == "mismatch" and not named_value(seen, cond):
            verdict = "unsure"  # a 'no' without an observed counter-value cannot be taught as a named mismatch
        if verdict == "mismatch" and first_mis is None:
            first_mis = (j + 1, seen, cond)
        lines.append(f"{j + 1}. {cond} | seen: {seen} | {verdict}")
    verdict = "fits" if first_mis is None else f"mismatch on {first_mis[0]}: {first_mis[1]}, not {first_mis[2]}"
    return "<audit>\n" + "\n".join(lines) + "\n</audit>\n<verdict>" + verdict + "</verdict>"


def audits_text(audits_raw: list[str], boxes_rel: list) -> str:
    """The K audit blocks with their boxes, as shown to the model in the answer step."""
    parts = []
    for k, (t, b) in enumerate(zip(audits_raw, boxes_rel), 1):
        parts.append(f"Candidate {k}, box {json.dumps(b)}:\n{t.strip()}")
    return "\n\n".join(parts)


def synthetic_table(audits: list[dict], boxes_rel: list, fits: list[bool], n_cond: int) -> str:
    """A <candidates> table equivalent to the audits (n_cond cells per row so the trace scorer accepts it): a fitting
    row is all yes; an excluded row carries the first named mismatch as a 'no' cell with its seen value."""
    rows = []
    for k, (a, b, f) in enumerate(zip(audits, boxes_rel, fits), 1):
        cells = [f"{j}: yes" for j in range(1, n_cond + 1)]
        if not f:
            nm = named_mismatches(a["lines"]) if a.get("format_ok") else []
            if nm:
                j = min(max(1, nm[0]["idx"]), n_cond)
                cells[j - 1] = f"{j}: no — {nm[0]['seen']}, not {nm[0]['claimed']}"
            else:
                cells[0] = "1: no — unparsable audit"
        rows.append(f'<object id="{k}" bbox="{json.dumps(b)}">' + "; ".join(cells) + (" -> fits" if f else " -> excluded") + "</object>")
    return "<candidates>\n" + "\n".join(rows) + "\n</candidates>\n"


_BANK = None


def clause_bank():
    """(run, group) -> {clause: {iid: verdict}} over all checked training items (lazy, cached)."""
    global _BANK
    if _BANK is None:
        from pathlib import Path

        from train import data as D
        from train.long_items import SOURCES, clause_bank as _cb

        items = []
        for f in SOURCES:
            if Path(f).is_file():
                items.extend(D.load_items(Path(f)))
        _BANK = {k: v["clauses"] for k, v in _cb(items).items()}
    return _BANK


def seen_from_bank(origin, claim: str) -> str | None:
    """For a 'no' cell without a rationale: a checked yes-clause of the same instance that shares a content word
    with the claim and names something beyond it (e.g. claim 'wearing a dark blue cap' -> 'wearing a red cap')."""
    if not origin or len(origin) < 3:
        return None
    clauses = clause_bank().get((origin[0], origin[1]))
    if not clauses:
        return None
    wc = set(norm(claim).split()) - NEG
    best = None
    for c, vs in clauses.items():
        if vs.get(origin[2]) != "yes" or c == claim:
            continue
        w = set(norm(c).split()) - NEG
        if (w & wc) and named_value(c, claim):
            if best is None or len(w & wc) > best[0]:
                best = (len(w & wc), c)
    return best[1] if best else None
