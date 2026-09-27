"""Shared helpers for the datagen stages: paths, resumable JSONL, drawing, parsing."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from idea91 import masks as M
from shared import paths

#: OpenImages labels that are whole objects a person can describe with several
#: checkable details.  Parts, body parts, clothing and "stuff" (Tree, Window,
#: Wheel, Human arm, Clothing ...) are excluded: they dominate the crowded scenes
#: in the bank but cannot carry a five-detail expression.
WHITELIST = {
    "Car", "Person", "Man", "Woman", "Boy", "Girl", "Boat", "Chair", "Bottle", "Table",
    "Bird", "Dog", "Cat", "Horse", "Cow", "Sheep", "Goat", "Pig", "Duck", "Goose",
    "Chicken", "Penguin", "Fish", "Elephant", "Zebra", "Giraffe", "Lion", "Tiger", "Bear",
    "Monkey", "Deer", "Rabbit", "Butterfly", "Bicycle", "Motorcycle", "Bus", "Truck",
    "Airplane", "Van", "Taxi", "Train", "Toy", "Flag", "Balloon", "Umbrella", "Lamp",
    "Cup", "Mug", "Coffee cup", "Wine glass", "Book", "Hat", "Bench", "House", "Cake",
    "Bread", "Cookie", "Traffic sign", "Poster", "Pillow", "Vase", "Candle",
    "Plate", "Bowl", "Handbag", "Backpack", "Suitcase", "Guitar", "Drum", "Sculpture",
    "Statue", "Doll", "Teddy bear", "Tomato", "Apple", "Orange", "Banana", "Strawberry",
    "Mushroom", "Flowerpot", "Houseplant", "Skateboard", "Surfboard", "Helmet",
    "Football", "Tennis ball", "Ball",
}
# Dropped after the first test50 run (2026-09-21): "Mammal" (OpenImages labels people as
# Mammal; the writer then described a person as "the animal"), and the generic labels
# Food / Snack / Fruit / Vegetable / Drink / Furniture / Vehicle / Land vehicle /
# Sports equipment / Tableware, whose head nouns read as "the food item that ...".
GENERIC_LABELS = {"Mammal", "Food", "Snack", "Fruit", "Vegetable", "Drink", "Furniture",
                  "Vehicle", "Land vehicle", "Sports equipment", "Tableware"}

#: Labels whose head noun should read more naturally than the OpenImages name.
HEAD_NOUN = {
    "Land vehicle": "vehicle", "Sports equipment": "piece of sports equipment",
    "Tableware": "piece of tableware", "Furniture": "piece of furniture",
    "Mammal": "animal", "Food": "food item", "Snack": "snack", "Coffee cup": "cup",
}

MIN_SIBLING_AREA = 0.0015  # 0.15% of the image: below this a 1024 px sibling is ~40 px
MIN_TARGET_AREA = 0.01     # a 1024 px target under 1% is ~100 px across: details get hallucinated
MAX_TARGET_ASPECT = 4.0    # slivers (a strip at the frame edge) are not describable objects
MIN_TARGET_AREA_AT_BORDER = 0.04  # a border-touching target must be large enough to be mostly in frame
MIN_CANDIDATES, MAX_CANDIDATES = 3, 8
MAX_NESTING = 0.7          # a candidate mostly inside another candidate is a part or a duplicate
MAX_TARGET_OVERLAP = 0.3   # a candidate overlapping the target this much makes the outline ambiguous
N_CLAUSES_CHOICES = (4, 5, 6)


def run_root(run: str) -> Path:
    root = paths.DATA_ROOT / "datagen" / run
    root.mkdir(parents=True, exist_ok=True)
    return root


def image_path(image_id: str) -> Path:
    return paths.RAW / "openimages" / "images" / f"{image_id}.jpg"


def head_noun(label: str) -> str:
    return HEAD_NOUN.get(label, label.lower())


# --- resumable JSONL ---------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()


def done_keys(path: Path, key=("image_id",)) -> set[tuple]:
    return {tuple(r.get(k) for k in key) for r in read_jsonl(path)}


def by_image(rows: list[dict]) -> dict[str, dict]:
    return {r["image_id"]: r for r in rows}


# --- drawing -----------------------------------------------------------------


def outline(image: Image.Image, box, colour: str = "red", width: int | None = None) -> Image.Image:
    """A copy of the image with one box outlined.  Width scales with the image."""
    img = image.copy()
    w, h = img.size
    width = width or max(3, round(min(w, h) / 200))
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    pad = width
    draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=colour, width=width)
    return img


def closeup(image: Image.Image, box, pad_frac: float = 0.25, min_side: int = 512) -> Image.Image:
    """A crop around the box with 25% context on each side, upscaled so the short side
    is at least ``min_side`` px (a 1024 px scene gives ~100 px objects; the writer and
    the checker both need to see the details they are asked about)."""
    w, h = image.size
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    px, py = bw * pad_frac, bh * pad_frac
    cx0, cy0 = max(0, int(x0 - px)), max(0, int(y0 - py))
    cx1, cy1 = min(w, int(x1 + px) + 1), min(h, int(y1 + py) + 1)
    crop = image.crop((cx0, cy0, cx1, cy1))
    cw, ch = crop.size
    scale = max(1.0, min_side / max(1, min(cw, ch)))
    if scale > 1.0:
        crop = crop.resize((round(cw * scale), round(ch * scale)), Image.LANCZOS)
    return crop


def containment(inner, outer) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def png_bytes(image: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# --- geometry ----------------------------------------------------------------


def box_iou(a, b) -> float:
    return M.box_iou(tuple(a), tuple(b))


def assign_box(box, instances: list[dict], thr: float = 0.5) -> tuple[int | None, float]:
    """Which candidate instance a predicted box lands on (max IoU >= thr), else None."""
    if not box:
        return None, 0.0
    best, best_iou = None, 0.0
    for inst in instances:
        v = box_iou(box, inst["box"])
        if v > best_iou:
            best, best_iou = inst["iid"], v
    return (best, best_iou) if best_iou >= thr else (None, best_iou)


def point_in_instance(point, instances: list[dict]) -> int | None:
    """The candidate whose mask contains the point; falls back to the box."""
    if point is None:
        return None
    x, y = int(round(point[0])), int(round(point[1]))
    for inst in instances:
        if "mask_rle" in inst:
            mask = M.decode_rle(inst["mask_rle"])
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                return inst["iid"]
    for inst in instances:
        x0, y0, x1, y1 = inst["box"]
        if x0 <= x <= x1 and y0 <= y <= y1:
            return inst["iid"]
    return None


# --- text parsing --------------------------------------------------------------

_QUOTES = "\"'“”‘’`"


def clean_line(text: str) -> str:
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return line.strip(_QUOTES).strip()


def parse_decomposition(text: str) -> tuple[str | None, list[str]]:
    head, clauses = None, []
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("HEAD:"):
            head = s[5:].strip().strip(_QUOTES)
        elif s.upper().startswith("CLAUSE:"):
            c = s[7:].strip().strip(_QUOTES).rstrip(".")
            if c:
                clauses.append(c)
    return head, clauses


_CHANGED = re.compile(r"CHANGED:\s*(?P<a>.+?)\s*->\s*(?P<b>.+)$", re.IGNORECASE)


def parse_flip(text: str) -> tuple[str, str | None, str | None]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    expr = lines[0].strip(_QUOTES).strip() if lines else ""
    a = b = None
    for ln in lines[1:]:
        m = _CHANGED.search(ln)
        if m:
            a, b = m.group("a").strip().strip(_QUOTES), m.group("b").strip().strip(_QUOTES)
            break
    return expr, a, b


_OLDNEW = re.compile(r"^(OLD|NEW[123]):\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_flip_v2(text: str) -> tuple[str | None, list[str]]:
    """OLD span and up to three NEW spans from the v2 flip prompt."""
    old, news = None, []
    for m in _OLDNEW.finditer(text):
        key, val = m.group(1).upper(), m.group(2).strip().strip(_QUOTES).strip()
        if key == "OLD":
            old = val
        elif val:
            news.append(val)
    return old, news


def substitute_once(expr: str, old: str, new: str) -> str | None:
    """expr with the single occurrence of old replaced by new; None unless exactly one match."""
    if not old or not new or old.strip().lower() == new.strip().lower():
        return None
    import re

    hits = list(re.finditer(r"(?<!\w)" + re.escape(old.strip()) + r"(?!\w)", expr, flags=re.IGNORECASE))
    if len(hits) != 1:  # exactly one whole-word occurrence ('red' must not match inside 'covered')
        return None
    i, j = hits[0].span()
    return expr[:i] + new + expr[j:]


def align_span(expr: str, clauses: list[str], old: str | None, new: str | None):
    """(old_span, new_span) such that old_span occurs exactly once (whole words) in expr and
    in exactly one clause, and new_span is the matching replacement; None if impossible.
    Handles the model answering in clause form ('has a white car parked nearby' ->
    'has a red car parked nearby') when the expression says '... a white car parked nearby':
    the shared words at both ends are stripped, then the differing core is widened with
    context until it is unique."""
    import re

    if not old or not new:
        return None
    ow, nw = old.strip().strip(".").split(), new.strip().strip(".").split()

    def norm(w: str) -> str:
        return w.strip(",.;:").lower()

    if not ow or not nw or [norm(w) for w in ow] == [norm(w) for w in nw]:
        return None
    p = 0
    while p < min(len(ow), len(nw)) and norm(ow[p]) == norm(nw[p]):
        p += 1
    s = 0
    while s < min(len(ow), len(nw)) - p and norm(ow[len(ow) - 1 - s]) == norm(nw[len(nw) - 1 - s]):
        s += 1
    if p == len(ow) or p == len(nw) or p + s >= len(ow):  # one is a prefix/suffix of the other: an append, not a change
        return None

    def once(span: str) -> bool:
        pat = re.compile(r"(?<!\w)" + re.escape(span) + r"(?!\w)", re.IGNORECASE)
        return len(pat.findall(expr)) == 1 and sum(1 for c in clauses if pat.search(c)) == 1

    # widen the core symmetrically with context words until unique
    for extra in range(0, max(p, s) + 1):
        for left, right in ((extra, extra), (extra + 1, extra), (extra, extra + 1)):
            lo, hi = p - left, len(ow) - s + right
            if lo < 0 or hi > len(ow) or lo >= hi:
                continue
            old_span = " ".join(ow[lo:hi])
            new_span = " ".join(nw[lo:len(nw) - s + right])
            if old_span.lower() == new_span.lower():
                continue
            if once(old_span):
                return old_span, new_span
    return None


def join_clauses(head: str, clauses: list[str]) -> str:
    """'the mug that has a blue stripe, has a chipped handle, and is left of the laptop'."""
    if not clauses:
        return f"the {head}"
    if len(clauses) == 1:
        return f"the {head} that {clauses[0]}"
    return f"the {head} that " + ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


def flipped_index(clauses_pos: list[str], clauses_neg: list[str]) -> int | None:
    """The one position where the negative's decomposition differs from the positive's."""
    if len(clauses_pos) != len(clauses_neg):
        return None
    diff = [i for i, (a, b) in enumerate(zip(clauses_pos, clauses_neg))
            if a.strip().lower() != b.strip().lower()]
    return diff[0] if len(diff) == 1 else None


def first_word(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"[a-zA-Z]+", text.strip().lower())
    return m.group(0) if m else None


def seeded(image_id: str, salt: str = "") -> random.Random:
    return random.Random(f"{image_id}:{salt}")


def yes_no_unclear(text: str | None) -> str:
    w = first_word(text)
    if w in ("yes", "yeah", "yep"):
        return "yes"
    if w in ("no", "nope"):
        return "no"
    if w and w.startswith("unclear"):
        return "unclear"
    return "unparsed"


def np_json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
