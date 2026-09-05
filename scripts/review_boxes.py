"""Where did the box go? The 182 verified-clean removals, judged by eye.

    python scripts/review_boxes.py --build     # render the panels once
    python scripts/review_boxes.py --serve     # then judge them in a browser

The removal review answered "did the object leave the picture". This one asks
the question that decides the paper's headline number, on the items where the
answer was yes and the model had the ORIGINAL box right:

    the referent is gone, so `none` is the only correct answer.
    The model gave a box instead. **Is that box a legitimate answer?**

`N+` is the relation: after the removal the output must be valid -- `none`, or a
new region a judge can confirm.  A box somewhere else is not automatically a
pass, and a box still covering the hole is the hardest failure there is.  The
automatic statistics cannot separate those, because the difference is about
meaning rather than geometry:

    same-box (IoU >= 0.5 with its own original box)   13%
    box still on the hole (IoU >= 0.3 with R)         17%
    gave a box when `none` was the only valid answer  52%

The headline lives somewhere between 17% and 52%, and only eyes can place it.

The panel shows the model's ORIGINAL box in green and the hole in amber; after
the removal, the same original box as a dashed ghost with the new box in red, so
a drift is visible without measuring it.  The zoom underneath is on the new box,
because "is there a {label} in there" is the whole question.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

PANEL = 520
ZOOM = 330
HOLE_COLOUR = (0, 200, 255)     # amber: what the editor removed
ORIG_COLOUR = (60, 200, 60)     # green: the model's box on the ORIGINAL
NEW_COLOUR = (60, 60, 235)      # red:   its box after the removal
GHOST_COLOUR = (225, 225, 225)  # dashed: where the original box was

#: One question, because only one thing here needs eyes: **is there a real
#: instance inside the red box?**
#:
#:   correct   the model declined, or its new box holds a genuine instance
#:   nothing   the new box holds nothing matching -- confabulated
#:   drop      the item should not be in this set at all: the removal was not
#:             clean after all, or the box never held the whole referent.  An
#:             escape hatch for the earlier review, which judged the hole
#:             rather than the answer and could not see this
#:   unsure
#:
#: Where the box landed is NOT asked.  A box still covering the hole and a box
#: that jumped elsewhere are both `N+` failures, and IoU with R separates them
#: exactly, so asking a person to do it only buys a second opinion to reconcile.
#:
#: There is no "the position was inferable" label, and that is deliberate.  When
#: the head is gone and the neck remains, "is there a head" is still no; reading
#: the head's place off the neck is a prior, which is the very thing this audit
#: is about.  Those items are not defective, they are the subset where a
#: pixel-driven model and a prior-driven model differ most.
LABELS = {"1": "correct", "2": "nothing", "3": "drop", "0": "unsure"}


def review_dir() -> Path:
    from shared import paths

    return paths.DATA_ROOT / "box_review"


def items() -> list[dict]:
    """The 182: human-clean removals where the ORIGINAL box was right.

    Boxed items come first, because that is where the judgement lives; the ones
    the model declined follow, and are one keypress each.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from idea91.masks import bbox_xyxy, decode_rle
    from shared import paths

    root = Path(__file__).resolve().parents[1]
    with open(root / "data/human/openimages_removal_labels.csv", encoding="utf-8") as fh:
        human = {r["image_id"]: r["label"].strip() for r in csv.DictReader(fh)}

    pilot = json.loads((root / "tables/pilot_does_it_look.json").read_text())["records"]

    shards = sorted((paths.EDITS_ROOT / "index").glob("shard-*.parquet"))
    table = pa.concat_tables([pq.read_table(s) for s in shards])
    col = {n: table.column(n).to_pylist() for n in table.column_names}
    edits = {col["image_id"][i]: {
        "window_xyxy_px": col["window_xyxy_px"][i],
        "window_path": col["window_path"][i],
        "mask_rle": col["mask_rle"][i],
    } for i in range(table.num_rows) if col["operator"][i] == "REMOVE"}

    out = []
    for rec in pilot:
        iid = rec["image_id"]
        if human.get(iid) != "clean" or (rec["original_iou_gt"] or 0) < 0.5:
            continue
        if iid not in edits:
            continue
        out.append({**rec, **edits[iid],
                    "hole_box": list(bbox_xyxy(decode_rle(edits[iid]["mask_rle"])))})
    out.sort(key=lambda r: (r["removed_type"] == "none", r["image_id"]))
    return out


def iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def build() -> None:
    from idea91.edits.build import load_edited, read_image
    from shared import paths

    out_dir = review_dir() / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = paths.RAW / "openimages" / "images"
    rows = items()
    print(f"rendering {len(rows)} panels -> {out_dir}")

    index = []
    for n, item in enumerate(rows, 1):
        path = out_dir / f"{item['image_id']}.jpg"
        if not path.exists():
            try:
                original = read_image(image_dir / f"{item['image_id']}.jpg")
                removed = load_edited(original, item, paths.EDITS_ROOT)
            except Exception as exc:
                print(f"  {item['image_id']}: {type(exc).__name__}: {exc}")
                continue
            cv2.imwrite(str(path), _panel(original, removed, item),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        index.append({
            "image_id": item["image_id"],
            "label": item["expr"],
            "declined": item["removed_type"] == "none",
            "same_box": round(iou(item.get("removed_box"), item.get("original_box")), 3),
            "on_hole": round(iou(item.get("removed_box"), item["hole_box"]), 3),
        })
        if n % 50 == 0:
            print(f"  [{n}/{len(rows)}]", flush=True)

    (review_dir() / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    boxed = sum(1 for i in index if not i["declined"])
    print(f"\n{len(index)} panels ready  ({boxed} with a box, {len(index) - boxed} declined)")
    print("now:  python scripts/review_boxes.py --serve")


def _fit(image: np.ndarray, side: int) -> tuple[np.ndarray, float]:
    """Fit into a `side`-wide box, keeping only the rows the image uses.

    Padding to a square wastes up to a fifth of the vertical space on a
    landscape photo, and this panel is looked at 182 times.
    """
    h, w = image.shape[:2]
    scale = side / max(h, w)
    out = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = np.full((out.shape[0], side, 3), 250, np.uint8)
    canvas[:, : out.shape[1]] = out
    return canvas, scale


def _rect(canvas, box, scale, colour, dashed=False, thick=2) -> None:
    if not box:
        return
    x0, y0, x1, y1 = (int(v * scale) for v in box)
    if not dashed:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thick)
        return
    for x in range(x0, max(x1, x0 + 1), 14):
        cv2.line(canvas, (x, y0), (min(x + 7, x1), y0), colour, 2)
        cv2.line(canvas, (x, y1), (min(x + 7, x1), y1), colour, 2)
    for y in range(y0, max(y1, y0 + 1), 14):
        cv2.line(canvas, (x0, y), (x0, min(y + 7, y1)), colour, 2)
        cv2.line(canvas, (x1, y), (x1, min(y + 7, y1)), colour, 2)


def _crop(image: np.ndarray, box, pad: float = 1.5) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(max(x1 - x0, y1 - y0) * pad, 64)
    a = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    b = min(w, int(cx + side / 2)), min(h, int(cy + side / 2))
    piece = image[a[1]:b[1], a[0]:b[0]]
    if piece.size == 0:
        piece = image
    return cv2.resize(piece, (ZOOM, ZOOM), interpolation=cv2.INTER_CUBIC)


def _panel(original: np.ndarray, removed: np.ndarray, item: dict) -> np.ndarray:
    o = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    r = cv2.cvtColor(removed, cv2.COLOR_RGB2BGR)
    hole = item["hole_box"]
    orig_box, new_box = item.get("original_box"), item.get("removed_box")

    left, scale = _fit(o, PANEL)
    _rect(left, hole, scale, HOLE_COLOUR)
    _rect(left, orig_box, scale, ORIG_COLOUR, thick=3)

    right, _ = _fit(r, PANEL)
    _rect(right, hole, scale, HOLE_COLOUR)
    _rect(right, orig_box, scale, GHOST_COLOUR, dashed=True)
    _rect(right, new_box, scale, NEW_COLOUR, thick=3)
    if new_box is None:
        cv2.putText(right, "DECLINED", (14, PANEL - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (80, 220, 120), 2, cv2.LINE_AA)
    top = np.hstack([left, right])

    # The zoom is on the new box, because that is what is being judged; with no
    # new box there is nothing to judge, so it falls back to the hole.
    bottom = np.hstack([_crop(o, orig_box or hole), _crop(r, new_box or hole)])
    lpad = (top.shape[1] - bottom.shape[1]) // 2
    bottom = np.hstack([
        np.full((ZOOM, lpad, 3), 250, np.uint8), bottom,
        np.full((ZOOM, top.shape[1] - bottom.shape[1] - lpad, 3), 250, np.uint8),
    ])

    strip = np.full((30, top.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, "BEFORE   green = the model's box", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    after = ("AFTER   the model DECLINED" if new_box is None else
             f"AFTER   red = its new box     IoU with its old box "
             f"{iou(new_box, orig_box):.2f}     with the hole {iou(new_box, hole):.2f}")
    cv2.putText(strip, after, (PANEL + 8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

    foot = np.full((26, top.shape[1], 3), 255, np.uint8)
    cv2.putText(foot, "zoom: the model's ORIGINAL box", (lpad + 6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(foot, "zoom: what it points at NOW" if new_box is not None
                else "zoom: the hole", (lpad + ZOOM + 6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
    return np.vstack([strip, top, bottom, foot])


# --- the server -------------------------------------------------------------

PAGE = """<!doctype html><meta charset=utf-8><title>box review</title>
<style>
 body{{margin:0;background:#111;color:#eee;font:14px system-ui;display:flex;
      flex-direction:column;height:100vh;overflow:hidden}}
 header{{padding:6px 12px;display:flex;gap:16px;align-items:center;background:#1b1b1b}}
 .q{{font-size:16px;font-weight:600}} .q b{{color:#ffd479}}
 .tag{{background:#2a4;color:#052;border-radius:4px;padding:2px 8px;font-weight:700;
       white-space:nowrap}}
 .tag.box{{background:#a33;color:#fee}}
 .bar{{flex:1;height:6px;background:#333;border-radius:3px;overflow:hidden}}
 .bar i{{display:block;height:100%;background:#4c9;width:{pct}%}}
 img{{flex:1;min-height:0;object-fit:contain;background:#111}}
 footer{{padding:7px 12px;background:#1b1b1b;display:flex;gap:16px;font-size:12.5px;
         flex-wrap:wrap}}
 kbd{{background:#333;border-radius:4px;padding:2px 7px;font-weight:700}}
 .done{{text-align:center;padding:60px;font-size:18px;line-height:1.8}}
</style>
<header>
  <span class="tag {cls}">{state}</span>
  <span class=q>The <b>{label}</b> is gone. Is the model's answer acceptable?</span>
  <div class=bar><i></i></div>
  <span>{done} / {total}</span>
</header>
<img src="/panel/{image_id}?v={done}">
<footer>
  <span><kbd>1</kbd> yes &mdash; declined, or the red box holds a real one</span>
  <span><kbd>2</kbd> no &mdash; nothing matching is in the red box</span>
  <span><kbd>3</kbd> drop it &mdash; not cleanly removed after all, or the box
        never held the whole thing</span>
  <span><kbd>0</kbd> unsure</span>
  <span><kbd>&larr;</kbd> undo</span>
</footer>
<script>
 const K = {keys};
 const send = (l) => fetch('/label', {{method:'POST', headers:{{'Content-Type':'application/json'}},
     body: JSON.stringify({{image_id:'{image_id}', label:l}})}}).then(()=>location.reload());
 addEventListener('keydown', e => {{
   if (K[e.key]) send(K[e.key]);
   else if (e.key === 'ArrowLeft' || e.key === 'Backspace')
     fetch('/undo', {{method:'POST'}}).then(()=>location.reload());
 }});
</script>"""


def serve(port: int) -> None:
    import http.server
    import socketserver

    root = review_dir()
    index = json.loads((root / "index.json").read_text())
    labels_path = root / "box_labels.csv"

    def read_labels() -> dict[str, str]:
        if not labels_path.is_file():
            return {}
        with open(labels_path, encoding="utf-8") as fh:
            return {r["image_id"]: r["label"] for r in csv.DictReader(fh)}

    def append(image_id: str, label: str) -> None:
        new = not labels_path.is_file()
        with open(labels_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new:
                writer.writerow(["image_id", "label"])
            writer.writerow([image_id, label])

    def drop_last() -> None:
        if not labels_path.is_file():
            return
        rows = labels_path.read_text(encoding="utf-8").splitlines()
        if len(rows) > 1:
            labels_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: A003 - quiet
            pass

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/panel/"):
                name = self.path.split("/panel/")[1].split("?")[0]
                path = root / "panels" / f"{name}.jpg"
                if not path.is_file():
                    self.send_error(404)
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            done = read_labels()
            todo = [i for i in index if i["image_id"] not in done]
            if not todo:
                by_id = {i["image_id"]: i for i in index}
                tally = {}
                for value in done.values():
                    tally[value] = tally.get(value, 0) + 1
                # The geometry the review deliberately did not ask for: of the
                # failures, how many still cover the hole.
                fails = [by_id[k] for k, v in done.items() if v == "nothing"]
                on_hole = sum(1 for f in fails if f["on_hole"] >= 0.3)
                kept = sum(tally.get(v, 0) for v in ("correct", "nothing"))
                lines = "<br>".join(
                    f"<b>{tally.get(v, 0)}</b> &nbsp; {v}"
                    for v in ("correct", "nothing", "drop", "unsure"))
                rate = (f"<br><br>N+ failure &nbsp; <b>{len(fails)}/{kept}</b> = "
                        f"<b>{len(fails)/max(kept,1):.0%}</b><br>"
                        f"of those, still on the hole {on_hole}, moved elsewhere "
                        f"{len(fails)-on_hole}")
                body = (f"<div class=done>All {len(index)} reviewed.<br><br>{lines}"
                        f"{rate}<br><br>{labels_path}</div>").encode()
            else:
                item = todo[0]
                body = PAGE.format(
                    label=item["label"], image_id=item["image_id"],
                    state="DECLINED" if item["declined"] else "GAVE A BOX",
                    cls="" if item["declined"] else "box",
                    done=len(done), total=len(index),
                    pct=round(100 * len(done) / max(len(index), 1), 1),
                    keys=json.dumps(LABELS),
                ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            if self.path == "/undo":
                drop_last()
            else:
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"])).decode())
                append(payload["image_id"], payload["label"])
            self.send_response(204)
            self.end_headers()

    boxed = sum(1 for i in index if not i["declined"])
    print(f"{len(index)} panels ({boxed} with a box first, then {len(index) - boxed} "
          f"declines), {len(read_labels())} already reviewed")
    print(f"\n  open  http://127.0.0.1:{port}\n")
    print(f"  labels append to {labels_path}\n")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8901)
    args = ap.parse_args()
    if args.build:
        build()
    if args.serve:
        serve(args.port)
    if not (args.build or args.serve):
        ap.error("pass --build then --serve")


if __name__ == "__main__":
    main()
