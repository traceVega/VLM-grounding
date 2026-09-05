"""Human review of removal quality, built for speed.

    python scripts/review_removals.py --build     # render the review panels once
    python scripts/review_removals.py --serve     # then judge them in a browser

One question per item: **did the object actually leave the picture?**  The
automatic verifier cannot answer it -- checked against the author's own labels
on 32 cases it agreed 28% of the time, rejecting clean removals 60% of the time
and accepting failed ones 92% of the time -- so this is done by eye, once, and
the result becomes the dataset everything downstream runs on.

The panel is laid out for the judgement rather than for looking pretty: the full
images give context, and the zoomed pair underneath shows the hole close enough
to see whether the object is gone or merely blurred.  That distinction is the
one that matters and the one the verifier kept missing: a blurred plate is still
a plate, and a model that finds it is right.

Keys are one-handed and the file is written through on every press, so a
closed tab costs nothing and the work resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PANEL = 460
ZOOM = 300
HOLE_COLOUR = (0, 200, 255)
#: ``bad_item`` is not a judgement about the removal, it is a judgement about
#: the *item*, and it exists because the pool's filter cannot be trusted.
#:
#: The filter counted instances carrying the same label in the instance store,
#: which is a count over annotations rather than over what is in the picture.
#: OpenImages labels are hierarchical and often box a part: the first panel
#: reviewed was labelled "Toy" with a box on a fallen shell, while the whole
#: radio-controlled car -- itself a toy -- stood beside it, never inside the
#: box. The shell was removed cleanly, and "is the Toy gone?" is still no.
#:
#: Such an item cannot be scored either way. It is not a failed removal, and
#: calling it clean would put an unanswerable question into the dataset, so it
#: leaves the pool entirely.
LABELS = {"1": "clean", "2": "not_clean", "3": "bad_item", "0": "unsure"}


def review_dir() -> Path:
    from shared import paths

    return paths.DATA_ROOT / "removal_review"


def items() -> list[dict]:
    """The pool: mask REMOVE edits whose class occurs once in the image.

    Same filter the pilot used, and for the same reason -- with two instances of
    the class the label is not a referring expression and the model cannot be
    marked wrong for pointing at the other one.
    """
    import sys

    sys.argv = [sys.argv[0]]
    from scripts.pilot_does_it_look import remove_edits, unambiguous_referents

    referents, edits = unambiguous_referents(), remove_edits()
    out = []
    for image_id in sorted(set(referents) & set(edits)):
        out.append({
            "image_id": image_id,
            "label": referents[image_id]["label"],
            **edits[image_id],
        })
    return out


def build() -> None:
    from idea91.edits.build import load_edited, read_image
    from idea91.masks import bbox_xyxy, decode_rle
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
                hole = bbox_xyxy(decode_rle(item["mask_rle"]))
            except Exception as exc:
                print(f"  {item['image_id']}: {type(exc).__name__}: {exc}")
                continue
            cv2.imwrite(str(path), _panel(original, removed, hole),
                        [cv2.IMWRITE_JPEG_QUALITY, 86])
        index.append({"image_id": item["image_id"], "label": item["label"]})
        if n % 100 == 0:
            print(f"  [{n}/{len(rows)}]", flush=True)

    (review_dir() / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\n{len(index)} panels ready")
    print(f"now:  python scripts/review_removals.py --serve")


def _fit(image: np.ndarray, side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = side / max(h, w)
    out = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = np.full((side, side, 3), 250, np.uint8)
    canvas[: out.shape[0], : out.shape[1]] = out
    return canvas, scale


def _crop(image: np.ndarray, hole, pad: float = 1.4) -> np.ndarray:
    """The hole with room around it, so the seam is visible too."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = hole
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * pad
    side = max(side, 48)
    a = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    b = min(w, int(cx + side / 2)), min(h, int(cy + side / 2))
    piece = image[a[1]:b[1], a[0]:b[0]]
    if piece.size == 0:
        piece = image
    return cv2.resize(piece, (ZOOM, ZOOM), interpolation=cv2.INTER_CUBIC)


def _panel(original: np.ndarray, removed: np.ndarray, hole) -> np.ndarray:
    o = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    r = cv2.cvtColor(removed, cv2.COLOR_RGB2BGR)
    top_l, scale = _fit(o, PANEL)
    top_r, _ = _fit(r, PANEL)
    for canvas in (top_l, top_r):
        x0, y0, x1, y1 = (int(v * scale) for v in hole)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), HOLE_COLOUR, 2)
    top = np.hstack([top_l, top_r])

    bottom = np.hstack([_crop(o, hole), _crop(r, hole)])
    pad = (top.shape[1] - bottom.shape[1]) // 2
    bottom = np.hstack([
        np.full((ZOOM, pad, 3), 250, np.uint8), bottom,
        np.full((ZOOM, top.shape[1] - bottom.shape[1] - pad, 3), 250, np.uint8),
    ])
    strip = np.full((26, top.shape[1], 3), 255, np.uint8)
    for text, x in (("BEFORE", 8), ("AFTER", PANEL + 8)):
        cv2.putText(strip, text, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([strip, top, bottom])


# --- the server -------------------------------------------------------------

PAGE = """<!doctype html><meta charset=utf-8><title>removal review</title>
<style>
 body{{margin:0;background:#111;color:#eee;font:14px system-ui;display:flex;
      flex-direction:column;height:100vh;overflow:hidden}}
 header{{padding:6px 12px;display:flex;gap:18px;align-items:center;background:#1b1b1b}}
 .q{{font-size:17px;font-weight:600}} .q b{{color:#ffd479}}
 .bar{{flex:1;height:6px;background:#333;border-radius:3px;overflow:hidden}}
 .bar i{{display:block;height:100%;background:#4c9;width:{pct}%}}
 img{{flex:1;min-height:0;object-fit:contain;background:#111}}
 footer{{padding:8px 12px;background:#1b1b1b;display:flex;gap:20px;font-size:13px}}
 kbd{{background:#333;border-radius:4px;padding:2px 7px;font-weight:700}}
 .done{{text-align:center;padding:60px;font-size:19px}}
</style>
<header>
  <span class=q>After the edit, is there still a <b>{label}</b> in the picture?</span>
  <div class=bar><i></i></div>
  <span>{done} / {total} &nbsp; clean {clean} &nbsp; dropped {bad}</span>
</header>
<img src="/panel/{image_id}?v={done}">
<footer>
  <span><kbd>1</kbd> no &mdash; it is gone, removed cleanly</span>
  <span><kbd>2</kbd> yes &mdash; still there, or only blurred</span>
  <span><kbd>3</kbd> bad item &mdash; the box never held the whole thing,
        or another one was always there</span>
  <span><kbd>0</kbd> unsure</span>
  <span><kbd>&larr;</kbd> undo</span>
</footer>
<script>
 const send = (l) => fetch('/label', {{method:'POST', headers:{{'Content-Type':'application/json'}},
     body: JSON.stringify({{image_id:'{image_id}', label:l}})}}).then(()=>location.reload());
 addEventListener('keydown', e => {{
   if (e.key === '1') send('clean');
   else if (e.key === '2') send('not_clean');
   else if (e.key === '3') send('bad_item');
   else if (e.key === '0') send('unsure');
   else if (e.key === 'ArrowLeft' || e.key === 'Backspace')
     fetch('/undo', {{method:'POST'}}).then(()=>location.reload());
 }});
</script>"""


def serve(port: int) -> None:
    import http.server
    import socketserver

    root = review_dir()
    index = json.loads((root / "index.json").read_text())
    labels_path = root / "removal_labels.csv"

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
            clean = sum(1 for v in done.values() if v == "clean")
            bad = sum(1 for v in done.values() if v == "bad_item")
            if not todo:
                body = (f"<div class=done>All {len(index)} reviewed.<br><br>"
                        f"<b>{clean}</b> clean removals &mdash; the dataset.<br>"
                        f"{bad} dropped as bad items.<br><br>"
                        f"{labels_path}</div>").encode()
            else:
                item = todo[0]
                body = PAGE.format(
                    label=item["label"], image_id=item["image_id"],
                    done=len(done), total=len(index), clean=clean, bad=bad,
                    pct=round(100 * len(done) / max(len(index), 1), 1),
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
                    self.rfile.read(int(self.headers["Content-Length"])).decode()
                )
                append(payload["image_id"], payload["label"])
            self.send_response(204)
            self.end_headers()

    done = len(read_labels())
    print(f"{len(index)} panels, {done} already reviewed")
    print(f"\n  open  http://127.0.0.1:{port}\n")
    print("  1 = gone   2 = still there   0 = unsure   <- = undo")
    print(f"  labels append to {labels_path}\n")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8900)
    args = ap.parse_args()
    if args.build:
        build()
    if args.serve:
        serve(args.port)
    if not (args.build or args.serve):
        ap.error("pass --build then --serve")


if __name__ == "__main__":
    main()
