"""Human review of the GroundingME removals: did the described object leave?

    python scripts/review_gme.py --build     # render the panels once
    python scripts/review_gme.py --serve     # then judge them, port 8902
    python scripts/review_gme.py --status    # counts, without serving

Same question as the OpenImages review and one new answer. There the referent
was named by a class label; here it is a 39-word description, and after the
removal something *else* in the picture may satisfy it -- the fourth leaf
becomes the third from the left, one of several similar objects is now the only
one left. Such an item cannot test necessity in either direction and leaves the
set, which is what key 3 is for.

The expression is shown in Chinese above the English. **The translation is for
the reviewer and reaches no model**: every experiment renders its prompt from
the English `expr`, per P12, which pins the primary protocol to GroundingME's
own words. The Chinese exists because 333 expressions have to be read at review
speed and that is where the reviewer's time goes.

The dimension is on every panel, so it is always clear which stratum is being
judged -- they behave differently and the reviewer should know which is which.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PANEL = 520
ZOOM = 330
HOLE_COLOUR = (0, 200, 255)   # amber: what the editor removed

#: `other_match` is a judgement about the item, not the removal. With a
#: description rather than a class label, removing the referent can promote
#: another object into satisfying it, and then a box is correct and an
#: abstention is wrong -- the item cannot score either way.
LABELS = {"1": "clean", "2": "not_clean", "3": "other_match", "0": "unsure"}
LABEL_ZH = {"clean": "干净移除", "not_clean": "还在", "other_match": "另有匹配", "unsure": "拿不准"}
DIM_ZH = {"Discriminative": "外观区分", "Spatial": "空间关系", "Limited": "信息有限"}


def review_dir() -> Path:
    from shared import paths

    return paths.DATA_ROOT / "gme_review"


def items() -> list[dict]:
    """The built removals, with their Chinese rendering when one exists."""
    rows = json.loads(Path("tables/gme_removals_index.json").read_text(encoding="utf-8"))
    zh = {}
    path = Path("tables/gme_expr_zh.jsonl")
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                zh[rec["item_id"]] = rec["zh"]
    for row in rows:
        row["zh"] = zh.get(row["item_id"], "")
    # Grouped by dimension so the reviewer stays in one kind of judgement at a
    # time, and the header can say how far through that stratum they are.
    order = {"Discriminative": 0, "Spatial": 1, "Limited": 2}
    rows.sort(key=lambda r: (order.get(r["dimension"], 9), r["item_id"]))
    return rows


def _fit(image: np.ndarray, side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = side / max(h, w)
    out = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = np.full((out.shape[0], side, 3), 250, np.uint8)
    canvas[:, : out.shape[1]] = out
    return canvas, scale


def _crop(image: np.ndarray, box, pad: float = 2.4) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(max(x1 - x0, y1 - y0) * pad, 96)
    a = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    b = min(w, int(cx + side / 2)), min(h, int(cy + side / 2))
    piece = image[a[1]:b[1], a[0]:b[0]]
    if piece.size == 0:
        piece = image
    return cv2.resize(piece, (ZOOM, ZOOM), interpolation=cv2.INTER_CUBIC)


def _panel(original: np.ndarray, edited: np.ndarray, hole) -> np.ndarray:
    o = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    e = cv2.cvtColor(edited, cv2.COLOR_RGB2BGR)
    left, scale = _fit(o, PANEL)
    right, _ = _fit(e, PANEL)
    for canvas in (left, right):
        x0, y0, x1, y1 = (int(v * scale) for v in hole)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), HOLE_COLOUR, 2)
    top = np.hstack([left, right])

    bottom = np.hstack([_crop(o, hole), _crop(e, hole)])
    lpad = (top.shape[1] - bottom.shape[1]) // 2
    bottom = np.hstack([
        np.full((ZOOM, lpad, 3), 250, np.uint8), bottom,
        np.full((ZOOM, top.shape[1] - bottom.shape[1] - lpad, 3), 250, np.uint8)])

    strip = np.full((28, top.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, "BEFORE", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(strip, "AFTER", (PANEL + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (40, 40, 40), 1, cv2.LINE_AA)
    foot = np.full((24, top.shape[1], 3), 255, np.uint8)
    cv2.putText(foot, "zoom on the amber box", (lpad + 6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
    return np.vstack([strip, top, bottom, foot])


def build() -> None:
    from idea91.edits.build import load_edited
    from idea91.masks import bbox_xyxy, decode_rle
    from scripts.build_gme_removals import edits_root
    from shared import paths

    out_dir = review_dir() / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = items()
    print(f"rendering {len(rows)} panels -> {out_dir}", flush=True)

    index = []
    for n, item in enumerate(rows, 1):
        path = out_dir / f"{item['item_id']}.jpg"
        if not path.exists():
            try:
                original = np.asarray(
                    Image.open(paths.DATA_ROOT / item["image_path"]).convert("RGB"))
                edited = load_edited(original, item, edits_root())
                hole = bbox_xyxy(decode_rle(item["mask_rle"]))
                cv2.imwrite(str(path), _panel(original, edited, hole),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
            except Exception as exc:
                print(f"  {item['item_id']}: {type(exc).__name__}: {exc}", flush=True)
                continue
        index.append({k: item[k] for k in ("item_id", "expr", "zh", "dimension", "size_bin")})
        if n % 25 == 0:
            print(f"  [{n}/{len(rows)}]", flush=True)

    (review_dir() / "index.json").write_text(
        json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    counts = collections.Counter(i["dimension"] for i in index)
    print(f"\n{len(index)} panels ready  {dict(counts)}")


# --- the server -------------------------------------------------------------

PAGE = """<!doctype html><meta charset=utf-8><title>GroundingME 移除审核</title>
<style>
 body{{margin:0;background:#111;color:#eee;font:14px/1.5 system-ui,"PingFang SC",
      "Microsoft YaHei",sans-serif;display:flex;flex-direction:column;height:100vh;
      overflow:hidden}}
 header{{padding:8px 14px;background:#1b1b1b;flex:0 0 auto}}
 .top{{display:flex;gap:14px;align-items:center;margin-bottom:6px}}
 .dim{{background:#3a5;color:#032;border-radius:4px;padding:2px 9px;font-weight:700;
       white-space:nowrap}}
 .bar{{flex:1;height:6px;background:#333;border-radius:3px;overflow:hidden}}
 .bar i{{display:block;height:100%;background:#4c9;width:{pct}%}}
 .zh{{font-size:16px;color:#ffd479;max-height:4.6em;overflow-y:auto}}
 .en{{font-size:11.5px;color:#888;margin-top:3px;max-height:3.4em;overflow-y:auto}}
 img{{flex:1;min-height:0;object-fit:contain;background:#111}}
 footer{{padding:7px 14px;background:#1b1b1b;display:flex;gap:18px;font-size:13px;
         flex-wrap:wrap;flex:0 0 auto}}
 kbd{{background:#333;border-radius:4px;padding:2px 8px;font-weight:700}}
 .done{{text-align:center;padding:60px;font-size:18px;line-height:1.9}}
</style>
<header>
  <div class=top>
    <span class=dim>{dim_zh} · {dim_done}/{dim_total}</span>
    <div class=bar><i></i></div>
    <span>总计 {done}/{total} &nbsp;·&nbsp; 干净 {clean} &nbsp; 还在 {not_clean}
          &nbsp; 另有匹配 {other_match}</span>
  </div>
  <div class=zh>{zh}</div>
  <div class=en>{en}</div>
</header>
<img src="/panel/{item_id}?v={done}">
<footer>
  <span><kbd>1</kbd> 走了 &mdash; 琥珀框里那个东西干净移除了</span>
  <span><kbd>2</kbd> 还在 &mdash; 没移除干净，或只是糊了</span>
  <span><kbd>3</kbd> 另有匹配 &mdash; 它走了，但画面里还有别的东西满足这段描述</span>
  <span><kbd>0</kbd> 拿不准</span>
  <span><kbd>&larr;</kbd> 撤销</span>
</footer>
<script>
 const K = {keys};
 const send = (l) => fetch('/label', {{method:'POST',
     headers:{{'Content-Type':'application/json'}},
     body: JSON.stringify({{item_id:'{item_id}', label:l}})}}).then(()=>location.reload());
 addEventListener('keydown', e => {{
   if (K[e.key]) send(K[e.key]);
   else if (e.key === 'ArrowLeft' || e.key === 'Backspace')
     fetch('/undo', {{method:'POST'}}).then(()=>location.reload());
 }});
</script>"""


def _labels_path() -> Path:
    return review_dir() / "gme_labels.csv"


def read_labels() -> dict[str, str]:
    path = _labels_path()
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["item_id"]: r["label"] for r in csv.DictReader(fh)}


def status() -> None:
    index_path = review_dir() / "index.json"
    if not index_path.is_file():
        print("no panels built yet")
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))
    done = read_labels()
    counts = collections.Counter(done.values())
    dims = collections.Counter(i["dimension"] for i in index)
    print(f"panels {len(index)}  {dict(dims)}")
    print(f"reviewed {len(done)}/{len(index)}  {dict(counts)}")
    print(f"labels -> {_labels_path()}")


def serve(port: int) -> None:
    import http.server
    import socketserver

    root = review_dir()
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    labels_path = _labels_path()
    dim_total = collections.Counter(i["dimension"] for i in index)

    def append(item_id: str, label: str) -> None:
        new = not labels_path.is_file()
        with open(labels_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new:
                writer.writerow(["item_id", "label"])
            writer.writerow([item_id, label])

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
            todo = [i for i in index if i["item_id"] not in done]
            counts = collections.Counter(done.values())
            if not todo:
                lines = "<br>".join(
                    f"<b>{counts.get(v, 0)}</b> &nbsp; {LABEL_ZH[v]} ({v})"
                    for v in ("clean", "not_clean", "other_match", "unsure"))
                body = (f"<div class=done>全部 {len(index)} 条审核完毕<br><br>{lines}"
                        f"<br><br>{labels_path}</div>").encode()
            else:
                item = todo[0]
                dim = item["dimension"]
                body = PAGE.format(
                    item_id=item["item_id"],
                    zh=item.get("zh") or "(未翻译)",
                    en=item["expr"],
                    dim_zh=DIM_ZH.get(dim, dim),
                    dim_done=sum(1 for i in index
                                 if i["dimension"] == dim and i["item_id"] in done),
                    dim_total=dim_total[dim],
                    done=len(done), total=len(index),
                    clean=counts.get("clean", 0),
                    not_clean=counts.get("not_clean", 0),
                    other_match=counts.get("other_match", 0),
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
                append(payload["item_id"], payload["label"])
            self.send_response(204)
            self.end_headers()

    print(f"{len(index)} panels, {len(read_labels())} reviewed")
    print(f"\n  open  http://127.0.0.1:{port}\n")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--port", type=int, default=8902)
    args = ap.parse_args()
    if args.build:
        build()
    if args.status:
        status()
    if args.serve:
        serve(args.port)
    if not (args.build or args.serve or args.status):
        ap.error("pass --build then --serve")


if __name__ == "__main__":
    main()
