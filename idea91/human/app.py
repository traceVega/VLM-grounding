"""Local annotation UI for the P11 human check.

    python -m idea91.human.app --tasks ~/vlmg-data/human/k2 --annotator A1

Then open http://127.0.0.1:8800 .

Deliberately local and dependency-free.  P11 puts the K2 images under a research
licence that forbids publishing them, so this binds to 127.0.0.1 by default and
serves only files under the task directory.  For the three paid annotators the
design's own answer is a private Label Studio instance on a VPS with HTTPS and
individual accounts; ``--host`` exists for that case, and refuses to bind
publicly without ``--i-understand-the-licence``.

What the annotator sees is fixed by P11: the edit window at 2x the hole, the
full image, the head noun and the expression.  Never the original, never the
condition, never another annotator's label.  Labels are written through to CSV
on every keystroke, so closing the tab loses nothing.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from idea91.human.tasks import LABELS, Task, order_for, read_tasks

LABEL_HELP = {
    "clean": "The object is gone. Whatever is there now looks like ordinary background.",
    "remnant": "Traces remain: a smear, a shadow, a piece of the object, a repeated texture.",
    "failed": "The object is still there, whole or nearly whole.",
}

_LOCK = threading.Lock()


def labels_path(task_dir: Path) -> Path:
    return Path(task_dir) / "human_labels.csv"


def append_label(task_dir: Path, row: dict) -> None:
    """Write through immediately; the design's store is a CSV (design Section 5)."""
    path = labels_path(task_dir)
    with _LOCK:
        exists = path.is_file()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["window_sha256", "annotator_id", "label", "timestamp",
                                "sample_stratum", "task_id"]
            )
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def read_labels(task_dir: Path) -> list[dict]:
    path = labels_path(task_dir)
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def done_for(task_dir: Path, annotator_id: str) -> dict[str, str]:
    """task_id -> label, so a reopened tab resumes and a re-label overwrites."""
    out: dict[str, str] = {}
    for row in read_labels(task_dir):
        if row.get("annotator_id") == annotator_id:
            out[row.get("task_id", "")] = row.get("label", "")
    return out


PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Removal check</title>
<style>
 :root {{ color-scheme: dark; --bg:#14161a; --panel:#1d2026; --line:#2c313a; --ink:#e8eaed;
          --muted:#9aa3ad; --clean:#2f9e5e; --remnant:#c08a2e; --failed:#c0492e; }}
 * {{ box-sizing:border-box }}
 /* One screen per item, never a scroll: an annotator does this 300 times. */
 html, body {{ height:100% }}
 body {{ margin:0; background:var(--bg); color:var(--ink); height:100%;
        display:flex; flex-direction:column; overflow:hidden;
        font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif }}
 header {{ display:flex; gap:16px; align-items:baseline; padding:10px 16px; flex:0 0 auto;
           border-bottom:1px solid var(--line); background:var(--panel) }}
 header b {{ font-size:15px }} .muted {{ color:var(--muted) }}
 .bar {{ flex:1; height:6px; background:var(--line); border-radius:3px; overflow:hidden }}
 .bar div {{ height:100%; background:var(--clean); width:{pct}% }}
 main {{ display:grid; grid-template-columns:1.35fr 1fr; gap:14px; padding:12px 16px;
         flex:1 1 auto; min-height:0 }}
 figure {{ margin:0; background:var(--panel); border:1px solid var(--line);
           border-radius:8px; padding:10px; display:flex; flex-direction:column;
           min-height:0 }}
 figure img {{ flex:1 1 auto; min-height:0; width:100%; height:100%;
               object-fit:contain; border-radius:4px; display:block }}
 figcaption {{ color:var(--muted); font-size:12px; margin-bottom:8px; flex:0 0 auto }}
 .q {{ padding:10px 16px; background:var(--panel); border-bottom:1px solid var(--line);
       flex:0 0 auto }}
 .q .noun {{ font-size:19px; font-weight:600 }}
 .q .expr {{ color:var(--muted); font-style:italic }}
 .actions {{ display:flex; gap:10px; padding:0 16px 14px; flex:0 0 auto }}
 .scroll {{ overflow:auto }}
 button {{ flex:1; padding:14px; font-size:15px; font-weight:600; color:#fff; cursor:pointer;
           border:0; border-radius:8px }}
 button small {{ display:block; font-weight:400; font-size:11.5px; opacity:.85; margin-top:3px }}
 .clean {{ background:var(--clean) }} .remnant {{ background:var(--remnant) }}
 .failed {{ background:var(--failed) }}
 .skip {{ background:#3a4048; flex:0 0 130px }}
 kbd {{ background:#00000055; border-radius:3px; padding:0 4px }}
 .done {{ padding:40px 16px; text-align:center }}
</style>
<header>
  <b>Removal check</b>
  <span class="muted">annotator {annotator}</span>
  <div class="bar"><div></div></div>
  <span class="muted">{done} / {total}</span>
  <a class="muted" href="/instructions">instructions</a>
</header>
{body}
<script>
 const keys = {{'1':'clean','2':'remnant','3':'failed','0':'skip'}};
 document.addEventListener('keydown', e => {{
   if (e.key === 'Backspace') {{ location = '/back'; return; }}
   const l = keys[e.key];
   if (l) {{ document.getElementById(l).click(); }}
 }});
 function send(label) {{
   const f = document.getElementById('form');
   document.getElementById('label').value = label;
   f.submit();
 }}
</script>
"""

ITEM_BODY = """
<div class="q">
  <div class="noun">Is there still a <u>{noun}</u> in this image?</div>
  <div class="expr">{expr}</div>
</div>
<main>
  <figure><figcaption>Edit region, close up</figcaption><img src="/img/{window}" alt=""></figure>
  <figure><figcaption>Whole image</figcaption><img src="/img/{full}" alt=""></figure>
</main>
<form id="form" method="post" action="/label" style="flex:0 0 auto">
  <input type="hidden" name="task_id" value="{task_id}">
  <input type="hidden" name="label" id="label">
  <div class="actions">
    <button type="button" class="clean" id="clean" onclick="send('clean')">Gone
      <small>nothing left of it &middot; <kbd>1</kbd></small></button>
    <button type="button" class="remnant" id="remnant" onclick="send('remnant')">Traces
      <small>smear, shadow, part of it &middot; <kbd>2</kbd></small></button>
    <button type="button" class="failed" id="failed" onclick="send('failed')">Still there
      <small>object largely intact &middot; <kbd>3</kbd></small></button>
    <button type="button" class="skip" id="skip" onclick="send('skip')">Unsure
      <small><kbd>0</kbd></small></button>
  </div>
</form>
"""

INSTRUCTIONS = """
<div class="q"><div class="noun">Instructions</div></div>
<main class="scroll" style="grid-template-columns:1fr; display:block">
<figure>
<p>Each image has had one object removed by an automatic editor. Your job is to say
   <b>how well it was removed</b> &mdash; not whether the picture looks nice.</p>
<p>You are told a noun, for example <i>cup</i>. Look at the close-up first, then the whole
   image, and answer: <b>is there still a cup there?</b></p>
<ul>
  <li><b>Gone</b> &mdash; {clean}</li>
  <li><b>Traces</b> &mdash; {remnant}</li>
  <li><b>Still there</b> &mdash; {failed}</li>
  <li><b>Unsure</b> &mdash; only if you genuinely cannot tell. These are reported separately.</li>
</ul>
<p>Judge only the region that was edited. Other objects of the same kind elsewhere in the
   picture do not matter &mdash; if the noun is <i>cup</i> and a different cup sits untouched
   across the table, that is not a failure.</p>
<p>Blurry or low quality is not the same as a trace. A smooth, plausible background is
   <b>Gone</b> even if it is soft.</p>
<p>Keyboard: <kbd>1</kbd> gone, <kbd>2</kbd> traces, <kbd>3</kbd> still there,
   <kbd>0</kbd> unsure, <kbd>Backspace</kbd> to go back and change your last answer.</p>
<p><a href="/">Start</a></p>
</figure></main>
"""


class Handler(BaseHTTPRequestHandler):
    task_dir: Path
    tasks: list[Task]
    annotator: str

    def log_message(self, *args):  # quiet
        pass

    # --- helpers ---
    def _send(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _page(self, body: str) -> str:
        done = len(done_for(self.task_dir, self.annotator))
        total = len(self.tasks)
        return PAGE.format(
            annotator=html.escape(self.annotator),
            done=done,
            total=total,
            pct=(100 * done / total) if total else 0,
            body=body,
        )

    def _next_task(self) -> Task | None:
        done = done_for(self.task_dir, self.annotator)
        for task in order_for(self.tasks, self.annotator):
            if task.task_id not in done:
                return task
        return None

    # --- routes ---
    def do_GET(self):  # noqa: N802
        route = urlparse(self.path)
        if route.path == "/instructions":
            self._send(self._page(INSTRUCTIONS.format(**LABEL_HELP)))
            return
        if route.path.startswith("/img/"):
            self._serve_image(route.path[len("/img/") :])
            return
        if route.path == "/back":
            self._undo_last()
            return
        if route.path == "/progress":
            done = done_for(self.task_dir, self.annotator)
            self._send(json.dumps({"done": len(done), "total": len(self.tasks)}))
            return

        task = self._next_task()
        if task is None:
            self._send(self._page(
                '<div class="done"><h2>All done.</h2>'
                f'<p class="muted">{len(self.tasks)} items labelled. '
                "You can close this tab.</p></div>"))
            return
        self._send(self._page(ITEM_BODY.format(
            noun=html.escape(task.head_noun),
            expr=html.escape(task.expr or ""),
            window=html.escape(task.window_image),
            full=html.escape(task.full_image),
            task_id=html.escape(task.task_id),
        )))

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        task_id = form.get("task_id", [""])[0]
        label = form.get("label", [""])[0]
        if label in LABELS or label == "skip":
            task = next((t for t in self.tasks if t.task_id == task_id), None)
            if task is not None:
                append_label(self.task_dir, {
                    "window_sha256": task.window_sha256,
                    "annotator_id": self.annotator,
                    "label": label,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "sample_stratum": task.stratum,
                    "task_id": task.task_id,
                })
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def _undo_last(self) -> None:
        """Drop this annotator's most recent label so it can be redone."""
        rows = read_labels(self.task_dir)
        mine = [i for i, r in enumerate(rows) if r.get("annotator_id") == self.annotator]
        if mine:
            rows.pop(mine[-1])
            with _LOCK, open(labels_path(self.task_dir), "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["window_sha256", "annotator_id", "label", "timestamp",
                                    "sample_stratum", "task_id"])
                writer.writeheader()
                writer.writerows(rows)
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def _serve_image(self, name: str) -> None:
        # only files inside the task directory, whatever the client asks for
        target = (self.task_dir / "images" / name).resolve()
        root = (self.task_dir / "images").resolve()
        if not str(target).startswith(str(root)) or not target.is_file():
            self.send_error(404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def serve(task_dir: Path, annotator: str, host: str = "127.0.0.1", port: int = 8800) -> None:
    tasks = read_tasks(task_dir)
    if not tasks:
        raise SystemExit(f"no tasks in {task_dir}; build them first")
    Handler.task_dir = Path(task_dir)
    Handler.tasks = tasks
    Handler.annotator = annotator
    done = len(done_for(Path(task_dir), annotator))
    print(f"{len(tasks)} tasks, {done} already labelled by {annotator}")
    print(f"open http://{host}:{port}   (labels -> {labels_path(Path(task_dir))})")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--i-understand-the-licence", action="store_true",
                    help="required to bind anywhere but localhost")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost") and not args.i_understand_the_licence:
        raise SystemExit(
            "P11's images are research-licensed and must not be published. Binding to "
            f"{args.host} would expose them. Use an SSH tunnel, or pass "
            "--i-understand-the-licence if this is the private VPS the design describes."
        )
    serve(args.tasks, args.annotator, args.host, args.port)


if __name__ == "__main__":
    main()
