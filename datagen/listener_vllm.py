"""Molmo2-8B listener through a vLLM OpenAI-compatible endpoint.

The card's remote modeling code is written against transformers 4.x and breaks
in three places under 5.16 (processor kwargs, RoPE registry, masking API).
vLLM 0.28 ships a native `Molmo2ForConditionalGeneration`, so the listener is
served by vLLM (from the other project's env, which has it pinned) and called
over HTTP -- the transport `shared/judges/serve.sh` already assumes.

    # terminal 1 (WSL):
    ~/ptr1-env/bin/python -m vllm.entrypoints.openai.api_server \
        --model allenai/Molmo2-8B --revision e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b \
        --dtype bfloat16 --max-model-len 8192 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.62 \
        --trust-remote-code --port 8010
    # terminal 2:
    python -m datagen.run listen --run test50 --backend vllm

Same prompt (`pointing_molmo2_primary`), same parser (`parse_molmo2`), same
record format as `datagen.listener`.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.request

from PIL import Image

from datagen import common as C
from shared.harness import parsers as P
from shared.harness import prompts

HF = "allenai/Molmo2-8B"
BASE_URL = "http://127.0.0.1:8010/v1"


def _jpeg_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def ask(image: Image.Image, text: str, max_tokens: int = 64, timeout: float = 180.0) -> str:
    payload = {
        "model": HF, "temperature": 0.0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(image)}"}},
            {"type": "text", "text": text}]}],
    }
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body["choices"][0]["message"]["content"] or ""


def wait_ready(timeout: float = 900.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/models", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise RuntimeError("vLLM endpoint did not come up")


def run(run: str, limit: int | None) -> None:
    root = C.run_root(run)
    scenes = C.by_image(C.read_jsonl(root / "scenes.jsonl"))
    written = C.by_image(C.read_jsonl(root / "write.jsonl"))
    sib = C.by_image(C.read_jsonl(root / "sibling.jsonl"))
    out = root / "listener.jsonl"
    done = C.done_keys(out)
    todo = [i for i in written if (i,) not in done]
    if limit:
        todo = todo[:limit]
    print(f"listen(vllm): {len(written)} written, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    tmpl = prompts.load("pointing_molmo2_primary")
    wait_ready()
    t0 = time.time()
    for n, image_id in enumerate(todo, 1):
        sc, w = scenes[image_id], written[image_id]
        image = Image.open(C.image_path(image_id)).convert("RGB")
        inst = sc["instances"]

        def hit(expr: str) -> dict:
            raw = ask(image, tmpl.render(expr=expr))
            parsed = P.parse_molmo2(raw, original_wh=image.size)
            pt = list(parsed.point_xy_px) if parsed.point_xy_px else None
            return {"raw": raw, "output_type": parsed.output_type, "point": pt,
                    "n_points": parsed.n_boxes, "hit_iid": C.point_in_instance(pt, inst)}

        r_t = hit(w["expr_target"])
        rec = {"image_id": image_id, "target": r_t,
               "unique_hit": r_t["hit_iid"] == sc["target_iid"] and r_t["n_points"] == 1}
        head = w["head"] or sc["category"]
        abl = []
        for k in range(len(w["clauses"])):
            rest = [c for j, c in enumerate(w["clauses"]) if j != k]
            r = hit(C.join_clauses(head, rest))
            abl.append({"dropped": k, **r, "still_hits_target": r["hit_iid"] == sc["target_iid"]})
        rec["ablation"] = abl
        rec["load_bearing"] = [not a["still_hits_target"] for a in abl]
        if image_id in sib:
            r_s = hit(sib[image_id]["expr_sibling"])
            rec["sibling"] = r_s
            rec["sibling_unique_hit"] = (r_s["hit_iid"] == sib[image_id]["sibling_iid"]
                                        and r_s["n_points"] == 1)
        rec["listener"] = f"{HF} via vllm"
        C.append_jsonl(out, rec)
        if n % 5 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {(time.time() - t0) / n:.1f}s/scene "
                  f"unique={rec['unique_hit']} lb={rec['load_bearing']}", flush=True)
