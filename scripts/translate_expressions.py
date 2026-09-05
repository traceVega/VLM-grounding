"""Chinese renderings of the expressions, for the reviewer's eyes only.

    python scripts/translate_expressions.py

**These translations never reach a model under test.** Every experiment sends
the English original: `gme_remove_infer.py` and `gme_original_precheck.py` both
render the prompt from `item["expr"]`, and the review UI is the only consumer of
this file. A translated prompt would be a different prompt, and P12 pins the
primary protocol to GroundingME's own words.

The expressions run to 39 words and 333 of them have to be read at review speed,
which is where the minutes go. Qwen3.5-9B is already on this machine, is strong
in Chinese, and is not a model under test here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

HF = "Qwen/Qwen3.5-9B"
OUT = Path("tables/gme_expr_zh.jsonl")

SYSTEM = (
    "You are a translator. Translate the user's English text into simplified "
    "Chinese. It describes one object in a photograph, for a human annotator "
    "who must find that object. Keep every spatial relation, colour, material, "
    "count and marking exactly; do not summarise, do not add, do not explain. "
    "Reply with the Chinese translation only."
)


def done_ids() -> set[str]:
    if not OUT.is_file():
        return set()
    return {json.loads(x)["item_id"]
            for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = json.loads(Path("tables/gme_removals_index.json").read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    done = done_ids()
    todo = [r for r in rows if r["item_id"] not in done]
    print(f"{len(rows)} expressions; {len(done)} translated, {len(todo)} to go\n")
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(HF)
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    started = time.time()
    for n, item in enumerate(todo, 1):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": item["expr"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=320, do_sample=False)
        zh = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"item_id": item["item_id"], "zh": zh},
                                ensure_ascii=False) + "\n")
            fh.flush()
        if n % 25 == 0:
            rate = (time.time() - started) / n
            print(f"  [{n}/{len(todo)}]  {rate:.1f}s/item, "
                  f"{(len(todo) - n) * rate / 60:.0f} min left", flush=True)

    print(f"\n-> {OUT}  ({len(done_ids())} translated)")


if __name__ == "__main__":
    main()
