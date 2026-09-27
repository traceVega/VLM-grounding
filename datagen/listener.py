"""Stage `listen`: Molmo2-8B points at the expression (uniqueness) and at each
expression with one detail removed (load-bearing test).

Molmo2's card-pinned processor was written against transformers 4.x and passes
its optional attributes into `ProcessorMixin.__init__`, which 5.16 rejects.  The
subclass below sets them on the instance itself; the rest of the remote code is
used unchanged.  Prompt: `pointing_molmo2_primary` (VERIFIED, the card's own
"Point to the {expr}."); parser: `shared.harness.parsers.parse_molmo2` (Q-2).
"""

from __future__ import annotations

import time

import torch
from PIL import Image

from datagen import common as C
from shared.harness import parsers as P
from shared.harness import prompts

HF = "allenai/Molmo2-8B"
REV = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"


def _tf_major() -> int:
    import transformers
    return int(transformers.__version__.split(".")[0])


def load_molmo2():
    """Load the pinned Molmo2-8B.

    Under transformers 4.57 (the version the model card targets; `~/molmo-env`)
    the card's remote code runs as published.  Under 5.16 (`~/vlmg-env`) four
    shims are needed (processor kwargs, RoPE registry, masking API, prefill
    cache_position) and even then generation degenerates into repeated text with
    no `<points>` tag (2026-09-21), so the 5.16 path is kept only for debugging
    and the datagen chain runs this stage from `~/molmo-env`.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t0 = time.time()
    if _tf_major() < 5:
        proc = AutoProcessor.from_pretrained(HF, revision=REV, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            HF, revision=REV, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()
        print(f"Molmo2-8B loaded in {time.time() - t0:.0f}s (transformers 4.x path), "
              f"{torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
        return proc, model
    return _load_molmo2_tf5()


def _load_molmo2_tf5():
    from transformers import AutoModelForImageTextToText
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.processing_utils import ProcessorMixin

    # transformers 5.16 renamed the masking helpers' `input_embeds` argument to
    # `inputs_embeds`; the card's modeling code (imported below, at from_pretrained)
    # still passes the old name.  Wrap the two helpers before that import binds them.
    from transformers import masking_utils as MU

    def _alias(fn):
        def wrapped(*args, **kwargs):
            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            kwargs.pop("cache_position", None)  # 5.16 derives it from position_ids / the cache
            return fn(*args, **kwargs)
        wrapped.__name__ = getattr(fn, "__name__", "wrapped")
        return wrapped

    if not getattr(MU.create_causal_mask, "_datagen_alias", False):
        MU.create_causal_mask = _alias(MU.create_causal_mask)
        MU.create_causal_mask._datagen_alias = True
        MU.create_masks_for_generate = _alias(MU.create_masks_for_generate)
        MU.create_masks_for_generate._datagen_alias = True

    Remote = get_class_from_dynamic_module("processing_molmo2.Molmo2Processor", HF, revision=REV)
    mod = __import__(Remote.__module__, fromlist=["IMAGE_PROMPT", "VIDEO_PROMPT", "IMAGE_TOKENS"])

    class Patched(Remote):  # type: ignore[misc,valid-type]
        def __init__(self, image_processor=None, video_processor=None, tokenizer=None,
                     chat_template=None, image_use_col_tokens=True, use_single_crop_col_tokens=None,
                     use_single_crop_start_token=True, video_use_col_tokens=False,
                     use_frame_special_tokens=True, **kwargs):
            ProcessorMixin.__init__(self, image_processor, video_processor, tokenizer,
                                    chat_template=chat_template)
            self.image_use_col_tokens = image_use_col_tokens
            self.use_single_crop_col_tokens = use_single_crop_col_tokens
            self.use_single_crop_start_token = use_single_crop_start_token
            self.video_use_col_tokens = video_use_col_tokens
            self.use_frame_special_tokens = use_frame_special_tokens
            self.image_placeholder_token = mod.IMAGE_PROMPT
            self.video_placeholder_token = mod.VIDEO_PROMPT
            self.image_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in mod.IMAGE_TOKENS]

    # transformers 5.16 dropped the "default" entry from ROPE_INIT_FUNCTIONS; the card's
    # modeling code looks it up for a config without rope_scaling.  Molmo2-8B's text
    # config: head_dim 128, rope_theta 1e6, rope_scaling None -> plain RoPE.
    from transformers import modeling_rope_utils as MRU

    def _default_rope(config, device=None, seq_len=None, **kw):
        base = float(getattr(config, "rope_theta", 10000.0))
        dim = int(getattr(config, "head_dim", None)
                  or config.hidden_size // config.num_attention_heads)
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv, 1.0

    MRU.ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope)

    t0 = time.time()
    proc = Patched.from_pretrained(HF, revision=REV, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        HF, revision=REV, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()

    # transformers 5.16's prefill calls prepare_inputs_for_generation with
    # cache_position=None and fills it inside the base implementation; the card's
    # override indexes the argument before that.  Read it back from the base's
    # result instead, and treat "no cache yet" as the prefill step.
    from transformers.generation import GenerationMixin

    ModelCls = type(model)

    def _prepare(self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None,
                 image_token_pooling=None, image_grids=None, image_num_crops=None,
                 pixel_values_videos=None, video_token_pooling=None, video_grids=None,
                 attention_mask=None, token_type_ids=None, cache_position=None,
                 logits_to_keep=None, **kwargs):
        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self, input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, cache_position=cache_position,
            logits_to_keep=logits_to_keep, token_type_ids=token_type_ids, **kwargs)
        cp = cache_position if cache_position is not None else model_inputs.get("cache_position")
        if cp is not None:
            prefill = bool(cp[0] == 0)
        else:
            prefill = past_key_values is None or getattr(past_key_values, "get_seq_length", lambda: 0)() == 0
        if prefill:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_token_pooling"] = image_token_pooling
            model_inputs["image_grids"] = image_grids
            model_inputs["image_num_crops"] = image_num_crops
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["video_token_pooling"] = video_token_pooling
            model_inputs["video_grids"] = video_grids
        return model_inputs

    ModelCls.prepare_inputs_for_generation = _prepare
    print(f"Molmo2-8B loaded in {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 1e9:.1f} GB",
          flush=True)
    return proc, model


def point(proc, model, image: Image.Image, text: str, max_new_tokens: int = 64) -> tuple[str, P.Parsed]:
    msgs = [{"role": "user", "content": [{"type": "text", "text": text},
                                         {"type": "image", "image": image}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw = proc.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return raw, P.parse_molmo2(raw, original_wh=image.size)


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
    print(f"listen: {len(written)} written, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    tmpl = prompts.load("pointing_molmo2_primary")
    proc, model = load_molmo2()
    t0 = time.time()
    for n, image_id in enumerate(todo, 1):
        sc, w = scenes[image_id], written[image_id]
        image = Image.open(C.image_path(image_id)).convert("RGB")
        inst = sc["instances"]

        def hit(expr: str) -> dict:
            raw, parsed = point(proc, model, image, tmpl.render(expr=expr))
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
            rec["sibling_unique_hit"] = r_s["hit_iid"] == sib[image_id]["sibling_iid"] and r_s["n_points"] == 1
        C.append_jsonl(out, rec)
        if n % 5 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {(time.time() - t0) / n:.1f}s/scene "
                  f"unique={rec['unique_hit']} lb={rec['load_bearing']}", flush=True)
