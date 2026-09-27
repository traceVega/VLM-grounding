"""Hard-sample data generation (arm A: text-side negatives, no image editing).

One source scene (an OpenImages image with >= 3 instances of one category) becomes
one master record carrying a target expression, a sibling expression, a
one-detail-false negative, per-instance detail verdicts, listener hits, the base
policy's own boxes and the gate flags.  See
notes/HARD-SAMPLE-METHODS-2026-09-16.md section 3.1 and the conversation of
2026-09-21 for the record format.

Stages run as separate processes so that one model is resident at a time:

    select  (CPU)          scenes.jsonl
    write   (Qwen3.5-9B)   write.jsonl      target expr, clauses, negative, flipped clause
    policy  (Qwen3-VL)     policy.jsonl     boxes + decision token on target / negative
    sibling (Qwen3.5-9B)   sibling.jsonl    sibling expr for the instance the base boxed
    policy --which sibling                  boxes on the sibling expr
    listen  (Molmo2-8B)    listener.jsonl   uniqueness + load-bearing ablation
    check   (Gemma4-12B)   checker.jsonl    detail x instance verdicts, zero-satisfier
    blind   (Qwen3.5-9B)   blind.jsonl      text-only detectability of the flip
    report  (CPU)          records.jsonl, gates.md, review/index.html
"""
