# Grounding benchmark landscape, 2026-09-05

Which referring-expression / grounding benchmarks are saturated, which are
not, and what has been shown to move the unsaturated ones. Numbers are from
the sources listed at the bottom, checked on 2026-09-05.

## Saturated

| Benchmark | Best reported | Why it is done |
|---|---:|---|
| RefCOCO/+/g (avg) | 93.5 (Qwen3.6 Plus); 87.9 for a 3B model | Ref-L4 paper measured ~14% label error, so ~93 is the ceiling |
| gRefCOCO no-target | N-acc >80 for best models | negatives are expressions from other images; easy |

## Not saturated

| Benchmark | Size | Best | Where the points are lost |
|---|---:|---:|---|
| GroundingME (CVPR 2026) | 1,005 (Disc 204 / Spatial 300 / Limited 300 / Rej 201) | 49.8 overall (Qwen3-VL-235B-A22B Thinking); Rejection best 9.5 | Rejection ~0 for 20/25 models; Limited 36-54; Spatial 47-50 without thinking, 70-74 with |
| Ref-Adv (2026, 2602.23898) | 5,000 (1,142 public as Ref-Adv-s); 11.5 words, 4 distractors, 21% negation | 58.3 Qwen2.5-VL-72B (89.9 on RefCOCOg); 63.7 GPT-4o+CoT | picks the hard distractor; drops 12.9 at >=7 distractors. No rejection subset |
| RefBench-PRO (2512.06276) | 6,000 (6 tasks x 1,000 incl. Reject) | Qwen3-VL-8B 62.2 all-task / 71.4 non-reject; Reject 15.8 (Qwen2.5-VL-72B Reject 23.6) | Reject subset |
| OpenRef (2605.25706) | 32,735 expr / 17,586 img; none-target built by swapping colour / orientation / noun / proper noun | Qwen3-VL-8B F1 63.7, N3R 84.6; multi-target 35.5 | multi-target; also proper nouns, dark / drone scenes |
| Ref-L4 (2024) | 45,341 ann, 24.2 words avg, 365 classes | 81.7 CogVLM-Grounding (2024); modern LMMs rarely report it | long expressions, small instances. Status: under-reported rather than known-unsaturated |
| D3 / OmniLabel | detection-style, descriptions may match zero objects | AP in the 30s-40s | detection community (Grounding DINO line), AP metric, not MLLM-native |
| KnowDR-REC, MC-Bench, FineCops-Ref, HumanRef, RefSpatial-Bench, ScreenSpot-Pro | - | - | knowledge / multi-image / compositional / multi-instance / robot spatial / GUI. Different sub-fields |

## What has moved the unsaturated ones

| Lever | Evidence | Cost on positives |
|---|---|---|
| Thinking / CoT | GroundingME +1.9 to +7.4 overall; Spatial +23 to +24 for 32B / A22B | none |
| In-distribution hard-positive RL (Ref-R1, Qwen2.5-VL-7B, 180k SFT + 80k RL) | RefBench-PRO 48.5 -> 67.5 all-task | RefCOCO/+/g +2.1 (no cost) |
| Naive negatives mixed 2:1 (GroundingME's own experiment, Qwen3-VL-8B) | Rejection 0 -> 27.9 | RefCOCOg 88.2 -> 83.1; other three dimensions fall hard |
| Easy negatives + refusal RL (RC-GRPO, Qwen3-VL-4B) | gRefCOCO precision 70.6 | about -4 |
| Training-free consistency check (MCC) | +7 to +8 F1; N3R +54 on a 2B model | none |

The same 8B model rejects well on OpenRef's swapped-noun negatives and
scores 0 on GroundingME Rejection. Rejection exists on easy negatives and
vanishes on hard ones. That is the whole picture from our own runs as well.

## The open cell

Nobody has shown hard negatives (referent present, one clause false) taught
without collapsing positives. GroundingME's attempt paid 5 points on RefCOCOg
and about 20 on its own positive dimensions. RC-GRPO stayed on easy negatives.
Ref-R1 has no negatives at all beyond the Reject subset.

Metric arithmetic for an 8B model on GroundingME (weights 20/30/30/20):
- Rejection 0 -> 50 with no positive loss: +10 overall
- Thinking-style RL on Spatial (+23 seen on larger models): +7 overall
- Together from our measured 41: roughly 55-58, above the current best 49.8

RefBench-PRO: Reject 15.8 -> 70 adds about +9 to Qwen3-VL-8B's 62.2.

## Sources

- GroundingME leaderboard: https://groundingme.github.io/ ; paper https://arxiv.org/abs/2512.17495
- Ref-Adv: https://arxiv.org/abs/2602.23898 ; https://ref-adv.github.io/
- RefBench-PRO / Ref-R1: https://arxiv.org/abs/2512.06276
- OpenRef / MCC: https://arxiv.org/abs/2605.25706
- Ref-L4: https://arxiv.org/abs/2406.16866 ; https://github.com/JierunChen/Ref-L4
- RC-GRPO: https://arxiv.org/abs/2608.04698
- RefCOCO-avg leaderboard: https://llm-stats.com/benchmarks/refcoco-avg
- OmniLabel: https://arxiv.org/abs/2304.11463
- KnowDR-REC: https://arxiv.org/abs/2508.14080 ; MC-Bench: https://arxiv.org/abs/2410.12332 ; FineCops-Ref: https://arxiv.org/abs/2409.14750
