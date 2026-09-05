# GroundingME on unmodified images: what the removal experiment has to work with

Run 2026-09-04, Qwen3-VL-8B-Instruct at `0c351dd0`, all 1,005 items, ORIGINAL
condition only, no editing and no annotation. 1,005 prompts, 1.4 s each, one
attempt. Exploratory, `--non-kill`: nothing here may enter a kill table or
change P8 to P21. Data in `tables/gme_original.jsonl`.

## The harness reproduces the benchmark

| | measured here | published |
|---|---:|---|
| overall accuracy, 804 positives | **41%** | best of 25 MLLMs 45.1% |
| abstention on the Rejection dimension | **0 / 201** | 20 of 25 models exactly 0% |

Prompt, parser, coordinate convention and the P21 resolution cap are therefore
doing what they should, which had to be established before building edits on
this set.

## Accuracy, and the sample it leaves

| dimension | n | ORIGINAL correct | declined | ordinal wording |
|---|---:|---:|---:|---:|
| Discriminative | 204 | **61%** | 0% | 1% |
| Limited | 300 | 41% | **13%** | 49% |
| Spatial | 300 | **29%** | 0% | 37% |

Only items the model got right can enter the removal experiment. Projecting a
0.6 removal success -- 0.46 held on OpenImages and these referents are far
smaller, so that is a floor:

| dimension | usable | to review |
|---|---:|---:|
| Discriminative | ~75 | 124 |
| Spatial | ~52 | 86 |
| Limited | ~74 | 123 |

Spatial against Discriminative at 52 and 75 resolves a 25-point difference at
about 2.9 sigma and a 20-point one at 2.3. It can settle whether the effect is
large; it cannot measure a small one.

Abstention tracks difficulty rather than absence: 13% on `Limited`, whose
referents have a median area of 0.1% of the image, against 0% on both others.

## The finding that matters, and the doubt it raises

The Rejection items are real photographs, unedited, whose expression describes
something that is not in them.

| condition | referent | edited | expression | median p(null) |
|---|---|---|---|---:|
| CONTROL_OBJ (earlier run) | present | yes | class label | 1.3e-18 |
| **Rejection** | **absent** | **no** | **39-word description** | **3.1e-18** |
| REMOVE (earlier run) | absent | yes | class label | 6.2e-01 |

**On rich descriptions of an object that is not there, the abstention signal
does not appear at all** -- p(null) sits where it sits when the object is
present, and a box is emitted 201 times out of 201.

The AUROC of 0.964 was measured on two-word class labels. Whether it survives
realistic referring expressions is now the open question, and three
explanations fit the table equally well:

1. **Expression style.** Long descriptions never trigger abstention, so the
   necessity relation works only on simple ones, and `IC` is weaker than the
   earlier run suggested.
2. **Absence versus removal.** A body with no head is an incoherent scene; a
   photograph that never contained a skyscraper is not. The signal may come
   from the edit rather than from checking presence.
3. **Partial matches.** A city photograph does contain buildings, so "tall,
   slender, blue-tinted glass" finds something to accept. A class label with no
   instance anywhere finds nothing. This is the author's failure modes 3 to 6 --
   a threshold under a person's -- and needs no absence signal to fail.

Nothing here separates them: expression, image distribution and the kind of
absence all differ at once. The removal experiment does separate them, because
it holds the expression and the image fixed and changes only whether the
referent survived. That is now its main purpose, ahead of the Spatial versus
Discriminative contrast it was designed for.
