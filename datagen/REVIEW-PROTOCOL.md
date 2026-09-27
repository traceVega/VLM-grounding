# Scene review protocol (hard-sample datagen)

You are reviewing generated grounding data. Each scene is one image with several
same-class candidates. The pipeline produced three sentences about it:

- **e_T**: a description of the target candidate T (4–7 checkable details).
- **e_S**: a description of the sibling candidate S.
- **e_T⁻**: e_T with exactly one detail changed so that it should describe *no* candidate.

Your job is to decide, per scene, whether **e_T⁻ is a usable negative** (a model shown
the image and e_T⁻ should answer "no such object"), and to write one short note in
Chinese explaining why.

## What to look at, in this order

1. `<image_id>_cands.jpg`: the close-up of every candidate at native resolution, labelled
   with its number and T / S. **Judge every detail here, not on the overview.** On
   test50c, four verdicts of "the writer hallucinated" were overturned by the close-ups:
   the writer saw them, the overview did not show them.
2. `<image_id>.jpg`: the overview with boxes (thick green = T, thick orange = S, grey =
   other candidates; thin boxes are the base model's answers; dots are the listener).
   Use it for spatial relations (left of, behind, in front of) and for whether the
   candidates are really distinct objects.
3. The text record: expressions, the checker's detail × candidate verdict matrix
   (rows = details of e_T, last row = the changed detail; columns = candidates), base
   model outputs, listener hits, blind-judge result, automatic gates.

## Rules (learned on test50c, keep them)

- A negative is usable only if **every non-flipped detail of e_T is true of T** (verify
  on the close-up) **and the flipped detail is false of every candidate that also
  satisfies the rest of e_T⁻**. Another candidate satisfying only the flipped detail is
  fine (it fails the rest of the sentence).
- A detail you cannot verify (too small, hidden) makes the scene **unsure**, not usable:
  the negative must be false only because of the flip.
- If the flipped detail is still true of T (e.g. "red shield" when the shield is half red),
  the scene is unusable.
- The checker is not ground truth: it says "no" to true details about 10–20% of the time
  (Gemma more than Gemini) and is lenient on spatial relations. Overrule it from the
  images, and say so in the note.
- Scenes where the target is a reflection, a picture inside a picture (newspaper, comic,
  screen, render), an unlabelled-wrong class (a pear labelled apple), or where two
  candidates are visually identical, are unusable regardless of the sentences.
- Grammatically broken e_T⁻ (e.g. "leaping with her red leotard raised high") is unusable.
- A self-contradictory e_T⁻ ("wearing a silver helmet ... wearing a black helmet", "the black hood ... and the white hood")
  is unusable too (`flip_failed`): a model can reject it from the text alone, which teaches nothing about looking.
- e_S copying e_T means the siblings could not be distinguished: mention it; it does not
  by itself kill the negative.
- `REFLIPPED` scenes had their e_T⁻ regenerated; judge the new e_T⁻ exactly like any other.

## Buckets (use these ids exactly)

| id | kind | meaning |
|---|---|---|
| `ok_strict` | ok | usable; every automatic gate also passed |
| `ok_override` | ok | usable by your judgement although an automatic gate failed (say which cell the checker got wrong) |
| `ok_conjunction` | ok | usable; another candidate satisfies only the flipped detail, not the rest |
| `unsure` | unsure | you cannot decide from the images (say what needs checking) |
| `flip_failed` | bad | no e_T⁻, or e_T⁻ is garbled / changed a word that was not there |
| `flip_true_of_target` | bad | the changed detail is still true of T |
| `flip_satisfied` | bad | another candidate satisfies the whole e_T⁻ |
| `not_unique` | bad | two or more candidates are indistinguishable; e_T fits several |
| `writer_wrong` | bad | e_T has details that are false of T (verified on the close-up) |
| `scene` | bad | reflection, picture-in-picture, wrong label, invisible target, base model already refuses |

## Output

Write a JSON file at the path given in your task, of the form

```json
{"<image_id>": {"bucket": "<id>", "note": "<one to four sentences in Chinese: what you saw on the close-up, why usable or not, which checker cells were wrong, what would fix it>"}, ...}
```

One entry per scene in your packet, no scene skipped. Notes are read by the researcher
who trains on this data: be concrete (name the detail and the candidate number).
