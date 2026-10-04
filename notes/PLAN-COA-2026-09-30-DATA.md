# Downloaded evaluation / training data (2026-09-30)

All under WSL `~/vlmg-data/raw/<name>/`, each with a SOURCE.txt (URLs, revision, license, files, counts, box format). Counts were verified by reading the files.

| Dataset | Role | Path | License | Size | Verified counts | Boxes |
|---|---|---|---|---|---|---|
| PR-Bench (was RefBench-PRO, arXiv 2512.06276 v3) | OOD eval (rejection + positives) | raw/refbench-pro/ | CC BY-NC 4.0, card says not for training | 2.7 GB | 6,000 items, 1,000 per task; Reject 1,000 (`bbox=[]`) on 764 images | xyxy px |
| Ref-Adv-s | OOD eval (hard positives, no no-target items) | raw/ref-adv/hf/data/ | CC BY 4.0 | 244 MB | 1,142 rows (OpenImages 883, COCO val2017 259), 0 no-target | xyxy px (+ 0-1000 normalized) |
| FineCops-Ref test | OOD eval (compositional negatives) | raw/finecops-ref/ | CC BY 4.0 (GQA/VG images) | 43 GB | 27,926 items: 9,605 positive, 18,321 no-target (9,814 negative expressions + 8,507 negative images) | xywh px; negatives keep a copied box, identify by negative_type / `neg_` file name |
| MMStar | general-ability eval | raw/mmstar/hf/ | no dataset license stated | 97 MB | 1,500 items, 250 per category | multiple choice |
| Visual Genome v1.4 | training (spatial relations) | raw/visual_genome/ | CC BY 4.0 | 31 GB | 108,077 images, 2.52M objects, 2.32M relationships, 5.41M regions | xywh px |
| OpenRef | OOD eval (none-target) | raw/openref/SOURCE.txt only | apache-2.0 | - | blocked: HF `Wuzj/OpenRef` is gated (login + agree to share contact info) | - |

- Train/test overlap: all 4,313 real images of FineCops-Ref test are Visual Genome images (md5-identical), and 98 of Ref-Adv-s's COCO images are in VG. The 4,400 VG image ids are in `raw/visual_genome/EVAL_OVERLAP_image_ids.json` and must be excluded from any VG-based training data. Even then, training on VG makes FineCops-Ref share the image source with training; report that with the FineCops-Ref numbers.
- visualgenome.org no longer hosts the project (unrelated site, old URLs 404); VG came from the authors' pages (homes.cs.washington.edu/~ranjay/visualgenome annotations, cs.stanford.edu/people/rak248 images).
- VG relationships use `name` or `names` for subject/object. PR-Bench images.zip had Windows paths (re-extracted into images/). FineCops-Ref images sit in two folders (final_neg_images/, gqa_images/images/).
- Original zips kept (VG zips 15.3 GB, GQA images.zip 20 GB, PR-Bench images.zip 1.4 GB), about 36 GB that can be deleted later.

## Decisions (user, 2026-09-30 ~13:00)
- Visual Genome approved for training (spatial relations); exclude the 4,400 image ids in EVAL_OVERLAP_image_ids.json and disclose the FineCops-Ref image-source overlap.
- OOD claim: main = PR-Bench Reject (1,000; FineHARD images; published Qwen3-VL-8B 15.8, Qwen2.5-VL-72B 23.6); secondary = FineCops-Ref negatives (vs RC-GRPO), OpenRef none-target if access is granted. OOD positives = PR-Bench's other five tasks (attribute, position, interaction, relation, commonsense), Ref-Adv-s, FineCops-Ref positives. General = MMStar. Retention = RefCOCO/+/g 300x3. GME = target benchmark (held out, style-matched), not OOD.
- OOD sets are evaluated only after the recipe is frozen; never used for model selection or training.
- OpenRef downloaded 13:00 after the user accepted the HF terms (cached token, account MagicOwO): raw/openref/, apache-2.0, 4.3 GB. Test split 3,540 items (single 1,857, multi 1,195, none 488); none-target expressions are ~2.6 words ("green truck"). Boxes are {x, y, width, height} in percent of the image. Our protocol answers one box or none, so we use none (488) + single-target files; multi-target is out of scope.

## Spatial training data (2026-09-30 13:30): `train/spatial_items.py` -> ~/vlmg-data/train/spatial_items.jsonl
- Goal: the Spatial dimension (GME Spatial positives 31.0 -> 24.3 under RL9). Every label is judged from boxes, so the RL reward stays verifiable: ordinal among same-class candidates ("the second barrel from the left"; centre gaps >= 5 %) and relations to a once-boxed reference object ("is to the left of the table"; yes = whole box on that side with a 2 % margin, no = centre on the other side, ambiguous -> the condition is not used). Positive = two conditions only the target satisfies; negative = one relation flipped to its opposite (or an ordinal to another position) with zero satisfiers; observed values per cell as in gme_export.
- Visual Genome was tried first and rejected on a check sheet: VG boxes only some instances of a class ("the rightmost man" with ten men and three boxed; "the second frame from the top" on a wall of frames), so ordinal and uniqueness labels are wrong. The default source is now the OpenImages validation boxes of our 10,000 local images (human-verified, every instance of an image's positive classes boxed); person-like classes merged into "person" and de-duplicated; a class with a crowd / depiction / inside box is skipped; own validation scenes excluded.
- Yield: 186 scenes -> 372 items (126 with 2 candidates, 60 with 3-5; negatives: 164 relation flips, 22 ordinal flips). A sheet of 8 random items was checked by eye: target, ordinal, relation and the falsified detail all correct; some references are semantically odd (a rower "to the left of the boat" = the motorboat in the background). Source "spatial" registered in traces.label_matrix.
- Use: RL pool only (the lesson from SFT4-7: new falsified data in SFT teaches a "relations are usually false" prior), next RL round after RL10. VG stays on disk (CC BY 4.0) for later relation data with a completeness check.

## gRefCOCO and HumanRef (downloaded 2026-09-30 ~14:30, user-approved; evaluation only)
| Dataset | Path | License | Counts | Boxes / metric |
|---|---|---|---|---|
| gRefCOCO (CVPR 2023) | raw/grefcoco/ (HF FudanCVL/gRefCOCO @81eede59 = Google Drive grefs, md5-identical); 3,000 val/testA/testB COCO train2014 images fetched individually | CC BY-NC-SA 4.0 (project page) | val 14,229 expr (8,905 no-target, 5,324 multi, **0 single**); testA 19,200 (4,448 / 8,835 / 5,917 single); testB 16,063 (4,673 / 5,744 / 5,646 single). File totals differ from both papers; both official copies agree | COCO xywh px; no-target ann_id [-1]. GREC metric Pr@(F1=1, IoU>=0.5) and N-acc (official mdetr refexp.py counts boxes with score >= 0.7, GIoU matching); copied to official_code/ |
| HumanRef (ICCV 2025) | raw/humanref/hf/ (HF IDEA-Research/HumanRef @9a7b1c1e) | IDEA License 1.0, non-commercial research; real people incl. celebrities | 6 x 1,000 (attribute, position, interaction, reasoning, celebrity, rejection); rejection items have empty answer_boxes, images still show >= 4 people | xyxy px; official metric hf/metric/recall_precision_densityf1.py; Rejection Score = empty predictions out of 1,000 (missing ids count as correct rejections: prediction files must cover all 6,000 ids). 17 celebrity file names repaired after extraction |
- Our protocol answers one box or none: evaluate no-target + single-target items (gRefCOCO N-acc on val/testA/testB, single-target accuracy on testA/testB; HumanRef rejection + single-person items). Multi-target is out of scope.

## Train/eval image overlap in RefCOCO training rows (found 2026-09-30 14:40)
- RefCOCO / RefCOCO+ / RefCOCOg use different image splits, so a training image of one set can be an evaluation image of another. By md5 against our RefCOCO/+/g val shards and the 3,000 gRefCOCO eval images: refcoco_train.jsonl (used by RL9 and earlier, @60) has 89 / 600 rows on evaluation images; refcoco_train_v2.jsonl had 138 / 1,200.
- Fix: refcoco_train_v2.jsonl rewritten without them (1,062 rows; original kept as .with_eval_overlap) before RL10 starts. Image names of both files' offending rows in raw/grefcoco/TRAIN_OVERLAP_images.json.
- Effect on existing numbers: RL9's pool sampled ~60 of the 600 rows, so about 9 contaminated rows; inflates RefCOCO slightly at most (RefCOCO is our weak side, so conclusions do not change). For reporting, RefCOCO and gRefCOCO scores of models trained on the old file must skip evaluation images that appear in training (md5 filter).
