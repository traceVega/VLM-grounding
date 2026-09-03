"""Build an annotation task directory from an edit bank.

    python -m idea91.human.build_tasks --out ~/vlmg-data/human/pilot --limit 20

The full P11 draw needs the K2 bank, the verifier answers and a scored ORIGINAL
condition, none of which exist while K1 is still running.  So this takes what is
there: it renders REMOVE edits from an edit index into the two views P11
specifies, tags them with a stratum, and writes the key file separately.  When
the K2 stores exist, ``--frame`` switches it to the real 200 + 100 draw.

B9's pilot -- ten items, three annotators, kappa reported before any K2 scoring
-- is exactly this command with ``--limit 10 --stratum pilot``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from idea91.edits.build import load_edited, read_image
from idea91.human.tasks import PILOT_STRATUM, Task, render_task_images, write_key, write_tasks
from idea91.masks import decode_rle
from shared import paths


def load_referent_labels(set_name: str) -> dict[str, str]:
    """``image_id -> the referent's class label``, from the instance store.

    Empty when the store is absent, in which case the question falls back to a
    generic noun and the caller should know the pilot is weaker for it.
    """
    directory = paths.PREPARED / set_name / "instances"
    shards = sorted(Path(directory).glob("shard-*.parquet"))
    if not shards:
        return {}
    out: dict[str, str] = {}
    for shard in shards:
        try:
            table = pq.read_table(shard, columns=["image_id", "source", "label"])
        except Exception:
            continue
        image_ids = table.column("image_id").to_pylist()
        sources = table.column("source").to_pylist()
        labels = table.column("label").to_pylist()
        for image_id, source, label in zip(image_ids, sources, labels):
            if source == "referent" and label:
                out.setdefault(image_id, label)
    return out


def build(
    out_dir: Path,
    *,
    limit: int,
    stratum: str,
    index_dir: Path | None = None,
    edits_root: Path | None = None,
    image_dir: Path | None = None,
    set_name: str = "openimages_pool",
) -> Path:
    index_dir = index_dir or (paths.EDITS_ROOT / "index")
    edits_root = edits_root or paths.EDITS_ROOT
    image_dir = image_dir or (paths.RAW / "openimages" / "images")

    shards = sorted(Path(index_dir).glob("shard-*.parquet"))
    if not shards:
        raise SystemExit(f"no edit index in {index_dir}; run `run_k1 edits` first")
    table = pa.concat_tables([pq.read_table(s) for s in shards])
    columns = {n: table.column(n).to_pylist() for n in table.column_names}

    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    tasks: list[Task] = []
    key_rows: list[dict] = []
    referent_labels = load_referent_labels(set_name)

    for i in range(table.num_rows):
        if len(tasks) >= limit:
            break
        operator = columns["operator"][i]
        if not operator.endswith("REMOVE"):
            continue  # P11 labels removals
        image_id = columns["image_id"][i]
        original = read_image(image_dir / f"{image_id}.jpg")
        edited = load_edited(
            original,
            {
                "window_xyxy_px": columns["window_xyxy_px"][i],
                "window_path": columns["window_path"][i],
            },
            edits_root,
        )
        hole = decode_rle(columns["mask_rle"][i])
        task_id = f"t{len(tasks):04d}"
        window_name, full_name = render_task_images(edited, hole, images_dir, task_id)

        # The annotator is told a noun and nothing else.  K1 has no referring
        # expression, so the noun is the referent's own class label, looked up in
        # the instance store; "is there still an object" would be unanswerable.
        head_noun = referent_labels.get(image_id) or "object"
        tasks.append(
            Task(
                task_id=task_id,
                window_sha256=columns["window_sha256"][i],
                head_noun=str(head_noun),
                expr="",
                window_image=window_name,
                full_image=full_name,
                stratum=stratum,
            )
        )
        key_rows.append(
            {
                "task_id": task_id,
                "window_sha256": columns["window_sha256"][i],
                "image_id": image_id,
                "operator": operator,
                "hole_type": columns["hole_type"][i],
                "set_or_pool": columns["set_or_pool"][i],
            }
        )

    if not tasks:
        raise SystemExit("no REMOVE edits found in the index")
    write_tasks(tasks, out_dir)
    write_key(key_rows, out_dir)
    print(f"{len(tasks)} tasks -> {out_dir}")
    print(f"  images   {images_dir}")
    print(f"  key      {out_dir / 'KEY_do_not_show_annotators.csv'} (never served)")
    print(f"\nstart the UI:\n  python -m idea91.human.app --tasks {out_dir} --annotator A1")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--stratum", default=PILOT_STRATUM)
    ap.add_argument("--index-dir", type=Path, default=None)
    ap.add_argument("--edits-root", type=Path, default=None)
    ap.add_argument("--image-dir", type=Path, default=None)
    args = ap.parse_args()
    build(
        args.out,
        limit=args.limit,
        stratum=args.stratum,
        index_dir=args.index_dir,
        edits_root=args.edits_root,
        image_dir=args.image_dir,
    )


if __name__ == "__main__":
    main()
