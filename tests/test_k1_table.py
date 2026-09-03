"""The K1 table renderer: a PASS never prints without its conditions."""

from __future__ import annotations

from idea91.analysis.k1 import CONTRAST_BG, CONTRAST_OBJ, K1Tables, verdict_from_results
from idea91.gate import ladders as L
from idea91.gate.train import RowResult
from idea91.gate.verdict import GATE_ROW_NAMES
from shared.stats import Interval


def row(name: str, hole: str, auroc: float, contrast: str = CONTRAST_OBJ, clf: str = "resnet18"):
    return RowResult(
        row=name,
        classifier=clf,
        hole_type=hole,
        contrast=contrast,
        auroc_mean=auroc,
        auroc_by_seed=[auroc, auroc, auroc],
        ci=Interval(auroc, auroc - 0.02, auroc + 0.02, 2000),
        n_images=2000,
        by_control_source={"labelled_other_class": auroc, "class_agnostic": auroc + 0.01},
    )


def results(auroc: float = 0.53):
    clf = {"i_resnet18_full_cap": "resnet18", "ii_vit_s16_tiles_native": "vit_s16",
           "iii_resnet18_1024": "resnet18", "iii_vit_s16_1024": "vit_s16"}
    return [
        row(name, hole, auroc, clf=clf[name])
        for hole in ("mask", "rect")
        for name in GATE_ROW_NAMES
    ] + [row("iii_resnet18_1024", "mask", 0.58, CONTRAST_BG)]


def test_a_clean_gate_renders_a_pass_with_its_ladder_conditions():
    v = verdict_from_results(
        results(0.53),
        global_ladder=L.GlobalLadderVerdict({"r": 0.96}, {"r": 0.78}),
        null_1b=L.NullVerdict({n: 0.5 for n in GATE_ROW_NAMES}),
        local_floors={b: 75 for b in L.HOLE_AREA_BINS},
        null_1c_asserted=True,
        freeze_version="k1-v1",
    )
    text = K1Tables(results=results(0.53), verdict=v, editor="big_lama").render()
    assert "K1: PASS" in text
    assert "check 1a global ladder: PASS" in text
    assert "local floor [0.5-1%]" in text
    assert "check 1c paired-crop null: asserted" in text
    assert "REMOVE versus CONTROL_BG" in text


def test_a_failing_gate_says_no_downstream_consumer_may_touch_the_bank():
    v = verdict_from_results(results(0.71), null_1c_asserted=True)
    text = K1Tables(results=results(0.71), verdict=v).render()
    assert "K1: FAIL" in text and "no downstream consumer" in text


def test_class_agnostic_controls_are_marked_unverified_in_the_removal_table():
    v = verdict_from_results(results(), null_1c_asserted=True)
    tables = K1Tables(
        results=results(),
        verdict=v,
        removal_success={
            ("REMOVE", "labelled_other_class"): Interval(0.93, 0.91, 0.95, 5000),
            ("CONTROL_OBJ", "class_agnostic"): Interval(0.0, 0.0, 0.0, 800),
        },
    )
    text = tables.render()
    assert "unverified (V2 only)" in text
    assert "never mixed into the verified-pairs row" in text


def test_dinov2_rows_are_reported_but_do_not_gate():
    rows = results(0.53) + [row("adv_dinov2b_1024", "mask", 0.81, clf="dinov2_b")]
    v = verdict_from_results(
        rows,
        global_ladder=L.GlobalLadderVerdict({"r": 0.96}, {"r": 0.78}),
        null_1b=L.NullVerdict({n: 0.5 for n in GATE_ROW_NAMES}),
        local_floors={"1-2%": 75},
        null_1c_asserted=True,
    )
    assert v.decisive
    assert "DINOv2-B adversary" in K1Tables(results=rows, verdict=v).render()


def test_the_table_writes_to_disk(tmp_path):
    v = verdict_from_results(results(), null_1c_asserted=True)
    path = K1Tables(results=results(), verdict=v).write(tmp_path / "k1.md")
    assert path.exists() and path.read_text(encoding="utf-8").startswith("# K1")
