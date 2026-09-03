"""The P11 human check: the draw, the blinding, the label store and the statistics."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from idea91.human import agreement as A
from idea91.human import tasks as T
from idea91.human.app import append_label, done_for, read_labels

from .conftest import disc, textured_image


def make_labels(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"window_sha256": w, "annotator_id": a, "label": l, "sample_stratum": s,
          "timestamp": "2026-09-03T00:00:00+00:00", "task_id": w}
         for w, a, l, s in rows]
    )


# --- the draw ----------------------------------------------------------------


def test_the_draw_is_seeded_and_reproducible():
    frame = pd.DataFrame({
        "window_sha256": [f"w{i}" for i in range(50)],
        "size_bin": ["tiny", "small", "medium", "large", "xl"] * 10,
    })
    a = T.stratified_draw(frame, 20, seed=0)
    b = T.stratified_draw(frame, 20, seed=0)
    c = T.stratified_draw(frame, 20, seed=1)
    assert list(a.window_sha256) == list(b.window_sha256)
    assert list(a.window_sha256) != list(c.window_sha256)
    assert len(a) == 20


def test_the_draw_is_stratified_by_size_bin():
    """P11 draws "stratified by size bin", so no bin may be starved."""
    frame = pd.DataFrame({
        "window_sha256": [f"w{i}" for i in range(100)],
        "size_bin": ["tiny"] * 60 + ["small"] * 30 + ["xl"] * 10,
    })
    drawn = T.stratified_draw(frame, 20, seed=0)
    counts = drawn.size_bin.value_counts().to_dict()
    assert set(counts) == {"tiny", "small", "xl"}
    assert counts["tiny"] > counts["small"] > counts["xl"]  # proportional
    assert sum(counts.values()) == 20


def test_a_frame_smaller_than_the_draw_is_returned_whole():
    frame = pd.DataFrame({"window_sha256": ["a", "b"], "size_bin": ["tiny", "xl"]})
    assert len(T.stratified_draw(frame, 200, seed=0)) == 2


def test_p11_draws_200_random_plus_100_verifier_negatives():
    frame = pd.DataFrame({
        "window_sha256": [f"w{i}" for i in range(1000)],
        "size_bin": ["tiny", "small", "medium", "large", "xl"] * 200,
        "verifier_clean": [i % 4 != 0 for i in range(1000)],  # a quarter are not clean
    })
    drawn = T.draw_p11(frame, seed=0)
    counts = drawn.sample_stratum.value_counts().to_dict()
    assert counts[T.RANDOM_STRATUM] == T.N_RANDOM
    assert counts[T.V1_NEGATIVE_STRATUM] == T.N_V1_NEGATIVE
    negatives = drawn[drawn.sample_stratum == T.V1_NEGATIVE_STRATUM]
    assert not negatives.verifier_clean.any(), "the negative stratum must be V1-negative items"
    assert drawn.window_sha256.nunique() == len(drawn), "no item drawn twice"


# --- blinding ----------------------------------------------------------------


def test_a_task_carries_no_condition_and_no_verifier_answer():
    """The app must be unable to leak what it never receives."""
    task = T.Task(task_id="t0", window_sha256="w0", head_noun="cup", expr="the red cup",
                  window_image="a.jpg", full_image="b.jpg", stratum=T.RANDOM_STRATUM)
    served = set(task.to_json())
    for forbidden in ("operator", "condition", "verifier_clean", "control_source",
                      "original_image", "image_id"):
        assert forbidden not in served


def test_item_order_differs_per_annotator():
    tasks = [T.Task(f"t{i}", f"w{i}", "cup", "", "a.jpg", "b.jpg", T.RANDOM_STRATUM)
             for i in range(40)]
    a = [t.task_id for t in T.order_for(tasks, "A1")]
    b = [t.task_id for t in T.order_for(tasks, "A2")]
    assert a != b
    assert sorted(a) == sorted(b)
    assert a == [t.task_id for t in T.order_for(tasks, "A1")]  # stable per annotator


def test_the_key_file_is_named_so_nobody_shows_it_to_an_annotator(tmp_path):
    path = T.write_key([{"task_id": "t0", "operator": "REMOVE"}], tmp_path)
    assert "do_not_show" in path.name
    assert path.suffix == ".csv"


def test_rendering_produces_two_views_and_never_the_original(tmp_path):
    edited = textured_image((300, 400), seed=1)
    hole = disc((300, 400), 200, 150, 25)
    window, full = T.render_task_images(edited, hole, tmp_path, "t0")
    assert (tmp_path / window).is_file() and (tmp_path / full).is_file()
    assert {p.name for p in tmp_path.glob("*.jpg")} == {window, full}, "only the edited views"


def test_the_window_view_is_a_zoom_on_the_hole(tmp_path):
    import cv2

    edited = textured_image((600, 800), seed=2)
    hole = disc((600, 800), 400, 300, 20)
    window, full = T.render_task_images(edited, hole, tmp_path, "t0")
    w = cv2.imread(str(tmp_path / window))
    f = cv2.imread(str(tmp_path / full))
    assert w.shape[:2] != f.shape[:2]


# --- the label store ---------------------------------------------------------


def test_labels_are_written_through_and_read_back(tmp_path):
    append_label(tmp_path, {"window_sha256": "w0", "annotator_id": "A1", "label": "clean",
                            "timestamp": "t", "sample_stratum": "random", "task_id": "t0"})
    append_label(tmp_path, {"window_sha256": "w1", "annotator_id": "A1", "label": "failed",
                            "timestamp": "t", "sample_stratum": "random", "task_id": "t1"})
    rows = read_labels(tmp_path)
    assert len(rows) == 2
    assert done_for(tmp_path, "A1") == {"t0": "clean", "t1": "failed"}
    assert done_for(tmp_path, "A2") == {}


def test_the_csv_matches_the_design_section_5_columns(tmp_path):
    append_label(tmp_path, {"window_sha256": "w0", "annotator_id": "A1", "label": "clean",
                            "timestamp": "t", "sample_stratum": "random", "task_id": "t0"})
    with open(tmp_path / "human_labels.csv", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    for column in ("window_sha256", "annotator_id", "label", "timestamp", "sample_stratum"):
        assert column in header


# --- the statistics ----------------------------------------------------------


def test_majority_needs_a_strict_majority():
    assert A.majority_label(["clean", "clean", "failed"]) == "clean"
    assert A.majority_label(["clean", "remnant", "failed"]) is None
    assert A.majority_label(["clean", "clean", "skip"]) == "clean"
    assert A.majority_label(["skip", "skip", "skip"]) is None


def test_binarise_is_clean_versus_everything_else():
    assert A.binarise("clean") is True
    assert A.binarise("remnant") is False
    assert A.binarise("failed") is False
    assert A.binarise("skip") is None
    assert A.binarise(None) is None


def test_unanimous_annotators_give_kappa_one():
    labels = make_labels([(f"w{i}", a, "clean" if i % 2 else "failed", "random")
                          for i in range(10) for a in ("A1", "A2", "A3")])
    report = A.analyse(labels)
    assert report.fleiss_3way == pytest.approx(1.0)
    assert report.n_annotators == 3


def test_check_3_passes_when_humans_and_the_verifier_agree():
    rows, v1 = [], {}
    for i in range(20):
        clean = i % 3 != 0
        for a in ("A1", "A2", "A3"):
            rows.append((f"w{i}", a, "clean" if clean else "failed", "random"))
        v1[f"w{i}"] = clean
    report = A.analyse(make_labels(rows), pd.Series(v1),
                       pd.Series({f"w{i}": "random" for i in range(20)}))
    assert report.check_3_applies and report.check_3_passes
    assert report.cohen_human_vs_v1 == pytest.approx(1.0)
    assert "verifier-clean is the decisive column" in report.verdict_line()


def test_check_3_failing_sends_the_headline_to_the_human_column():
    rows, v1 = [], {}
    for i in range(20):
        for a in ("A1", "A2", "A3"):
            rows.append((f"w{i}", a, "clean" if i % 2 else "failed", "random"))
        v1[f"w{i}"] = i % 2 == 0  # the verifier disagrees with the humans every time
    report = A.analyse(make_labels(rows), pd.Series(v1),
                       pd.Series({f"w{i}": "random" for i in range(20)}))
    assert report.check_3_applies and not report.check_3_passes
    assert "human-clean carries the headline" in report.verdict_line()
    assert "escalates to 500" in report.verdict_line()


def test_check_3_does_not_apply_to_a_pilot():
    """B9's pilot reports human-human agreement only; check 3 lives on the random 200."""
    rows = [(f"w{i}", a, "clean", "pilot") for i in range(6) for a in ("A1", "A2", "A3")]
    strata = pd.Series({f"w{i}": "pilot" for i in range(6)})
    report = A.analyse(make_labels(rows), pd.Series({f"w{i}": True for i in range(6)}), strata)
    assert not report.check_3_applies
    assert "not applicable" in report.verdict_line()
    assert "pilot" in report.fleiss_by_stratum


def test_unsure_labels_leave_the_kappas_and_are_counted():
    rows = [(f"w{i}", a, "clean", "random") for i in range(5) for a in ("A1", "A2")]
    rows += [(f"w{i}", "A3", "skip", "random") for i in range(5)]
    report = A.analyse(make_labels(rows))
    assert report.skipped == 5
    # every item has a skip, so no item is fully rated for the 3-way Fleiss
    assert np.isnan(report.fleiss_3way)
