"""The K2 run loop: what it records, what it refuses, and what it never invents."""

from __future__ import annotations

import numpy as np
import pytest

from idea91 import freeze as F
from shared.harness import conditions as C
from shared.harness import parsers as P
from shared.harness import run as R
from shared.harness import schema as S


class FakeClient:
    """Answers from a script, and remembers what it was shown."""

    def __init__(self, answers=None) -> None:
        self.answers = answers or ['{"bbox_2d": [100, 100, 300, 300]}']
        self.calls: list[tuple[str, tuple[int, int]]] = []

    def generate(self, prompt: str, image: np.ndarray) -> R.Answer:
        self.calls.append((prompt, (image.shape[1], image.shape[0])))
        text = self.answers[(len(self.calls) - 1) % len(self.answers)]
        return R.Answer(text=text, latency_ms=7, tokens_in=900, tokens_out=12)


@pytest.fixture
def item(tmp_path):
    import cv2

    path = tmp_path / "img.jpg"
    cv2.imwrite(str(path), np.full((800, 1000, 3), 128, np.uint8))
    return R.Item(
        item_id="gme_00001",
        image_path=path,
        expr="The object is a red cup.",
        gt_boxes_xyxy_px=[(100.0, 100.0, 300.0, 300.0)],
        pair_id="p1",
        head_noun="cup",
    )


def cfg(**over):
    base = dict(
        run_id="smoke", model_id="qwen3vl-8b-instruct",
        coordinate_convention=P.ABSOLUTE_RESIZED, non_kill=True,
        conditions=(C.ORIGINAL,),
    )
    base.update(over)
    return R.RunConfig(**base)


# --- what lands in outputs.parquet -------------------------------------------


def test_a_run_produces_schema_valid_rows(item):
    rows = R.run([item], cfg(), FakeClient())
    S.validate(S.pa.Table.from_pylist(rows, schema=S.OUTPUTS_SCHEMA), "outputs")
    assert len(rows) == 1


def test_image_px_sent_records_what_the_model_actually_saw(item):
    client = FakeClient()
    rows = R.run([item], cfg(max_pixels=100_000), client)
    sent = tuple(rows[0]["image_px_sent"])
    assert sent == client.calls[0][1], "the row must match what was sent"
    # full_at_cap rounds each side up to a whole pixel, so it can land a
    # fraction of a percent over. Immaterial here: the processor applies its own
    # cap, and P21's budget is checked against the processor.
    assert sent[0] * sent[1] == pytest.approx(100_000, rel=0.01)
    assert sent != (1000, 800), "and it did resize"


def test_a_small_image_is_not_upscaled(item):
    rows = R.run([item], cfg(), FakeClient())
    assert tuple(rows[0]["image_px_sent"]) == (1000, 800)


def test_iou_is_scored_on_original_only(item):
    rows = R.run([item], cfg(), FakeClient())
    assert rows[0]["iou_gt"] == pytest.approx(1.0)
    assert rows[0]["is_failure"] is False


def test_a_non_original_condition_carries_no_ground_truth_score(item):
    """P14 compares those boxes against the ORIGINAL box, not against truth."""
    rows = R.run([item], cfg(conditions=(C.T_NULL,)), FakeClient())
    assert rows[0]["iou_gt"] is None
    assert rows[0]["is_failure"] is None
    S.validate(S.pa.Table.from_pylist(rows, schema=S.OUTPUTS_SCHEMA), "outputs")


def test_an_abstention_on_original_is_a_failure_not_a_gap(item):
    rows = R.run([item], cfg(), FakeClient(['{"bbox_2d": null}']))
    assert rows[0]["output_type"] == "none"
    assert rows[0]["is_failure"] is True


def test_an_unparseable_answer_still_becomes_a_row(item):
    rows = R.run([item], cfg(), FakeClient(["somewhere on the left"]))
    assert len(rows) == 1
    assert rows[0]["parse_ok"] is False and rows[0]["output_type"] == "invalid"


def test_the_raw_text_is_kept_verbatim(item):
    rows = R.run([item], cfg(), FakeClient(["  odd  spacing {} "]))
    assert rows[0]["raw_text"] == "  odd  spacing {} "


# --- conditions ---------------------------------------------------------------


def test_t_null_sends_the_replacement_expression(item):
    client = FakeClient()
    R.run([item], cfg(conditions=(C.T_NULL,)), client)
    assert "the object" in client.calls[0][0]
    assert "red cup" not in client.calls[0][0]


def test_an_item_that_cannot_carry_a_condition_is_skipped_not_faked(item, capsys):
    """T_HEAD needs a category SAM 3 found absent; nothing may invent one."""
    rows = R.run([item], cfg(conditions=(C.ORIGINAL, C.T_HEAD)), FakeClient())
    assert len(rows) == 1
    assert "skipped" in capsys.readouterr().out


def test_an_image_condition_without_a_composer_is_an_error(item):
    with pytest.raises(ValueError, match="needs an edited image"):
        R.run([item], cfg(conditions=(C.REMOVE,)), FakeClient())


def test_the_composer_is_asked_for_the_right_operator(item):
    seen = []

    def compose(it, operator):
        seen.append(operator)
        return np.full((800, 1000, 3), 64, np.uint8)

    R.run([item], cfg(conditions=(C.REMOVE, C.CONTROL_BG)), FakeClient(), compose=compose)
    assert seen == ["REMOVE", "CONTROL_BG"]


# --- protocols ----------------------------------------------------------------


def test_each_protocol_gets_its_own_row_and_prompt_version(item):
    rows = R.run([item], cfg(protocols=("primary", "secondary")), FakeClient())
    assert len(rows) == 2
    assert {r["abstain_protocol"] for r in rows} == {"primary", "secondary"}
    assert rows[0]["prompt_version"] != rows[1]["prompt_version"], (
        "the secondary is a different file, so its hash differs"
    )


def test_the_secondary_prompt_is_a_file_not_an_appended_suffix(item):
    """Two ways to build one string is how T_NULL's template drifted."""
    client = FakeClient()
    R.run([item], cfg(protocols=("secondary",)), client)
    assert client.calls[0][0].endswith("If the object is not present, output none.")


def test_a_protocol_without_a_prompt_file_is_refused(item):
    with pytest.raises(ValueError, match="no prompt file named"):
        R.run([item], cfg(protocols=("tertiary",)), FakeClient())


# --- P20's K2 freeze ----------------------------------------------------------


def test_scoring_original_refuses_without_b0b(item, tmp_path, monkeypatch):
    """P20: 'before the first ORIGINAL item is scored'. The K1 guard was added
    only after a probe walked past it; this one is here from the start."""
    monkeypatch.setattr(F.paths, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="no freeze record exists"):
        R.run([item], cfg(non_kill=False), FakeClient())


def test_a_non_kill_run_may_score_original_before_the_freeze(item, tmp_path, monkeypatch):
    """P12's dev slices and check 4's parity run are non-kill by definition and
    must be able to run first; the manifest records which kind a run was."""
    monkeypatch.setattr(F.paths, "REPO_ROOT", tmp_path)
    assert R.run([item], cfg(non_kill=True), FakeClient())


def test_the_other_conditions_do_not_need_the_freeze(item, tmp_path, monkeypatch):
    monkeypatch.setattr(F.paths, "REPO_ROOT", tmp_path)
    assert R.run([item], cfg(non_kill=False, conditions=(C.T_NULL,)), FakeClient())


# --- P21's rejection rule -----------------------------------------------------


def test_a_run_that_would_be_rejected_refuses_to_start():
    with pytest.raises(SystemExit, match="no request may be rejected"):
        R.check_budget([(7680, 7680)], prompt_tokens=258, cfg=cfg(max_model_len=500))


def test_a_run_that_fits_starts():
    R.check_budget([(7680, 7680)], prompt_tokens=258, cfg=cfg())  # must not raise
