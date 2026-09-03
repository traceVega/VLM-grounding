"""The instance stack: backend capabilities, the SAM 2 contingency path, the
noun-phrase parse, and the K1/K2 scenes."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from idea91 import masks as M
from idea91 import schemas
from idea91.edits.sampler import build_exclusion, plan_k1_image
from idea91.instances.backend import RawInstance, Segmenter, SegmenterUnsupported, deduplicate
from idea91.instances.build import build_k1_scene, build_k2_scene, scene_rows
from idea91.instances.nounphrase import (
    HeuristicNounPhraseParser,
    LLMNounPhraseParser,
    ParseError,
    parse_json_output,
)
from idea91.instances.sam import Sam2Segmenter, Sam3Segmenter, get_segmenter

from .conftest import disc, textured_image

SHAPE = (800, 1000)


class FakeSegmenter(Segmenter):
    """A segmenter with a scripted world, so the scene builders are testable."""

    name = "fake"
    revision = "test"

    def __init__(self, world: dict[str, list[np.ndarray]], *, concept=True, generic=True,
                 pinned=True):
        self.world = world
        self.supports_concept = concept
        self.supports_generic = generic
        self.is_pinned_backend = pinned

    def concept(self, image, phrase):
        return [RawInstance(mask=m, prompt=phrase) for m in self.world.get(phrase, [])]

    def from_box(self, image, box):
        return M.mask_from_box(box, image.shape[:2]) & self.world["_solid"][0]

    def generic(self, image, min_area_frac: float = 0.005):
        return [
            RawInstance(mask=m, origin="generic")
            for m in self.world.get("_generic", [])
            if M.area_frac(m) > min_area_frac
        ]


@pytest.fixture
def world():
    cat = disc(SHAPE, 300, 400, 45)
    cat2 = disc(SHAPE, 640, 400, 42)  # a second instance of the same noun
    dog = disc(SHAPE, 700, 250, 44)
    table = disc(SHAPE, 500, 700, 90)
    speck = disc(SHAPE, 60, 60, 6)
    return {
        "cat": [cat],
        "dog": [dog],
        "table": [table],
        "_generic": [cat, dog, table, speck],
        "_solid": [np.ones(SHAPE, dtype=bool)],
        "_shapes": {"cat": cat, "cat2": cat2, "dog": dog, "table": table, "speck": speck},
    }


# --- backend contract --------------------------------------------------------


def test_sam2_declares_no_concept_mode_and_says_what_to_do():
    seg = Sam2Segmenter(checkpoint="/nonexistent")
    assert seg.supports_concept is False and seg.supports_generic is True
    assert seg.is_pinned_backend is False
    with pytest.raises(SegmenterUnsupported, match="ground-truth boxes"):
        seg.concept(np.zeros((4, 4, 3), np.uint8), "cat")


def test_sam3_is_the_pinned_backend_and_reports_the_gate():
    seg = Sam3Segmenter()
    assert seg.is_pinned_backend is True and seg.supports_concept is True
    assert seg.supports_generic is False  # design O6, until the release is read
    with pytest.raises(SegmenterUnsupported, match="design O6"):
        seg.generic(np.zeros((4, 4, 3), np.uint8))
    assert seg.describe()["hf_path"] == "facebook/sam3"
    assert len(seg.revision) == 40  # a commit SHA, never a branch name


def test_sam3_pins_the_release_that_transformers_can_load():
    """facebook/sam3.1 is newer but ships no safetensors (DEVIATIONS D-16)."""
    from idea91.instances.sam import SAM3_HF_PATH, SAM3_REVISION

    assert SAM3_HF_PATH == "facebook/sam3"
    assert SAM3_REVISION == "3c879f39826c281e95690f02c7821c4de09afae7"


# --- SAM 3 tensor plumbing, exercised without the gated weights --------------


class FakeTensor:
    """Just enough of the torch surface the adapter touches."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def to(self, *_a, **_k):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeSam3Processor:
    """Records what it was called with and returns scripted instances."""

    def __init__(self, results):
        self.results = results
        self.calls = []
        self.post_kwargs = None

    def __call__(self, images=None, return_tensors=None, **prompt):
        self.calls.append({"image_shape": np.asarray(images).shape, **prompt})
        return {"pixel_values": FakeTensor(np.zeros((1, 3, 8, 8)))}

    def post_process_instance_segmentation(self, outputs, **kwargs):
        self.post_kwargs = kwargs
        return self.results


class FakeSam3Model:
    def __call__(self, **_kw):
        return object()


def sam3_with(results):
    processor = FakeSam3Processor(results)
    return Sam3Segmenter(processor=processor, model=FakeSam3Model()), processor


def test_sam3_concept_passes_the_phrase_and_returns_masks_at_original_size(world):
    cat, dog = world["_shapes"]["cat"], world["_shapes"]["dog"]
    seg, processor = sam3_with(
        [{"masks": [FakeTensor(cat), FakeTensor(dog)], "scores": [0.91, 0.42]}]
    )
    out = seg.concept(textured_image(SHAPE), "cat")

    assert processor.calls[0]["text"] == "cat"
    assert processor.post_kwargs["target_sizes"] == [SHAPE]  # (height, width)
    assert processor.post_kwargs["threshold"] == 0.3
    assert [round(i.score, 2) for i in out] == [0.91, 0.42]
    assert all(i.origin == "concept" and i.prompt == "cat" for i in out)
    assert out[0].mask.dtype == bool and np.array_equal(out[0].mask, cat)


def test_sam3_box_prompt_takes_the_highest_scoring_instance(world):
    cat, dog = world["_shapes"]["cat"], world["_shapes"]["dog"]
    seg, processor = sam3_with(
        [{"masks": [FakeTensor(dog), FakeTensor(cat)], "scores": [0.20, 0.88]}]
    )
    box = M.bbox_xyxy(cat)
    mask = seg.from_box(textured_image(SHAPE), box)

    assert np.array_equal(mask, cat)  # the 0.88 one, not the first
    assert processor.calls[0]["input_boxes"] == [[list(box)]]
    assert processor.calls[0]["input_boxes_labels"] == [[[1]]]


def test_sam3_box_prompt_returns_an_empty_mask_when_nothing_is_found():
    """build_k2_scene drops these as empty_referent_mask rather than editing air."""
    seg, _ = sam3_with([{"masks": [], "scores": []}])
    mask = seg.from_box(textured_image(SHAPE), (10.0, 10.0, 50.0, 50.0))
    assert mask.shape == SHAPE and not mask.any()


def test_sam3_generic_refuses_even_when_the_flag_is_forced():
    seg = Sam3Segmenter(has_generic_mode=True, processor=object(), model=object())
    with pytest.raises(SegmenterUnsupported, match="no generate-everything call"):
        seg.generic(np.zeros((4, 4, 3), np.uint8))


def test_get_segmenter_names():
    assert isinstance(get_segmenter("sam2", checkpoint="/nonexistent"), Sam2Segmenter)
    assert isinstance(get_segmenter("sam3"), Sam3Segmenter)
    with pytest.raises(ValueError):
        get_segmenter("segment-everything")


def test_deduplicate_keeps_the_highest_scoring_of_a_duplicate_group(world):
    cat = world["_shapes"]["cat"]
    raws = [
        RawInstance(mask=cat, score=0.5, prompt="cat"),
        RawInstance(mask=cat.copy(), score=0.9, prompt="animal"),
        RawInstance(mask=world["_shapes"]["dog"], score=0.7, prompt="dog"),
    ]
    kept = deduplicate(raws)
    assert len(kept) == 2
    assert {k.prompt for k in kept} == {"animal", "dog"}


# --- K1 scene ----------------------------------------------------------------


def test_k1_scene_picks_a_referent_in_the_p1_area_band(world):
    seg = FakeSegmenter(world)
    scene = build_k1_scene(textured_image(SHAPE), "img1", ["cat", "dog", "table"], seg, seed=0)
    assert scene.drop_reason is None
    assert 0.005 <= M.area_frac(scene.referent.mask) <= 0.15
    assert scene.kill_grade is True


def test_k1_referent_choice_is_seeded_and_stable(world):
    seg = FakeSegmenter(world)
    img = textured_image(SHAPE)
    a = build_k1_scene(img, "img1", ["cat", "dog", "table"], seg, seed=0)
    b = build_k1_scene(img, "img1", ["cat", "dog", "table"], seg, seed=0)
    assert np.array_equal(a.referent.mask, b.referent.mask)


def test_k1_other_classes_become_control_candidates_and_same_class_does_not(world):
    world = dict(world)
    world["cat"] = [world["_shapes"]["cat"], world["_shapes"]["cat2"]]
    seg = FakeSegmenter(world)
    scene = build_k1_scene(textured_image(SHAPE), "img1", ["cat", "dog"], seg, seed=0)
    sources = {i.instance_id: i.source for i in scene.instances}
    assert "labelled_other_class" in sources.values()
    if scene.referent.label == "cat":
        assert "same_class_other_instance" in sources.values()


def test_k1_scene_adds_class_agnostic_masks_above_the_area_floor(world):
    seg = FakeSegmenter(world)
    scene = build_k1_scene(textured_image(SHAPE), "img1", ["cat"], seg, seed=0)
    agnostic = [i for i in scene.instances if i.source == "class_agnostic"]
    assert agnostic
    assert all(M.area_frac(i.mask) > 0.005 for i in agnostic)  # the speck is dropped


def test_k1_drops_an_image_with_nothing_in_the_band():
    tiny = {"cat": [disc(SHAPE, 100, 100, 4)]}
    seg = FakeSegmenter(tiny, generic=False)
    scene = build_k1_scene(textured_image(SHAPE), "img1", ["cat"], seg, seed=0)
    assert scene.drop_reason == "no_instance_in_the_p1_area_band"


def test_the_sam2_contingency_path_needs_boxes_and_is_not_kill_grade(world):
    seg = FakeSegmenter(world, concept=False, pinned=False)
    img = textured_image(SHAPE)
    boxes = {"cat": [M.bbox_xyxy(world["_shapes"]["cat"])]}
    scene = build_k1_scene(img, "img1", ["cat"], seg, seed=0, gt_boxes=boxes)
    assert scene.drop_reason is None
    assert scene.kill_grade is False  # labelled as such, per design Section 8

    without = build_k1_scene(img, "img1", ["cat"], seg, seed=0)
    assert without.drop_reason == "no_concept_mode_and_no_boxes"


def test_a_k1_scene_feeds_the_sampler_and_check_2(world):
    seg = FakeSegmenter(world)
    scene = build_k1_scene(textured_image(SHAPE), "img1", ["cat", "dog", "table"], seg, seed=0)
    plans = plan_k1_image(scene.referent, scene.instances, rng=np.random.default_rng(0))
    if plans is not None:  # the fake world may not always yield a matched control
        from idea91.edits.sampler import assert_no_exclusion_overlap

        assert_no_exclusion_overlap(plans, scene.referent, scene.instances)


# --- K2 scene ----------------------------------------------------------------


def parse_for(expr="the cat on the table"):
    return HeuristicNounPhraseParser().parse(expr)


def test_k2_scene_uses_a_mask_hole_when_box_to_mask_iou_is_high(world):
    seg = FakeSegmenter(world)
    box = M.bbox_xyxy(world["_shapes"]["cat"])
    scene = build_k2_scene(textured_image(SHAPE), "img1", parse_for(), box, seg)
    assert scene.hole_type == "mask" and scene.box_inpaint_flag is False
    assert scene.box_to_mask_iou >= 0.5


def test_k2_scene_falls_back_to_a_rectangular_hole(world):
    """P8: an item whose box-to-mask IoU is below 0.5 gets a rect hole and a flag."""

    class ThinMask(FakeSegmenter):
        def from_box(self, image, box):
            m = np.zeros(image.shape[:2], dtype=bool)
            x0, y0, x1, y1 = (int(v) for v in box)
            m[y0:y1, x0 : x0 + max(1, (x1 - x0) // 8)] = True  # a sliver in the box
            return m

    seg = ThinMask(world)
    box = M.bbox_xyxy(world["_shapes"]["table"])
    scene = build_k2_scene(textured_image(SHAPE), "img1", parse_for(), box, seg)
    assert scene.box_to_mask_iou < 0.5
    assert scene.hole_type == "rect" and scene.box_inpaint_flag is True
    assert np.array_equal(scene.referent.mask, M.mask_from_box(box, SHAPE))


def test_k2_drops_a_referent_above_the_p8_area_cap(world):
    class Huge(FakeSegmenter):
        def from_box(self, image, box):
            return np.ones(image.shape[:2], dtype=bool)

    scene = build_k2_scene(
        textured_image(SHAPE), "img1", parse_for(), (0.0, 0.0, 999.0, 799.0), Huge(world)
    )
    assert scene.drop_reason == "referent_area_above_p8_cap"


def test_k2_finds_same_noun_neighbours_for_the_p10_routing(world):
    world = dict(world)
    world["cat"] = [world["_shapes"]["cat"], world["_shapes"]["cat2"]]
    seg = FakeSegmenter(world)
    box = M.bbox_xyxy(world["_shapes"]["cat"])
    scene = build_k2_scene(textured_image(SHAPE), "img1", parse_for("the cat"), box, seg)
    assert len(scene.head_noun_neighbours) == 1
    assert scene.n_head_noun_instances == 2  # P15 REDUNDANT reads this


def test_k2_context_phrases_become_exclusion_set_members(world):
    seg = FakeSegmenter(world)
    box = M.bbox_xyxy(world["_shapes"]["cat"])
    parse = LLMNounPhraseParser(
        ask=lambda _: '{"head_noun": "cat", "noun_phrases": ["cat", "table"]}',
        model="qwen3.5-9b", revision="test", prompt_version="v",
    ).parse("the cat on the table", "prompt")
    scene = build_k2_scene(textured_image(SHAPE), "img1", parse, box, seg)
    assert any(i.source == "noun_phrase_instance" and i.label == "table" for i in scene.instances)
    exclusion = build_exclusion(scene.referent, scene.instances)
    assert np.any(exclusion[world["_shapes"]["table"]])


# --- noun-phrase parsing -----------------------------------------------------


def test_json_parse_inserts_a_missing_head_noun_into_the_phrase_list():
    head, phrases = parse_json_output('prose {"head_noun": "cup", "noun_phrases": ["table"]}')
    assert head == "cup" and phrases[0] == "cup" and "table" in phrases


def test_json_parse_rejects_rubbish():
    with pytest.raises(ParseError, match="no JSON object"):
        parse_json_output("sorry, I cannot")
    with pytest.raises(ParseError, match="no head_noun"):
        parse_json_output('{"noun_phrases": ["table"]}')


def test_the_heuristic_parser_is_never_kill_grade():
    p = HeuristicNounPhraseParser().parse("the red cup on the table")
    assert p.head_noun == "cup" and p.kill_grade is False


def test_the_referents_own_phrase_is_never_a_control_candidate():
    """"red cup" names the referent, so P3 must not sample a control from it."""
    p = LLMNounPhraseParser(
        ask=lambda _: '{"head_noun": "cup", "noun_phrases": ["red cup", "wooden table"]}',
        model="qwen3.5-9b", revision="abc", prompt_version="v1",
    ).parse("the red cup on the wooden table", "prompt")
    assert p.other_phrases() == ["wooden table"]


def test_the_llm_parser_stores_its_raw_output():
    p = LLMNounPhraseParser(
        ask=lambda _: '{"head_noun": "cup", "noun_phrases": ["cup", "table"]}',
        model="qwen3.5-9b", revision="abc", prompt_version="v1",
    ).parse("the cup on the table", "prompt text")
    assert p.kill_grade is True and p.raw_output.startswith("{")
    assert p.other_phrases() == ["table"]


# --- the store ---------------------------------------------------------------


def test_scene_rows_validate_against_the_instances_schema(world):
    seg = FakeSegmenter(world)
    scene = build_k1_scene(textured_image(SHAPE), "img1", ["cat", "dog", "table"], seg, seed=0)
    rows = scene_rows(scene, "openimages_pool")
    table = pa.Table.from_pylist(rows, schema=schemas.INSTANCES_SCHEMA)
    schemas.validate(table, "instances")
    assert {r["source"] for r in rows} <= set(schemas.INSTANCE_SOURCES)


def test_an_instance_source_outside_the_list_is_refused(world):
    seg = FakeSegmenter(world)
    scene = build_k1_scene(textured_image(SHAPE), "img1", ["cat"], seg, seed=0)
    rows = scene_rows(scene, "openimages_pool")
    rows[0]["source"] = "vibes"
    with pytest.raises(Exception, match="source"):
        schemas.validate(pa.Table.from_pylist(rows, schema=schemas.INSTANCES_SCHEMA), "instances")
