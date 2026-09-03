"""Dataset registry: what to fetch, how big, under what licence (design 4.2).

Every row carries the licence and the allowed uses, because two of them decide
what may ever be released: OpenImages is CC-BY and makes the K1 bank the only
releasable edited artefact (P1), while SA-1B, GroundingME and COCO-Search18
forbid redistributing derivatives, so any bank built on them stays local.

``locator`` is left ``PIN_REQUIRED`` wherever the exact file path or repository
revision has not been read off the source.  ``download.py`` refuses to fetch a
source whose locator is unpinned rather than guessing a URL, and
``licenses.py`` renders the whole table into ``data/LICENSES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PIN = "PIN_REQUIRED"

MANUAL = "manual"  # a signed link or a form; cannot be fetched by script
HF = "huggingface"
HTTP = "http"


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    kind: str  # HF | HTTP | MANUAL
    locator: str  # repo id, URL, or PIN_REQUIRED
    revision: str  # commit SHA / version, or PIN_REQUIRED
    approx_gb: float
    licence: str
    allowed: tuple[str, ...]  # evaluate | publish_numbers | release_images | release_derivatives
    used_by: tuple[str, ...]
    date_read: str = ""  # when the licence text was actually read
    attribution: str = ""
    notes: str = ""
    optional: bool = False

    @property
    def pinned(self) -> bool:
        return PIN not in (self.locator, self.revision)

    @property
    def may_release_derivatives(self) -> bool:
        return "release_derivatives" in self.allowed


SOURCES: tuple[Source, ...] = (
    Source(
        key="openimages",
        name="Open Images subset (K1 pool)",
        kind=MANUAL,
        locator=PIN,
        revision=PIN,
        approx_gb=15.0,
        licence="CC-BY 4.0 per image (annotations CC BY 4.0)",
        allowed=("evaluate", "publish_numbers", "release_images", "release_derivatives"),
        used_by=("P1 K1 pool",),
        attribution="author, URL and licence recorded per image from the images CSV (P1)",
        notes=(
            "Fetched by image id with the official downloader; the images CSV carries the "
            "attribution columns P1 requires. Pin the CSV release and the downloader commit. "
            "This is the only source whose edited bank may be released."
        ),
    ),
    Source(
        key="sa1b",
        name="SA-1B shards (K1 fallback pool)",
        kind=MANUAL,
        locator=PIN,
        revision=PIN,
        approx_gb=20.0,
        licence="SA-1B Research License",
        allowed=("evaluate", "publish_numbers"),
        used_by=("P1 fallback",),
        notes="Signed download link. Using it makes the K1 bank non-releasable (P1).",
        optional=True,
    ),
    Source(
        key="groundingme",
        name="GroundingME",
        kind=HF,
        locator="lirang04/GroundingME",
        revision=PIN,
        approx_gb=3.0,
        licence="research use, under SA-1B and HR-Bench terms",
        allowed=("evaluate", "publish_numbers"),
        used_by=("P8 K2 items", "P12 primary protocol"),
        notes=(
            "Images run 1,500 to 7,680 px, which is why P2 edits a window. Its evaluator "
            "repository is also needed, at a recorded commit, for the primary protocol's "
            "instruction and null-box convention (OPEN-QUESTIONS Q-1)."
        ),
    ),
    Source(
        key="openref",
        name="OpenRef",
        kind=MANUAL,
        locator=PIN,
        revision=PIN,
        approx_gb=3.5,
        licence=PIN,
        allowed=(),
        used_by=("P8 K2 items",),
        notes=(
            "Release of arXiv 2605.25706. Design O1: format and licence unread. If it is "
            "unavailable, K2 runs on GroundingME alone and the P15 CI half-widths are restated."
        ),
        optional=True,
    ),
    Source(
        key="ref_l4",
        name="Ref-L4 (dev slice (a))",
        kind=HF,
        locator=PIN,
        revision=PIN,
        approx_gb=21.0,
        licence="CC BY-NC 4.0 (re-ships Objects365 images)",
        allowed=("evaluate", "publish_numbers"),
        used_by=("acceptance check 4 parity run",),
        notes=(
            "Materialized by IDEA-11's shared/data; this idea only needs dev slice (a), the "
            "1,000 items of IDEA-11 D2(a), for the P21 backend parity run."
        ),
    ),
    Source(
        key="coco_train2014",
        name="COCO train2014 images",
        kind=HTTP,
        locator="http://images.cocodataset.org/zips/train2014.zip",
        revision="train2014",
        approx_gb=13.0,
        licence="COCO terms; images CC BY 4.0 with Flickr terms",
        allowed=("evaluate", "publish_numbers"),
        used_by=("p12 dev slice (gRefCOCO no-target)", "P18 secondary statistic mapping"),
        notes="The only COCO image download in either idea; val2014 is a P18 contingency.",
    ),
    Source(
        key="coco_val2014",
        name="COCO val2014 images",
        kind=HTTP,
        locator="http://images.cocodataset.org/zips/val2014.zip",
        revision="val2014",
        approx_gb=6.0,
        licence="COCO terms",
        allowed=("evaluate", "publish_numbers"),
        used_by=("P18 contingency",),
        notes="Only needed if the P18 pHash mapping misses against train2014.",
        optional=True,
    ),
    Source(
        key="coco_ann2017",
        name="COCO 2017 instance annotations",
        kind=HTTP,
        locator="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        revision="2017",
        approx_gb=0.25,
        licence="CC BY 4.0",
        allowed=("evaluate", "publish_numbers"),
        used_by=("P18 secondary statistic",),
    ),
    Source(
        key="grefcoco",
        name="gRefCOCO no-target annotations",
        kind=MANUAL,
        locator=PIN,
        revision=PIN,
        approx_gb=0.2,
        licence=PIN,
        allowed=("evaluate", "publish_numbers"),
        used_by=("p12 dev slice",),
        notes="100 no-target items and 100 positives, run under --non-kill (P12).",
    ),
    Source(
        key="cocosearch18",
        name="COCO-Search18 (edit-free row)",
        kind=MANUAL,
        locator=PIN,
        revision=PIN,
        approx_gb=5.0,
        licence="MIT data terms: no commercial use, no redistribution",
        allowed=("evaluate", "publish_numbers"),
        used_by=("P18",),
        notes=(
            "Its own distributed images, 1,680 x 1,050 padded, for all 18 categories, "
            "target-present and target-absent. The letterbox transform gets a 20-image "
            "visual check (P18)."
        ),
    ),
)

BY_KEY = {s.key: s for s in SOURCES}

#: Non-dataset licence rows the design requires in the same file.
EXTRA_LICENCE_ROWS: tuple[dict[str, str], ...] = (
    {
        "name": "SAM 3 (model and checkpoint)",
        "licence": "custom; the GitHub text permits derivative distribution with a copy of "
        "the agreement and acknowledgement",
        "allowed": "to be confirmed against the full text behind the gate (design O7)",
        "used_by": "instances (P1, P8, P9)",
    },
    {
        "name": "big-LaMa",
        "licence": "PIN_REQUIRED",
        "allowed": "read before any bank is released",
        "used_by": "the editor (P2)",
    },
    {
        "name": "Alibaba Model Studio (Qwen3-VL-235B-Instruct)",
        "licence": "provider terms; only a tier with no retention of inputs and no training "
        "on inputs may be used (P17)",
        "allowed": "record the tier and retention terms per provider",
        "used_by": "P17 frontier subset",
    },
    {
        "name": "Google API (Gemini 3 Pro)",
        "licence": "provider terms; same no-retention, no-training requirement (P17)",
        "allowed": "record the tier and retention terms per provider",
        "used_by": "P17 frontier subset",
    },
)


def unpinned() -> list[Source]:
    return [s for s in SOURCES if not s.pinned]


def core(include_optional: bool = False) -> list[Source]:
    return [s for s in SOURCES if include_optional or not s.optional]


def total_gb(sources: list[Source] | None = None) -> float:
    return round(sum(s.approx_gb for s in (sources or core())), 1)
