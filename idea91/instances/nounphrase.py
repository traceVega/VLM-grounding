"""Head noun and noun phrases from a referring expression (design 4.3).

    for K2 the head noun and every noun phrase of the expression, parsed by the
    text-only LLM Qwen3.5-9B with the parser output stored

Two consumers, and they pull in opposite directions, which is why the parse is
stored rather than recomputed:

* **P3's exclusion set** wants *every* noun phrase, referent and context alike,
  so a control hole never lands on an object the expression mentions.
* **P10's clean question** wants only the *head noun*, because relational context
  usually lies outside the edit window and asking the full expression there
  would answer "no" whether or not remnants remain (design deviation 4).

The parser is judge A (Qwen3.5-9B), which is a different lineage from the edit
verifier (judge B, Gemma4-12B) -- the lineage rule of design 4.5, enforced in
``shared/judges/lineage.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

PROMPT_NAME = "nounphrase_parse"

_ARTICLES = ("the ", "a ", "an ")
_STOP_TAIL = re.compile(
    r"\b(that|which|who|on|in|at|under|above|behind|next|near|beside|between|to|of|with|"
    r"left|right|top|bottom|middle|centre|center|front|back)\b"
)


@dataclass
class NounPhraseParse:
    """One expression's parse, stored so the two consumers see the same words."""

    expr: str
    head_noun: str
    noun_phrases: list[str]
    parser_model: str
    parser_revision: str
    prompt_version: str
    raw_output: str = ""
    kill_grade: bool = True
    meta: dict = field(default_factory=dict)

    def other_phrases(self) -> list[str]:
        """The P3 non-referent prompts: noun phrases that do not name the referent.

        A phrase containing the head noun as a word ("red cup" for head noun
        "cup") is the referent's own phrase and is dropped, so a CONTROL_OBJ can
        never be sampled from something the expression names.  Those instances
        still reach the exclusion set through the head-noun concept prompt.
        """
        head = self.head_noun.strip().lower()
        return [
            p
            for p in self.noun_phrases
            if head not in re.findall(r"[a-z]+", p.strip().lower())
        ]

    def as_row(self) -> dict:
        return {
            "expr": self.expr,
            "head_noun": self.head_noun,
            "noun_phrases": list(self.noun_phrases),
            "parser_model": self.parser_model,
            "parser_revision": self.parser_revision,
            "prompt_version": self.prompt_version,
            "raw_output": self.raw_output,
            "kill_grade": self.kill_grade,
        }


class ParseError(RuntimeError):
    pass


def parse_json_output(text: str) -> tuple[str, list[str]]:
    """Pull ``head_noun`` and ``noun_phrases`` out of the model's reply."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ParseError(f"no JSON object in the parser's reply: {text[:200]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ParseError(f"parser reply is not valid JSON: {exc}") from exc
    head = str(data.get("head_noun", "")).strip().lower()
    phrases = [str(p).strip().lower() for p in data.get("noun_phrases", []) if str(p).strip()]
    if not head:
        raise ParseError(f"parser returned no head_noun: {data}")
    if head not in phrases:
        phrases.insert(0, head)
    return head, phrases


class LLMNounPhraseParser:
    """Judge A, text-only, through the judge service (design 4.3, 4.5)."""

    kill_grade = True

    def __init__(self, ask, model: str, revision: str, prompt_version: str) -> None:
        #: ``ask(prompt_text) -> reply text``; the judge client supplies it
        self.ask = ask
        self.model = model
        self.revision = revision
        self.prompt_version = prompt_version

    def parse(self, expr: str, prompt_text: str) -> NounPhraseParse:
        reply = self.ask(prompt_text)
        head, phrases = parse_json_output(reply)
        return NounPhraseParse(
            expr=expr,
            head_noun=head,
            noun_phrases=phrases,
            parser_model=self.model,
            parser_revision=self.revision,
            prompt_version=self.prompt_version,
            raw_output=reply,
        )


class HeuristicNounPhraseParser:
    """A crude parser for dev-slice smoke runs.  Never kill-grade.

    Takes the last word before the first relational cue as the head noun and the
    text up to that cue as the only noun phrase.  It gets "the red cup on the
    table" right and much else wrong, which is why anything it produces is
    stamped ``kill_grade=False`` and travels with that flag into the store.
    """

    kill_grade = False

    def parse(self, expr: str, prompt_text: str = "") -> NounPhraseParse:
        text = expr.strip().lower().rstrip(".")
        for article in _ARTICLES:
            if text.startswith(article):
                text = text[len(article) :]
                break
        cut = _STOP_TAIL.search(text)
        phrase = (text[: cut.start()] if cut else text).strip()
        words = [w for w in re.findall(r"[a-z]+", phrase) if w]
        if not words:
            raise ParseError(f"nothing noun-like in {expr!r}")
        return NounPhraseParse(
            expr=expr,
            head_noun=words[-1],
            noun_phrases=[phrase] if phrase else [words[-1]],
            parser_model="heuristic",
            parser_revision="n/a",
            prompt_version="n/a",
            raw_output="",
            kill_grade=False,
        )
