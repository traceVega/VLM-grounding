"""Judge client with an on-disk cache (design 4.5).

    a client with a cache keyed by sha256 of (model, prompt version, image
    bytes, question)

The cache is what makes the verifier affordable to re-run: K2's ~10,000 calls
take about 45 minutes on the 5090, and every re-analysis after that is free.  It
is keyed by image *bytes*, so a re-built edit bank invalidates itself.

Transport is the OpenAI-compatible endpoint vLLM serves (``serve.sh``), so no
provider SDK is needed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from shared import paths
from shared.judges.lineage import Judge, for_role

DEFAULT_BASE_URL = "http://127.0.0.1:8010/v1"
MAX_TOKENS = 4  # P10
TEMPERATURE = 0.0  # P10


def cache_key(model: str, prompt_version: str, image_bytes: bytes, question: str) -> str:
    h = hashlib.sha256()
    for part in (model.encode(), prompt_version.encode(), question.encode()):
        h.update(part)
        h.update(b"\x00")
    h.update(image_bytes)
    return h.hexdigest()


class JudgeCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (paths.DATA_ROOT / "judge_cache.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS answers ("
            "key TEXT PRIMARY KEY, answer TEXT, model TEXT, question TEXT, created REAL)"
        )
        self.db.commit()

    def get(self, key: str) -> str | None:
        row = self.db.execute("SELECT answer FROM answers WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, answer: str, model: str, question: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO answers VALUES (?,?,?,?,?)",
            (key, answer, model, question, time.time()),
        )
        self.db.commit()

    def stats(self) -> dict[str, int]:
        (n,) = self.db.execute("SELECT COUNT(*) FROM answers").fetchone()
        return {"entries": int(n)}


@dataclass
class JudgeAnswer:
    text: str
    cached: bool
    latency_ms: int


class JudgeClient:
    """One judge, one role, one endpoint."""

    def __init__(
        self,
        role: str = "edit_verifier",
        base_url: str = DEFAULT_BASE_URL,
        cache: JudgeCache | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.judge: Judge = for_role(role)
        self.role = role
        self.base_url = base_url.rstrip("/")
        self.cache = cache or JudgeCache()
        self.timeout = timeout
        self.calls = 0
        self.cache_hits = 0

    def ask(
        self,
        question_text: str,
        image_png: bytes,
        *,
        prompt_version: str,
        question_key: str,
        max_tokens: int = MAX_TOKENS,
    ) -> JudgeAnswer:
        key = cache_key(self.judge.hf_path, prompt_version, image_png, question_key)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return JudgeAnswer(text=hit, cached=True, latency_ms=0)

        payload = {
            "model": self.judge.hf_path,
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(image_png).decode()
                            },
                        },
                        {"type": "text", "text": question_text},
                    ],
                }
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"judge endpoint {self.base_url} unreachable: {exc}. "
                f"Start it with shared/judges/serve.sh {self.role}"
            ) from exc
        latency = int((time.perf_counter() - started) * 1000)
        text = body["choices"][0]["message"]["content"] or ""
        self.calls += 1
        self.cache.put(key, text, self.judge.hf_path, question_key)
        return JudgeAnswer(text=text, cached=False, latency_ms=latency)

    def verifier_row(
        self,
        *,
        window_sha256: str,
        question_key: str,
        prompt_version: str,
        answer: JudgeAnswer,
    ) -> dict:
        """One row of ``edits/verifier.parquet`` (design Section 5)."""
        from idea91.relations.verifier_rules import parse_yes_no

        parsed = parse_yes_no(answer.text)
        return {
            "window_sha256": window_sha256,
            "verifier_model": self.judge.name,
            "verifier_commit": self.judge.revision,
            "prompt_version": prompt_version,
            "question": question_key,
            "answer_raw": answer.text,
            "answer_bool": parsed,
            "parsed_ok": parsed is not None,
            "logprob_yes": None,
        }
