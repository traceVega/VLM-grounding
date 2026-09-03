"""Versioned prompt files (SPEC Section 1; design P10, P12, P13, P18).

Every prompt is a file, and ``prompt_version`` on an outputs row is the sha256 of
that file, so a table can always be traced to the exact words the model saw.

Two of the prompts are not ours to write: P12 pins Qwen3-VL's primary protocol to
"GroundingME's own instruction and null-box convention" and Molmo2's to "its
native pointing instruction ... with its native abstention (parsed per its
card)".  Until those are copied from the benchmark repository and the model card,
their files carry ``status: UNVERIFIED`` in the header and :func:`load` refuses to
serve them to a kill run.  A placeholder that silently became "the benchmark's
protocol" is exactly the failure the pre-registration exists to prevent.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from shared import paths

HEADER = re.compile(r"^#\s*(?P<key>[a-z_]+):\s*(?P<value>.*)$")


class PromptError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prompt:
    name: str
    path: Path
    text: str
    meta: dict[str, str]

    @property
    def version(self) -> str:
        """sha256 of the whole file, header included."""
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    @property
    def verified(self) -> bool:
        return self.meta.get("status", "").upper() == "VERIFIED"

    def render(self, **fields) -> str:
        try:
            return self.text.format(**fields)
        except KeyError as exc:
            raise PromptError(f"{self.name}: prompt needs field {exc}") from exc


def _parse(path: Path) -> Prompt:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body: list[str] = []
    for line in raw.splitlines():
        m = HEADER.match(line)
        if m and not body:
            meta[m.group("key")] = m.group("value").strip()
        elif line.strip() or body:
            body.append(line)
    return Prompt(name=path.stem, path=path, text="\n".join(body).strip(), meta=meta)


def load(name: str, *, non_kill: bool = False, root: Path | None = None) -> Prompt:
    """Load a prompt.  A kill run refuses an UNVERIFIED file."""
    root = root or paths.PROMPTS_ROOT
    path = (root / name).with_suffix(".txt") if not name.endswith(".txt") else root / name
    if not path.is_file():
        raise PromptError(f"no prompt file at {path}")
    prompt = _parse(path)
    if not prompt.verified and not non_kill:
        raise PromptError(
            f"{path.name} is marked '{prompt.meta.get('status', 'missing status')}': "
            f"{prompt.meta.get('todo', 'copy the authoritative text before a kill run')}. "
            "Use --non-kill for smoke and dev-slice runs."
        )
    return prompt


def hashes(names: list[str], root: Path | None = None) -> dict[str, str]:
    """The ``prompt_hashes`` map a manifest carries (SPEC Section 2)."""
    return {n: load(n, non_kill=True, root=root).version for n in names}
