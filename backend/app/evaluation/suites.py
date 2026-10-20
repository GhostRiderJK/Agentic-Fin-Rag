"""Evaluation suites the HTTP API is allowed to read.

The API takes a suite *name*, never a path. The JSONL loader reads whatever it
is handed, so a caller-supplied path would let an admin have the server open
any file on disk — this module is the allowlist that stops that.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import NotFoundError, ValidationError

PACKAGED_SUITE = Path(__file__).resolve().parent.parent / "seed" / "gold_qa.jsonl"
PACKAGED_ID = "packaged"

# No separators, so an id can only ever name a file directly inside the suite
# directory once ``.jsonl`` is appended.
_SUITE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")

# Accepted as the packaged suite for callers that still say "default".
_ALIASES = {"", "default", PACKAGED_ID}


def suites_dir() -> Path:
    return Path(get_settings().eval_suites_dir).resolve()


def resolve(suite_id: str) -> tuple[Path, str]:
    """Return (path, canonical id) for a known suite, or raise NotFound/Validation."""
    sid = (suite_id or "").strip()
    if sid in _ALIASES:
        return PACKAGED_SUITE, PACKAGED_ID
    if not _SUITE_ID_RE.fullmatch(sid):
        raise ValidationError(f"not a valid suite name: {suite_id!r}")
    path = suites_dir() / f"{sid}.jsonl"
    if not path.is_file():
        raise NotFoundError(f"eval suite not found: {sid}")
    return path, sid


def list_suites() -> list[dict]:
    out = [{"id": PACKAGED_ID, "packaged": True,
            "n_items": _count_items(PACKAGED_SUITE)}]
    d = suites_dir()
    if d.is_dir():
        out.extend(
            {"id": p.stem, "packaged": False, "n_items": _count_items(p)}
            for p in sorted(d.glob("*.jsonl"))
        )
    return out


def _count_items(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    return sum(1 for line in text.splitlines() if line.strip())
