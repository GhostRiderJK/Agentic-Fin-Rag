"""Parent-child chunking strategy (PRD v2 §5).

We split documents into:

  Section (parent unit)
    └─ Paragraph (parent chunk — what the LLM sees)
        └─ Sentence group / small window (child chunk — what we embed)

Retrieval searches over child chunks; on a hit we expand to the parent for
the LLM context. This gives accurate retrieval (small embedded units) and
strong reasoning context (large parent units).

We also keep the source table markdown attached to whichever chunk overlaps
the table's page so the LLM sees the structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_SECTION_HEADERS_RE = re.compile(
    r"(?m)^\s{0,4}(?P<header>(?:[0-9]+\.?){1,4}\s+)?"
    r"(?P<title>(?:ITEM\s+\d+[A-Z]?|RISK FACTORS|MANAGEMENT[\'\u2019]S DISCUSSION|"
    r"BUSINESS OVERVIEW|FINANCIAL HIGHLIGHTS|CONSOLIDATED|OUTLOOK|"
    r"CORPORATE GOVERNANCE|INCOME STATEMENT|BALANCE SHEET|CASH FLOW|"
    r"BUSINESS STRATEGY|EXECUTIVE SUMMARY|RESULTS OF OPERATIONS|"
    r"PROPERTIES|MARKET FOR REGISTRANT|NOTE\s+\d+)[A-Z0-9 \-\u2013]{0,80})\s*$"
)

# Common "section" patterns used in earnings transcripts
_TRANSCRIPT_HEADER_RE = re.compile(
    r"(?m)^\s*(?P<title>(?:OPERATOR|QUESTION AND ANSWER|PREPARED REMARKS|"
    r"MANAGEMENT DISCUSSION|OPENING REMARKS|CONFERENCE CALL|EARNINGS CALL)[A-Z ]{0,60})\s*$"
)

_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

_CHILD_TARGET_CHARS = 350   # roughly ~80-100 tokens
_CHILD_MAX_CHARS = 600
_PARENT_MAX_CHARS = 2400


@dataclass
class ChildChunk:
    text: str
    section: str
    page: int
    char_start: int
    char_end: int
    tables_md: str = ""


@dataclass
class ParentChunk:
    text: str
    section: str
    page: int
    char_start: int
    char_end: int
    tables_md: str = ""
    children: list[ChildChunk] = field(default_factory=list)


@dataclass
class ChunkingResult:
    parents: list[ParentChunk]


def chunk_document(
    text: str,
    *,
    tables_by_page: dict[int, list[str]] | None = None,
) -> ChunkingResult:
    """Top-level entry: split ``text`` into parent chunks with children."""
    tables_by_page = tables_by_page or {}
    sections = _split_sections(text)
    parents: list[ParentChunk] = []
    for section_title, section_text, section_page in sections:
        for paragraph in _split_paragraphs(section_text):
            if len(paragraph.strip()) < 30:
                continue
            tables_md = _tables_for(section_page, tables_by_page)
            parent = ParentChunk(
                text=paragraph.strip()[:_PARENT_MAX_CHARS],
                section=section_title,
                page=section_page,
                char_start=0,
                char_end=len(paragraph),
                tables_md=tables_md,
            )
            for child in _split_children(paragraph):
                parent.children.append(
                    ChildChunk(
                        text=child.strip(),
                        section=section_title,
                        page=section_page,
                        char_start=0,
                        char_end=len(child),
                        tables_md="",  # only attach tables to parent
                    )
                )
            if not parent.children:
                # paragraph too short to split — embed it directly as one child
                parent.children.append(
                    ChildChunk(
                        text=parent.text,
                        section=section_title,
                        page=section_page,
                        char_start=0,
                        char_end=len(parent.text),
                    )
                )
            parents.append(parent)
    # Fallback: if no sections matched, treat the whole text as one parent
    if not parents and text.strip():
        parent = ParentChunk(text=text.strip()[:_PARENT_MAX_CHARS], section="",
                             page=1, char_start=0, char_end=len(text))
        for child in _split_children(text):
            parent.children.append(ChildChunk(text=child, section="", page=1,
                                              char_start=0, char_end=len(child)))
        if not parent.children:
            parent.children.append(ChildChunk(text=parent.text, section="", page=1,
                                              char_start=0, char_end=len(parent.text)))
        parents.append(parent)
    return ChunkingResult(parents=parents)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _split_sections(text: str) -> list[tuple[str, str, int]]:
    """Return [(section_title, section_text, page=1), ...].

    Sections are detected by ALL-CAPS / 'ITEM N' style headers. Text before
    the first header is assigned section "Introduction". Pages are not yet
    tracked at this layer; page detection happens earlier (parser) — we keep
    section-level page=1 here and rely on metadata extraction to attach
    page numbers if available.
    """
    matches = list(_SECTION_HEADERS_RE.finditer(text)) or list(
        _TRANSCRIPT_HEADER_RE.finditer(text)
    )
    if not matches:
        return [("", text, 1)]
    sections: list[tuple[str, str, int]] = []
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append(("Introduction", pre, 1))
    for i, m in enumerate(matches):
        title = (m.group("title") or "").strip()[:120]
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append((title, body, 1))
    return sections or [("", text, 1)]


def _split_paragraphs(section_text: str) -> list[str]:
    parts = _PARAGRAPH_SPLIT_RE.split(section_text)
    out: list[str] = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Glue short list/table-ish lines together until ~PARENT_MAX
        if len(buf) + len(p) + 2 <= _PARENT_MAX_CHARS:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def _split_children(paragraph: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 > _CHILD_MAX_CHARS and buf:
            chunks.append(buf)
            buf = s
        else:
            buf = (buf + " " + s) if buf else s
        if len(buf) >= _CHILD_TARGET_CHARS:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _tables_for(page: int, tables_by_page: dict[int, list[str]]) -> str:
    tbls = tables_by_page.get(page) or []
    if not tbls:
        return ""
    return "\n\n".join(tbls)
