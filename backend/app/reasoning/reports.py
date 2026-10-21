"""Report generators: earnings summary + investment thesis (PRD §5.7, §5.8).

These are heavier, async-run reports. They run the retrieval pipeline over
several sections and synthesize a structured report. Same agent/tool
abstractions as the chat orchestrator.
"""

from __future__ import annotations

import re

from app.db.mongodb.tenant_context import TenantContext
from app.providers.llm import BaseLLM, Message, get_llm
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_understanding import understand_query
from app.reasoning.tools import build_dispatcher, list_metrics
from app.reasoning.agents import run_tool_loop


SUMMARY_SECTIONS = [
    "Revenue and growth",
    "Earnings per share",
    "Cash flow highlights",
    "Key risks",
    "Outlook and guidance",
]

THESIS_SECTIONS = [
    "Business overview",
    "Growth drivers",
    "Risks and mitigants",
    "Valuation",
    "Competitive advantage",
    "Near-term catalysts",
]


def build_report(
    ctx: TenantContext, *, company: str,
    report_type: str = "summary", year: int | None = None,
    quarter: str | None = None,
) -> dict:
    llm = get_llm()
    retriever = HybridRetriever(ctx)
    period = ""
    if year:
        period = f"{year}"
        if quarter:
            period = f"{quarter} {year}"

    sections = SUMMARY_SECTIONS if report_type == "summary" else THESIS_SECTIONS
    pieces: list[str] = []
    citations: list[dict] = []
    # chunk_id -> its number in ``citations``. Every section retrieves its own
    # top chunks and the model cites them [1]..[n] against *that* section's
    # context, so those local tags are remapped to report-wide numbers below.
    number_by_chunk: dict[str, int] = {}
    metrics_snapshot: dict = {}
    try:
        metrics_snapshot = list_metrics(ctx, company=company, fiscal_year=year)
    except Exception:
        pass

    dispatch, tool_specs = build_dispatcher(ctx)

    for section in sections:
        q = f"{company} {period} {section}".strip()
        plan = understand_query(q)
        if year:
            plan.years = [year]
        result = retriever.retrieve(q, plan=plan)
        # Only the chunks actually serialized into the prompt can be cited.
        # ``result.chunks`` is wider than that and includes budget-truncated
        # and never-supplied retrievals.
        context, included = result.context_with_chunks(max_chunks=5)
        local_to_report: dict[int, int] = {}
        for local, cit in enumerate(
                result.citation_list(chunks=included), start=1):
            number = number_by_chunk.get(cit["chunk_id"])
            if number is None:
                number = len(citations) + 1
                number_by_chunk[cit["chunk_id"]] = number
                citations.append({**cit, "index": number})
            local_to_report[local] = number

        prompt = (
            f"You are an equity research analyst writing the '{section}' "
            f"section of a {report_type} for {company} {period}. "
            f"Use the context below. Cite as [n]. If missing, say so briefly.\n\n"
            f"Context:\n{context}\n\nSection ({section}):"
        )
        msgs = [Message(role="user", content=prompt)]
        text, _ = run_tool_loop(llm, msgs, tools=tool_specs, dispatch=dispatch,
                                max_iterations=1)
        body = (text.strip() or "(insufficient data)")
        pieces.append(f"## {section}\n\n{_renumber(body, local_to_report)}")

    header = (f"# {'Earnings Summary' if report_type == 'summary' else 'Investment Thesis'}"
              f" — {company} {period}".strip())
    body = "\n\n".join(pieces)
    sources = _format_sources(citations)
    report = f"{header}\n\n{body}\n\n---\n\n## Sources\n{sources}"
    return {
        "report": report,
        "citations": citations,
        "metrics": metrics_snapshot,
    }


def _format_sources(citations: list[dict]) -> str:
    if not citations:
        return "No sources available."
    lines = []
    for i, c in enumerate(citations, start=1):
        label = c.get("company") or c.get("source_filename") or "Unknown"
        parts = " ".join(str(p) for p in
                         (label, c.get("year"), c.get("quarter"), c.get("doc_type"))
                         if p)
        lines.append(f"[{i}] {parts} p.{c.get('page', '?')}")
    return "\n".join(lines)


_CITATION_MARK = re.compile(r"\[(\d{1,3})\]")
# What a dropped marker leaves behind: "saw it [9]." becomes "saw it ."
_HSPACE_RUN = re.compile(r"[^\S\n]{2,}")
_ORPHAN_PUNCT = re.compile(r"[^\S\n]+([.,;:])")


def _renumber(text: str, local_to_report: dict[int, int]) -> str:
    """Rewrite one section's ``[1]``-style markers to report-wide numbers.

    A bracket that carries no local meaning was invented by the model, so it is
    dropped rather than kept: in the assembled report ``[9]`` would otherwise
    resolve to whichever source happens to be ninth overall, citing a document
    that section never saw. A four-digit bracket is a year and never matches.
    """
    def repl(match: re.Match[str]) -> str:
        number = local_to_report.get(int(match.group(1)))
        return "" if number is None else f"[{number}]"

    out = _CITATION_MARK.sub(repl, text)
    if out == text:
        return out
    return _ORPHAN_PUNCT.sub(r"\1", _HSPACE_RUN.sub(" ", out))
