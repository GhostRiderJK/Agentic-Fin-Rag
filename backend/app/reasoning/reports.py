"""Report generators: earnings summary + investment thesis (PRD §5.7, §5.8).

These are heavier, async-run reports. They run the retrieval pipeline over
several sections and synthesize a structured report. Same agent/tool
abstractions as the chat orchestrator.
"""

from __future__ import annotations

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
    seen_chunk_ids: set[str] = set()
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
        context = result.context_text(max_chunks=5) if result.chunks else ""
        # collect citations
        for c in result.citation_list():
            if c["chunk_id"] not in seen_chunk_ids:
                citations.append(c)
                seen_chunk_ids.add(c["chunk_id"])

        prompt = (
            f"You are an equity research analyst writing the '{section}' "
            f"section of a {report_type} for {company} {period}. "
            f"Use the context below. Cite as [n]. If missing, say so briefly.\n\n"
            f"Context:\n{context}\n\nSection ({section}):"
        )
        msgs = [Message(role="user", content=prompt)]
        text, _ = run_tool_loop(llm, msgs, tools=tool_specs, dispatch=dispatch,
                                max_iterations=1)
        pieces.append(f"## {section}\n\n{text.strip() or '(insufficient data)'}")

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
        lines.append(
            f"[{i}] {c.get('company') or 'Unknown'} "
            f"{c.get('year') or ''} {c.get('quarter') or ''} "
            f"{c.get('doc_type') or ''} p.{c.get('page', '?')}"
        )
    return "\n".join(lines)
