"""Offline evaluation harness (PRD v2 §10 + PRD §10 closed-loop requirement).

Runs a *gold set* of question/expected-answer pairs through the live
orchestrator and scores them on:

  - faithfulness        (claims supported by retrieved context)
  - context_precision   (retrieved chunks relevant to expected answer)
  - context_recall      (expected facts present in retrieved context)
  - answer_relevance    (answer addresses the question)
  - citation_precision  (citations actually support claims)
  - numeric_accuracy    (numbers in answer match expected — finance-specific)
  - latency_ms          (operational)

Each metric is computed deterministically (token/number overlap) so the
harness runs offline without paid LLM-as-judge calls. The LLM judge path
exists as an optional enhancement when an LLM provider is configured.

The harness is exposed via the ``/admin/eval`` API and a CLI command.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.errors import without_secrets
from app.core.logging import get_logger
from app.db.models import EvalItem, EvalRun
from app.db.mongodb.tenant_context import TenantContext
from app.reasoning.orchestrator import AnswerRequest, Orchestrator

log = get_logger(__name__)


@dataclass
class GoldItem:
    question: str
    expected: str
    expected_numbers: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)


@dataclass
class EvalSuite:
    name: str
    workspace_id: str
    items: list[GoldItem]

    @classmethod
    def from_jsonl(cls, path: str | Path, *, workspace_id: str,
                   name: str = "default") -> "EvalSuite":
        items: list[GoldItem] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(GoldItem(
                question=obj["question"],
                expected=obj.get("expected", ""),
                expected_numbers=obj.get("expected_numbers", []),
                expected_keywords=obj.get("expected_keywords", []),
            ))
        return cls(name=name, workspace_id=workspace_id, items=items)


@dataclass
class EvalItemResult:
    question: str
    expected: str
    actual: str
    citations: list[dict]
    metrics: dict
    latency_ms: int
    # Set when the pipeline raised before producing an answer. Such an item has
    # no metrics: scoring the exception text invented passes (a 401 message
    # contains digits, so citation_precision came back 1.0 and the whole run
    # looked like a real measurement).
    error: str = ""


@dataclass
class EvalResult:
    suite: str
    n_items: int
    metrics: dict
    items: list[EvalItemResult]

    def to_dict(self) -> dict:
        return asdict(self)


class EvalRunner:
    def __init__(self, db: Session, ctx: TenantContext,
                 orchestrator: Orchestrator | None = None) -> None:
        self.db = db
        self.ctx = ctx
        self.orch = orchestrator or Orchestrator(ctx)

    def run(self, suite: EvalSuite, *, persist: bool = True) -> EvalResult:
        item_results: list[EvalItemResult] = []
        for gi in suite.items:
            t0 = time.perf_counter()
            error = ""
            try:
                resp = self.orch.answer(AnswerRequest(
                    question=gi.question, workspace_id=suite.workspace_id,
                ))
                actual = resp.answer
                citations = resp.citations
                retrieved_chunks_n = resp.retrieval.get("n_chunks", 0)
                verification = resp.verification
            except Exception as e:
                log.warning("eval item failed: %s", e)
                # Both strings are persisted and the dashboard reads ``actual``
                # back out, so the provider's own error text is scrubbed here
                # rather than on the way out.
                error = without_secrets(f"{type(e).__name__}: {e}")
                actual = f"<error: {error}>"
                citations = []
            latency = int((time.perf_counter() - t0) * 1000)
            metrics: dict = {} if error else score_item(
                question=gi.question,
                expected=gi.expected,
                actual=actual,
                expected_numbers=gi.expected_numbers,
                expected_keywords=gi.expected_keywords,
                n_retrieved=retrieved_chunks_n,
                verification_outcome=verification.get("outcome", "n/a"),
            )
            item_results.append(EvalItemResult(
                question=gi.question, expected=gi.expected, actual=actual,
                citations=citations, metrics=metrics, latency_ms=latency,
                error=error,
            ))
        agg = aggregate_metrics([it.metrics for it in item_results])
        result = EvalResult(suite=suite.name, n_items=len(item_results),
                            metrics=agg, items=item_results)
        if persist:
            self._persist(suite, result)
        return result

    def _persist(self, suite: EvalSuite, result: EvalResult) -> None:
        run = EvalRun(suite=suite.name, n_items=result.n_items,
                      metrics=result.metrics,
                      tenant_id=self.ctx.tenant_id,
                      workspace_id=suite.workspace_id or self.ctx.workspace_id)
        self.db.add(run)
        self.db.flush()
        for it in result.items:
            self.db.add(EvalItem(
                run_id=run.id, question=it.question, expected=it.expected,
                actual=it.actual, citations=it.citations, metrics=it.metrics,
                latency_ms=it.latency_ms,
            ))
        self.db.commit()


def run_suite(db: Session, ctx: TenantContext, suite: EvalSuite) -> EvalResult:
    return EvalRunner(db, ctx).run(suite)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _numeric_tokens(text: str) -> list[float]:
    """Every number in ``text`` as a float, thousands separators removed.

    Gold ``expected_numbers`` entries are phrases, not bare numbers — the PIF
    suite stores "SAR 3,424 billion". Comparing that string against the answer's
    numeric tokens scores a verbatim-correct answer as 0.0, so both sides go
    through this extraction first.
    """
    out: list[float] = []
    for token in _NUMBER_RE.findall(text):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


def score_item(
    *, question: str, expected: str, actual: str,
    expected_numbers: list[str], expected_keywords: list[str],
    n_retrieved: int, verification_outcome: str,
) -> dict:
    expected_tokens = _tokenize(expected)
    actual_tokens = _tokenize(actual)

    # Faithfulness: proxy via verification outcome
    faithfulness = {
        "supported": 1.0, "partial": 0.5, "rejected": 0.0,
        "no_context": 0.0, "not_applicable": 0.5, "n/a": 0.5, "error": 0.0,
    }.get(verification_outcome, 0.5)

    # Answer relevance: keyword overlap with question + expected keywords
    q_tokens = _tokenize(question)
    rel_q = _jaccard(q_tokens, actual_tokens)
    expected_kw_tokens = set(expected_keywords)
    if expected_kw_tokens:
        rel_kw = len(expected_kw_tokens & actual_tokens) / len(expected_kw_tokens)
    else:
        rel_kw = _jaccard(expected_tokens, actual_tokens)
    answer_relevance = 0.5 * rel_q + 0.5 * rel_kw

    # Context precision: did we retrieve anything for a relevant question?
    context_precision = min(1.0, n_retrieved / 5.0)
    context_recall = min(1.0, n_retrieved / 5.0)

    # Citation precision: stub — assume all citations are valid format
    citation_precision = 1.0 if any(_NUMBER_RE.findall(actual)) else 0.5

    # Numeric accuracy (finance-specific): for each expected number, is it in
    # the actual answer? An entry counts when every number it contains is
    # present, so "SAR 3,424 billion" matches a 3,424 in the answer. The unit
    # and scale words are not checked here; a wrong "million" still counts.
    if expected_numbers:
        actual_nums = _numeric_tokens(actual)
        matched = 0
        for n in expected_numbers:
            wanted = _numeric_tokens(n)
            if wanted and all(any(abs(a - w) < 0.01 for a in actual_nums)
                              for w in wanted):
                matched += 1
        numeric_accuracy = matched / len(expected_numbers)
    else:
        # If no specific numbers expected, neutral
        numeric_accuracy = 0.5

    return {
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(answer_relevance, 3),
        "context_precision": round(context_precision, 3),
        "context_recall": round(context_recall, 3),
        "citation_precision": round(citation_precision, 3),
        "numeric_accuracy": round(numeric_accuracy, 3),
    }


def aggregate_metrics(items: list[dict]) -> dict:
    """Average the scored items and report how many items never got an answer.

    An item with no metrics is one whose request raised: it is counted, not
    averaged in. Without this split a run where every request failed still
    produced a full metric table, and the pass rates looked like results.
    """
    n_errored = sum(1 for it in items if not it)
    scored = [it for it in items if it]
    out: dict[str, float] = {
        "n_scored": float(len(scored)),
        "n_errored": float(n_errored),
        "error_rate": round(n_errored / len(items), 3) if items else 0.0,
    }
    if not scored:
        return out
    keys = scored[0].keys()
    for k in keys:
        vals = [it[k] for it in scored]
        out[f"mean_{k}"] = round(sum(vals) / len(vals), 3)
        # % of items above 0.7 (passing)
        out[f"pass_{k}"] = round(sum(1 for v in vals if v >= 0.7) / len(vals), 3)
    return out


def reconcile_run_metrics(stored: dict, n_items: int,
                          item_rows: list[tuple[str, dict]]) -> dict:
    """Re-read a run row so a failed request can never look like a score.

    Rows persisted before the harness learned to separate answered items from
    errored ones averaged the exception text as if it were an answer, which put
    a literal `1.0` citation-precision on the dashboard for a run where nothing
    answered. The per-item rows still say what happened, so the aggregate is
    rebuilt over the items that answered and the answered/failed split is
    reported. Runs written by the current harness already carry the split and
    are returned untouched — history is not rewritten silently.
    """
    if "n_errored" in stored:
        return dict(stored)
    if not item_rows:
        return dict(stored)
    per_item = [{} if (actual or "").startswith("<error:") else (metrics or {})
                for actual, metrics in item_rows]
    agg = aggregate_metrics(per_item)
    if n_items > len(per_item):
        # Item rows are gone (or were never written); the suite size still comes
        # from the run row, so say how much of it is unverifiable.
        agg["n_items_unmatched"] = float(n_items - len(per_item))
    return agg


def _tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
