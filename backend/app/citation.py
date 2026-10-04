"""Citation verification (PRD v2 §7 + PRD §9 critique: "Citations are just
format, not a guarantee of correctness").

We split the draft answer into atomic claims, then check each claim against
the cited chunks via a small Natural Language Inference (NLI) check:

  - SUPPORTED: claim is entailed by at least one cited chunk
  - UNSUPPORTED: not entailed by any cited chunk

The NLI check is implemented two ways:
  1. A heuristic lexical/number-overlap check (default, deterministic)
  2. An LLM-as-judge fallback (used when an LLM provider is configured)

When UNSUPPORTED claims exceed a threshold the outcome is REJECTED and the
orchestrator flags the answer for the user.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field

from app.providers.llm import BaseLLM, Message
from app.retrieval.hybrid import RetrievedChunk


class VerificationOutcome(str, enum.Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    REJECTED = "rejected"
    NOT_APPLICABLE = "not_applicable"  # no chunks / no answer


@dataclass
class VerificationResult:
    outcome: VerificationOutcome
    supported: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    note: str = ""


# Claim-splitting: simple sentence-ish boundaries.
_CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_REJECT_RATIO = 0.5  # >50% unsupported → reject


class CitationVerifier:
    """Lightweight NLI verifier."""

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self.llm = llm

    def verify(self, draft: str, chunks: list[RetrievedChunk]) -> VerificationResult:
        claims = self._split_claims(draft)
        if not claims:
            return VerificationResult(
                outcome=VerificationOutcome.NOT_APPLICABLE,
                note="no claims found in draft",
            )
        context = "\n\n".join(
            (c.parent_text or c.text) for c in chunks if c.parent_text or c.text
        )
        if not context:
            return VerificationResult(
                outcome=VerificationOutcome.NOT_APPLICABLE,
                note="no context to verify against",
            )

        supported, unsupported = self._check_claims(claims, context)
        if not claims:
            ratio = 0.0
        else:
            ratio = len(unsupported) / len(claims)
        if not unsupported:
            outcome = VerificationOutcome.SUPPORTED
        elif ratio >= _REJECT_RATIO:
            outcome = VerificationOutcome.REJECTED
        else:
            outcome = VerificationOutcome.PARTIAL
        return VerificationResult(
            outcome=outcome,
            supported=supported,
            unsupported=unsupported,
            note=f"{len(supported)}/{len(claims)} claims supported",
        )

    # ------------------------------------------------------------------
    def _split_claims(self, text: str) -> list[str]:
        out: list[str] = []
        for raw in _CLAIM_SPLIT_RE.split(text):
            s = raw.strip().strip("•*-").strip()
            if len(s) < 12:
                continue
            # Skip pure non-claim lines (headers, source lists, disclaimers)
            low = s.lower()
            if low.startswith(("source", "note:", "sources", "warning", "⚠")):
                continue
            out.append(s)
        return out

    def _check_claims(self, claims: list[str], context: str) -> tuple[list[str], list[str]]:
        # Prefer the LLM judge when available; otherwise fall back to overlap.
        if self.llm is not None and self.llm.name not in {"stub", "base"}:
            try:
                return self._llm_judge(claims, context)
            except Exception as e:  # pragma: no cover
                pass
        return self._heuristic_judge(claims, context)

    def _heuristic_judge(self, claims: list[str], context: str) -> tuple[list[str], list[str]]:
        supported: list[str] = []
        unsupported: list[str] = []
        ctx_tokens = _tokenize_set(context)
        for claim in claims:
            claim_tokens = _tokenize_set(claim)
            numbers = set(_NUMBER_RE.findall(claim))
            if not claim_tokens:
                continue
            # Overlap-based entailment approximation
            overlap = len(claim_tokens & ctx_tokens) / max(1, len(claim_tokens))
            number_ok = True
            if numbers:
                ctx_numbers = set(_NUMBER_RE.findall(context))
                # Allow claim numbers to be a subset of context numbers
                # (rounded/abbreviated forms may differ; require ≥50%)
                matched = sum(1 for n in numbers if n in ctx_numbers
                              or any(n.startswith(cn[:3]) for cn in ctx_numbers))
                number_ok = matched / len(numbers) >= 0.5
            if overlap >= 0.55 and number_ok:
                supported.append(claim)
            else:
                unsupported.append(claim)
        return supported, unsupported

    def _llm_judge(self, claims: list[str], context: str) -> tuple[list[str], list[str]]:
        import json

        joined = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        prompt = (
            "You are a strict fact-checker. For each claim below, decide if it "
            "is SUPPORTED by the provided context. Return a JSON list of "
            "objects: [{\"index\": int, \"supported\": bool}].\n\n"
            f"Claims:\n{joined}\n\nContext:\n{context[:6000]}\n"
        )
        msgs = [Message(role="user", content=prompt)]
        resp = self.llm.chat(msgs, temperature=0.0)
        text = (resp.content or "").strip()
        # extract first JSON list
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return self._heuristic_judge(claims, context)
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return self._heuristic_judge(claims, context)
        supported: list[str] = []
        unsupported: list[str] = []
        for entry in data:
            try:
                idx = int(entry["index"]) - 1
            except (KeyError, ValueError, TypeError):
                continue
            if 0 <= idx < len(claims):
                if bool(entry.get("supported")):
                    supported.append(claims[idx])
                else:
                    unsupported.append(claims[idx])
        return supported, unsupported


def _tokenize_set(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))
