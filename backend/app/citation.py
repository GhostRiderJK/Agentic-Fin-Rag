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
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_UNICODE_CITATION_RE = re.compile(r"【(\d+)】")
_MALFORMED_CITATION_RE = re.compile(r"【[^】]*】|\[(?!\d+\])[^\]]*\]")


def canonicalize_citation_markers(text: str) -> str:
    """Normalize numeric full-width citation markers to canonical ASCII [n].

    This is syntax normalization only: it never adds, changes, or reassigns an
    index, and all indexes are still resolved against the supplied context.
    """
    return _UNICODE_CITATION_RE.sub(lambda match: f"[{match.group(1)}]", text)
_REJECT_RATIO = 0.5  # >50% unsupported → reject
_CURRENCY_RE = re.compile(r"\b(?:SAR|USD|CAD|AUD|EUR|GBP|JPY|CNY|INR)\b", re.I)
_SCALE_RE = re.compile(r"\b(?:thousand|million|billion|trillion|bn)\b", re.I)
_PERCENT_RE = re.compile(r"%|\bpercent(?:age)?\b", re.I)
_PERCENTAGE_POINT_RE = re.compile(r"\bpercentage\s+points?\b", re.I)


class CitationVerifier:
    """Lightweight NLI verifier."""

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self.llm = llm

    def verify(self, draft: str, chunks: list[RetrievedChunk]) -> VerificationResult:
        draft = canonicalize_citation_markers(draft)
        claims = self._split_claims(draft)
        if not claims:
            return VerificationResult(
                outcome=VerificationOutcome.NOT_APPLICABLE,
                note="no claims found in draft",
            )
        supported: list[str] = []
        unsupported: list[str] = []
        missing_or_invalid: list[str] = []
        for claim in claims:
            if _MALFORMED_CITATION_RE.search(claim):
                unsupported.append(claim)
                missing_or_invalid.append("claim has a malformed citation marker")
                continue
            references = [int(value) for value in _CITATION_RE.findall(claim)]
            if not references:
                unsupported.append(claim)
                missing_or_invalid.append("claim has no citation marker")
                continue
            if any(ref < 1 or ref > len(chunks) for ref in references):
                unsupported.append(claim)
                missing_or_invalid.append("claim references an unavailable citation")
                continue
            cited_chunks = [chunks[ref - 1] for ref in dict.fromkeys(references)]
            evidence = [c.evidence_text() for c in cited_chunks if c.evidence_text()]
            context = "\n\n".join(evidence)
            if not evidence:
                unsupported.append(claim)
                missing_or_invalid.append("cited evidence is empty")
                continue
            judge_claim = _CITATION_RE.sub("", claim).strip()
            claim_supported, claim_unsupported = self._check_claims(
                [judge_claim], context, evidence
            )
            if claim_unsupported or not claim_supported:
                unsupported.append(claim)
            else:
                supported.append(claim)
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
            note=(f"{len(supported)}/{len(claims)} claims supported by their cited "
                  "chunks" + (f"; {'; '.join(dict.fromkeys(missing_or_invalid))}"
                              if missing_or_invalid else "")),
        )

    # ------------------------------------------------------------------
    def _split_claims(self, text: str) -> list[str]:
        out: list[str] = []
        for raw in _CLAIM_SPLIT_RE.split(text):
            s = raw.strip().strip("•*-").strip()
            s = re.sub(r"\s+", " ", s)
            if not s:
                continue
            # Skip pure non-claim lines (headers, source lists, disclaimers)
            low = s.lower()
            if low.startswith(("source", "note:", "sources", "warning", "⚠",
                               "offline test stub:")):
                continue
            out.append(s)
        return out

    def _check_claims(
        self, claims: list[str], context: str, evidence: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        # The lexical/LLM judge is not allowed to override an exact financial
        # value or unit conflict. Check those deterministically first.
        safe_claims = []
        unsupported = []
        for claim in claims:
            if _has_numeric_or_unit_mismatch(claim, context, evidence or [context]):
                unsupported.append(claim)
            else:
                safe_claims.append(claim)
        if not safe_claims:
            return [], unsupported
        # Prefer the LLM judge when available; otherwise fall back to overlap.
        if self.llm is not None and self.llm.name not in {"stub", "base"}:
            try:
                supported, rejected = self._llm_judge(safe_claims, context)
                return supported, unsupported + rejected
            except Exception:  # pragma: no cover
                pass
        supported, rejected = self._heuristic_judge(safe_claims, context)
        return supported, unsupported + rejected

    def _heuristic_judge(self, claims: list[str], context: str) -> tuple[list[str], list[str]]:
        supported: list[str] = []
        unsupported: list[str] = []
        ctx_tokens = _tokenize_set(context)
        for claim in claims:
            claim_tokens = _tokenize_set(claim)
            if not claim_tokens:
                continue
            # Overlap-based entailment approximation
            overlap = len(claim_tokens & ctx_tokens) / max(1, len(claim_tokens))
            if overlap >= 0.55:
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
        # Malformed or incomplete judge output must fail closed. In
        # particular, bool("false") is True in Python and cannot be accepted.
        if not isinstance(data, list) or len(data) != len(claims):
            return [], list(claims)
        decisions: dict[int, bool] = {}
        for entry in data:
            if not isinstance(entry, dict):
                return [], list(claims)
            idx = entry.get("index")
            decision = entry.get("supported")
            if (type(idx) is not int or not 1 <= idx <= len(claims)
                    or type(decision) is not bool or idx in decisions):
                return [], list(claims)
            decisions[idx] = decision
        if len(decisions) != len(claims):
            return [], list(claims)
        supported = [claim for idx, claim in enumerate(claims, start=1)
                     if decisions[idx]]
        unsupported = [claim for idx, claim in enumerate(claims, start=1)
                       if not decisions[idx]]
        return supported, unsupported


def _tokenize_set(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _normalized_numbers(text: str) -> set[str]:
    """Return exact numeric values with grouping commas removed."""
    return {value.replace(",", "") for value in _NUMBER_RE.findall(text)}


def _has_numeric_or_unit_mismatch(
    claim: str, context: str, evidence: list[str] | None = None,
) -> bool:
    claim_numbers = _normalized_numbers(claim)
    context_numbers = _normalized_numbers(context)
    if not claim_numbers.issubset(context_numbers):
        return True

    # Bind each stated financial unit to the same amount in one cited chunk.
    # Units elsewhere in another cited page cannot be borrowed for this value.
    chunks = evidence if evidence is not None else [context]
    for amount in _NUMBER_RE.finditer(claim):
        before = claim[max(0, amount.start() - 24):amount.start()]
        after = claim[amount.end():amount.end() + 32]
        currency_match = re.search(r"\b(SAR|USD|CAD|AUD|EUR|GBP|JPY|CNY|INR)\s*$", before, re.I)
        scale_match = re.match(r"\s*(thousand|million|billion|trillion|bn)\b", after, re.I)
        if not currency_match and not scale_match:
            continue
        found = False
        for chunk in chunks:
            amount_pattern = r"(?<!\w)" + re.escape(amount.group(0)) + r"(?!\w)"
            for occurrence in re.finditer(amount_pattern, chunk, re.I):
                neighborhood = chunk[max(0, occurrence.start() - 32):occurrence.end() + 40]
                if currency_match:
                    requested = currency_match.group(1).casefold()
                    currencies = {
                        m.group(0).casefold()
                        for m in _CURRENCY_RE.finditer(neighborhood)
                    }
                    # The source PDF's ª glyph is the visually confirmed SAR symbol.
                    preceding = chunk[max(0, occurrence.start() - 4):occurrence.start()]
                    if requested == "sar" and re.search(r"ª\s*$", preceding):
                        currencies.add("sar")
                    if requested not in currencies:
                        continue
                if scale_match:
                    requested_scale = scale_match.group(1).casefold()
                    scales = {m.group(0).casefold() for m in _SCALE_RE.finditer(neighborhood)}
                    if requested_scale == "billion":
                        scales.add("billion") if "bn" in scales else None
                    if requested_scale == "bn":
                        scales.add("bn") if "billion" in scales else None
                    if requested_scale not in scales:
                        continue
                found = True
                break
            if found:
                break
        if not found:
            return True

    if _PERCENTAGE_POINT_RE.search(claim) and not _PERCENTAGE_POINT_RE.search(context):
        return True
    if _PERCENT_RE.search(claim) and not _PERCENT_RE.search(context):
        return True
    return False
