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
_CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_UNICODE_CITATION_RE = re.compile(r"【(\d+)】")
_MALFORMED_CITATION_RE = re.compile(r"【[^】]*】|\[(?!\d+\])[^\]]*\]")

# A period inside one of these is abbreviation punctuation, not a sentence end.
_ABBREV_SENTINEL = "\x01"
_ABBREV_DOT_RE = re.compile(
    r"\b(?:vs|cf|approx|inc|ltd|corp|co|no|nos|fig|vol|pp|et al|e\.g|i\.e|u\.s)"
    r"\.(?=\s|$)",
    re.I,
)


def _mask_abbreviation_dots(text: str) -> str:
    return _ABBREV_DOT_RE.sub(
        lambda m: m.group(0)[:-1] + _ABBREV_SENTINEL, text)


def _unmask_abbreviation_dots(text: str) -> str:
    return text.replace(_ABBREV_SENTINEL, ".")


_NON_CLAIM_PREFIXES = ("source", "sources", "note:", "warning", "⚠",
                       "offline test stub:")


def _normalize_claim(part: str) -> str:
    """Return a claim's display text, or "" when it asserts nothing.

    List markers and emphasis are kept because the answer is rendered as
    markdown: stripping "- " from every supported bullet made a trimmed answer
    collapse into one run-on paragraph in the UI.
    """
    text = _unmask_abbreviation_dots(part)
    text = re.sub(r"[ \t\xa0]+", " ", text).strip()
    if not text:
        return ""
    bare = text.strip("•*-").strip().lower()
    if bare.startswith(_NON_CLAIM_PREFIXES):
        return ""
    # A digit-free line that ends in a colon introduces the claims below it.
    if bare.endswith((":", "：")) and not _NUMBER_RE.search(text):
        return ""
    return text


@dataclass
class Claim:
    """One verifiable assertion and the context indexes it cites.

    ``text`` carries the markers the claim will be judged by. When a model puts
    the citation only at the end of a bullet, the markers are shared with the
    bullet's other fragments, so they appear on every fragment of that line.
    """

    text: str
    refs: list[int] = field(default_factory=list)

    @property
    def judged_text(self) -> str:
        """The claim with citation syntax removed, as the judge reads it.

        A marker sits mid-sentence, so removing it leaves "rose 20% ." behind;
        the stray space is taken with it, as the judge compares wording, not
        punctuation.
        """
        stripped = _CITATION_RE.sub("", self.text)
        return re.sub(r"\s+([.,;:!?)])", r"\1", stripped).strip()


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
            if _MALFORMED_CITATION_RE.search(claim.text):
                unsupported.append(claim.text)
                missing_or_invalid.append("claim has a malformed citation marker")
                continue
            if not claim.refs:
                unsupported.append(claim.text)
                missing_or_invalid.append("claim has no citation marker")
                continue
            if any(ref < 1 or ref > len(chunks) for ref in claim.refs):
                unsupported.append(claim.text)
                missing_or_invalid.append("claim references an unavailable citation")
                continue
            cited_chunks = [chunks[ref - 1] for ref in dict.fromkeys(claim.refs)]
            evidence = [c.evidence_text() for c in cited_chunks if c.evidence_text()]
            context = "\n\n".join(evidence)
            if not evidence:
                unsupported.append(claim.text)
                missing_or_invalid.append("cited evidence is empty")
                continue
            judge_claim = claim.judged_text
            claim_supported, claim_unsupported = self._check_claims(
                [judge_claim], context, evidence
            )
            if claim_unsupported or not claim_supported:
                unsupported.append(claim.text)
            else:
                supported.append(claim.text)
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
    def _split_claims(self, text: str) -> list[Claim]:
        """Split a draft into claims, keeping each one with its citations.

        Two typography rules, both from live answers that were correct but
        rejected:

        * A line is the unit a model cites. It writes "FY2024 revenue was SAR
          284,005m vs. SAR 237,575m in FY2023 (+20%) [1]": splitting at the
          abbreviation period in "vs." left a fragment with the marker and a
          fragment without one, so one correct bullet scored as two
          unsupported claims. Fragments of a cited line share its markers.
        * A digit-free line ending in a colon is a heading ("Breakdown of the
          change:"), not an assertion, so it cannot be cited and is not a claim.
        """
        claims: list[Claim] = []
        for raw_line in _mask_abbreviation_dots(text).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in _CLAIM_SPLIT_RE.split(line) if p.strip()]
            if not parts:
                parts = [line]
            line_refs = list(dict.fromkeys(
                int(value) for part in parts for value in _CITATION_RE.findall(part)
            ))
            for part in parts:
                own_refs = [int(v) for v in _CITATION_RE.findall(part)]
                normalized = _normalize_claim(part)
                if not normalized:
                    continue
                if own_refs:
                    claims.append(Claim(text=normalized, refs=own_refs))
                elif line_refs:
                    claims.append(Claim(
                        text=f"{normalized} " + "".join(f"[{r}]" for r in line_refs),
                        refs=line_refs,
                    ))
                else:
                    claims.append(Claim(text=normalized))
        return claims

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


_CURRENCY_ALTS = "SAR|USD|CAD|AUD|EUR|GBP|JPY|CNY|INR"
_SCALE_ALTS = "thousand|million|billion|trillion|bn"
# A unit label and the figure it qualifies are separated by whitespace and, in
# markdown evidence, by emphasis or table pipes.
_UNIT_GAP = r"[\s*`|:>\xa0]*"


def _normalized_numbers(text: str) -> set[str]:
    """Return the numeric magnitudes a text asserts.

    Grouping commas and a leading sign are typography, not magnitude: PDF text
    layers lose a sign as often as they keep one, and "(+25%)" and "25%" report
    the same change. Direction lives in the words around the number, which the
    judge reads; this set only answers whether the figure is in the source at
    all.
    """
    return {value.strip("+-").replace(",", "")
            for value in _NUMBER_RE.findall(text)
            if value.strip("+-")}


def _governing_units(text: str, start: int, end: int) -> tuple[str, str]:
    """Return the currency and scale an amount is stated with, on either side.

    "SAR 413,423 million" and "413,423 SAR million" state the same thing, so
    reading units from one fixed position both waves wrong units past the
    check and rejects correct ones, depending on how the answer happened to
    read.
    """
    before = text[max(0, start - 24):start]
    after = text[end:end + 24]
    currency = ""
    trailing = re.search(rf"\b({_CURRENCY_ALTS}){_UNIT_GAP}$", before, re.I)
    leading = re.match(rf"{_UNIT_GAP}({_CURRENCY_ALTS})\b", after, re.I)
    if trailing:
        currency = trailing.group(1).upper()
    elif leading:
        currency = leading.group(1).upper()
    scale = ""
    after_scale = re.match(
        rf"{_UNIT_GAP}(?:(?:{_CURRENCY_ALTS})\b{_UNIT_GAP})?({_SCALE_ALTS})\b",
        after, re.I)
    before_scale = re.search(rf"\b({_SCALE_ALTS}){_UNIT_GAP}$", before, re.I)
    if after_scale:
        scale = after_scale.group(1).lower()
    elif before_scale:
        scale = before_scale.group(1).lower()
    return currency, scale


def _declared_units(chunk: str) -> tuple[set[str], set[str]]:
    """Collect units declared by the digit-free lines of an evidence chunk.

    A financial table names its unit once above the columns ("Amounts in SAR
    million") and leaves every cell bare, so a table figure quoted correctly as
    "SAR 413,423 million" looked like a unit mismatch to a reader that only
    looked beside the number.
    """
    currencies: set[str] = set()
    scales: set[str] = set()
    for line in chunk.splitlines():
        if _NUMBER_RE.search(line):
            continue  # a row that asserts figures asserts no units
        currencies |= {m.group(0).upper() for m in _CURRENCY_RE.finditer(line)}
        scales |= {m.group(0).lower() for m in _SCALE_RE.finditer(line)}
    return currencies, scales


def _units_bound_to_amount(
    chunks: list[str],
    declarations: list[tuple[set[str], set[str]]],
    amount: str,
    currency: str,
    scale: str,
) -> bool:
    """Find one occurrence of the amount whose units cover what was claimed."""
    occurrence_re = re.compile(r"(?<!\w)" + re.escape(amount) + r"(?!\w)")
    for chunk, (declared_currencies, declared_scales) in zip(chunks, declarations):
        for occurrence in occurrence_re.finditer(chunk):
            window = chunk[max(0, occurrence.start() - 32):occurrence.end() + 40]
            currencies = {m.group(0).upper() for m in _CURRENCY_RE.finditer(window)}
            scales = {m.group(0).lower() for m in _SCALE_RE.finditer(window)}
            # The source PDF renders the Riyal sign as an ordinal-a glyph.
            preceding = chunk[max(0, occurrence.start() - 4):occurrence.start()]
            if currency == "SAR" and re.search(r"\xaa\s*$", preceding):
                currencies.add("SAR")
            if declared_currencies or declared_scales:
                currencies |= declared_currencies
                scales |= declared_scales
            if currency and currency not in currencies:
                continue
            if scale:
                if scale == "billion" and "bn" in scales:
                    scales.add("billion")
                elif scale == "bn" and "billion" in scales:
                    scales.add("bn")
                if scale not in scales:
                    continue
            return True
    return False


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
    declarations = [_declared_units(chunk) for chunk in chunks]
    for amount in _NUMBER_RE.finditer(claim):
        currency, scale = _governing_units(claim, amount.start(), amount.end())
        if not currency and not scale:
            continue
        if not _units_bound_to_amount(chunks, declarations, amount.group(0),
                                      currency, scale):
            return True

    if _PERCENTAGE_POINT_RE.search(claim) and not _PERCENTAGE_POINT_RE.search(context):
        return True
    if _PERCENT_RE.search(claim) and not _PERCENT_RE.search(context):
        return True
    return False
