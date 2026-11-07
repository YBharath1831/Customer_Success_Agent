"""Blocking safety and grounding guardrails for generated responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence

from .models import NormalizedTicket


class PassageLike(Protocol):
    """Minimum interface required from a retrieved passage."""

    passage_id: str
    text: str


class GuardrailName(str, Enum):
    """Names of the deterministic guardrails."""

    ROUTING_PERMISSION = "routing_permission"
    ANSWER_PRESENT = "answer_present"
    PROMPT_INJECTION = "prompt_injection"
    SENSITIVE_DATA = "sensitive_data"
    PROHIBITED_COMMITMENT = "prohibited_commitment"
    CITATION_INTEGRITY = "citation_integrity"
    CLAIM_GROUNDING = "claim_grounding"


@dataclass(frozen=True)
class CitedClaim:
    """
    A factual claim made by the generated response.

    Greetings and conversational text do not need to be represented as claims.
    Every factual claim should cite one or more retrieved passage IDs.
    """

    text: str
    passage_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Claim text must be a string.")

        cleaned_ids = tuple(
            passage_id.strip()
            for passage_id in self.passage_ids
            if isinstance(passage_id, str) and passage_id.strip()
        )

        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "passage_ids", cleaned_ids)


@dataclass(frozen=True)
class GuardrailFinding:
    """Result produced by one guardrail check."""

    name: GuardrailName
    passed: bool
    reason: str
    evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "passed": self.passed,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class GuardrailReport:
    """Combined result of all guardrail checks."""

    findings: tuple[GuardrailFinding, ...]

    @property
    def blocked(self) -> bool:
        """A single failed guardrail blocks the response."""

        return any(not finding.passed for finding in self.findings)

    @property
    def blocking_guardrails(self) -> tuple[str, ...]:
        return tuple(
            finding.name.value
            for finding in self.findings
            if not finding.passed
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "blocking_guardrails": list(self.blocking_guardrails),
            "findings": [
                finding.as_dict()
                for finding in self.findings
            ],
        }


# We record only the category of sensitive information found.
# The actual matching value is deliberately not placed in logs.
SENSITIVE_DATA_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email_address",
        re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "phone_number",
        re.compile(
            r"(?<!\d)(?:\+?\d[\s().-]?){9,14}\d(?!\d)"
        ),
    ),
    (
        "credit_card_or_account_number",
        re.compile(
            r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"
        ),
    ),
    (
        "api_key_or_token",
        re.compile(
            r"\b(?:api[_ -]?key|access[_ -]?token|secret)"
            r"\s*[:=]\s*[A-Za-z0-9_\-./]{8,}",
            re.IGNORECASE,
        ),
    ),
    (
        "openai_style_secret",
        re.compile(
            r"\bsk-[A-Za-z0-9_-]{16,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            re.IGNORECASE,
        ),
    ),
)


PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bignore (?:all |any |the )?(?:previous|prior|above) "
        r"(?:instructions|prompts|rules)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdisregard (?:all |any |the )?(?:previous|prior|above) "
        r"(?:instructions|prompts|rules)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\breveal (?:the |your )?(?:system prompt|hidden instructions)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:show|print|repeat) (?:the |your )?"
        r"(?:system prompt|hidden instructions)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\byou are now (?:an?|the)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"<\s*(?:system|assistant|developer)\s*>",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdeveloper message\b",
        re.IGNORECASE,
    ),
)


PROHIBITED_COMMITMENT_PATTERNS: tuple[
    tuple[str, re.Pattern[str]], ...
] = (
    (
        "refund_commitment",
        re.compile(
            r"\b(?:your )?refund (?:has been|is|will be) "
            r"(?:issued|approved|processed|guaranteed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "compensation_commitment",
        re.compile(
            r"\b(?:we|i) (?:will|shall) "
            r"(?:compensate|reimburse|credit) you\b",
            re.IGNORECASE,
        ),
    ),
    (
        "guaranteed_resolution",
        re.compile(
            r"\b(?:we|i) guarantee\b",
            re.IGNORECASE,
        ),
    ),
    (
        "guaranteed_timeline",
        re.compile(
            r"\b(?:definitely|guaranteed|certainly) "
            r"(?:resolved|fixed|completed|available) "
            r"(?:by|within)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "legal_commitment",
        re.compile(
            r"\b(?:we|our company) (?:accept|admit) "
            r"(?:liability|legal responsibility)\b",
            re.IGNORECASE,
        ),
    ),
)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "we",
    "will",
    "with",
    "you",
    "your",
}


def _meaningful_tokens(text: str) -> set[str]:
    """Convert text into normalized tokens for deterministic comparison."""

    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOP_WORDS
        and len(token) > 1
    }


def _grounding_overlap(claim_text: str, source_text: str) -> float:
    """
    Calculate how much of a claim is represented in its cited source.

    This is intentionally simple and deterministic. Later, an LLM judge may
    supplement it, but it should not replace this blocking check.
    """

    claim_tokens = _meaningful_tokens(claim_text)

    if not claim_tokens:
        return 0.0

    source_tokens = _meaningful_tokens(source_text)
    shared_tokens = claim_tokens.intersection(source_tokens)

    return len(shared_tokens) / len(claim_tokens)


class GuardrailEngine:
    """Run deterministic blocking checks on a generated response."""

    def __init__(
        self,
        minimum_grounding_overlap: float = 0.15,
    ) -> None:
        if not 0.0 <= minimum_grounding_overlap <= 1.0:
            raise ValueError(
                "minimum_grounding_overlap must be between 0.0 and 1.0."
            )

        self.minimum_grounding_overlap = minimum_grounding_overlap

    def evaluate(
        self,
        *,
        ticket: NormalizedTicket,
        answer_text: str,
        claims: Sequence[CitedClaim],
        retrieved_passages: Sequence[PassageLike],
        routing_allows_generation: bool,
    ) -> GuardrailReport:
        """
        Evaluate all guardrails.

        The caller must not send the response when report.blocked is True.
        """

        findings = (
            self._check_routing_permission(routing_allows_generation),
            self._check_answer_present(answer_text),
            self._check_prompt_injection(ticket),
            self._check_sensitive_data(answer_text),
            self._check_prohibited_commitments(answer_text),
            self._check_citation_integrity(
                claims,
                retrieved_passages,
            ),
            self._check_claim_grounding(
                claims,
                retrieved_passages,
            ),
        )

        return GuardrailReport(findings=findings)

    @staticmethod
    def _check_routing_permission(
        routing_allows_generation: bool,
    ) -> GuardrailFinding:
        if not routing_allows_generation:
            return GuardrailFinding(
                name=GuardrailName.ROUTING_PERMISSION,
                passed=False,
                reason=(
                    "Routing policy did not authorize an automated response."
                ),
            )

        return GuardrailFinding(
            name=GuardrailName.ROUTING_PERMISSION,
            passed=True,
            reason="Routing policy authorized response generation.",
        )

    @staticmethod
    def _check_answer_present(
        answer_text: str,
    ) -> GuardrailFinding:
        if not isinstance(answer_text, str) or not answer_text.strip():
            return GuardrailFinding(
                name=GuardrailName.ANSWER_PRESENT,
                passed=False,
                reason="Generated answer is empty.",
            )

        return GuardrailFinding(
            name=GuardrailName.ANSWER_PRESENT,
            passed=True,
            reason="Generated answer contains text.",
        )

    @staticmethod
    def _check_prompt_injection(
        ticket: NormalizedTicket,
    ) -> GuardrailFinding:
        ticket_text = ticket.text or ""

        detected_patterns = tuple(
            f"pattern_{index}"
            for index, pattern in enumerate(
                PROMPT_INJECTION_PATTERNS,
                start=1,
            )
            if pattern.search(ticket_text)
        )

        if detected_patterns:
            return GuardrailFinding(
                name=GuardrailName.PROMPT_INJECTION,
                passed=False,
                reason=(
                    "The incoming ticket contains possible prompt-injection "
                    "instructions."
                ),
                evidence=detected_patterns,
            )

        return GuardrailFinding(
            name=GuardrailName.PROMPT_INJECTION,
            passed=True,
            reason="No prompt-injection pattern was detected.",
        )

    @staticmethod
    def _check_sensitive_data(
        answer_text: str,
    ) -> GuardrailFinding:
        detected_categories = tuple(
            category
            for category, pattern in SENSITIVE_DATA_PATTERNS
            if pattern.search(answer_text)
        )

        if detected_categories:
            return GuardrailFinding(
                name=GuardrailName.SENSITIVE_DATA,
                passed=False,
                reason=(
                    "Generated answer contains potentially sensitive data."
                ),
                evidence=detected_categories,
            )

        return GuardrailFinding(
            name=GuardrailName.SENSITIVE_DATA,
            passed=True,
            reason="No sensitive-data pattern was detected.",
        )

    @staticmethod
    def _check_prohibited_commitments(
        answer_text: str,
    ) -> GuardrailFinding:
        detected_commitments = tuple(
            category
            for category, pattern in PROHIBITED_COMMITMENT_PATTERNS
            if pattern.search(answer_text)
        )

        if detected_commitments:
            return GuardrailFinding(
                name=GuardrailName.PROHIBITED_COMMITMENT,
                passed=False,
                reason=(
                    "Generated answer makes a commitment that requires "
                    "human approval."
                ),
                evidence=detected_commitments,
            )

        return GuardrailFinding(
            name=GuardrailName.PROHIBITED_COMMITMENT,
            passed=True,
            reason="No prohibited commitment was detected.",
        )

    @staticmethod
    def _check_citation_integrity(
        claims: Sequence[CitedClaim],
        retrieved_passages: Sequence[PassageLike],
    ) -> GuardrailFinding:
        if not claims:
            return GuardrailFinding(
                name=GuardrailName.CITATION_INTEGRITY,
                passed=False,
                reason=(
                    "No structured factual claims were supplied for "
                    "citation validation."
                ),
            )

        retrieved_ids = {
            passage.passage_id
            for passage in retrieved_passages
        }

        uncited_claims: list[str] = []
        unknown_citations: list[str] = []

        for index, claim in enumerate(claims, start=1):
            if not claim.text:
                uncited_claims.append(f"claim_{index}_empty")
                continue

            if not claim.passage_ids:
                uncited_claims.append(f"claim_{index}")
                continue

            for passage_id in claim.passage_ids:
                if passage_id not in retrieved_ids:
                    unknown_citations.append(passage_id)

        evidence = tuple(
            dict.fromkeys(
                uncited_claims + unknown_citations
            )
        )

        if evidence:
            return GuardrailFinding(
                name=GuardrailName.CITATION_INTEGRITY,
                passed=False,
                reason=(
                    "One or more factual claims are missing citations or "
                    "cite passages that were not retrieved."
                ),
                evidence=evidence,
            )

        return GuardrailFinding(
            name=GuardrailName.CITATION_INTEGRITY,
            passed=True,
            reason=(
                "Every factual claim cites at least one retrieved passage."
            ),
        )

    def _check_claim_grounding(
        self,
        claims: Sequence[CitedClaim],
        retrieved_passages: Sequence[PassageLike],
    ) -> GuardrailFinding:
        passage_map = {
            passage.passage_id: passage.text
            for passage in retrieved_passages
        }

        unsupported_claims: list[str] = []

        for index, claim in enumerate(claims, start=1):
            source_parts = [
                passage_map[passage_id]
                for passage_id in claim.passage_ids
                if passage_id in passage_map
            ]

            if not claim.text or not source_parts:
                unsupported_claims.append(f"claim_{index}")
                continue

            combined_source = "\n".join(source_parts)
            overlap = _grounding_overlap(
                claim.text,
                combined_source,
            )

            if overlap < self.minimum_grounding_overlap:
                unsupported_claims.append(
                    f"claim_{index}:overlap={overlap:.3f}"
                )

        if unsupported_claims:
            return GuardrailFinding(
                name=GuardrailName.CLAIM_GROUNDING,
                passed=False,
                reason=(
                    "One or more factual claims are not sufficiently "
                    "supported by their cited passages."
                ),
                evidence=tuple(unsupported_claims),
            )

        return GuardrailFinding(
            name=GuardrailName.CLAIM_GROUNDING,
            passed=True,
            reason="All factual claims passed the grounding check.",
        )


def evaluate_guardrails(
    *,
    ticket: NormalizedTicket,
    answer_text: str,
    claims: Sequence[CitedClaim],
    retrieved_passages: Sequence[PassageLike],
    routing_allows_generation: bool,
    minimum_grounding_overlap: float = 0.15,
) -> GuardrailReport:
    """Convenience function for evaluating the standard guardrail set."""

    engine = GuardrailEngine(
        minimum_grounding_overlap=minimum_grounding_overlap
    )

    return engine.evaluate(
        ticket=ticket,
        answer_text=answer_text,
        claims=claims,
        retrieved_passages=retrieved_passages,
        routing_allows_generation=routing_allows_generation,
    )