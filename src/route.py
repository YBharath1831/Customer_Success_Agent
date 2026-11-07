"""Deterministic ticket-routing policy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from src.models import NormalizedTicket


class RoutingAction(str, Enum):
    """Available routing outcomes before answer generation."""

    AUTO_RESPOND = "auto_respond"
    ESCALATE = "escalate"


class RoutingReasonCode(str, Enum):
    """Machine-readable explanation for a routing decision."""

    AUTO_RESPONSE_ALLOWED = "auto_response_allowed"
    KILL_SWITCH_ENABLED = "kill_switch_enabled"
    EMPTY_TICKET = "empty_ticket"
    MISSING_INTENT = "missing_intent"
    MANDATORY_ESCALATION_INTENT = (
        "mandatory_escalation_intent"
    )
    MISSING_CONFIDENCE = "missing_confidence"
    INVALID_CONFIDENCE = "invalid_confidence"
    LOW_CONFIDENCE = "low_confidence"
    NO_RETRIEVAL_RESULTS = "no_retrieval_results"
    INVALID_RETRIEVAL_SCORE = (
        "invalid_retrieval_score"
    )
    LOW_RETRIEVAL_SCORE = "low_retrieval_score"


DEFAULT_ALWAYS_ESCALATE_INTENTS = frozenset(
    {
        "compliance_request",
        "feature_request",
        "security_incident",
        "unclear_request",
    }
)


@dataclass(frozen=True)
class RoutingPolicy:
    """Thresholds and safety rules used by the router."""

    confidence_threshold: float
    retrieval_threshold: float
    kill_switch: bool = False

    always_escalate_intents: frozenset[str] = field(
        default_factory=lambda: (
            DEFAULT_ALWAYS_ESCALATE_INTENTS
        )
    )

    def __post_init__(self) -> None:
        _validate_probability(
            "confidence_threshold",
            self.confidence_threshold,
        )
        _validate_probability(
            "retrieval_threshold",
            self.retrieval_threshold,
        )

        normalized_intents = frozenset(
            intent.strip().lower()
            for intent in self.always_escalate_intents
            if intent.strip()
        )

        object.__setattr__(
            self,
            "always_escalate_intents",
            normalized_intents,
        )


@dataclass(frozen=True)
class RouteDecision:
    """Complete, auditable routing result."""

    action: RoutingAction
    reason_code: RoutingReasonCode
    reason: str

    predicted_intent: str | None
    classification_confidence: float | None
    confidence_threshold: float

    top_retrieval_score: float | None
    retrieval_threshold: float

    policy_checks: dict[str, bool]
    requirement_ids: tuple[str, ...] = (
        "A5",
        "FR-ROUTE-01",
    )

    @property
    def should_generate(self) -> bool:
        """Whether the pipeline may generate an answer."""

        return self.action is RoutingAction.AUTO_RESPOND


def decide_route(
    *,
    ticket: NormalizedTicket,
    predicted_intent: str | None,
    classification_confidence: float | None,
    retrieval_scores: Sequence[float],
    policy: RoutingPolicy,
) -> RouteDecision:
    """
    Decide whether a ticket may receive an automatic response.

    The order is intentional. Safety rules are checked before confidence and
    retrieval quality, so a high model score cannot override mandatory
    escalation.
    """

    normalized_intent = _normalize_intent(
        predicted_intent
    )

    checks = {
        "kill_switch_disabled": (
            not policy.kill_switch
        ),
        "ticket_has_text": bool(
            ticket.text.strip()
        ),
        "intent_present": (
            normalized_intent is not None
        ),
        "intent_is_safe": (
            normalized_intent
            not in policy.always_escalate_intents
        ),
        "confidence_present": (
            classification_confidence is not None
        ),
        "confidence_valid": (
            _is_valid_probability(
                classification_confidence
            )
        ),
        "confidence_passed": False,
        "retrieval_present": bool(
            retrieval_scores
        ),
        "retrieval_scores_valid": False,
        "retrieval_passed": False,
    }

    top_retrieval_score = _top_valid_score(
        retrieval_scores
    )

    if policy.kill_switch:
        return _escalation(
            code=RoutingReasonCode.KILL_SWITCH_ENABLED,
            reason=(
                "Automatic responses are disabled by the "
                "operational kill switch."
            ),
            intent=normalized_intent,
            confidence=classification_confidence,
            top_retrieval_score=top_retrieval_score,
            policy=policy,
            checks=checks,
        )

    if not ticket.text.strip():
        return _escalation(
            code=RoutingReasonCode.EMPTY_TICKET,
            reason=(
                "The ticket contains no subject or message "
                "text that can be classified safely."
            ),
            intent=normalized_intent,
            confidence=classification_confidence,
            top_retrieval_score=top_retrieval_score,
            policy=policy,
            checks=checks,
        )

    if normalized_intent is None:
        return _escalation(
            code=RoutingReasonCode.MISSING_INTENT,
            reason=(
                "No intent classification was available."
            ),
            intent=None,
            confidence=classification_confidence,
            top_retrieval_score=top_retrieval_score,
            policy=policy,
            checks=checks,
        )

    if (
        normalized_intent
        in policy.always_escalate_intents
    ):
        return _escalation(
            code=(
                RoutingReasonCode
                .MANDATORY_ESCALATION_INTENT
            ),
            reason=(
                f"Intent {normalized_intent!r} is governed "
                "by a mandatory human-review policy."
            ),
            intent=normalized_intent,
            confidence=classification_confidence,
            top_retrieval_score=top_retrieval_score,
            policy=policy,
            checks=checks,
        )

    if classification_confidence is None:
        return _escalation(
            code=RoutingReasonCode.MISSING_CONFIDENCE,
            reason=(
                "The classifier did not provide a confidence "
                "score."
            ),
            intent=normalized_intent,
            confidence=None,
            top_retrieval_score=top_retrieval_score,
            policy=policy,
            checks=checks,
        )

    if not _is_valid_probability(
        classification_confidence
    ):
        return _escalation(
            code=RoutingReasonCode.INVALID_CONFIDENCE,
            reason=(
                "The classifier confidence was not a finite "
                "number between zero and one."
            ),
            intent=normalized_intent,
            confidence=classification_confidence,
            top_retrieval_score=top_retrieval_score,
            policy=policy,
            checks=checks,
        )

    checks["confidence_passed"] = (
        classification_confidence
        >= policy.confidence_threshold
    )

    if not checks["confidence_passed"]:
        return _escalation(
            code=RoutingReasonCode.LOW_CONFIDENCE,
            reason=(
                "Classification confidence "
                f"{classification_confidence:.3f} is below "
                "the configured threshold "
                f"{policy.confidence_threshold:.3f}."
            ),
            intent=normalized_intent,
            confidence=classification_confidence,
            top_retrieval_score=top_retrieval_score,
            policy=policy,
            checks=checks,
        )

    if not retrieval_scores:
        return _escalation(
            code=(
                RoutingReasonCode.NO_RETRIEVAL_RESULTS
            ),
            reason=(
                "Retrieval returned no documentation "
                "passages that could ground an answer."
            ),
            intent=normalized_intent,
            confidence=classification_confidence,
            top_retrieval_score=None,
            policy=policy,
            checks=checks,
        )

    if not _all_valid_scores(retrieval_scores):
        return _escalation(
            code=(
                RoutingReasonCode
                .INVALID_RETRIEVAL_SCORE
            ),
            reason=(
                "Retrieval returned at least one invalid "
                "relevance score."
            ),
            intent=normalized_intent,
            confidence=classification_confidence,
            top_retrieval_score=top_retrieval_score,
            policy=policy,
            checks=checks,
        )

    checks["retrieval_scores_valid"] = True
    checks["retrieval_passed"] = (
        top_retrieval_score is not None
        and top_retrieval_score
        >= policy.retrieval_threshold
    )

    if not checks["retrieval_passed"]:
        return _escalation(
            code=RoutingReasonCode.LOW_RETRIEVAL_SCORE,
            reason=(
                "Best retrieval relevance score "
                f"{top_retrieval_score:.3f} is below "
                "the configured threshold "
                f"{policy.retrieval_threshold:.3f}."
            ),
            intent=normalized_intent,
            confidence=classification_confidence,
            top_retrieval_score=top_retrieval_score,
            policy=policy,
            checks=checks,
        )

    return RouteDecision(
        action=RoutingAction.AUTO_RESPOND,
        reason_code=(
            RoutingReasonCode.AUTO_RESPONSE_ALLOWED
        ),
        reason=(
            "The intent is eligible for automation, "
            f"classification confidence "
            f"{classification_confidence:.3f} passed "
            f"the {policy.confidence_threshold:.3f} "
            "threshold, and retrieval relevance "
            f"{top_retrieval_score:.3f} passed "
            f"the {policy.retrieval_threshold:.3f} "
            "threshold."
        ),
        predicted_intent=normalized_intent,
        classification_confidence=(
            classification_confidence
        ),
        confidence_threshold=(
            policy.confidence_threshold
        ),
        top_retrieval_score=top_retrieval_score,
        retrieval_threshold=(
            policy.retrieval_threshold
        ),
        policy_checks=dict(checks),
    )


def _escalation(
    *,
    code: RoutingReasonCode,
    reason: str,
    intent: str | None,
    confidence: float | None,
    top_retrieval_score: float | None,
    policy: RoutingPolicy,
    checks: dict[str, bool],
) -> RouteDecision:
    """Create a consistent escalation decision."""

    return RouteDecision(
        action=RoutingAction.ESCALATE,
        reason_code=code,
        reason=reason,
        predicted_intent=intent,
        classification_confidence=confidence,
        confidence_threshold=(
            policy.confidence_threshold
        ),
        top_retrieval_score=top_retrieval_score,
        retrieval_threshold=(
            policy.retrieval_threshold
        ),
        policy_checks=dict(checks),
    )


def _normalize_intent(
    predicted_intent: str | None,
) -> str | None:
    """Normalize classifier intent values."""

    if predicted_intent is None:
        return None

    normalized = predicted_intent.strip().lower()
    return normalized or None


def _is_valid_probability(
    value: float | None,
) -> bool:
    """Check for a finite value between zero and one."""

    if value is None:
        return False

    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    return (
        math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _all_valid_scores(
    scores: Sequence[float],
) -> bool:
    """Validate every retrieval relevance score."""

    return all(
        _is_valid_probability(score)
        for score in scores
    )


def _top_valid_score(
    scores: Sequence[float],
) -> float | None:
    """Return the highest valid score, if one exists."""

    valid_scores = [
        float(score)
        for score in scores
        if _is_valid_probability(score)
    ]

    if not valid_scores:
        return None

    return max(valid_scores)


def _validate_probability(
    name: str,
    value: float,
) -> None:
    """Validate policy threshold values."""

    if not _is_valid_probability(value):
        raise ValueError(
            f"{name} must be a finite number "
            "between zero and one."
        )