"""Tests for deterministic ticket routing."""

import math

import pytest

from src.models import NormalizedTicket
from src.route import (
    DEFAULT_ALWAYS_ESCALATE_INTENTS,
    RouteDecision,
    RoutingAction,
    RoutingPolicy,
    RoutingReasonCode,
    decide_route,
)


def make_ticket(
    *,
    ticket_id: str = "TEST-001",
    channel: str = "email",
    subject: str = "Login problem",
    body: str = "I cannot sign in.",
    text: str | None = None,
    labels: dict | None = None,
) -> NormalizedTicket:
    """Create a normalized ticket for routing tests."""

    if text is None:
        text = (
            f"Subject: {subject}\n\n"
            f"Message: {body}"
        ).strip()

    return NormalizedTicket(
        ticket_id=ticket_id,
        channel=channel,
        subject=subject,
        body=body,
        text=text,
        labels=labels,
    )


def make_policy(
    *,
    confidence_threshold: float = 0.80,
    retrieval_threshold: float = 0.40,
    kill_switch: bool = False,
) -> RoutingPolicy:
    """Create the standard routing policy."""

    return RoutingPolicy(
        confidence_threshold=confidence_threshold,
        retrieval_threshold=retrieval_threshold,
        kill_switch=kill_switch,
    )


def make_auto_response_decision(
) -> RouteDecision:
    """Create a routing decision that should permit generation."""

    return decide_route(
        ticket=make_ticket(),
        predicted_intent="authentication_failure",
        classification_confidence=0.91,
        retrieval_scores=[0.87, 0.62],
        policy=make_policy(),
    )


def test_safe_ticket_is_allowed_to_generate() -> None:
    decision = make_auto_response_decision()

    assert decision.action is (
        RoutingAction.AUTO_RESPOND
    )
    assert decision.reason_code is (
        RoutingReasonCode.AUTO_RESPONSE_ALLOWED
    )
    assert decision.should_generate is True

    assert decision.predicted_intent == (
        "authentication_failure"
    )
    assert decision.classification_confidence == 0.91
    assert decision.top_retrieval_score == 0.87

    assert decision.policy_checks[
        "confidence_passed"
    ] is True

    assert decision.policy_checks[
        "retrieval_passed"
    ] is True


def test_routing_is_deterministic() -> None:
    arguments = {
        "ticket": make_ticket(),
        "predicted_intent": "authentication_failure",
        "classification_confidence": 0.91,
        "retrieval_scores": [0.87, 0.62],
        "policy": make_policy(),
    }

    first = decide_route(**arguments)
    second = decide_route(**arguments)

    assert first == second


def test_kill_switch_forces_escalation() -> None:
    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent="authentication_failure",
        classification_confidence=0.99,
        retrieval_scores=[0.99],
        policy=make_policy(kill_switch=True),
    )

    assert decision.action is RoutingAction.ESCALATE
    assert decision.reason_code is (
        RoutingReasonCode.KILL_SWITCH_ENABLED
    )
    assert decision.should_generate is False


def test_kill_switch_has_priority_over_other_checks() -> None:
    decision = decide_route(
        ticket=make_ticket(text=""),
        predicted_intent=None,
        classification_confidence=None,
        retrieval_scores=[],
        policy=make_policy(kill_switch=True),
    )

    assert decision.reason_code is (
        RoutingReasonCode.KILL_SWITCH_ENABLED
    )


def test_empty_ticket_is_escalated() -> None:
    decision = decide_route(
        ticket=make_ticket(
            subject="",
            body="",
            text="",
        ),
        predicted_intent="authentication_failure",
        classification_confidence=0.99,
        retrieval_scores=[0.99],
        policy=make_policy(),
    )

    assert decision.action is RoutingAction.ESCALATE
    assert decision.reason_code is (
        RoutingReasonCode.EMPTY_TICKET
    )


@pytest.mark.parametrize(
    "intent",
    [
        None,
        "",
        "   ",
    ],
)
def test_missing_intent_is_escalated(
    intent,
) -> None:
    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent=intent,
        classification_confidence=0.99,
        retrieval_scores=[0.99],
        policy=make_policy(),
    )

    assert decision.action is RoutingAction.ESCALATE
    assert decision.reason_code is (
        RoutingReasonCode.MISSING_INTENT
    )


@pytest.mark.parametrize(
    "intent",
    sorted(DEFAULT_ALWAYS_ESCALATE_INTENTS),
)
def test_governed_intents_are_always_escalated(
    intent: str,
) -> None:
    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent=intent,
        classification_confidence=1.0,
        retrieval_scores=[1.0],
        policy=make_policy(),
    )

    assert decision.action is RoutingAction.ESCALATE
    assert decision.reason_code is (
        RoutingReasonCode
        .MANDATORY_ESCALATION_INTENT
    )


def test_intent_normalization_is_case_insensitive() -> None:
    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent="  SECURITY_INCIDENT  ",
        classification_confidence=0.99,
        retrieval_scores=[0.99],
        policy=make_policy(),
    )

    assert decision.predicted_intent == (
        "security_incident"
    )
    assert decision.reason_code is (
        RoutingReasonCode
        .MANDATORY_ESCALATION_INTENT
    )


def test_missing_confidence_is_escalated() -> None:
    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent="authentication_failure",
        classification_confidence=None,
        retrieval_scores=[0.99],
        policy=make_policy(),
    )

    assert decision.reason_code is (
        RoutingReasonCode.MISSING_CONFIDENCE
    )


@pytest.mark.parametrize(
    "confidence",
    [
        -0.01,
        1.01,
        math.nan,
        math.inf,
        -math.inf,
        True,
    ],
)
def test_invalid_confidence_is_escalated(
    confidence,
) -> None:
    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent="authentication_failure",
        classification_confidence=confidence,
        retrieval_scores=[0.99],
        policy=make_policy(),
    )

    assert decision.action is RoutingAction.ESCALATE
    assert decision.reason_code is (
        RoutingReasonCode.INVALID_CONFIDENCE
    )


def test_low_confidence_is_escalated() -> None:
    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent="authentication_failure",
        classification_confidence=0.79,
        retrieval_scores=[0.99],
        policy=make_policy(
            confidence_threshold=0.80
        ),
    )

    assert decision.action is RoutingAction.ESCALATE
    assert decision.reason_code is (
        RoutingReasonCode.LOW_CONFIDENCE
    )
    assert "0.790" in decision.reason
    assert "0.800" in decision.reason


def test_confidence_equal_to_threshold_passes() -> None:
    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent="authentication_failure",
        classification_confidence=0.80,
        retrieval_scores=[0.90],
        policy=make_policy(
            confidence_threshold=0.80
        ),
    )

    assert decision.action is (
        RoutingAction.AUTO_RESPOND
    )


def test_no_retrieval_results_causes_escalation() -> None:
    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent="authentication_failure",
        classification_confidence=0.95,
        retrieval_scores=[],
        policy=make_policy(),
    )

    assert decision.action is RoutingAction.ESCALATE
    assert decision.reason_code is (
        RoutingReasonCode.NO_RETRIEVAL_RESULTS
    )
    assert decision.top_retrieval_score is None


@pytest.mark.parametrize(
    "scores",
    [
        [math.nan],
        [math.inf],
        [-0.01],
        [1.01],
        [True],
        [0.90, math.nan],
        [0.90, 1.50],
    ],
)
def test_invalid_retrieval_scores_cause_escalation(
    scores,
) -> None:
    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent="authentication_failure",
        classification_confidence=0.95,
        retrieval_scores=scores,
        policy=make_policy(),
    )

    assert decision.action is RoutingAction.ESCALATE
    assert decision.reason_code is (
        RoutingReasonCode.INVALID_RETRIEVAL_SCORE
    )


def test_low_retrieval_score_causes_escalation() -> None:
    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent="authentication_failure",
        classification_confidence=0.95,
        retrieval_scores=[0.39, 0.20],
        policy=make_policy(
            retrieval_threshold=0.40
        ),
    )

    assert decision.action is RoutingAction.ESCALATE
    assert decision.reason_code is (
        RoutingReasonCode.LOW_RETRIEVAL_SCORE
    )
    assert decision.top_retrieval_score == 0.39
    assert "0.390" in decision.reason
    assert "0.400" in decision.reason


def test_highest_retrieval_score_is_used() -> None:
    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent="authentication_failure",
        classification_confidence=0.95,
        retrieval_scores=[
            0.41,
            0.88,
            0.63,
        ],
        policy=make_policy(),
    )

    assert decision.action is (
        RoutingAction.AUTO_RESPOND
    )
    assert decision.top_retrieval_score == 0.88


def test_retrieval_equal_to_threshold_passes() -> None:
    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent="authentication_failure",
        classification_confidence=0.95,
        retrieval_scores=[0.40],
        policy=make_policy(
            retrieval_threshold=0.40
        ),
    )

    assert decision.action is (
        RoutingAction.AUTO_RESPOND
    )


@pytest.mark.parametrize(
    ("confidence_threshold", "retrieval_threshold"),
    [
        (-0.01, 0.40),
        (1.01, 0.40),
        (0.80, -0.01),
        (0.80, 1.01),
        (math.nan, 0.40),
        (0.80, math.inf),
    ],
)
def test_invalid_policy_thresholds_are_rejected(
    confidence_threshold,
    retrieval_threshold,
) -> None:
    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        RoutingPolicy(
            confidence_threshold=confidence_threshold,
            retrieval_threshold=retrieval_threshold,
        )


def test_custom_mandatory_intents_are_normalized() -> None:
    policy = RoutingPolicy(
        confidence_threshold=0.80,
        retrieval_threshold=0.40,
        always_escalate_intents=frozenset(
            {
                "  CUSTOM_RISK  ",
                "",
            }
        ),
    )

    assert policy.always_escalate_intents == (
        frozenset({"custom_risk"})
    )

    decision = decide_route(
        ticket=make_ticket(),
        predicted_intent="custom_risk",
        classification_confidence=0.99,
        retrieval_scores=[0.99],
        policy=policy,
    )

    assert decision.reason_code is (
        RoutingReasonCode
        .MANDATORY_ESCALATION_INTENT
    )


def test_router_does_not_use_ground_truth_labels() -> None:
    """
    Evaluation labels must never influence live routing.

    This ticket's label says escalate, but the actual routing inputs satisfy
    the automation policy. The router must use predictions, not answers from
    the evaluation dataset.
    """

    ticket = make_ticket(
        labels={
            "intent": "security_incident",
            "urgency": "high",
            "expected_route": "escalate",
            "answerable_from_docs": False,
            "expected_doc_ids": [],
            "must_not_auto_respond": True,
        }
    )

    decision = decide_route(
        ticket=ticket,
        predicted_intent="authentication_failure",
        classification_confidence=0.95,
        retrieval_scores=[0.90],
        policy=make_policy(),
    )

    assert decision.action is (
        RoutingAction.AUTO_RESPOND
    )


def test_requirement_ids_are_recorded() -> None:
    decision = make_auto_response_decision()

    assert "A5" in decision.requirement_ids