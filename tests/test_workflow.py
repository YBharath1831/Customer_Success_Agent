"""Tests for the end-to-end customer-success workflow."""

from pathlib import Path

import pytest

from src.classify import (
    ClassificationAlternative,
    ClassificationResult,
    LabelPrediction,
)
from src.generate import (
    GeneratedDraft,
    GenerationError,
)
from src.guardrails import (
    CitedClaim,
    GuardrailEngine,
)
from src.ingest import normalize_ticket
from src.logging_store import DecisionStore
from src.retrieve import RetrievedMatch
from src.route import RoutingPolicy
from src.workflow import (
    CustomerSuccessWorkflow,
    WorkflowOutcome,
)


class FakeClassifier:
    """Configurable deterministic classifier."""

    def __init__(
        self,
        *,
        intent: str = "account_access",
        urgency: str = "normal",
        confidence: float = 0.95,
        intent_confidence: float | None = None,
        urgency_confidence: float | None = None,
        error: Exception | None = None,
    ) -> None:
        self.intent = intent
        self.urgency = urgency
        self.intent_confidence = (
            confidence
            if intent_confidence is None
            else intent_confidence
        )
        self.urgency_confidence = (
            confidence
            if urgency_confidence is None
            else urgency_confidence
        )
        self.error = error
        self.calls = 0

    def classify(self, ticket):
        self.calls += 1

        if self.error is not None:
            raise self.error

        return ClassificationResult(
            intent=LabelPrediction(
                value=self.intent,
                confidence=self.intent_confidence,
                alternatives=(
                    ClassificationAlternative(
                        value="unclear_request",
                        confidence=0.05,
                    ),
                ),
            ),
            urgency=LabelPrediction(
                value=self.urgency,
                confidence=self.urgency_confidence,
                alternatives=(
                    ClassificationAlternative(
                        value="high",
                        confidence=0.05,
                    ),
                ),
            ),
            model_name="fake-classifier",
            model_version="test-v1",
        )


class FakeRetriever:
    """Configurable deterministic retriever."""

    def __init__(
        self,
        matches=None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.matches = (
            list(matches)
            if matches is not None
            else [make_match()]
        )
        self.error = error
        self.calls = 0
        self.last_query = None
        self.last_top_k = None
        self.last_minimum_relevance = None

    def search(
        self,
        query_text,
        *,
        top_k=5,
        minimum_relevance=0.0,
    ):
        self.calls += 1
        self.last_query = query_text
        self.last_top_k = top_k
        self.last_minimum_relevance = (
            minimum_relevance
        )

        if self.error is not None:
            raise self.error

        return self.matches[:top_k]


class FakeGenerator:
    """Configurable deterministic response generator."""

    def __init__(
        self,
        *,
        answer_text: str | None = None,
        claim_text: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.answer_text = answer_text or (
            "You can reset your password from the account "
            "settings page [DOC001#p001]."
        )
        self.claim_text = claim_text or (
            "Users can reset their password from the account "
            "settings page."
        )
        self.error = error
        self.calls = 0
        self.last_intent = None
        self.last_urgency = None
        self.last_passages = None

    def generate(
        self,
        *,
        ticket,
        intent,
        urgency,
        passages,
    ):
        self.calls += 1
        self.last_intent = intent
        self.last_urgency = urgency
        self.last_passages = tuple(passages)

        if self.error is not None:
            raise self.error

        return GeneratedDraft(
            answer_text=self.answer_text,
            claims=(
                CitedClaim(
                    text=self.claim_text,
                    passage_ids=("DOC001#p001",),
                ),
            ),
            model="fake-generation-model",
            raw_response="{}",
        )


def make_ticket(
    ticket_id: str = "WF-001",
):
    return normalize_ticket(
        {
            "ticket_id": ticket_id,
            "channel": "email",
            "subject": "Password reset",
            "body": "How can I reset my password?",
        }
    )


def make_match(
    *,
    relevance_score: float = 0.91,
) -> RetrievedMatch:
    return RetrievedMatch(
        passage_id="DOC001#p001",
        doc_id="DOC001",
        title="Resetting a password",
        category="account_access",
        chunk_index=1,
        text=(
            "Users can reset their password from the account "
            "settings page. Select Security and then choose "
            "Reset Password."
        ),
        relevance_score=relevance_score,
        distance=1.0 - relevance_score,
    )


def make_store(
    tmp_path: Path,
) -> DecisionStore:
    return DecisionStore(
        tmp_path / "workflow-decisions.db"
    )


def make_workflow(
    tmp_path: Path,
    *,
    classifier=None,
    retriever=None,
    generator=None,
    guardrail_engine=None,
    routing_policy=None,
    retrieval_top_k: int = 5,
):
    store = make_store(tmp_path)

    workflow = CustomerSuccessWorkflow(
        classifier=classifier or FakeClassifier(),
        retriever=retriever or FakeRetriever(),
        generator=generator or FakeGenerator(),
        guardrail_engine=(
            guardrail_engine or GuardrailEngine()
        ),
        routing_policy=(
            routing_policy
            or RoutingPolicy(
                confidence_threshold=0.80,
                retrieval_threshold=0.40,
            )
        ),
        decision_store=store,
        retrieval_top_k=retrieval_top_k,
    )

    return workflow, store


def decision_stages(
    store: DecisionStore,
    *,
    run_id: str,
    ticket_id: str,
) -> set[str]:
    decisions = store.fetch_decisions(
        run_id=run_id,
        ticket_id=ticket_id,
    )

    return {
        decision["stage"]
        for decision in decisions
    }


def test_successful_workflow_auto_responds(
    tmp_path: Path,
) -> None:
    workflow, store = make_workflow(tmp_path)

    result = workflow.process(
        make_ticket(),
        run_id="RUN-SUCCESS",
    )

    assert result.outcome is WorkflowOutcome.AUTO_RESPOND
    assert result.may_send_response is True
    assert result.answer_text is not None
    assert "[DOC001#p001]" in result.answer_text
    assert result.cited_passage_ids == (
        "DOC001#p001",
    )
    assert result.classification is not None
    assert result.route_decision is not None
    assert result.route_decision.should_generate is True
    assert result.guardrail_report is not None
    assert result.guardrail_report.blocked is False
    assert result.error_type is None
    assert result.latency_seconds >= 0

    ticket_run = store.fetch_ticket_run(
        run_id="RUN-SUCCESS",
        ticket_id="WF-001",
    )

    assert ticket_run is not None
    assert ticket_run["status"] == "completed"
    assert ticket_run["terminal_outcome"] == (
        "auto_respond"
    )

    assert decision_stages(
        store,
        run_id="RUN-SUCCESS",
        ticket_id="WF-001",
    ) == {
        "ingestion",
        "classification",
        "retrieval",
        "routing",
        "generation",
        "validation",
    }


def test_low_classification_confidence_escalates(
    tmp_path: Path,
) -> None:
    classifier = FakeClassifier(
        confidence=0.45
    )
    generator = FakeGenerator()

    workflow, store = make_workflow(
        tmp_path,
        classifier=classifier,
        generator=generator,
    )

    result = workflow.process(
        make_ticket(),
        run_id="RUN-LOW-CONFIDENCE",
    )

    assert result.outcome is WorkflowOutcome.ESCALATE
    assert result.may_send_response is False
    assert result.answer_text is None
    assert result.route_decision is not None
    assert (
        result.route_decision.reason_code.value
        == "low_confidence"
    )
    assert generator.calls == 0

    assert decision_stages(
        store,
        run_id="RUN-LOW-CONFIDENCE",
        ticket_id="WF-001",
    ) == {
        "ingestion",
        "classification",
        "retrieval",
        "routing",
    }


def test_low_retrieval_score_escalates(
    tmp_path: Path,
) -> None:
    retriever = FakeRetriever(
        matches=[
            make_match(
                relevance_score=0.20
            )
        ]
    )
    generator = FakeGenerator()

    workflow, _ = make_workflow(
        tmp_path,
        retriever=retriever,
        generator=generator,
    )

    result = workflow.process(
        make_ticket(),
        run_id="RUN-LOW-RETRIEVAL",
    )

    assert result.outcome is WorkflowOutcome.ESCALATE
    assert result.may_send_response is False
    assert (
        result.route_decision.reason_code.value
        == "low_retrieval_score"
    )
    assert generator.calls == 0


def test_no_retrieval_results_escalates(
    tmp_path: Path,
) -> None:
    retriever = FakeRetriever(
        matches=[]
    )
    generator = FakeGenerator()

    workflow, _ = make_workflow(
        tmp_path,
        retriever=retriever,
        generator=generator,
    )

    result = workflow.process(
        make_ticket(),
        run_id="RUN-NO-RESULTS",
    )

    assert result.outcome is WorkflowOutcome.ESCALATE
    assert result.answer_text is None
    assert (
        result.route_decision.reason_code.value
        == "no_retrieval_results"
    )
    assert generator.calls == 0


def test_mandatory_escalation_intent_never_generates(
    tmp_path: Path,
) -> None:
    classifier = FakeClassifier(
        intent="security_incident",
        confidence=0.99,
    )
    generator = FakeGenerator()

    workflow, _ = make_workflow(
        tmp_path,
        classifier=classifier,
        generator=generator,
    )

    result = workflow.process(
        make_ticket(),
        run_id="RUN-SECURITY",
    )

    assert result.outcome is WorkflowOutcome.ESCALATE
    assert (
        result.route_decision.reason_code.value
        == "mandatory_escalation_intent"
    )
    assert generator.calls == 0


def test_kill_switch_prevents_generation(
    tmp_path: Path,
) -> None:
    generator = FakeGenerator()

    policy = RoutingPolicy(
        confidence_threshold=0.80,
        retrieval_threshold=0.40,
        kill_switch=True,
    )

    workflow, _ = make_workflow(
        tmp_path,
        generator=generator,
        routing_policy=policy,
    )

    result = workflow.process(
        make_ticket(),
        run_id="RUN-KILL-SWITCH",
    )

    assert result.outcome is WorkflowOutcome.ESCALATE
    assert (
        result.route_decision.reason_code.value
        == "kill_switch_enabled"
    )
    assert generator.calls == 0


def test_classifier_failure_escalates_and_is_logged(
    tmp_path: Path,
) -> None:
    classifier = FakeClassifier(
        error=RuntimeError(
            "classifier unavailable"
        )
    )
    retriever = FakeRetriever()
    generator = FakeGenerator()

    workflow, store = make_workflow(
        tmp_path,
        classifier=classifier,
        retriever=retriever,
        generator=generator,
    )

    result = workflow.process(
        make_ticket(),
        run_id="RUN-CLASSIFIER-FAILURE",
    )

    assert result.outcome is WorkflowOutcome.ESCALATE
    assert result.answer_text is None
    assert result.error_type == "RuntimeError"
    assert retriever.calls == 0
    assert generator.calls == 0

    ticket_run = store.fetch_ticket_run(
        run_id="RUN-CLASSIFIER-FAILURE",
        ticket_id="WF-001",
    )

    assert ticket_run["terminal_outcome"] == "escalate"
    assert ticket_run["error_type"] == "RuntimeError"

    decisions = store.fetch_decisions(
        run_id="RUN-CLASSIFIER-FAILURE",
        ticket_id="WF-001",
    )

    classification_decision = next(
        decision
        for decision in decisions
        if decision["stage"] == "classification"
    )

    assert (
        classification_decision["action_taken"]
        == "failed"
    )


def test_retrieval_failure_escalates(
    tmp_path: Path,
) -> None:
    retriever = FakeRetriever(
        error=RuntimeError(
            "vector database unavailable"
        )
    )
    generator = FakeGenerator()

    workflow, store = make_workflow(
        tmp_path,
        retriever=retriever,
        generator=generator,
    )

    result = workflow.process(
        make_ticket(),
        run_id="RUN-RETRIEVAL-FAILURE",
    )

    assert result.outcome is WorkflowOutcome.ESCALATE
    assert result.answer_text is None
    assert result.error_type == "RuntimeError"
    assert "retrieval failed" in result.reason.lower()
    assert generator.calls == 0

    ticket_run = store.fetch_ticket_run(
        run_id="RUN-RETRIEVAL-FAILURE",
        ticket_id="WF-001",
    )

    assert ticket_run["terminal_outcome"] == "escalate"
    assert ticket_run["error_type"] == "RuntimeError"


def test_generation_failure_escalates(
    tmp_path: Path,
) -> None:
    generator = FakeGenerator(
        error=GenerationError(
            "model request failed"
        )
    )

    workflow, store = make_workflow(
        tmp_path,
        generator=generator,
    )

    result = workflow.process(
        make_ticket(),
        run_id="RUN-GENERATION-FAILURE",
    )

    assert result.outcome is WorkflowOutcome.ESCALATE
    assert result.answer_text is None
    assert result.may_send_response is False
    assert result.error_type == "GenerationError"
    assert result.reason == "model request failed"

    ticket_run = store.fetch_ticket_run(
        run_id="RUN-GENERATION-FAILURE",
        ticket_id="WF-001",
    )

    assert ticket_run["terminal_outcome"] == "escalate"


def test_sensitive_generated_answer_is_blocked(
    tmp_path: Path,
) -> None:
    generator = FakeGenerator(
        answer_text=(
            "You can reset your password from account settings "
            "[DOC001#p001]. Contact private@example.com."
        ),
        claim_text=(
            "Users can reset their password from the account "
            "settings page."
        ),
    )

    workflow, store = make_workflow(
        tmp_path,
        generator=generator,
    )

    result = workflow.process(
        make_ticket(),
        run_id="RUN-BLOCKED",
    )

    assert result.outcome is WorkflowOutcome.BLOCK
    assert result.may_send_response is False

    # Unsafe draft text must not be exposed as sendable output.
    assert result.answer_text is None
    assert result.guardrail_report is not None
    assert result.guardrail_report.blocked is True
    assert (
        "sensitive_data"
        in result.guardrail_report.blocking_guardrails
    )

    ticket_run = store.fetch_ticket_run(
        run_id="RUN-BLOCKED",
        ticket_id="WF-001",
    )

    assert ticket_run["terminal_outcome"] == "block"


class FailingGuardrailEngine:
    def evaluate(self, **kwargs):
        raise RuntimeError(
            "guardrail service failed"
        )


def test_guardrail_engine_failure_blocks_response(
    tmp_path: Path,
) -> None:
    workflow, store = make_workflow(
        tmp_path,
        guardrail_engine=FailingGuardrailEngine(),
    )

    result = workflow.process(
        make_ticket(),
        run_id="RUN-GUARDRAIL-FAILURE",
    )

    assert result.outcome is WorkflowOutcome.BLOCK
    assert result.answer_text is None
    assert result.may_send_response is False
    assert result.error_type == "RuntimeError"

    ticket_run = store.fetch_ticket_run(
        run_id="RUN-GUARDRAIL-FAILURE",
        ticket_id="WF-001",
    )

    assert ticket_run["terminal_outcome"] == "block"


def test_retrieval_receives_normalized_ticket_text(
    tmp_path: Path,
) -> None:
    retriever = FakeRetriever()

    workflow, _ = make_workflow(
        tmp_path,
        retriever=retriever,
        retrieval_top_k=3,
    )

    ticket = make_ticket()

    workflow.process(
        ticket,
        run_id="RUN-QUERY",
    )

    assert retriever.last_query == ticket.text
    assert retriever.last_top_k == 3
    assert retriever.last_minimum_relevance == 0.0


def test_result_serializes_to_dictionary(
    tmp_path: Path,
) -> None:
    workflow, _ = make_workflow(tmp_path)

    result = workflow.process(
        make_ticket(),
        run_id="RUN-SERIALIZE",
    )

    serialized = result.as_dict()

    assert serialized["run_id"] == "RUN-SERIALIZE"
    assert serialized["ticket_id"] == "WF-001"
    assert serialized["outcome"] == "auto_respond"
    assert serialized["may_send_response"] is True
    assert serialized["classification"]["intent"] == (
        "account_access"
    )
    assert serialized["route"]["action"] == (
        "auto_respond"
    )
    assert serialized["guardrails"]["blocked"] is False


def test_batch_processes_every_ticket(
    tmp_path: Path,
) -> None:
    workflow, store = make_workflow(tmp_path)

    tickets = [
        make_ticket("BATCH-001"),
        make_ticket("BATCH-002"),
        make_ticket("BATCH-003"),
    ]

    results = workflow.process_batch(
        tickets,
        run_id="RUN-BATCH",
    )

    assert len(results) == 3
    assert {
        result.ticket_id
        for result in results
    } == {
        "BATCH-001",
        "BATCH-002",
        "BATCH-003",
    }

    assert all(
        result.outcome
        is WorkflowOutcome.AUTO_RESPOND
        for result in results
    )

    for ticket in tickets:
        ticket_run = store.fetch_ticket_run(
            run_id="RUN-BATCH",
            ticket_id=ticket.ticket_id,
        )

        assert ticket_run is not None
        assert ticket_run["status"] == "completed"


def test_batch_continues_after_ticket_failure(
    tmp_path: Path,
) -> None:
    class SelectiveClassifier(FakeClassifier):
        def classify(self, ticket):
            if ticket.ticket_id == "FAIL-001":
                raise RuntimeError(
                    "selected classifier failure"
                )

            return super().classify(ticket)

    workflow, _ = make_workflow(
        tmp_path,
        classifier=SelectiveClassifier(),
    )

    results = workflow.process_batch(
        [
            make_ticket("PASS-001"),
            make_ticket("FAIL-001"),
            make_ticket("PASS-002"),
        ],
        run_id="RUN-MIXED-BATCH",
    )

    assert len(results) == 3
    assert results[0].outcome is (
        WorkflowOutcome.AUTO_RESPOND
    )
    assert results[1].outcome is (
        WorkflowOutcome.ESCALATE
    )
    assert results[2].outcome is (
        WorkflowOutcome.AUTO_RESPOND
    )


def test_invalid_retrieval_top_k_is_rejected(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)

    with pytest.raises(
        ValueError,
        match="retrieval_top_k",
    ):
        CustomerSuccessWorkflow(
            classifier=FakeClassifier(),
            retriever=FakeRetriever(),
            generator=FakeGenerator(),
            guardrail_engine=GuardrailEngine(),
            routing_policy=RoutingPolicy(
                confidence_threshold=0.80,
                retrieval_threshold=0.40,
            ),
            decision_store=store,
            retrieval_top_k=0,
        )


def test_routing_uses_intent_confidence_not_urgency_confidence(
    tmp_path: Path,
) -> None:
    classifier = FakeClassifier(
        intent="account_access",
        intent_confidence=0.91,
        urgency_confidence=0.45,
    )
    generator = FakeGenerator()

    workflow, store = make_workflow(
        tmp_path,
        classifier=classifier,
        generator=generator,
    )

    result = workflow.process(
        make_ticket(),
        run_id="RUN-SEPARATE-CONFIDENCE",
    )

    assert result.outcome is WorkflowOutcome.AUTO_RESPOND
    assert result.route_decision is not None
    assert (
        result.route_decision.classification_confidence
        == pytest.approx(0.91)
    )
    assert generator.calls == 1
