"""End-to-end customer-success ticket workflow."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .classify import (
    ClassificationResult,
    TicketClassifier,
)
from .generate import (
    GeneratedDraft,
    GenerationError,
    ResponseGenerator,
)
from .guardrails import (
    GuardrailEngine,
    GuardrailReport,
)
from .logging_store import (
    DecisionRecord,
    DecisionStore,
    hash_input,
)
from .models import NormalizedTicket
from .retrieve import (
    ChromaRetriever,
    RetrievedMatch,
)
from .route import (
    RouteDecision,
    RoutingPolicy,
    decide_route,
)


class WorkflowOutcome(str, Enum):
    """Terminal outcomes produced by the workflow."""

    AUTO_RESPOND = "auto_respond"
    ESCALATE = "escalate"
    BLOCK = "block"


@dataclass(frozen=True)
class WorkflowResult:
    """Final result for one processed ticket."""

    run_id: str
    ticket_id: str
    outcome: WorkflowOutcome
    reason: str

    answer_text: str | None = None
    cited_passage_ids: tuple[str, ...] = ()

    classification: ClassificationResult | None = None
    retrieval_matches: tuple[RetrievedMatch, ...] = ()
    route_decision: RouteDecision | None = None
    guardrail_report: GuardrailReport | None = None

    error_type: str | None = None
    latency_seconds: float = 0.0

    @property
    def may_send_response(self) -> bool:
        """Only this condition authorizes sending the generated answer."""

        return (
            self.outcome is WorkflowOutcome.AUTO_RESPOND
            and bool(self.answer_text)
        )

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "ticket_id": self.ticket_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "answer_text": self.answer_text,
            "cited_passage_ids": list(
                self.cited_passage_ids
            ),
            "classification": (
                {
                    "intent": (
                        self.classification.intent.value
                    ),
                    "intent_confidence": (
                        self.classification.intent.confidence
                    ),
                    "urgency": (
                        self.classification.urgency.value
                    ),
                    "urgency_confidence": (
                        self.classification.urgency.confidence
                    ),
                    "combined_confidence": (
                        self.classification.confidence
                    ),
                    "model_name": (
                        self.classification.model_name
                    ),
                    "model_version": (
                        self.classification.model_version
                    ),
                    "fallback_used": (
                        self.classification.fallback_used
                    ),
                }
                if self.classification is not None
                else None
            ),
            "retrieval_matches": [
                {
                    "passage_id": match.passage_id,
                    "title": match.title,
                    "category": match.category,
                    "relevance_score": (
                        match.relevance_score
                    ),
                }
                for match in self.retrieval_matches
            ],
            "route": (
                {
                    "action": (
                        self.route_decision.action.value
                    ),
                    "reason_code": (
                        self.route_decision.reason_code.value
                    ),
                    "reason": (
                        self.route_decision.reason
                    ),
                }
                if self.route_decision is not None
                else None
            ),
            "guardrails": (
                self.guardrail_report.as_dict()
                if self.guardrail_report is not None
                else None
            ),
            "error_type": self.error_type,
            "latency_seconds": self.latency_seconds,
            "may_send_response": self.may_send_response,
        }


class CustomerSuccessWorkflow:
    """
    Execute the complete ticket decision pipeline.

    Pipeline:

        normalize before entry
            -> classify
            -> retrieve
            -> route
            -> generate
            -> guardrails
            -> terminal outcome

    Every important decision is written to SQLite.
    """

    def __init__(
        self,
        *,
        classifier: TicketClassifier,
        retriever: ChromaRetriever,
        generator: ResponseGenerator,
        guardrail_engine: GuardrailEngine,
        routing_policy: RoutingPolicy,
        decision_store: DecisionStore,
        retrieval_top_k: int = 5,
    ) -> None:
        if retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than zero."
            )

        self.classifier = classifier
        self.retriever = retriever
        self.generator = generator
        self.guardrail_engine = guardrail_engine
        self.routing_policy = routing_policy
        self.decision_store = decision_store
        self.retrieval_top_k = retrieval_top_k

        # Initialization is idempotent and ensures that we cannot process
        # tickets without an available audit schema.
        self.decision_store.initialize()

    def process(
        self,
        ticket: NormalizedTicket,
        *,
        run_id: str | None = None,
    ) -> WorkflowResult:
        """Process one normalized ticket to exactly one terminal outcome."""

        resolved_run_id = run_id or str(uuid.uuid4())
        started_at = time.perf_counter()

        classification: ClassificationResult | None = None
        matches: list[RetrievedMatch] = []
        route_decision: RouteDecision | None = None
        guardrail_report: GuardrailReport | None = None

        self.decision_store.start_ticket(
            run_id=resolved_run_id,
            ticket_id=ticket.ticket_id,
            channel=ticket.channel.value,
        )

        self._log_ingestion(
            run_id=resolved_run_id,
            ticket=ticket,
        )

        # Classification failure must escalate rather than terminate the
        # entire batch or silently invent a classification.
        try:
            classification = self.classifier.classify(
                ticket
            )
        except Exception as exc:
            self._log_failure(
                run_id=resolved_run_id,
                ticket=ticket,
                stage="classification",
                reason="Ticket classification failed.",
                exc=exc,
            )

            route_decision = decide_route(
                ticket=ticket,
                predicted_intent=None,
                classification_confidence=None,
                retrieval_scores=[],
                policy=self.routing_policy,
            )

            self._log_routing(
                run_id=resolved_run_id,
                ticket=ticket,
                decision=route_decision,
            )

            return self._finish(
                started_at=started_at,
                run_id=resolved_run_id,
                ticket=ticket,
                outcome=WorkflowOutcome.ESCALATE,
                reason="Classification failed; human review required.",
                route_decision=route_decision,
                error=exc,
            )

        self._log_classification(
            run_id=resolved_run_id,
            ticket=ticket,
            classification=classification,
        )

        # Retrieval errors are converted into an empty result. The
        # deterministic router will then escalate due to missing grounding.
        retrieval_error: Exception | None = None

        try:
            matches = self.retriever.search(
                ticket.text,
                top_k=self.retrieval_top_k,
                minimum_relevance=0.0,
            )
        except Exception as exc:
            retrieval_error = exc
            matches = []

            self._log_failure(
                run_id=resolved_run_id,
                ticket=ticket,
                stage="retrieval",
                reason="Documentation retrieval failed.",
                exc=exc,
            )
        else:
            self._log_retrieval(
                run_id=resolved_run_id,
                ticket=ticket,
                matches=matches,
            )

        route_decision = decide_route(
            ticket=ticket,
            predicted_intent=classification.intent.value,
            classification_confidence=(
                classification.intent.confidence
            ),
            retrieval_scores=[
                match.relevance_score
                for match in matches
            ],
            policy=self.routing_policy,
        )

        self._log_routing(
            run_id=resolved_run_id,
            ticket=ticket,
            decision=route_decision,
        )

        if not route_decision.should_generate:
            reason = route_decision.reason

            if retrieval_error is not None:
                reason = (
                    "Documentation retrieval failed; "
                    "human review required."
                )

            return self._finish(
                started_at=started_at,
                run_id=resolved_run_id,
                ticket=ticket,
                outcome=WorkflowOutcome.ESCALATE,
                reason=reason,
                classification=classification,
                matches=matches,
                route_decision=route_decision,
                error=retrieval_error,
            )

        # The router authorizes generation, but generation itself can still
        # fail. Such failures escalate and never produce a sendable answer.
        try:
            draft = self.generator.generate(
                ticket=ticket,
                intent=classification.intent.value,
                urgency=classification.urgency.value,
                passages=matches,
            )
        except Exception as exc:
            reason = (
                "Response generation failed; "
                "human review required."
            )

            if isinstance(exc, GenerationError):
                reason = str(exc)

            self._log_failure(
                run_id=resolved_run_id,
                ticket=ticket,
                stage="generation",
                reason=reason,
                exc=exc,
            )

            return self._finish(
                started_at=started_at,
                run_id=resolved_run_id,
                ticket=ticket,
                outcome=WorkflowOutcome.ESCALATE,
                reason=reason,
                classification=classification,
                matches=matches,
                route_decision=route_decision,
                error=exc,
            )

        self._log_generation(
            run_id=resolved_run_id,
            ticket=ticket,
            draft=draft,
        )

        # Guardrail failures are BLOCK outcomes, not ordinary escalations.
        # The draft is deliberately omitted from the returned answer_text.
        try:
            guardrail_report = (
                self.guardrail_engine.evaluate(
                    ticket=ticket,
                    answer_text=draft.answer_text,
                    claims=draft.claims,
                    retrieved_passages=matches,
                    routing_allows_generation=(
                        route_decision.should_generate
                    ),
                )
            )
        except Exception as exc:
            self._log_failure(
                run_id=resolved_run_id,
                ticket=ticket,
                stage="validation",
                reason="Guardrail evaluation failed.",
                exc=exc,
            )

            return self._finish(
                started_at=started_at,
                run_id=resolved_run_id,
                ticket=ticket,
                outcome=WorkflowOutcome.BLOCK,
                reason=(
                    "Guardrail evaluation failed; "
                    "the response was blocked."
                ),
                classification=classification,
                matches=matches,
                route_decision=route_decision,
                error=exc,
            )

        self._log_guardrails(
            run_id=resolved_run_id,
            ticket=ticket,
            report=guardrail_report,
        )

        if guardrail_report.blocked:
            blocking_names = ", ".join(
                guardrail_report.blocking_guardrails
            )

            return self._finish(
                started_at=started_at,
                run_id=resolved_run_id,
                ticket=ticket,
                outcome=WorkflowOutcome.BLOCK,
                reason=(
                    "Generated response was blocked by: "
                    f"{blocking_names}."
                ),
                classification=classification,
                matches=matches,
                route_decision=route_decision,
                guardrail_report=guardrail_report,
            )

        return self._finish(
            started_at=started_at,
            run_id=resolved_run_id,
            ticket=ticket,
            outcome=WorkflowOutcome.AUTO_RESPOND,
            reason=(
                "Classification, retrieval, routing, generation, "
                "and guardrail checks passed."
            ),
            answer_text=draft.answer_text,
            cited_passage_ids=draft.cited_passage_ids,
            classification=classification,
            matches=matches,
            route_decision=route_decision,
            guardrail_report=guardrail_report,
        )

    def process_batch(
        self,
        tickets: Sequence[NormalizedTicket],
        *,
        run_id: str | None = None,
    ) -> list[WorkflowResult]:
        """
        Process all tickets without stopping after a ticket-level failure.

        A shared run ID groups the records in the audit database.
        """

        resolved_run_id = run_id or str(uuid.uuid4())
        results: list[WorkflowResult] = []

        for ticket in tickets:
            try:
                result = self.process(
                    ticket,
                    run_id=resolved_run_id,
                )
            except Exception as exc:
                # This path is reserved for infrastructure failures such as
                # an unavailable audit database. We retain an in-memory result
                # so the remaining evaluation tickets can still run.
                result = WorkflowResult(
                    run_id=resolved_run_id,
                    ticket_id=ticket.ticket_id,
                    outcome=WorkflowOutcome.ESCALATE,
                    reason=(
                        "Unexpected workflow infrastructure "
                        "failure; human review required."
                    ),
                    error_type=type(exc).__name__,
                )

            results.append(result)

        return results

    def _log_ingestion(
        self,
        *,
        run_id: str,
        ticket: NormalizedTicket,
    ) -> None:
        self.decision_store.log_decision(
            DecisionRecord(
                run_id=run_id,
                ticket_id=ticket.ticket_id,
                stage="ingestion",
                action_taken="normalized",
                reason=(
                    "Ticket passed schema validation and "
                    "normalization."
                ),
                input_summary=self._ticket_summary(ticket),
                input_hash=hash_input(ticket.text),
                requirement_ids=["A2"],
            )
        )

    def _log_classification(
        self,
        *,
        run_id: str,
        ticket: NormalizedTicket,
        classification: ClassificationResult,
    ) -> None:
        alternatives = []

        for target_name, prediction in (
            ("intent", classification.intent),
            ("urgency", classification.urgency),
        ):
            alternatives.extend(
                {
                    "target": target_name,
                    "value": alternative.value,
                    "confidence": alternative.confidence,
                }
                for alternative in prediction.alternatives
            )

        self.decision_store.log_decision(
            DecisionRecord(
                run_id=run_id,
                ticket_id=ticket.ticket_id,
                stage="classification",
                action_taken="classified",
                reason=(
                    "Intent and urgency were predicted by "
                    "the local classifier."
                ),
                input_summary=self._ticket_summary(ticket),
                input_hash=hash_input(ticket.text),
                model_name=classification.model_name,
                model_version=classification.model_version,
                prediction={
                    "intent": classification.intent.value,
                    "intent_confidence": (
                        classification.intent.confidence
                    ),
                    "urgency": classification.urgency.value,
                    "urgency_confidence": (
                        classification.urgency.confidence
                    ),
                    "fallback_used": (
                        classification.fallback_used
                    ),
                    "failure_reason": (
                        classification.failure_reason
                    ),
                },
                confidence=classification.confidence,
                alternatives=alternatives,
                requirement_ids=["A3"],
            )
        )

    def _log_retrieval(
        self,
        *,
        run_id: str,
        ticket: NormalizedTicket,
        matches: Sequence[RetrievedMatch],
    ) -> None:
        sources = [
            {
                "passage_id": match.passage_id,
                "doc_id": match.doc_id,
                "title": match.title,
                "category": match.category,
                "relevance_score": match.relevance_score,
            }
            for match in matches
        ]

        top_score = (
            matches[0].relevance_score
            if matches
            else None
        )

        self.decision_store.log_decision(
            DecisionRecord(
                run_id=run_id,
                ticket_id=ticket.ticket_id,
                stage="retrieval",
                action_taken=(
                    "retrieved"
                    if matches
                    else "no_results"
                ),
                reason=(
                    f"Retrieved {len(matches)} documentation "
                    "passage(s)."
                ),
                input_summary=(
                    "Semantic search over normalized ticket text."
                ),
                input_hash=hash_input(ticket.text),
                prediction={
                    "result_count": len(matches),
                    "top_relevance_score": top_score,
                },
                sources_used=sources,
                confidence=top_score,
                requirement_ids=["A4"],
            )
        )

    def _log_routing(
        self,
        *,
        run_id: str,
        ticket: NormalizedTicket,
        decision: RouteDecision,
    ) -> None:
        self.decision_store.log_decision(
            DecisionRecord(
                run_id=run_id,
                ticket_id=ticket.ticket_id,
                stage="routing",
                action_taken=decision.action.value,
                reason=decision.reason,
                prediction={
                    "reason_code": (
                        decision.reason_code.value
                    ),
                    "predicted_intent": (
                        decision.predicted_intent
                    ),
                    "policy_checks": (
                        decision.policy_checks
                    ),
                    "top_retrieval_score": (
                        decision.top_retrieval_score
                    ),
                },
                confidence=(
                    decision.classification_confidence
                ),
                threshold_applied=(
                    decision.confidence_threshold
                ),
                requirement_ids=list(
                    decision.requirement_ids
                ),
            )
        )

    def _log_generation(
        self,
        *,
        run_id: str,
        ticket: NormalizedTicket,
        draft: GeneratedDraft,
    ) -> None:
        self.decision_store.log_decision(
            DecisionRecord(
                run_id=run_id,
                ticket_id=ticket.ticket_id,
                stage="generation",
                action_taken="draft_generated",
                reason=(
                    "A structured response draft was generated "
                    "with passage citations."
                ),
                model_name=draft.model,
                prediction={
                    "answer_character_count": len(
                        draft.answer_text
                    ),
                    "claim_count": len(draft.claims),
                    "cited_passage_ids": list(
                        draft.cited_passage_ids
                    ),
                },
                sources_used=[
                    {
                        "passage_id": passage_id,
                    }
                    for passage_id in (
                        draft.cited_passage_ids
                    )
                ],
                prompt_version="generation-v1",
                requirement_ids=["A6"],
            )
        )

    def _log_guardrails(
        self,
        *,
        run_id: str,
        ticket: NormalizedTicket,
        report: GuardrailReport,
    ) -> None:
        self.decision_store.log_decision(
            DecisionRecord(
                run_id=run_id,
                ticket_id=ticket.ticket_id,
                stage="validation",
                action_taken=(
                    "blocked"
                    if report.blocked
                    else "approved"
                ),
                reason=(
                    "One or more blocking guardrails failed."
                    if report.blocked
                    else "All blocking guardrails passed."
                ),
                guardrail_results=report.as_dict(),
                requirement_ids=["A7"],
            )
        )

    def _log_failure(
        self,
        *,
        run_id: str,
        ticket: NormalizedTicket,
        stage: str,
        reason: str,
        exc: Exception,
    ) -> None:
        self.decision_store.log_decision(
            DecisionRecord(
                run_id=run_id,
                ticket_id=ticket.ticket_id,
                stage=stage,
                action_taken="failed",
                reason=reason,
                error=type(exc).__name__,
            )
        )

    def _finish(
        self,
        *,
        started_at: float,
        run_id: str,
        ticket: NormalizedTicket,
        outcome: WorkflowOutcome,
        reason: str,
        answer_text: str | None = None,
        cited_passage_ids: tuple[str, ...] = (),
        classification: ClassificationResult | None = None,
        matches: Sequence[RetrievedMatch] = (),
        route_decision: RouteDecision | None = None,
        guardrail_report: GuardrailReport | None = None,
        error: Exception | None = None,
    ) -> WorkflowResult:
        latency_seconds = max(
            time.perf_counter() - started_at,
            0.0,
        )

        self.decision_store.complete_ticket(
            run_id=run_id,
            ticket_id=ticket.ticket_id,
            terminal_outcome=outcome.value,
            latency_seconds=latency_seconds,
            error_type=(
                type(error).__name__
                if error is not None
                else None
            ),
            error_message=(
                str(error)[:500]
                if error is not None
                else None
            ),
        )

        return WorkflowResult(
            run_id=run_id,
            ticket_id=ticket.ticket_id,
            outcome=outcome,
            reason=reason,
            answer_text=answer_text,
            cited_passage_ids=cited_passage_ids,
            classification=classification,
            retrieval_matches=tuple(matches),
            route_decision=route_decision,
            guardrail_report=guardrail_report,
            error_type=(
                type(error).__name__
                if error is not None
                else None
            ),
            latency_seconds=latency_seconds,
        )

    @staticmethod
    def _ticket_summary(
        ticket: NormalizedTicket,
    ) -> str:
        """
        Produce useful audit metadata without storing ticket contents.

        Raw subject and body text are represented by a SHA-256 hash elsewhere.
        """

        return (
            f"channel={ticket.channel.value}; "
            f"subject_characters={len(ticket.subject)}; "
            f"body_characters={len(ticket.body)}"
        )