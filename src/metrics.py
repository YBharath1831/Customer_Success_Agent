"""Prometheus-compatible operational metrics."""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


SUPPORTED_CHANNELS = {
    "email",
    "chat",
    "docs_comment",
    "forum",
}

SUPPORTED_OUTCOMES = {
    "auto_respond",
    "escalate",
    "block",
}

SUPPORTED_STAGES = {
    "ingestion",
    "classification",
    "retrieval",
    "routing",
    "generation",
    "validation",
    "api",
}


class ApplicationMetrics:
    """
    Metrics for application health, workflow decisions, and latency.

    Ticket IDs, customer IDs, subjects, bodies, answers, and other
    high-cardinality or private values must never be metric labels.
    """

    def __init__(
        self,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self.registry = (
            registry or CollectorRegistry()
        )

        self.tickets_total = Counter(
            "customer_success_tickets_total",
            (
                "Number of tickets reaching a terminal "
                "workflow outcome."
            ),
            labelnames=(
                "channel",
                "outcome",
            ),
            registry=self.registry,
        )

        self.workflow_latency_seconds = Histogram(
            "customer_success_workflow_latency_seconds",
            (
                "End-to-end workflow latency in seconds."
            ),
            labelnames=("outcome",),
            buckets=(
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
                30.0,
                60.0,
            ),
            registry=self.registry,
        )

        self.classification_confidence = Histogram(
            "customer_success_classification_confidence",
            (
                "Conservative combined classifier confidence."
            ),
            buckets=(
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ),
            registry=self.registry,
        )

        self.retrieval_relevance = Histogram(
            "customer_success_retrieval_relevance",
            (
                "Top documentation retrieval relevance score."
            ),
            buckets=(
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ),
            registry=self.registry,
        )

        self.guardrail_blocks_total = Counter(
            "customer_success_guardrail_blocks_total",
            (
                "Number of response blocks grouped by "
                "blocking guardrail."
            ),
            labelnames=("guardrail",),
            registry=self.registry,
        )

        self.stage_failures_total = Counter(
            "customer_success_stage_failures_total",
            (
                "Number of failures grouped by workflow stage "
                "and exception class."
            ),
            labelnames=(
                "stage",
                "error_type",
            ),
            registry=self.registry,
        )

        self.http_requests_total = Counter(
            "customer_success_http_requests_total",
            (
                "Number of HTTP requests grouped by method, "
                "endpoint, and response status."
            ),
            labelnames=(
                "method",
                "endpoint",
                "status_code",
            ),
            registry=self.registry,
        )

        self.application_ready = Gauge(
            "customer_success_application_ready",
            (
                "Whether application dependencies are ready: "
                "1 for ready and 0 for not ready."
            ),
            registry=self.registry,
        )

        self.application_ready.set(0)

    @property
    def content_type(self) -> str:
        """Prometheus exposition response content type."""

        return CONTENT_TYPE_LATEST

    def set_application_ready(
        self,
        ready: bool,
    ) -> None:
        self.application_ready.set(
            1 if ready else 0
        )

    def record_http_request(
        self,
        *,
        method: str,
        endpoint: str,
        status_code: int,
    ) -> None:
        """
        Record an API request.

        The caller should pass a route template such as /tickets/process,
        never a user-controlled URL.
        """

        normalized_method = (
            method.strip().upper() or "UNKNOWN"
        )
        normalized_endpoint = (
            endpoint.strip() or "unknown"
        )

        self.http_requests_total.labels(
            method=normalized_method,
            endpoint=normalized_endpoint,
            status_code=str(status_code),
        ).inc()

    def record_stage_failure(
        self,
        *,
        stage: str,
        error: Exception | str,
    ) -> None:
        normalized_stage = _bounded_label(
            stage,
            allowed=SUPPORTED_STAGES,
        )

        error_type = (
            type(error).__name__
            if isinstance(error, Exception)
            else str(error).strip()
        )

        if not error_type:
            error_type = "UnknownError"

        # Bound label length to protect the metrics backend.
        error_type = error_type[:64]

        self.stage_failures_total.labels(
            stage=normalized_stage,
            error_type=error_type,
        ).inc()

    def record_workflow_result(
        self,
        *,
        result: Any,
        channel: str,
    ) -> None:
        """
        Record one completed WorkflowResult.

        Any object exposing the WorkflowResult fields may be supplied, which
        keeps this module easy to test without real models.
        """

        outcome_value = getattr(
            getattr(result, "outcome", None),
            "value",
            getattr(result, "outcome", None),
        )

        normalized_outcome = _bounded_label(
            str(outcome_value or ""),
            allowed=SUPPORTED_OUTCOMES,
        )

        normalized_channel = _bounded_label(
            channel,
            allowed=SUPPORTED_CHANNELS,
        )

        self.tickets_total.labels(
            channel=normalized_channel,
            outcome=normalized_outcome,
        ).inc()

        latency = getattr(
            result,
            "latency_seconds",
            None,
        )

        if _is_non_negative_number(latency):
            self.workflow_latency_seconds.labels(
                outcome=normalized_outcome
            ).observe(float(latency))

        classification = getattr(
            result,
            "classification",
            None,
        )

        confidence = getattr(
            classification,
            "confidence",
            None,
        )

        if _is_probability(confidence):
            self.classification_confidence.observe(
                float(confidence)
            )

        matches = getattr(
            result,
            "retrieval_matches",
            (),
        ) or ()

        relevance_scores = [
            getattr(match, "relevance_score", None)
            for match in matches
        ]

        valid_scores = [
            float(score)
            for score in relevance_scores
            if _is_probability(score)
        ]

        if valid_scores:
            self.retrieval_relevance.observe(
                max(valid_scores)
            )

        guardrail_report = getattr(
            result,
            "guardrail_report",
            None,
        )

        if (
            guardrail_report is not None
            and getattr(
                guardrail_report,
                "blocked",
                False,
            )
        ):
            blocking_guardrails = getattr(
                guardrail_report,
                "blocking_guardrails",
                (),
            )

            for guardrail_name in (
                blocking_guardrails
            ):
                safe_guardrail_name = (
                    str(guardrail_name).strip()
                    or "unknown"
                )[:64]

                self.guardrail_blocks_total.labels(
                    guardrail=safe_guardrail_name
                ).inc()

        error_type = getattr(
            result,
            "error_type",
            None,
        )

        if error_type:
            self.record_stage_failure(
                stage="api",
                error=str(error_type),
            )

    def render(self) -> bytes:
        """Render metrics using Prometheus text exposition format."""

        return generate_latest(
            self.registry
        )


def _bounded_label(
    value: str,
    *,
    allowed: set[str],
) -> str:
    normalized = str(value).strip().lower()

    if normalized in allowed:
        return normalized

    return "other"


def _is_non_negative_number(
    value: Any,
) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
    )


def _is_probability(
    value: Any,
) -> bool:
    return (
        _is_non_negative_number(value)
        and value <= 1
    )