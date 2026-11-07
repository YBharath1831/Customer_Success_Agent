"""Tests for Prometheus-compatible operational metrics."""

from types import SimpleNamespace

from prometheus_client import CollectorRegistry

from src.metrics import ApplicationMetrics
from src.workflow import WorkflowOutcome


def render_text(
    metrics: ApplicationMetrics,
) -> str:
    return metrics.render().decode("utf-8")


def make_result(
    *,
    outcome=WorkflowOutcome.AUTO_RESPOND,
    latency_seconds=1.25,
    confidence=0.87,
    relevance_score=0.76,
    blocked=False,
    blocking_guardrails=(),
    error_type=None,
):
    classification = SimpleNamespace(
        confidence=confidence
    )

    retrieval_match = SimpleNamespace(
        relevance_score=relevance_score
    )

    guardrail_report = SimpleNamespace(
        blocked=blocked,
        blocking_guardrails=(
            blocking_guardrails
        ),
    )

    return SimpleNamespace(
        outcome=outcome,
        latency_seconds=latency_seconds,
        classification=classification,
        retrieval_matches=(
            retrieval_match,
        ),
        guardrail_report=guardrail_report,
        error_type=error_type,
        ticket_id="PRIVATE-TICKET-ID",
        subject="Private subject must not appear",
    )


def test_metrics_render_prometheus_format() -> None:
    metrics = ApplicationMetrics()

    rendered = render_text(metrics)

    assert (
        "# HELP customer_success_tickets_total"
        in rendered
    )
    assert (
        "# TYPE customer_success_tickets_total counter"
        in rendered
    )
    assert (
        "# HELP customer_success_application_ready"
        in rendered
    )


def test_application_ready_starts_at_zero() -> None:
    metrics = ApplicationMetrics()

    rendered = render_text(metrics)

    assert (
        "customer_success_application_ready 0.0"
        in rendered
    )


def test_application_ready_can_be_set() -> None:
    metrics = ApplicationMetrics()

    metrics.set_application_ready(True)

    assert (
        "customer_success_application_ready 1.0"
        in render_text(metrics)
    )

    metrics.set_application_ready(False)

    assert (
        "customer_success_application_ready 0.0"
        in render_text(metrics)
    )


def test_http_request_is_recorded() -> None:
    metrics = ApplicationMetrics()

    metrics.record_http_request(
        method="get",
        endpoint="/health",
        status_code=200,
    )

    rendered = render_text(metrics)

    assert (
        'customer_success_http_requests_total'
        in rendered
    )
    assert 'method="GET"' in rendered
    assert 'endpoint="/health"' in rendered
    assert 'status_code="200"' in rendered


def test_http_request_counter_increments() -> None:
    metrics = ApplicationMetrics()

    for _ in range(3):
        metrics.record_http_request(
            method="POST",
            endpoint="/tickets/process",
            status_code=200,
        )

    rendered = render_text(metrics)

    matching_line = next(
        line
        for line in rendered.splitlines()
        if line.startswith(
            "customer_success_http_requests_total"
        )
        and 'method="POST"' in line
        and 'endpoint="/tickets/process"' in line
        and 'status_code="200"' in line
    )

    assert matching_line.endswith(" 3.0")


def test_stage_failure_records_exception_type() -> None:
    metrics = ApplicationMetrics()

    metrics.record_stage_failure(
        stage="retrieval",
        error=RuntimeError(
            "private error details"
        ),
    )

    rendered = render_text(metrics)

    assert (
        "customer_success_stage_failures_total"
        in rendered
    )
    assert 'stage="retrieval"' in rendered
    assert 'error_type="RuntimeError"' in rendered

    # Exception messages must not become metric labels.
    assert "private error details" not in rendered


def test_unknown_stage_is_bounded_to_other() -> None:
    metrics = ApplicationMetrics()

    metrics.record_stage_failure(
        stage="customer-controlled-stage",
        error="ExampleError",
    )

    rendered = render_text(metrics)

    assert 'stage="other"' in rendered
    assert (
        "customer-controlled-stage"
        not in rendered
    )


def test_successful_workflow_result_is_recorded() -> None:
    metrics = ApplicationMetrics()
    result = make_result()

    metrics.record_workflow_result(
        result=result,
        channel="email",
    )

    rendered = render_text(metrics)

    ticket_line = next(
        line
        for line in rendered.splitlines()
        if line.startswith(
            "customer_success_tickets_total"
        )
        and 'channel="email"' in line
        and 'outcome="auto_respond"' in line
    )

    assert ticket_line.endswith(" 1.0")

    assert (
        "customer_success_workflow_latency_seconds_count"
        in rendered
    )
    assert (
        "customer_success_classification_confidence_count 1.0"
        in rendered
    )
    assert (
        "customer_success_retrieval_relevance_count 1.0"
        in rendered
    )


def test_blocking_guardrails_are_recorded() -> None:
    metrics = ApplicationMetrics()

    result = make_result(
        outcome=WorkflowOutcome.BLOCK,
        blocked=True,
        blocking_guardrails=(
            "sensitive_data",
            "claim_grounding",
        ),
    )

    metrics.record_workflow_result(
        result=result,
        channel="chat",
    )

    rendered = render_text(metrics)

    assert (
        'customer_success_guardrail_blocks_total'
        '{guardrail="sensitive_data"} 1.0'
        in rendered
    )
    assert (
        'customer_success_guardrail_blocks_total'
        '{guardrail="claim_grounding"} 1.0'
        in rendered
    )


def test_workflow_error_type_is_recorded() -> None:
    metrics = ApplicationMetrics()

    result = make_result(
        outcome=WorkflowOutcome.ESCALATE,
        error_type="GenerationError",
    )

    metrics.record_workflow_result(
        result=result,
        channel="forum",
    )

    rendered = render_text(metrics)

    assert 'stage="api"' in rendered
    assert 'error_type="GenerationError"' in rendered


def test_unknown_channel_and_outcome_use_other() -> None:
    metrics = ApplicationMetrics()

    result = make_result(
        outcome="unknown-outcome",
    )

    metrics.record_workflow_result(
        result=result,
        channel="unknown-channel",
    )

    rendered = render_text(metrics)

    assert 'channel="other"' in rendered
    assert 'outcome="other"' in rendered
    assert "unknown-channel" not in rendered
    assert "unknown-outcome" not in rendered


def test_invalid_observations_are_ignored() -> None:
    metrics = ApplicationMetrics()

    result = make_result(
        latency_seconds=-1,
        confidence=1.5,
        relevance_score=-0.1,
    )

    metrics.record_workflow_result(
        result=result,
        channel="email",
    )

    rendered = render_text(metrics)

    assert (
        "customer_success_workflow_latency_seconds_count"
        not in rendered
    )
    assert (
        "customer_success_classification_confidence_count 0.0"
        in rendered
    )
    assert (
        "customer_success_retrieval_relevance_count 0.0"
        in rendered
    )


def test_private_ticket_values_are_not_exported() -> None:
    metrics = ApplicationMetrics()

    result = make_result()

    metrics.record_workflow_result(
        result=result,
        channel="docs_comment",
    )

    rendered = render_text(metrics)

    assert "PRIVATE-TICKET-ID" not in rendered
    assert (
        "Private subject must not appear"
        not in rendered
    )


def test_error_type_label_is_length_bounded() -> None:
    metrics = ApplicationMetrics()

    long_error_type = "E" * 200

    metrics.record_stage_failure(
        stage="generation",
        error=long_error_type,
    )

    rendered = render_text(metrics)

    assert long_error_type not in rendered
    assert ("E" * 64) in rendered


def test_separate_registries_do_not_conflict() -> None:
    first_registry = CollectorRegistry()
    second_registry = CollectorRegistry()

    first_metrics = ApplicationMetrics(
        registry=first_registry
    )
    second_metrics = ApplicationMetrics(
        registry=second_registry
    )

    first_metrics.record_http_request(
        method="GET",
        endpoint="/health",
        status_code=200,
    )

    second_metrics.record_http_request(
        method="GET",
        endpoint="/ready",
        status_code=503,
    )

    first_rendered = render_text(
        first_metrics
    )
    second_rendered = render_text(
        second_metrics
    )

    assert 'endpoint="/health"' in first_rendered
    assert 'endpoint="/ready"' not in first_rendered

    assert 'endpoint="/ready"' in second_rendered
    assert 'endpoint="/health"' not in second_rendered


def test_content_type_is_prometheus_format() -> None:
    metrics = ApplicationMetrics()

    assert "text/plain" in metrics.content_type
    assert "version=" in metrics.content_type