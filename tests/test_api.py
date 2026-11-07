"""Tests for the FastAPI service layer."""

from fastapi.testclient import TestClient
from src.metrics import ApplicationMetrics
import pytest

from src.api import (
    API_TITLE,
    API_VERSION,
    ApplicationProvider,
    create_api,
)
from src.app import ApplicationBuildError
from src.ingest import normalize_ticket
from src.workflow import (
    WorkflowOutcome,
    WorkflowResult,
)


class FakeApplication:
    """Offline application used by API tests."""

    def __init__(
        self,
        *,
        outcome: WorkflowOutcome = (
            WorkflowOutcome.AUTO_RESPOND
        ),
        processing_error: Exception | None = None,
    ) -> None:
        self.outcome = outcome
        self.processing_error = processing_error
        self.process_calls = []
        self.summary_calls = 0

    def startup_summary(self):
        self.summary_calls += 1

        return {
            "documentation_passage_count": 12,
            "index": {
                "collection_name": (
                    "test-collection"
                ),
                "passage_count": 12,
                "rebuilt": False,
            },
            "configuration": {
                "openrouter_api_key_configured": True,
            },
        }

    def process_raw_ticket(
        self,
        record,
        *,
        run_id=None,
    ):
        if self.processing_error is not None:
            raise self.processing_error

        ticket = normalize_ticket(record)

        self.process_calls.append(
            {
                "ticket": ticket,
                "run_id": run_id,
            }
        )

        if (
            self.outcome
            is WorkflowOutcome.AUTO_RESPOND
        ):
            return WorkflowResult(
                run_id=run_id or "GENERATED-RUN",
                ticket_id=ticket.ticket_id,
                outcome=(
                    WorkflowOutcome.AUTO_RESPOND
                ),
                reason="All checks passed.",
                answer_text=(
                    "Reset your password from account "
                    "settings [DOC001#p001]."
                ),
                cited_passage_ids=(
                    "DOC001#p001",
                ),
                latency_seconds=0.5,
            )

        if (
            self.outcome
            is WorkflowOutcome.BLOCK
        ):
            return WorkflowResult(
                run_id=run_id or "GENERATED-RUN",
                ticket_id=ticket.ticket_id,
                outcome=WorkflowOutcome.BLOCK,
                reason="Response blocked.",
                answer_text=None,
                latency_seconds=0.5,
            )

        return WorkflowResult(
            run_id=run_id or "GENERATED-RUN",
            ticket_id=ticket.ticket_id,
            outcome=WorkflowOutcome.ESCALATE,
            reason="Human review required.",
            answer_text=None,
            latency_seconds=0.5,
        )


def valid_payload():
    return {
        "ticket_id": "API-001",
        "channel": "email",
        "subject": "Password reset",
        "body": "How do I reset my password?",
    }


def test_health_does_not_initialize_application() -> None:
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return FakeApplication()

    provider = ApplicationProvider(
        factory=factory
    )
    client = TestClient(
        create_api(provider=provider)
    )

    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload == {
        "status": "alive",
        "service": API_TITLE,
        "version": API_VERSION,
        "application_initialized": False,
        "initialization_error_type": None,
    }

    assert factory_calls == []


def test_ready_initializes_application() -> None:
    application = FakeApplication()
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return application

    provider = ApplicationProvider(
        factory=factory
    )
    client = TestClient(
        create_api(provider=provider)
    )

    response = client.get("/ready")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ready"
    assert payload["service"] == API_TITLE
    assert payload[
        "documentation_passage_count"
    ] == 12
    assert payload["index"]["passage_count"] == 12

    assert factory_calls == [True]
    assert provider.initialized is True


def test_application_is_initialized_only_once() -> None:
    application = FakeApplication()
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return application

    provider = ApplicationProvider(
        factory=factory
    )
    client = TestClient(
        create_api(provider=provider)
    )

    first_response = client.get("/ready")
    second_response = client.get("/ready")
    health_response = client.get("/health")

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    assert factory_calls == [True]
    assert health_response.json()[
        "application_initialized"
    ] is True


def test_ready_returns_503_when_initialization_fails() -> None:
    def factory():
        raise RuntimeError(
            "native model initialization failed"
        )

    provider = ApplicationProvider(
        factory=factory
    )
    client = TestClient(
        create_api(provider=provider)
    )

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "status": "not_ready",
        "error_type": "RuntimeError",
    }

    health_response = client.get("/health")

    assert health_response.json()[
        "initialization_error_type"
    ] == "RuntimeError"


def test_process_ticket_returns_auto_response() -> None:
    application = FakeApplication()

    client = TestClient(
        create_api(application=application)
    )

    response = client.post(
        "/tickets/process",
        json=valid_payload(),
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["ticket_id"] == "API-001"
    assert payload["outcome"] == "auto_respond"
    assert payload["may_send_response"] is True
    assert payload["answer_text"] is not None
    assert payload["cited_passage_ids"] == [
        "DOC001#p001"
    ]

    assert len(application.process_calls) == 1


def test_run_id_header_is_forwarded() -> None:
    application = FakeApplication()
    client = TestClient(
        create_api(application=application)
    )

    response = client.post(
        "/tickets/process",
        json=valid_payload(),
        headers={
            "X-Run-ID": "API-TEST-RUN",
        },
    )

    assert response.status_code == 200
    assert response.json()["run_id"] == (
        "API-TEST-RUN"
    )

    assert application.process_calls[0][
        "run_id"
    ] == "API-TEST-RUN"


@pytest.mark.parametrize(
    "outcome",
    [
        WorkflowOutcome.ESCALATE,
        WorkflowOutcome.BLOCK,
    ],
)
def test_non_auto_outcomes_have_no_sendable_answer(
    outcome,
) -> None:
    application = FakeApplication(
        outcome=outcome
    )
    client = TestClient(
        create_api(application=application)
    )

    response = client.post(
        "/tickets/process",
        json=valid_payload(),
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["outcome"] == outcome.value
    assert payload["may_send_response"] is False
    assert payload["answer_text"] is None


def test_invalid_ticket_returns_422() -> None:
    application = FakeApplication()
    client = TestClient(
        create_api(application=application)
    )

    response = client.post(
        "/tickets/process",
        json={
            "ticket_id": "",
            "channel": "email",
            "body": "Missing a valid ticket ID.",
        },
    )

    assert response.status_code == 422

    detail = response.json()["detail"]

    assert detail["error"] == (
        "Ticket validation failed."
    )
    assert "validation_error" in detail


def test_non_object_request_body_returns_422() -> None:
    client = TestClient(
        create_api(
            application=FakeApplication()
        )
    )

    response = client.post(
        "/tickets/process",
        json=[
            valid_payload(),
        ],
    )

    assert response.status_code == 422


def test_application_build_failure_returns_503() -> None:
    def factory():
        raise ApplicationBuildError(
            "secret internal build details"
        )

    client = TestClient(
        create_api(
            provider=ApplicationProvider(
                factory=factory
            )
        )
    )

    response = client.post(
        "/tickets/process",
        json=valid_payload(),
    )

    assert response.status_code == 503

    detail = response.json()["detail"]

    assert detail["error"] == (
        "Application dependencies are not ready."
    )
    assert detail["error_type"] == (
        "ApplicationBuildError"
    )

    # Internal exception text is not exposed.
    assert "secret internal" not in str(detail)


def test_unexpected_processing_failure_returns_500() -> None:
    application = FakeApplication(
        processing_error=RuntimeError(
            "private infrastructure details"
        )
    )

    client = TestClient(
        create_api(application=application)
    )

    response = client.post(
        "/tickets/process",
        json=valid_payload(),
    )

    assert response.status_code == 500

    detail = response.json()["detail"]

    assert detail == {
        "error": "Ticket processing failed.",
        "error_type": "RuntimeError",
    }

    assert "private infrastructure" not in str(
        detail
    )


def test_provider_caches_application() -> None:
    application = FakeApplication()
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return application

    provider = ApplicationProvider(
        factory=factory
    )

    first = provider.get()
    second = provider.get()

    assert first is application
    assert second is application
    assert factory_calls == [True]


def test_provider_records_initialization_error() -> None:
    error = RuntimeError("initialization failed")

    provider = ApplicationProvider(
        factory=lambda: (_ for _ in ()).throw(
            error
        )
    )

    with pytest.raises(RuntimeError):
        provider.get()

    assert (
        provider.initialization_error
        is error
    )
    assert provider.initialized is False


def test_create_api_rejects_two_injection_methods() -> None:
    with pytest.raises(
        ValueError,
        match="either provider or application",
    ):
        create_api(
            provider=ApplicationProvider(),
            application=FakeApplication(),
        )


def test_openapi_document_is_available() -> None:
    client = TestClient(
        create_api(
            application=FakeApplication()
        )
    )

    response = client.get("/openapi.json")

    assert response.status_code == 200

    document = response.json()

    assert document["info"]["title"] == API_TITLE
    assert document["info"]["version"] == (
        API_VERSION
    )
    assert "/health" in document["paths"]
    assert "/ready" in document["paths"]
    assert (
        "/tickets/process"
        in document["paths"]
    )

def test_metrics_endpoint_returns_prometheus_text() -> None:
    metrics = ApplicationMetrics()

    client = TestClient(
        create_api(
            application=FakeApplication(),
            metrics=metrics,
        )
    )

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers[
        "content-type"
    ]

    assert (
        "# HELP customer_success_tickets_total"
        in response.text
    )
    assert (
        "customer_success_application_ready 0.0"
        in response.text
    )


def test_health_request_is_recorded() -> None:
    metrics = ApplicationMetrics()

    client = TestClient(
        create_api(
            application=FakeApplication(),
            metrics=metrics,
        )
    )

    health_response = client.get("/health")

    assert health_response.status_code == 200

    rendered = metrics.render().decode("utf-8")

    request_line = next(
        line
        for line in rendered.splitlines()
        if line.startswith(
            "customer_success_http_requests_total"
        )
        and 'method="GET"' in line
        and 'endpoint="/health"' in line
        and 'status_code="200"' in line
    )

    assert request_line.endswith(" 1.0")


def test_ready_endpoint_sets_readiness_metric() -> None:
    metrics = ApplicationMetrics()

    client = TestClient(
        create_api(
            application=FakeApplication(),
            metrics=metrics,
        )
    )

    response = client.get("/ready")

    assert response.status_code == 200

    rendered = metrics.render().decode("utf-8")

    assert (
        "customer_success_application_ready 1.0"
        in rendered
    )


def test_failed_readiness_records_failure_metric() -> None:
    metrics = ApplicationMetrics()

    def factory():
        raise RuntimeError(
            "model initialization failed"
        )

    client = TestClient(
        create_api(
            provider=ApplicationProvider(
                factory=factory
            ),
            metrics=metrics,
        )
    )

    response = client.get("/ready")

    assert response.status_code == 503

    rendered = metrics.render().decode("utf-8")

    assert (
        "customer_success_application_ready 0.0"
        in rendered
    )
    assert (
        "customer_success_stage_failures_total"
        in rendered
    )
    assert 'stage="api"' in rendered
    assert 'error_type="RuntimeError"' in rendered
    assert (
        "model initialization failed"
        not in rendered
    )


def test_successful_ticket_records_workflow_metrics() -> None:
    metrics = ApplicationMetrics()

    client = TestClient(
        create_api(
            application=FakeApplication(),
            metrics=metrics,
        )
    )

    response = client.post(
        "/tickets/process",
        json=valid_payload(),
    )

    assert response.status_code == 200

    rendered = metrics.render().decode("utf-8")

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

    request_line = next(
        line
        for line in rendered.splitlines()
        if line.startswith(
            "customer_success_http_requests_total"
        )
        and 'method="POST"' in line
        and 'endpoint="/tickets/process"' in line
        and 'status_code="200"' in line
    )

    assert request_line.endswith(" 1.0")


def test_escalation_records_escalate_outcome() -> None:
    metrics = ApplicationMetrics()

    client = TestClient(
        create_api(
            application=FakeApplication(
                outcome=(
                    WorkflowOutcome.ESCALATE
                )
            ),
            metrics=metrics,
        )
    )

    response = client.post(
        "/tickets/process",
        json=valid_payload(),
    )

    assert response.status_code == 200

    rendered = metrics.render().decode("utf-8")

    assert 'outcome="escalate"' in rendered
    assert 'channel="email"' in rendered


def test_block_records_block_outcome() -> None:
    metrics = ApplicationMetrics()

    client = TestClient(
        create_api(
            application=FakeApplication(
                outcome=WorkflowOutcome.BLOCK
            ),
            metrics=metrics,
        )
    )

    response = client.post(
        "/tickets/process",
        json=valid_payload(),
    )

    assert response.status_code == 200

    rendered = metrics.render().decode("utf-8")

    assert 'outcome="block"' in rendered
    assert 'channel="email"' in rendered


def test_invalid_ticket_records_ingestion_failure() -> None:
    metrics = ApplicationMetrics()

    client = TestClient(
        create_api(
            application=FakeApplication(),
            metrics=metrics,
        )
    )

    response = client.post(
        "/tickets/process",
        json={
            "ticket_id": "",
            "channel": "email",
            "body": "Invalid ticket ID.",
        },
    )

    assert response.status_code == 422

    rendered = metrics.render().decode("utf-8")

    failure_line = next(
        line
        for line in rendered.splitlines()
        if line.startswith(
            "customer_success_stage_failures_total"
        )
        and 'stage="ingestion"' in line
        and (
            'error_type="TicketIngestionError"'
            in line
        )
    )

    assert failure_line.endswith(" 1.0")

    request_line = next(
        line
        for line in rendered.splitlines()
        if line.startswith(
            "customer_success_http_requests_total"
        )
        and 'endpoint="/tickets/process"' in line
        and 'status_code="422"' in line
    )

    assert request_line.endswith(" 1.0")


def test_unknown_url_uses_bounded_metric_label() -> None:
    metrics = ApplicationMetrics()

    client = TestClient(
        create_api(
            application=FakeApplication(),
            metrics=metrics,
        )
    )

    response = client.get(
        "/customer-controlled/not-found"
    )

    assert response.status_code == 404

    rendered = metrics.render().decode("utf-8")

    assert 'endpoint="unmatched"' in rendered
    assert (
        "/customer-controlled/not-found"
        not in rendered
    )


def test_metrics_endpoint_is_hidden_from_openapi() -> None:
    client = TestClient(
        create_api(
            application=FakeApplication()
        )
    )

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/metrics" not in response.json()[
        "paths"
    ]