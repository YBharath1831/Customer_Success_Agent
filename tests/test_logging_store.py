"""Tests for persistent decision logging."""

import sqlite3
from datetime import datetime

import pytest

from src.logging_store import (
    DecisionRecord,
    DecisionStore,
    DecisionStoreError,
    hash_input,
    utc_now,
)


@pytest.fixture
def store(tmp_path) -> DecisionStore:
    """Create an isolated SQLite decision store."""

    database_path = tmp_path / "storage" / "decisions.db"
    decision_store = DecisionStore(database_path)
    decision_store.initialize()

    return decision_store


def start_test_ticket(
    store: DecisionStore,
    *,
    run_id: str = "RUN-001",
    ticket_id: str = "TICKET-001",
    channel: str = "email",
) -> None:
    """Register a ticket for decision tests."""

    store.start_ticket(
        run_id=run_id,
        ticket_id=ticket_id,
        channel=channel,
    )


def make_decision(
    *,
    run_id: str = "RUN-001",
    ticket_id: str = "TICKET-001",
    stage: str = "classification",
) -> DecisionRecord:
    """Create a representative decision record."""

    return DecisionRecord(
        run_id=run_id,
        ticket_id=ticket_id,
        stage=stage,
        action_taken="continue",
        reason="The classification completed successfully.",
        input_summary="Customer cannot sign in.",
        input_hash=hash_input(
            "Customer cannot sign in."
        ),
        model_name="local-classifier",
        model_version="1.0",
        prediction={
            "intent": "authentication_failure",
            "urgency": "high",
        },
        confidence=0.91,
        alternatives=[
            {
                "value": "account_access",
                "confidence": 0.06,
            }
        ],
        sources_used=[],
        threshold_applied=0.80,
        guardrail_results={},
        prompt_version=None,
        requirement_ids=["A3"],
        latency_seconds=0.05,
    )


def test_initialize_creates_database_and_tables(
    tmp_path,
) -> None:
    database_path = (
        tmp_path
        / "nested"
        / "storage"
        / "decisions.db"
    )

    store = DecisionStore(database_path)
    store.initialize()

    assert database_path.exists()

    with sqlite3.connect(database_path) as connection:
        table_rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()

    table_names = {
        row[0]
        for row in table_rows
    }

    assert "schema_metadata" in table_names
    assert "ticket_runs" in table_names
    assert "decision_events" in table_names


def test_initialize_is_idempotent(
    store: DecisionStore,
) -> None:
    store.initialize()
    store.initialize()

    start_test_ticket(store)

    ticket = store.fetch_ticket_run(
        run_id="RUN-001",
        ticket_id="TICKET-001",
    )

    assert ticket is not None


def test_start_ticket_creates_processing_record(
    store: DecisionStore,
) -> None:
    start_test_ticket(store)

    ticket = store.fetch_ticket_run(
        run_id="RUN-001",
        ticket_id="TICKET-001",
    )

    assert ticket is not None
    assert ticket["run_id"] == "RUN-001"
    assert ticket["ticket_id"] == "TICKET-001"
    assert ticket["channel"] == "email"
    assert ticket["status"] == "processing"
    assert ticket["terminal_outcome"] is None
    assert ticket["completed_at"] is None


def test_duplicate_ticket_in_same_run_is_rejected(
    store: DecisionStore,
) -> None:
    start_test_ticket(store)

    with pytest.raises(
        DecisionStoreError,
        match="already exists",
    ):
        start_test_ticket(store)


def test_same_ticket_can_exist_in_different_runs(
    store: DecisionStore,
) -> None:
    start_test_ticket(
        store,
        run_id="RUN-001",
        ticket_id="TICKET-001",
    )

    start_test_ticket(
        store,
        run_id="RUN-002",
        ticket_id="TICKET-001",
    )

    first = store.fetch_ticket_run(
        run_id="RUN-001",
        ticket_id="TICKET-001",
    )

    second = store.fetch_ticket_run(
        run_id="RUN-002",
        ticket_id="TICKET-001",
    )

    assert first is not None
    assert second is not None


def test_decision_is_persisted_and_deserialized(
    store: DecisionStore,
) -> None:
    start_test_ticket(store)

    decision = make_decision()
    store.log_decision(decision)

    records = store.fetch_decisions(
        run_id="RUN-001",
        ticket_id="TICKET-001",
    )

    assert len(records) == 1

    saved = records[0]

    assert saved["decision_id"] == (
        decision.decision_id
    )
    assert saved["stage"] == "classification"
    assert saved["confidence"] == 0.91

    assert saved["prediction"] == {
        "intent": "authentication_failure",
        "urgency": "high",
    }

    assert saved["alternatives"] == [
        {
            "confidence": 0.06,
            "value": "account_access",
        }
    ]

    assert saved["requirement_ids"] == ["A3"]


def test_multiple_stage_decisions_are_preserved(
    store: DecisionStore,
) -> None:
    start_test_ticket(store)

    store.log_decision(
        make_decision(stage="classification")
    )

    store.log_decision(
        DecisionRecord(
            run_id="RUN-001",
            ticket_id="TICKET-001",
            stage="retrieval",
            action_taken="continue",
            reason="Relevant passages were found.",
            sources_used=[
                {
                    "passage_id": "DOC-AUTH-001#p001",
                    "doc_id": "DOC-AUTH-001",
                    "score": 0.87,
                }
            ],
            threshold_applied=0.40,
            requirement_ids=["A4"],
        )
    )

    records = store.fetch_decisions(
        run_id="RUN-001",
        ticket_id="TICKET-001",
    )

    assert len(records) == 2

    stages = {
        record["stage"]
        for record in records
    }

    assert stages == {
        "classification",
        "retrieval",
    }


def test_decision_requires_existing_ticket_run(
    store: DecisionStore,
) -> None:
    decision = make_decision()

    with pytest.raises(
        DecisionStoreError,
        match="start_ticket",
    ):
        store.log_decision(decision)


def test_ticket_can_be_completed(
    store: DecisionStore,
) -> None:
    start_test_ticket(store)
    store.log_decision(make_decision())

    store.complete_ticket(
        run_id="RUN-001",
        ticket_id="TICKET-001",
        terminal_outcome="escalate",
        latency_seconds=0.25,
    )

    ticket = store.fetch_ticket_run(
        run_id="RUN-001",
        ticket_id="TICKET-001",
    )

    assert ticket is not None
    assert ticket["status"] == "completed"
    assert ticket["terminal_outcome"] == "escalate"
    assert ticket["latency_seconds"] == 0.25
    assert ticket["completed_at"] is not None


@pytest.mark.parametrize(
    "outcome",
    [
        "auto_respond",
        "escalate",
        "block",
    ],
)
def test_all_terminal_outcomes_are_supported(
    store: DecisionStore,
    outcome: str,
) -> None:
    ticket_id = f"TICKET-{outcome}"

    start_test_ticket(
        store,
        ticket_id=ticket_id,
    )

    store.log_decision(
        make_decision(
            ticket_id=ticket_id,
        )
    )

    store.complete_ticket(
        run_id="RUN-001",
        ticket_id=ticket_id,
        terminal_outcome=outcome,
        latency_seconds=0.10,
    )

    ticket = store.fetch_ticket_run(
        run_id="RUN-001",
        ticket_id=ticket_id,
    )

    assert ticket is not None
    assert ticket["terminal_outcome"] == outcome


def test_invalid_terminal_outcome_is_rejected(
    store: DecisionStore,
) -> None:
    start_test_ticket(store)

    with pytest.raises(
        ValueError,
        match="terminal_outcome",
    ):
        store.complete_ticket(
            run_id="RUN-001",
            ticket_id="TICKET-001",
            terminal_outcome="discard",
            latency_seconds=0.10,
        )


def test_negative_terminal_latency_is_rejected(
    store: DecisionStore,
) -> None:
    start_test_ticket(store)

    with pytest.raises(
        ValueError,
        match="latency_seconds",
    ):
        store.complete_ticket(
            run_id="RUN-001",
            ticket_id="TICKET-001",
            terminal_outcome="escalate",
            latency_seconds=-1,
        )


def test_ticket_cannot_be_completed_twice(
    store: DecisionStore,
) -> None:
    start_test_ticket(store)

    store.complete_ticket(
        run_id="RUN-001",
        ticket_id="TICKET-001",
        terminal_outcome="escalate",
        latency_seconds=0.10,
    )

    with pytest.raises(
        DecisionStoreError,
        match="already be completed",
    ):
        store.complete_ticket(
            run_id="RUN-001",
            ticket_id="TICKET-001",
            terminal_outcome="escalate",
            latency_seconds=0.20,
        )


def test_reconciliation_succeeds_for_complete_run(
    store: DecisionStore,
) -> None:
    outcomes = [
        ("TICKET-001", "auto_respond"),
        ("TICKET-002", "escalate"),
        ("TICKET-003", "block"),
    ]

    for ticket_id, outcome in outcomes:
        start_test_ticket(
            store,
            ticket_id=ticket_id,
        )

        store.log_decision(
            make_decision(
                ticket_id=ticket_id,
            )
        )

        store.complete_ticket(
            run_id="RUN-001",
            ticket_id=ticket_id,
            terminal_outcome=outcome,
            latency_seconds=0.10,
        )

    summary = store.reconciliation_summary(
        run_id="RUN-001"
    )

    assert summary["tickets_started"] == 3
    assert summary["tickets_completed"] == 3
    assert summary["tickets_incomplete"] == 0
    assert summary["auto_responded"] == 1
    assert summary["escalated"] == 1
    assert summary["blocked"] == 1
    assert summary["terminal_outcomes"] == 3
    assert summary["tickets_with_decisions"] == 3
    assert summary["tickets_without_decisions"] == 0
    assert summary["reconciled"] is True


def test_reconciliation_detects_incomplete_ticket(
    store: DecisionStore,
) -> None:
    start_test_ticket(store)
    store.log_decision(make_decision())

    summary = store.reconciliation_summary(
        run_id="RUN-001"
    )

    assert summary["tickets_started"] == 1
    assert summary["tickets_completed"] == 0
    assert summary["tickets_incomplete"] == 1
    assert summary["reconciled"] is False


def test_reconciliation_detects_missing_decisions(
    store: DecisionStore,
) -> None:
    start_test_ticket(store)

    store.complete_ticket(
        run_id="RUN-001",
        ticket_id="TICKET-001",
        terminal_outcome="escalate",
        latency_seconds=0.10,
    )

    summary = store.reconciliation_summary(
        run_id="RUN-001"
    )

    assert summary["tickets_completed"] == 1
    assert summary["tickets_without_decisions"] == 1
    assert summary["reconciled"] is False


@pytest.mark.parametrize(
    "invalid_stage",
    [
        "",
        "prediction",
        "unknown",
    ],
)
def test_invalid_decision_stage_is_rejected(
    invalid_stage: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="decision stage",
    ):
        make_decision(stage=invalid_stage)


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01],
)
def test_invalid_confidence_is_rejected(
    confidence: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="confidence",
    ):
        DecisionRecord(
            run_id="RUN-001",
            ticket_id="TICKET-001",
            stage="classification",
            action_taken="continue",
            reason="Testing invalid confidence.",
            confidence=confidence,
        )


def test_non_serializable_decision_is_rejected(
    store: DecisionStore,
) -> None:
    start_test_ticket(store)

    decision = DecisionRecord(
        run_id="RUN-001",
        ticket_id="TICKET-001",
        stage="classification",
        action_taken="continue",
        reason="Testing JSON validation.",
        prediction={
            "invalid": object(),
        },
    )

    with pytest.raises(
        DecisionStoreError,
        match="not JSON serializable",
    ):
        store.log_decision(decision)


def test_long_text_fields_are_truncated(
    store: DecisionStore,
) -> None:
    start_test_ticket(store)

    decision = DecisionRecord(
        run_id="RUN-001",
        ticket_id="TICKET-001",
        stage="ingestion",
        action_taken="continue",
        reason="Ticket was ingested.",
        input_summary="x" * 1500,
        error="y" * 2500,
    )

    store.log_decision(decision)

    saved = store.fetch_decisions(
        run_id="RUN-001",
        ticket_id="TICKET-001",
    )[0]

    assert len(saved["input_summary"]) == 1000
    assert len(saved["error"]) == 2000


def test_records_persist_across_store_instances(
    tmp_path,
) -> None:
    database_path = tmp_path / "decisions.db"

    first_store = DecisionStore(database_path)
    first_store.initialize()
    start_test_ticket(first_store)
    first_store.log_decision(make_decision())

    second_store = DecisionStore(database_path)

    ticket = second_store.fetch_ticket_run(
        run_id="RUN-001",
        ticket_id="TICKET-001",
    )

    decisions = second_store.fetch_decisions(
        run_id="RUN-001",
        ticket_id="TICKET-001",
    )

    assert ticket is not None
    assert len(decisions) == 1


def test_hash_input_is_deterministic() -> None:
    first = hash_input("same input")
    second = hash_input("same input")
    different = hash_input("different input")

    assert first == second
    assert first != different
    assert len(first) == 64


def test_utc_now_returns_timezone_aware_timestamp() -> None:
    timestamp = utc_now()
    parsed = datetime.fromisoformat(timestamp)

    assert parsed.tzinfo is not None