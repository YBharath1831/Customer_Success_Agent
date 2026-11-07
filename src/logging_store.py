"""Persistent SQLite audit logging."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_STAGES = {
    "ingestion",
    "classification",
    "retrieval",
    "routing",
    "generation",
    "validation",
}

TERMINAL_OUTCOMES = {
    "auto_respond",
    "escalate",
    "block",
}


class DecisionStoreError(Exception):
    """Raised when the persistent decision store cannot operate safely."""


@dataclass(frozen=True)
class DecisionRecord:
    """One auditable decision made during ticket processing."""

    run_id: str
    ticket_id: str
    stage: str
    action_taken: str
    reason: str

    input_summary: str | None = None
    input_hash: str | None = None

    model_name: str | None = None
    model_version: str | None = None

    prediction: dict[str, Any] | str | None = None
    confidence: float | None = None
    alternatives: list[dict[str, Any]] = field(
        default_factory=list
    )

    sources_used: list[dict[str, Any]] = field(
        default_factory=list
    )
    threshold_applied: float | None = None
    guardrail_results: dict[str, Any] = field(
        default_factory=dict
    )

    prompt_version: str | None = None
    requirement_ids: list[str] = field(
        default_factory=list
    )

    latency_seconds: float | None = None
    error: str | None = None

    decision_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: str = field(
        default_factory=lambda: utc_now()
    )

    def __post_init__(self) -> None:
        """Validate values before they reach SQLite."""

        if not self.run_id.strip():
            raise ValueError(
                "run_id cannot be empty."
            )

        if not self.ticket_id.strip():
            raise ValueError(
                "ticket_id cannot be empty."
            )

        if self.stage not in ALLOWED_STAGES:
            raise ValueError(
                f"Unsupported decision stage: {self.stage!r}"
            )

        if not self.action_taken.strip():
            raise ValueError(
                "action_taken cannot be empty."
            )

        if not self.reason.strip():
            raise ValueError(
                "reason cannot be empty."
            )

        _validate_optional_probability(
            "confidence",
            self.confidence,
        )
        _validate_optional_probability(
            "threshold_applied",
            self.threshold_applied,
        )

        if (
            self.latency_seconds is not None
            and self.latency_seconds < 0
        ):
            raise ValueError(
                "latency_seconds cannot be negative."
            )


class DecisionStore:
    """SQLite-backed store for ticket runs and decision events."""

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        self.database_path = (
            Path(database_path)
            .expanduser()
            .resolve()
        )

    def initialize(self) -> None:
        """Create the database directory and required tables."""

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        schema = """
        CREATE TABLE IF NOT EXISTS schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        INSERT OR REPLACE INTO schema_metadata (
            key,
            value
        ) VALUES (
            'schema_version',
            '1'
        );

        CREATE TABLE IF NOT EXISTS ticket_runs (
            run_id TEXT NOT NULL,
            ticket_id TEXT NOT NULL,
            channel TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL DEFAULT 'processing',
            terminal_outcome TEXT,
            latency_seconds REAL,
            error_type TEXT,
            error_message TEXT,
            PRIMARY KEY (run_id, ticket_id),
            CHECK (
                status IN ('processing', 'completed')
            ),
            CHECK (
                terminal_outcome IS NULL
                OR terminal_outcome IN (
                    'auto_respond',
                    'escalate',
                    'block'
                )
            ),
            CHECK (
                latency_seconds IS NULL
                OR latency_seconds >= 0
            )
        );

        CREATE TABLE IF NOT EXISTS decision_events (
            decision_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            ticket_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            stage TEXT NOT NULL,
            input_summary TEXT,
            input_hash TEXT,
            model_name TEXT,
            model_version TEXT,
            prediction_json TEXT,
            confidence REAL,
            alternatives_json TEXT NOT NULL,
            sources_used_json TEXT NOT NULL,
            threshold_applied REAL,
            action_taken TEXT NOT NULL,
            reason TEXT NOT NULL,
            guardrail_results_json TEXT NOT NULL,
            prompt_version TEXT,
            requirement_ids_json TEXT NOT NULL,
            latency_seconds REAL,
            error TEXT,
            FOREIGN KEY (
                run_id,
                ticket_id
            ) REFERENCES ticket_runs (
                run_id,
                ticket_id
            ) ON DELETE CASCADE,
            CHECK (
                stage IN (
                    'ingestion',
                    'classification',
                    'retrieval',
                    'routing',
                    'generation',
                    'validation'
                )
            ),
            CHECK (
                confidence IS NULL
                OR (
                    confidence >= 0
                    AND confidence <= 1
                )
            ),
            CHECK (
                threshold_applied IS NULL
                OR (
                    threshold_applied >= 0
                    AND threshold_applied <= 1
                )
            ),
            CHECK (
                latency_seconds IS NULL
                OR latency_seconds >= 0
            )
        );

        CREATE INDEX IF NOT EXISTS
            idx_ticket_runs_run
        ON ticket_runs (
            run_id
        );

        CREATE INDEX IF NOT EXISTS
            idx_decision_events_ticket
        ON decision_events (
            run_id,
            ticket_id
        );

        CREATE INDEX IF NOT EXISTS
            idx_decision_events_stage
        ON decision_events (
            run_id,
            stage
        );
        """

        with self._connection() as connection:
            connection.executescript(schema)

    def start_ticket(
        self,
        *,
        run_id: str,
        ticket_id: str,
        channel: str | None,
        started_at: str | None = None,
    ) -> None:
        """Register a ticket before processing starts."""

        if not run_id.strip():
            raise ValueError(
                "run_id cannot be empty."
            )

        if not ticket_id.strip():
            raise ValueError(
                "ticket_id cannot be empty."
            )

        timestamp = started_at or utc_now()

        query = """
        INSERT INTO ticket_runs (
            run_id,
            ticket_id,
            channel,
            started_at,
            status
        ) VALUES (?, ?, ?, ?, 'processing')
        """

        try:
            with self._connection() as connection:
                connection.execute(
                    query,
                    (
                        run_id.strip(),
                        ticket_id.strip(),
                        channel,
                        timestamp,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DecisionStoreError(
                "A ticket run already exists for "
                f"run_id={run_id!r}, "
                f"ticket_id={ticket_id!r}."
            ) from exc

    def log_decision(
        self,
        record: DecisionRecord,
    ) -> None:
        """Persist one stage-level decision."""

        query = """
        INSERT INTO decision_events (
            decision_id,
            run_id,
            ticket_id,
            timestamp,
            stage,
            input_summary,
            input_hash,
            model_name,
            model_version,
            prediction_json,
            confidence,
            alternatives_json,
            sources_used_json,
            threshold_applied,
            action_taken,
            reason,
            guardrail_results_json,
            prompt_version,
            requirement_ids_json,
            latency_seconds,
            error
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """

        values = (
            record.decision_id,
            record.run_id,
            record.ticket_id,
            record.timestamp,
            record.stage,
            _truncate(record.input_summary, 1000),
            record.input_hash,
            record.model_name,
            record.model_version,
            _json_dumps(record.prediction),
            record.confidence,
            _json_dumps(record.alternatives),
            _json_dumps(record.sources_used),
            record.threshold_applied,
            record.action_taken,
            record.reason,
            _json_dumps(record.guardrail_results),
            record.prompt_version,
            _json_dumps(record.requirement_ids),
            record.latency_seconds,
            _truncate(record.error, 2000),
        )

        try:
            with self._connection() as connection:
                connection.execute(query, values)
        except sqlite3.IntegrityError as exc:
            raise DecisionStoreError(
                "Could not log decision. Confirm that "
                "start_ticket() was called first and that "
                "the decision ID is unique."
            ) from exc

    def complete_ticket(
        self,
        *,
        run_id: str,
        ticket_id: str,
        terminal_outcome: str,
        latency_seconds: float,
        completed_at: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Record exactly one terminal result for a ticket."""

        if terminal_outcome not in TERMINAL_OUTCOMES:
            raise ValueError(
                "terminal_outcome must be one of: "
                + ", ".join(sorted(TERMINAL_OUTCOMES))
            )

        if latency_seconds < 0:
            raise ValueError(
                "latency_seconds cannot be negative."
            )

        query = """
        UPDATE ticket_runs
        SET
            completed_at = ?,
            status = 'completed',
            terminal_outcome = ?,
            latency_seconds = ?,
            error_type = ?,
            error_message = ?
        WHERE
            run_id = ?
            AND ticket_id = ?
            AND status = 'processing'
        """

        with self._connection() as connection:
            cursor = connection.execute(
                query,
                (
                    completed_at or utc_now(),
                    terminal_outcome,
                    latency_seconds,
                    error_type,
                    _truncate(error_message, 2000),
                    run_id,
                    ticket_id,
                ),
            )

            if cursor.rowcount != 1:
                raise DecisionStoreError(
                    "Ticket could not be completed. It may "
                    "not exist or may already be completed."
                )

    def fetch_ticket_run(
        self,
        *,
        run_id: str,
        ticket_id: str,
    ) -> dict[str, Any] | None:
        """Fetch one ticket-run record."""

        query = """
        SELECT *
        FROM ticket_runs
        WHERE run_id = ? AND ticket_id = ?
        """

        with self._connection() as connection:
            row = connection.execute(
                query,
                (run_id, ticket_id),
            ).fetchone()

        return dict(row) if row is not None else None

    def fetch_decisions(
        self,
        *,
        run_id: str,
        ticket_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch all decisions for a ticket in timestamp order."""

        query = """
        SELECT *
        FROM decision_events
        WHERE run_id = ? AND ticket_id = ?
        ORDER BY timestamp, decision_id
        """

        with self._connection() as connection:
            rows = connection.execute(
                query,
                (run_id, ticket_id),
            ).fetchall()

        return [
            _deserialize_decision(dict(row))
            for row in rows
        ]

    def reconciliation_summary(
        self,
        *,
        run_id: str,
    ) -> dict[str, Any]:
        """
        Reconcile ticket inputs, terminal outcomes, and decision coverage.
        """

        ticket_query = """
        SELECT
            COUNT(*) AS tickets_started,
            SUM(
                CASE WHEN status = 'completed'
                THEN 1 ELSE 0 END
            ) AS tickets_completed,
            SUM(
                CASE WHEN terminal_outcome = 'auto_respond'
                THEN 1 ELSE 0 END
            ) AS auto_responded,
            SUM(
                CASE WHEN terminal_outcome = 'escalate'
                THEN 1 ELSE 0 END
            ) AS escalated,
            SUM(
                CASE WHEN terminal_outcome = 'block'
                THEN 1 ELSE 0 END
            ) AS blocked
        FROM ticket_runs
        WHERE run_id = ?
        """

        decisions_query = """
        SELECT
            COUNT(*) AS decision_events,
            COUNT(DISTINCT ticket_id)
                AS tickets_with_decisions
        FROM decision_events
        WHERE run_id = ?
        """

        with self._connection() as connection:
            ticket_row = connection.execute(
                ticket_query,
                (run_id,),
            ).fetchone()

            decision_row = connection.execute(
                decisions_query,
                (run_id,),
            ).fetchone()

        tickets_started = (
            ticket_row["tickets_started"] or 0
        )
        tickets_completed = (
            ticket_row["tickets_completed"] or 0
        )
        auto_responded = (
            ticket_row["auto_responded"] or 0
        )
        escalated = ticket_row["escalated"] or 0
        blocked = ticket_row["blocked"] or 0

        terminal_outcomes = (
            auto_responded + escalated + blocked
        )

        tickets_with_decisions = (
            decision_row["tickets_with_decisions"] or 0
        )

        return {
            "run_id": run_id,
            "tickets_started": tickets_started,
            "tickets_completed": tickets_completed,
            "tickets_incomplete": (
                tickets_started - tickets_completed
            ),
            "auto_responded": auto_responded,
            "escalated": escalated,
            "blocked": blocked,
            "terminal_outcomes": terminal_outcomes,
            "decision_events": (
                decision_row["decision_events"] or 0
            ),
            "tickets_with_decisions": (
                tickets_with_decisions
            ),
            "tickets_without_decisions": max(
                0,
                tickets_started
                - tickets_with_decisions,
            ),
            "reconciled": (
                tickets_started
                == tickets_completed
                == terminal_outcomes
                and tickets_started
                == tickets_with_decisions
            ),
        }

    @contextmanager
    def _connection(
        self,
    ) -> Iterator[sqlite3.Connection]:
        """Open a configured SQLite transaction."""

        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row

        try:
            connection.execute(
                "PRAGMA foreign_keys = ON"
            )
            connection.execute(
                "PRAGMA busy_timeout = 30000"
            )
            connection.execute(
                "PRAGMA journal_mode = WAL"
            )

            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def hash_input(value: str) -> str:
    """Hash input text for audit correlation without duplicating it."""

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def _json_dumps(value: Any) -> str:
    """Serialize structured fields consistently."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DecisionStoreError(
            "Decision data is not JSON serializable."
        ) from exc


def _json_loads(value: str | None) -> Any:
    """Deserialize a structured SQLite value."""

    if value is None:
        return None

    return json.loads(value)


def _deserialize_decision(
    row: dict[str, Any],
) -> dict[str, Any]:
    """Convert JSON columns back into Python structures."""

    json_columns = {
        "prediction_json": "prediction",
        "alternatives_json": "alternatives",
        "sources_used_json": "sources_used",
        "guardrail_results_json": "guardrail_results",
        "requirement_ids_json": "requirement_ids",
    }

    for database_name, output_name in json_columns.items():
        row[output_name] = _json_loads(
            row.pop(database_name)
        )

    return row


def _truncate(
    value: str | None,
    maximum_length: int,
) -> str | None:
    """Bound text stored in audit records."""

    if value is None:
        return None

    if len(value) <= maximum_length:
        return value

    return value[:maximum_length]


def _validate_optional_probability(
    name: str,
    value: float | None,
) -> None:
    """Validate optional values expected between zero and one."""

    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )