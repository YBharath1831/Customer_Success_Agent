"""Automatic evaluation metrics for completed workflow runs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import get_settings
from .ingest import load_and_normalize_tickets
from .run import (
    resolve_project_path,
    write_json_report,
)


EVALUATION_SCHEMA_VERSION = 1


class EvaluationError(RuntimeError):
    """Raised when a run report cannot be evaluated safely."""


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def percentile(
    values: Sequence[float],
    probability: float,
) -> float | None:
    """Calculate a percentile using linear interpolation."""

    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * probability
    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )
    fraction = position - lower_index

    return (
        ordered[lower_index]
        + (
            ordered[upper_index]
            - ordered[lower_index]
        )
        * fraction
    )


def load_json_object(
    path: str | Path,
    *,
    description: str,
) -> dict[str, Any]:
    resolved_path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not resolved_path.exists():
        raise EvaluationError(
            f"{description} does not exist: {resolved_path}"
        )

    try:
        payload = json.loads(
            resolved_path.read_text(
                encoding="utf-8-sig"
            )
        )
    except json.JSONDecodeError as exc:
        raise EvaluationError(
            (
                f"{description} contains invalid JSON at "
                f"line {exc.lineno}, column {exc.colno}."
            )
        ) from exc

    if not isinstance(payload, dict):
        raise EvaluationError(
            f"{description} must contain a JSON object."
        )

    return payload


def normalize_actual_route(
    outcome: str | None,
) -> str | None:
    """
    Convert workflow outcomes to the labelled routing space.

    A blocked response is a safe non-automatic outcome, so it is evaluated
    as escalation for route-decision accuracy.
    """

    if outcome == "auto_respond":
        return "auto_respond"

    if outcome in {
        "escalate",
        "block",
    }:
        return "escalate"

    return None


def passage_id_to_doc_id(
    passage_id: str,
) -> str:
    """Convert DOC-ID#p001 into DOC-ID."""

    return passage_id.split(
        "#",
        maxsplit=1,
    )[0]


def index_run_results(
    run_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_results = run_report.get("results")

    if not isinstance(raw_results, list):
        raise EvaluationError(
            "Run report is missing a results array."
        )

    indexed: dict[str, dict[str, Any]] = {}

    for index, result in enumerate(
        raw_results,
        start=1,
    ):
        if not isinstance(result, dict):
            raise EvaluationError(
                (
                    "Run report result "
                    f"{index} must be an object."
                )
            )

        ticket_id = result.get("ticket_id")

        if not isinstance(ticket_id, str) or not ticket_id:
            raise EvaluationError(
                (
                    "Run report result "
                    f"{index} has no ticket_id."
                )
            )

        if ticket_id in indexed:
            raise EvaluationError(
                (
                    "Run report contains duplicate ticket ID: "
                    f"{ticket_id}"
                )
            )

        indexed[ticket_id] = result

    return indexed


def evaluate_run(
    *,
    tickets_path: str | Path,
    run_report_path: str | Path,
) -> dict[str, Any]:
    """Compare one completed run against labelled tickets."""

    ingestion = load_and_normalize_tickets(
        tickets_path
    )

    if ingestion.failures:
        raise EvaluationError(
            (
                "Evaluation ticket file contains "
                f"{len(ingestion.failures)} malformed record(s)."
            )
        )

    run_report = load_json_object(
        run_report_path,
        description="Run report",
    )
    indexed_results = index_run_results(
        run_report
    )

    rows: list[dict[str, Any]] = []
    unlabeled_ticket_ids: list[str] = []

    for ticket in ingestion.tickets:
        labels = ticket.labels

        if labels is None:
            unlabeled_ticket_ids.append(
                ticket.ticket_id
            )
            continue

        result = indexed_results.get(
            ticket.ticket_id
        )

        rows.append(
            build_evaluation_row(
                ticket=ticket,
                result=result,
            )
        )

    metrics = calculate_metrics(rows)

    expected_ticket_ids = {
        ticket.ticket_id
        for ticket in ingestion.tickets
        if ticket.labels is not None
    }

    extra_result_ids = sorted(
        set(indexed_results) - expected_ticket_ids
    )

    missing_result_ids = sorted(
        row["ticket_id"]
        for row in rows
        if not row["result_present"]
    )

    reconciliation = run_report.get(
        "reconciliation",
        {}
    )

    checks = build_evaluation_checks(
        metrics=metrics,
        labelled_ticket_count=len(rows),
        missing_result_ids=missing_result_ids,
        run_reconciliation=reconciliation,
    )

    return {
        "schema_version": (
            EVALUATION_SCHEMA_VERSION
        ),
        "generated_at": utc_now(),
        "run_id": run_report.get("run_id"),
        "tickets_path": str(
            Path(tickets_path).expanduser().resolve()
        ),
        "run_report_path": str(
            Path(run_report_path)
            .expanduser()
            .resolve()
        ),
        "dataset": {
            "total_records": (
                ingestion.total_records
            ),
            "labelled_records": len(rows),
            "unlabelled_records": len(
                unlabeled_ticket_ids
            ),
            "unlabelled_ticket_ids": (
                unlabeled_ticket_ids
            ),
            "matched_results": (
                len(rows) - len(missing_result_ids)
            ),
            "missing_result_ids": (
                missing_result_ids
            ),
            "extra_result_ids": extra_result_ids,
        },
        "metrics": metrics,
        "checks": checks,
        "details": rows,
    }


def build_evaluation_row(
    *,
    ticket,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    labels = ticket.labels

    if labels is None:
        raise ValueError(
            "Evaluation row requires ticket labels."
        )

    classification = (
        result.get("classification")
        if result is not None
        else None
    )

    if not isinstance(classification, dict):
        classification = {}

    predicted_intent = classification.get(
        "intent"
    )
    predicted_urgency = classification.get(
        "urgency"
    )

    intent_confidence = classification.get(
        "intent_confidence"
    )
    urgency_confidence = classification.get(
        "urgency_confidence"
    )
    combined_confidence = classification.get(
        "combined_confidence"
    )

    outcome = (
        result.get("outcome")
        if result is not None
        else None
    )

    actual_route = normalize_actual_route(
        outcome
    )

    retrieval_matches = (
        result.get("retrieval_matches", [])
        if result is not None
        else []
    )

    if not isinstance(retrieval_matches, list):
        retrieval_matches = []

    retrieved_passage_ids = [
        match.get("passage_id")
        for match in retrieval_matches
        if isinstance(match, dict)
        and isinstance(
            match.get("passage_id"),
            str,
        )
    ]

    retrieved_doc_ids = {
        passage_id_to_doc_id(passage_id)
        for passage_id in retrieved_passage_ids
    }

    expected_doc_ids = set(
        labels.expected_doc_ids
    )

    retrieved_expected_docs = (
        expected_doc_ids.intersection(
            retrieved_doc_ids
        )
    )

    cited_passage_ids = (
        result.get("cited_passage_ids", [])
        if result is not None
        else []
    )

    if not isinstance(cited_passage_ids, list):
        cited_passage_ids = []

    cited_doc_ids = {
        passage_id_to_doc_id(passage_id)
        for passage_id in cited_passage_ids
        if isinstance(passage_id, str)
    }

    cited_expected_docs = (
        expected_doc_ids.intersection(
            cited_doc_ids
        )
    )

    latency = (
        result.get("latency_seconds")
        if result is not None
        else None
    )

    return {
        "ticket_id": ticket.ticket_id,
        "channel": ticket.channel.value,
        "result_present": result is not None,
        "expected": {
            "intent": labels.intent,
            "urgency": labels.urgency.value,
            "route": (
                labels.expected_route.value
            ),
            "answerable_from_docs": (
                labels.answerable_from_docs
            ),
            "expected_doc_ids": sorted(
                expected_doc_ids
            ),
            "must_not_auto_respond": (
                labels.must_not_auto_respond
            ),
        },
        "actual": {
            "intent": predicted_intent,
            "urgency": predicted_urgency,
            "outcome": outcome,
            "route": actual_route,
            "intent_confidence": (
                intent_confidence
            ),
            "urgency_confidence": (
                urgency_confidence
            ),
            "combined_confidence": (
                combined_confidence
            ),
            "retrieved_doc_ids": sorted(
                retrieved_doc_ids
            ),
            "cited_doc_ids": sorted(
                cited_doc_ids
            ),
            "latency_seconds": latency,
        },
        "correct": {
            "intent": (
                predicted_intent
                == labels.intent
            ),
            "urgency": (
                predicted_urgency
                == labels.urgency.value
            ),
            "route": (
                actual_route
                == labels.expected_route.value
            ),
        },
        "safety": {
            "unsafe_auto_response": (
                labels.must_not_auto_respond
                and outcome == "auto_respond"
            ),
            "unexpected_auto_response": (
                labels.expected_route.value
                == "escalate"
                and outcome == "auto_respond"
            ),
        },
        "retrieval": {
            "retrieved_expected_doc_count": len(
                retrieved_expected_docs
            ),
            "expected_doc_count": len(
                expected_doc_ids
            ),
            "expected_doc_recall": (
                safe_ratio(
                    len(retrieved_expected_docs),
                    len(expected_doc_ids),
                )
            ),
            "any_expected_doc_retrieved": bool(
                retrieved_expected_docs
            ),
        },
        "citations": {
            "citation_count": len(
                cited_passage_ids
            ),
            "cited_expected_doc_count": len(
                cited_expected_docs
            ),
            "any_expected_doc_cited": bool(
                cited_expected_docs
            ),
        },
    }


def calculate_metrics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate detailed evaluation rows."""

    matched_rows = [
        row
        for row in rows
        if row["result_present"]
    ]

    intent_rows = [
        row
        for row in matched_rows
        if row["actual"]["intent"] is not None
    ]

    urgency_rows = [
        row
        for row in matched_rows
        if row["actual"]["urgency"] is not None
    ]

    route_rows = [
        row
        for row in matched_rows
        if row["actual"]["route"] is not None
    ]

    intent_correct = sum(
        row["correct"]["intent"]
        for row in intent_rows
    )

    urgency_correct = sum(
        row["correct"]["urgency"]
        for row in urgency_rows
    )

    route_correct = sum(
        row["correct"]["route"]
        for row in route_rows
    )

    unsafe_auto_responses = sum(
        row["safety"]["unsafe_auto_response"]
        for row in matched_rows
    )

    unexpected_auto_responses = sum(
        row["safety"][
            "unexpected_auto_response"
        ]
        for row in matched_rows
    )

    must_not_auto_rows = [
        row
        for row in matched_rows
        if row["expected"][
            "must_not_auto_respond"
        ]
    ]

    expected_auto_rows = [
        row
        for row in matched_rows
        if row["expected"]["route"]
        == "auto_respond"
    ]

    missed_auto_responses = sum(
        row["actual"]["route"] != "auto_respond"
        for row in expected_auto_rows
    )

    expected_doc_rows = [
        row
        for row in matched_rows
        if row["retrieval"][
            "expected_doc_count"
        ] > 0
    ]

    document_recalls = [
        row["retrieval"][
            "expected_doc_recall"
        ]
        for row in expected_doc_rows
        if row["retrieval"][
            "expected_doc_recall"
        ] is not None
    ]

    expected_doc_hits = sum(
        row["retrieval"][
            "any_expected_doc_retrieved"
        ]
        for row in expected_doc_rows
    )

    auto_response_rows = [
        row
        for row in matched_rows
        if row["actual"]["outcome"]
        == "auto_respond"
    ]

    cited_auto_responses = sum(
        row["citations"]["citation_count"] > 0
        for row in auto_response_rows
    )

    latencies = [
        float(row["actual"]["latency_seconds"])
        for row in matched_rows
        if isinstance(
            row["actual"]["latency_seconds"],
            (int, float),
        )
        and row["actual"]["latency_seconds"] >= 0
    ]

    outcomes = Counter(
        row["actual"]["outcome"]
        for row in matched_rows
        if row["actual"]["outcome"] is not None
    )

    intent_confusion = confusion_matrix(
        rows=intent_rows,
        expected_field="intent",
        actual_field="intent",
    )

    urgency_confusion = confusion_matrix(
        rows=urgency_rows,
        expected_field="urgency",
        actual_field="urgency",
    )

    route_confusion = confusion_matrix(
        rows=route_rows,
        expected_field="route",
        actual_field="route",
    )

    return {
        "coverage": {
            "labelled_tickets": len(rows),
            "matched_results": len(
                matched_rows
            ),
            "result_coverage": safe_ratio(
                len(matched_rows),
                len(rows),
            ),
        },
        "classification": {
            "intent": {
                "evaluated": len(intent_rows),
                "correct": intent_correct,
                "accuracy": safe_ratio(
                    intent_correct,
                    len(intent_rows),
                ),
                "confusion_matrix": (
                    intent_confusion
                ),
            },
            "urgency": {
                "evaluated": len(urgency_rows),
                "correct": urgency_correct,
                "accuracy": safe_ratio(
                    urgency_correct,
                    len(urgency_rows),
                ),
                "confusion_matrix": (
                    urgency_confusion
                ),
            },
        },
        "routing": {
            "evaluated": len(route_rows),
            "correct": route_correct,
            "accuracy": safe_ratio(
                route_correct,
                len(route_rows),
            ),
            "unexpected_auto_responses": (
                unexpected_auto_responses
            ),
            "unsafe_auto_responses": (
                unsafe_auto_responses
            ),
            "must_not_auto_respond_tickets": (
                len(must_not_auto_rows)
            ),
            "unsafe_auto_response_rate": (
                safe_ratio(
                    unsafe_auto_responses,
                    len(must_not_auto_rows),
                )
            ),
            "expected_auto_responses": len(
                expected_auto_rows
            ),
            "missed_auto_responses": (
                missed_auto_responses
            ),
            "confusion_matrix": route_confusion,
        },
        "retrieval": {
            "tickets_with_expected_docs": len(
                expected_doc_rows
            ),
            "tickets_with_expected_doc_hit": (
                expected_doc_hits
            ),
            "expected_doc_hit_rate": safe_ratio(
                expected_doc_hits,
                len(expected_doc_rows),
            ),
            "macro_expected_doc_recall": (
                statistics.fmean(
                    document_recalls
                )
                if document_recalls
                else None
            ),
        },
        "generation": {
            "auto_responses": len(
                auto_response_rows
            ),
            "auto_responses_with_citations": (
                cited_auto_responses
            ),
            "citation_coverage": safe_ratio(
                cited_auto_responses,
                len(auto_response_rows),
            ),
        },
        "outcomes": {
            "auto_respond": outcomes.get(
                "auto_respond",
                0,
            ),
            "escalate": outcomes.get(
                "escalate",
                0,
            ),
            "block": outcomes.get(
                "block",
                0,
            ),
        },
        "latency_seconds": {
            "evaluated": len(latencies),
            "mean": (
                statistics.fmean(latencies)
                if latencies
                else None
            ),
            "p50": percentile(
                latencies,
                0.50,
            ),
            "p95": percentile(
                latencies,
                0.95,
            ),
            "maximum": (
                max(latencies)
                if latencies
                else None
            ),
        },
        "per_channel": calculate_per_channel(
            matched_rows
        ),
    }


def confusion_matrix(
    *,
    rows: Sequence[dict[str, Any]],
    expected_field: str,
    actual_field: str,
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}

    for row in rows:
        expected = row["expected"][
            expected_field
        ]
        actual = row["actual"][
            actual_field
        ]

        expected_row = matrix.setdefault(
            str(expected),
            {},
        )

        actual_key = str(actual)

        expected_row[actual_key] = (
            expected_row.get(actual_key, 0)
            + 1
        )

    return matrix


def calculate_per_channel(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    channels = sorted(
        {
            row["channel"]
            for row in rows
        }
    )

    output: dict[str, Any] = {}

    for channel in channels:
        channel_rows = [
            row
            for row in rows
            if row["channel"] == channel
        ]

        intent_correct = sum(
            row["correct"]["intent"]
            for row in channel_rows
            if row["actual"]["intent"] is not None
        )
        intent_evaluated = sum(
            row["actual"]["intent"] is not None
            for row in channel_rows
        )

        route_correct = sum(
            row["correct"]["route"]
            for row in channel_rows
            if row["actual"]["route"] is not None
        )
        route_evaluated = sum(
            row["actual"]["route"] is not None
            for row in channel_rows
        )

        output[channel] = {
            "tickets": len(channel_rows),
            "intent_accuracy": safe_ratio(
                intent_correct,
                intent_evaluated,
            ),
            "route_accuracy": safe_ratio(
                route_correct,
                route_evaluated,
            ),
            "unsafe_auto_responses": sum(
                row["safety"][
                    "unsafe_auto_response"
                ]
                for row in channel_rows
            ),
        }

    return output


def build_evaluation_checks(
    *,
    metrics: dict[str, Any],
    labelled_ticket_count: int,
    missing_result_ids: Sequence[str],
    run_reconciliation: Any,
) -> dict[str, Any]:
    if not isinstance(
        run_reconciliation,
        dict,
    ):
        run_reconciliation = {}

    tickets_started = run_reconciliation.get(
        "tickets_started"
    )
    tickets_completed = run_reconciliation.get(
        "tickets_completed"
    )
    terminal_outcomes = run_reconciliation.get(
        "terminal_outcomes"
    )

    all_auto_cited = (
        metrics["generation"][
            "citation_coverage"
        ]
        in {
            None,
            1.0,
        }
    )

    return {
        "all_labelled_tickets_have_results": (
            not missing_result_ids
        ),
        "no_unsafe_auto_responses": (
            metrics["routing"][
                "unsafe_auto_responses"
            ]
            == 0
        ),
        "all_auto_responses_have_citations": (
            all_auto_cited
        ),
        "all_started_tickets_completed": (
            tickets_started is not None
            and tickets_started
            == tickets_completed
        ),
        "one_terminal_outcome_per_started_ticket": (
            tickets_started is not None
            and tickets_started
            == terminal_outcomes
        ),
        "labelled_ticket_count": (
            labelled_ticket_count
        ),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a completed customer-success workflow run."
        )
    )

    parser.add_argument(
        "--tickets",
        required=True,
        help="Labelled validation ticket JSON file.",
    )
    parser.add_argument(
        "--run-report",
        required=True,
        help="JSON report produced by src.run.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination metrics JSON file.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the metrics file.",
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        settings = get_settings()

        tickets_path = resolve_project_path(
            arguments.tickets,
            project_root=settings.project_root,
        )
        run_report_path = resolve_project_path(
            arguments.run_report,
            project_root=settings.project_root,
        )
        output_path = resolve_project_path(
            arguments.output,
            project_root=settings.project_root,
        )

        evaluation = evaluate_run(
            tickets_path=tickets_path,
            run_report_path=run_report_path,
        )

        write_json_report(
            output_path=output_path,
            report=evaluation,
            pretty=arguments.pretty,
        )

    except Exception as exc:
        print(
            (
                "Evaluation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            file=sys.stderr,
        )
        return 1

    metrics = evaluation["metrics"]

    print(
        "Intent accuracy:",
        metrics["classification"]["intent"][
            "accuracy"
        ],
    )
    print(
        "Urgency accuracy:",
        metrics["classification"]["urgency"][
            "accuracy"
        ],
    )
    print(
        "Routing accuracy:",
        metrics["routing"]["accuracy"],
    )
    print(
        "Expected-document hit rate:",
        metrics["retrieval"][
            "expected_doc_hit_rate"
        ],
    )
    print(
        "Unsafe auto-responses:",
        metrics["routing"][
            "unsafe_auto_responses"
        ],
    )
    print(f"Metrics written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())