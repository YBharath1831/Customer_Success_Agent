"""Unattended command-line runner for customer-success tickets."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .app import (
    ApplicationBuildError,
    ApplicationComponents,
    build_application,
)
from .config import Settings, get_settings
from .ingest import (
    BatchIngestionResult,
    IngestionFailure,
    TicketFileError,
    load_and_normalize_tickets,
)
from .logging_store import DecisionRecord
from .workflow import WorkflowOutcome, WorkflowResult


REPORT_SCHEMA_VERSION = 1


def utc_now() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Process customer-success tickets through "
            "classification, retrieval, routing, generation, "
            "guardrails, and audit logging."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "JSON file containing an array of customer tickets."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Destination JSON file for run results."
        ),
    )

    parser.add_argument(
        "--classifier-artifact",
        default=None,
        help=(
            "Optional classifier artifact path. Relative paths "
            "are resolved from the project root."
        ),
    )

    parser.add_argument(
        "--documentation",
        default=None,
        help=(
            "Optional documentation corpus path. Relative paths "
            "are resolved from the project root."
        ),
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional audit run ID. A UUID is generated when "
            "this is omitted."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Number of passages to retrieve. The default is "
            "RETRIEVAL_TOP_K from .env."
        ),
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON.",
    )

    return parser


def run_ticket_file(
    *,
    input_path: str | Path,
    output_path: str | Path,
    settings: Settings | None = None,
    classifier_artifact_path: str | Path | None = None,
    documentation_path: str | Path | None = None,
    run_id: str | None = None,
    retrieval_top_k: int | None = None,
    build_index: bool = False,
    pretty: bool = False,
) -> dict[str, Any]:
    """
    Process a complete ticket file and write a JSON report.

    Ticket-level errors are represented as outcomes. File-level and
    application-startup errors are raised to the CLI as fatal failures.
    """

    if (
        retrieval_top_k is not None
        and retrieval_top_k <= 0
    ):
        raise ValueError(
            "retrieval_top_k must be greater than zero."
        )

    current_settings = settings or get_settings()
    resolved_run_id = run_id or str(uuid.uuid4())

    resolved_input_path = resolve_project_path(
        input_path,
        project_root=current_settings.project_root,
    )
    resolved_output_path = resolve_project_path(
        output_path,
        project_root=current_settings.project_root,
    )

    ingestion_result = load_and_normalize_tickets(
        resolved_input_path
    )

    application = build_application(
        settings=current_settings,
        classifier_artifact_path=(
            classifier_artifact_path
        ),
        documentation_path=documentation_path,
        build_index=build_index,
        retrieval_top_k=retrieval_top_k,
    )

    workflow_results = (
        application.workflow.process_batch(
            ingestion_result.tickets,
            run_id=resolved_run_id,
        )
    )

    ingestion_failure_results = (
        log_ingestion_failures(
            application=application,
            ingestion_result=ingestion_result,
            run_id=resolved_run_id,
        )
    )

    report = build_run_report(
        application=application,
        ingestion_result=ingestion_result,
        workflow_results=workflow_results,
        ingestion_failure_results=(
            ingestion_failure_results
        ),
        run_id=resolved_run_id,
        input_path=resolved_input_path,
    )

    write_json_report(
        output_path=resolved_output_path,
        report=report,
        pretty=pretty,
    )

    return report


def log_ingestion_failures(
    *,
    application: ApplicationComponents,
    ingestion_result: BatchIngestionResult,
    run_id: str,
) -> list[dict[str, Any]]:
    """
    Persist malformed records as terminal escalations.

    No input record should disappear merely because it could not be
    normalized.
    """

    results: list[dict[str, Any]] = []

    used_ticket_ids = {
        ticket.ticket_id
        for ticket in ingestion_result.tickets
    }

    for failure in ingestion_result.failures:
        audit_ticket_id = unique_failure_ticket_id(
            failure,
            used_ticket_ids=used_ticket_ids,
        )
        used_ticket_ids.add(audit_ticket_id)

        application.decision_store.start_ticket(
            run_id=run_id,
            ticket_id=audit_ticket_id,
            channel=None,
        )

        application.decision_store.log_decision(
            DecisionRecord(
                run_id=run_id,
                ticket_id=audit_ticket_id,
                stage="ingestion",
                action_taken="failed",
                reason=(
                    "Ticket could not be normalized and "
                    "requires human review."
                ),
                prediction={
                    "record_index": (
                        failure.record_index
                    ),
                    "reported_ticket_id": (
                        failure.ticket_id
                    ),
                },
                error=failure.error,
                requirement_ids=["A2", "A11"],
            )
        )

        application.decision_store.complete_ticket(
            run_id=run_id,
            ticket_id=audit_ticket_id,
            terminal_outcome="escalate",
            latency_seconds=0.0,
            error_type="TicketIngestionError",
            error_message=failure.error,
        )

        results.append(
            {
                "run_id": run_id,
                "ticket_id": failure.ticket_id,
                "audit_ticket_id": audit_ticket_id,
                "record_index": failure.record_index,
                "outcome": "escalate",
                "reason": (
                    "Ticket could not be normalized; "
                    "human review required."
                ),
                "ingestion_error": failure.error,
                "answer_text": None,
                "may_send_response": False,
            }
        )

    return results


def unique_failure_ticket_id(
    failure: IngestionFailure,
    *,
    used_ticket_ids: set[str],
) -> str:
    """Return a unique database identifier for a malformed record."""

    candidate = failure.ticket_id.strip()

    if not candidate:
        candidate = (
            f"MALFORMED-{failure.record_index:04d}"
        )

    if candidate not in used_ticket_ids:
        return candidate

    suffix_candidate = (
        f"{candidate}__record_{failure.record_index}"
    )

    counter = 2

    while suffix_candidate in used_ticket_ids:
        suffix_candidate = (
            f"{candidate}__record_"
            f"{failure.record_index}_{counter}"
        )
        counter += 1

    return suffix_candidate


def build_run_report(
    *,
    application: ApplicationComponents,
    ingestion_result: BatchIngestionResult,
    workflow_results: Sequence[WorkflowResult],
    ingestion_failure_results: Sequence[
        dict[str, Any]
    ],
    run_id: str,
    input_path: Path,
) -> dict[str, Any]:
    """Build the final serializable run report."""

    serialized_workflow_results = [
        result.as_dict()
        for result in workflow_results
    ]

    all_results = (
        serialized_workflow_results
        + list(ingestion_failure_results)
    )

    outcome_counts = {
        "auto_respond": 0,
        "escalate": 0,
        "block": 0,
    }

    for result in all_results:
        outcome = result.get("outcome")

        if outcome in outcome_counts:
            outcome_counts[outcome] += 1

    reconciliation = (
        application.decision_store
        .reconciliation_summary(
            run_id=run_id
        )
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": utc_now(),
        "input_path": str(input_path),
        "summary": {
            "total_input_records": (
                ingestion_result.total_records
            ),
            "normalized_records": (
                ingestion_result.successful_records
            ),
            "ingestion_failures": (
                ingestion_result.failed_records
            ),
            "processed_results": len(all_results),
            "auto_responded": (
                outcome_counts["auto_respond"]
            ),
            "escalated": outcome_counts["escalate"],
            "blocked": outcome_counts["block"],
        },
        "reconciliation": reconciliation,
        "startup": application.startup_summary(),
        "results": all_results,
    }


def write_json_report(
    *,
    output_path: str | Path,
    report: dict[str, Any],
    pretty: bool = False,
) -> None:
    """
    Write the report atomically.

    The temporary file is replaced only after JSON serialization succeeds.
    """

    resolved_output = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

    resolved_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = resolved_output.with_suffix(
        resolved_output.suffix + ".tmp"
    )

    indentation = 2 if pretty else None

    serialized = json.dumps(
        report,
        indent=indentation,
        ensure_ascii=False,
        sort_keys=pretty,
    )

    try:
        temporary_path.write_text(
            serialized + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(
            resolved_output
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def resolve_project_path(
    value: str | Path,
    *,
    project_root: Path,
) -> Path:
    """Resolve relative CLI paths from the project root."""

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def print_run_summary(
    report: dict[str, Any],
    *,
    output_path: Path,
) -> None:
    """Print a concise, non-sensitive run summary."""

    summary = report["summary"]

    print(f"Run ID: {report['run_id']}")
    print(
        "Input records: "
        f"{summary['total_input_records']}"
    )
    print(
        "Auto-responded: "
        f"{summary['auto_responded']}"
    )
    print(
        f"Escalated: {summary['escalated']}"
    )
    print(
        f"Blocked: {summary['blocked']}"
    )
    print(f"Results written to: {output_path}")


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """CLI entry point."""

    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        settings = get_settings()

        resolved_output_path = resolve_project_path(
            arguments.output,
            project_root=settings.project_root,
        )

        report = run_ticket_file(
            input_path=arguments.input,
            output_path=resolved_output_path,
            settings=settings,
            classifier_artifact_path=(
                arguments.classifier_artifact
            ),
            documentation_path=(
                arguments.documentation
            ),
            run_id=arguments.run_id,
            retrieval_top_k=arguments.top_k,
            build_index=False,
            pretty=arguments.pretty,
        )

    except (
        TicketFileError,
        ApplicationBuildError,
        ValueError,
    ) as exc:
        print(
            f"Run failed: {exc}",
            file=sys.stderr,
        )
        return 1

    except Exception as exc:
        print(
            (
                "Run failed unexpectedly: "
                f"{type(exc).__name__}: {exc}"
            ),
            file=sys.stderr,
        )
        return 2

    print_run_summary(
        report,
        output_path=resolved_output_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
