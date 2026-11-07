"""One-command unattended evaluation harness."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

from src.config import (
    Settings,
    get_settings,
)
from src.evaluate import evaluate_run
from src.run import (
    resolve_project_path,
    run_ticket_file,
    write_json_report,
)


CRITICAL_CHECKS = (
    "all_labelled_tickets_have_results",
    "no_unsafe_auto_responses",
    "all_auto_responses_have_citations",
    "all_started_tickets_completed",
    "one_terminal_outcome_per_started_ticket",
)


class EvaluationHarnessError(RuntimeError):
    """Raised when the unattended evaluation cannot finish."""


def run_evaluation_harness(
    *,
    tickets_path: str | Path,
    run_report_path: str | Path,
    metrics_output_path: str | Path,
    summary_output_path: str | Path,
    settings: Settings | None = None,
    classifier_artifact_path: str | Path | None = None,
    documentation_path: str | Path | None = None,
    run_id: str | None = None,
    retrieval_top_k: int | None = None,
    build_index: bool = False,
    pretty: bool = True,
) -> dict[str, Any]:
    """
    Run the complete workflow and calculate evaluation metrics.

    This function is suitable for local validation and automated CI.
    """

    current_settings = settings or get_settings()
    resolved_run_id = run_id or str(uuid.uuid4())

    resolved_tickets_path = resolve_project_path(
        tickets_path,
        project_root=current_settings.project_root,
    )
    resolved_run_report_path = resolve_project_path(
        run_report_path,
        project_root=current_settings.project_root,
    )
    resolved_metrics_path = resolve_project_path(
        metrics_output_path,
        project_root=current_settings.project_root,
    )
    resolved_summary_path = resolve_project_path(
        summary_output_path,
        project_root=current_settings.project_root,
    )

    run_report = run_ticket_file(
        input_path=resolved_tickets_path,
        output_path=resolved_run_report_path,
        settings=current_settings,
        classifier_artifact_path=(
            classifier_artifact_path
        ),
        documentation_path=documentation_path,
        run_id=resolved_run_id,
        retrieval_top_k=retrieval_top_k,
        build_index=build_index,
        pretty=pretty,
    )

    evaluation = evaluate_run(
        tickets_path=resolved_tickets_path,
        run_report_path=resolved_run_report_path,
    )

    write_json_report(
        output_path=resolved_metrics_path,
        report=evaluation,
        pretty=pretty,
    )

    checks = evaluation.get(
        "checks",
        {},
    )

    failed_checks = [
        check_name
        for check_name in CRITICAL_CHECKS
        if checks.get(check_name) is not True
    ]

    metrics = evaluation["metrics"]

    summary = {
        "run_id": resolved_run_id,
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "paths": {
            "tickets": str(
                resolved_tickets_path
            ),
            "run_report": str(
                resolved_run_report_path
            ),
            "metrics": str(
                resolved_metrics_path
            ),
            "summary": str(
                resolved_summary_path
            ),
        },
        "run_summary": run_report["summary"],
        "key_metrics": {
            "intent_accuracy": (
                metrics["classification"][
                    "intent"
                ]["accuracy"]
            ),
            "urgency_accuracy": (
                metrics["classification"][
                    "urgency"
                ]["accuracy"]
            ),
            "routing_accuracy": (
                metrics["routing"]["accuracy"]
            ),
            "expected_doc_hit_rate": (
                metrics["retrieval"][
                    "expected_doc_hit_rate"
                ]
            ),
            "citation_coverage": (
                metrics["generation"][
                    "citation_coverage"
                ]
            ),
            "unsafe_auto_responses": (
                metrics["routing"][
                    "unsafe_auto_responses"
                ]
            ),
            "mean_latency_seconds": (
                metrics["latency_seconds"]["mean"]
            ),
            "p95_latency_seconds": (
                metrics["latency_seconds"]["p95"]
            ),
        },
        "checks": checks,
    }

    write_json_report(
        output_path=resolved_summary_path,
        report=summary,
        pretty=pretty,
    )

    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the evaluation-harness CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete customer-success agent "
            "and automatically evaluate its results."
        )
    )

    parser.add_argument(
        "--tickets",
        default="data/validation_tickets.json",
        help=(
            "Labelled ticket dataset. Default: "
            "data/validation_tickets.json"
        ),
    )

    parser.add_argument(
        "--run-report",
        default=(
            "evaluation/results/"
            "validation_run.json"
        ),
        help="Workflow run-report destination.",
    )

    parser.add_argument(
        "--metrics-output",
        default=(
            "evaluation/results/"
            "validation_metrics.json"
        ),
        help="Detailed metrics destination.",
    )

    parser.add_argument(
        "--summary-output",
        default=(
            "evaluation/results/"
            "validation_summary.json"
        ),
        help="Concise evaluation summary destination.",
    )

    parser.add_argument(
        "--classifier-artifact",
        default=None,
        help=(
            "Optional classifier artifact path."
        ),
    )

    parser.add_argument(
        "--documentation",
        default=None,
        help=(
            "Optional documentation corpus path."
        ),
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional audit run ID. A UUID is generated "
            "when omitted."
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
        "--compact",
        action="store_true",
        help=(
            "Write compact JSON instead of indented JSON."
        ),
    )

    return parser


def print_harness_summary(
    summary: dict[str, Any],
) -> None:
    """Print a concise evaluation result."""

    metrics = summary["key_metrics"]

    print(f"Run ID: {summary['run_id']}")
    print(
        f"Evaluation passed: {summary['passed']}"
    )
    print(
        "Intent accuracy:",
        metrics["intent_accuracy"],
    )
    print(
        "Urgency accuracy:",
        metrics["urgency_accuracy"],
    )
    print(
        "Routing accuracy:",
        metrics["routing_accuracy"],
    )
    print(
        "Expected-document hit rate:",
        metrics["expected_doc_hit_rate"],
    )
    print(
        "Citation coverage:",
        metrics["citation_coverage"],
    )
    print(
        "Unsafe auto-responses:",
        metrics["unsafe_auto_responses"],
    )

    if summary["failed_checks"]:
        print(
            "Failed checks:",
            ", ".join(
                summary["failed_checks"]
            ),
        )

    print(
        "Summary report:",
        summary["paths"]["summary"],
    )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the evaluation harness from the command line."""

    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        summary = run_evaluation_harness(
            tickets_path=arguments.tickets,
            run_report_path=arguments.run_report,
            metrics_output_path=(
                arguments.metrics_output
            ),
            summary_output_path=(
                arguments.summary_output
            ),
            classifier_artifact_path=(
                arguments.classifier_artifact
            ),
            documentation_path=(
                arguments.documentation
            ),
            run_id=arguments.run_id,
            retrieval_top_k=arguments.top_k,
            build_index=False,
            pretty=not arguments.compact,
        )

    except Exception as exc:
        print(
            (
                "Evaluation harness failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            file=sys.stderr,
        )
        return 1

    print_harness_summary(summary)

    if not summary["passed"]:
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
