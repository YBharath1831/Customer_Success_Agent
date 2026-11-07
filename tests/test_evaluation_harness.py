"""Tests for the one-command evaluation harness."""

import json
from pathlib import Path

import pytest

import evaluation.harness as harness_module
from evaluation.harness import (
    CRITICAL_CHECKS,
    run_evaluation_harness,
)
from src.config import Settings


def make_settings(
    tmp_path: Path,
) -> Settings:
    project_root = tmp_path / "project"

    return Settings(
        project_root=project_root,
        data_dir=project_root / "data",
        storage_dir=project_root / "storage",
        evaluation_results_dir=(
            project_root / "evaluation" / "results"
        ),
        chroma_path=(
            project_root / "storage" / "chroma"
        ),
        database_url=(
            "sqlite:///./storage/decisions.db"
        ),
        database_path=(
            project_root
            / "storage"
            / "decisions.db"
        ),
        openrouter_api_key="test-key",
    )


def successful_checks():
    return {
        "all_labelled_tickets_have_results": True,
        "no_unsafe_auto_responses": True,
        "all_auto_responses_have_citations": True,
        "all_started_tickets_completed": True,
        "one_terminal_outcome_per_started_ticket": True,
        "labelled_ticket_count": 3,
    }


def make_evaluation(
    *,
    checks=None,
):
    return {
        "schema_version": 1,
        "run_id": "HARNESS-RUN",
        "dataset": {
            "total_records": 3,
            "labelled_records": 3,
            "matched_results": 3,
        },
        "metrics": {
            "classification": {
                "intent": {
                    "accuracy": 0.80,
                },
                "urgency": {
                    "accuracy": 0.70,
                },
            },
            "routing": {
                "accuracy": 0.90,
                "unsafe_auto_responses": 0,
            },
            "retrieval": {
                "expected_doc_hit_rate": 0.75,
            },
            "generation": {
                "citation_coverage": 1.0,
            },
            "latency_seconds": {
                "mean": 1.5,
                "p95": 2.8,
            },
        },
        "checks": (
            checks
            if checks is not None
            else successful_checks()
        ),
        "details": [],
    }


def install_fake_run_and_evaluation(
    monkeypatch,
    *,
    evaluation=None,
):
    run_calls = []
    evaluation_calls = []

    def fake_run_ticket_file(**kwargs):
        run_calls.append(kwargs)

        return {
            "run_id": kwargs["run_id"],
            "summary": {
                "total_input_records": 3,
                "normalized_records": 3,
                "ingestion_failures": 0,
                "processed_results": 3,
                "auto_responded": 2,
                "escalated": 1,
                "blocked": 0,
            },
        }

    def fake_evaluate_run(**kwargs):
        evaluation_calls.append(kwargs)

        return (
            evaluation
            if evaluation is not None
            else make_evaluation()
        )

    monkeypatch.setattr(
        harness_module,
        "run_ticket_file",
        fake_run_ticket_file,
    )
    monkeypatch.setattr(
        harness_module,
        "evaluate_run",
        fake_evaluate_run,
    )

    return run_calls, evaluation_calls


def test_successful_harness_writes_all_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)

    run_calls, evaluation_calls = (
        install_fake_run_and_evaluation(
            monkeypatch
        )
    )

    summary = run_evaluation_harness(
        tickets_path="data/validation.json",
        run_report_path=(
            "evaluation/results/run.json"
        ),
        metrics_output_path=(
            "evaluation/results/metrics.json"
        ),
        summary_output_path=(
            "evaluation/results/summary.json"
        ),
        settings=settings,
        run_id="HARNESS-RUN",
    )

    results_directory = (
        settings.project_root
        / "evaluation"
        / "results"
    )

    metrics_path = (
        results_directory / "metrics.json"
    )
    summary_path = (
        results_directory / "summary.json"
    )

    assert metrics_path.exists()
    assert summary_path.exists()

    assert summary["run_id"] == "HARNESS-RUN"
    assert summary["passed"] is True
    assert summary["failed_checks"] == []

    assert summary["key_metrics"][
        "intent_accuracy"
    ] == 0.80
    assert summary["key_metrics"][
        "urgency_accuracy"
    ] == 0.70
    assert summary["key_metrics"][
        "routing_accuracy"
    ] == 0.90
    assert summary["key_metrics"][
        "expected_doc_hit_rate"
    ] == 0.75
    assert summary["key_metrics"][
        "citation_coverage"
    ] == 1.0
    assert summary["key_metrics"][
        "unsafe_auto_responses"
    ] == 0

    saved_metrics = json.loads(
        metrics_path.read_text(
            encoding="utf-8"
        )
    )
    saved_summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert saved_metrics["run_id"] == (
        "HARNESS-RUN"
    )
    assert saved_summary["passed"] is True

    assert len(run_calls) == 1
    assert len(evaluation_calls) == 1


def test_harness_forwards_runtime_options(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)

    run_calls, _ = (
        install_fake_run_and_evaluation(
            monkeypatch
        )
    )

    run_evaluation_harness(
        tickets_path="tickets.json",
        run_report_path="run.json",
        metrics_output_path="metrics.json",
        summary_output_path="summary.json",
        settings=settings,
        classifier_artifact_path=(
            "models/custom.joblib"
        ),
        documentation_path="docs/custom.json",
        run_id="OPTIONS-RUN",
        retrieval_top_k=7,
        build_index=False,
        pretty=False,
    )

    call = run_calls[0]

    assert call["run_id"] == "OPTIONS-RUN"
    assert call["retrieval_top_k"] == 7
    assert call["build_index"] is False
    assert call["pretty"] is False
    assert call[
        "classifier_artifact_path"
    ] == "models/custom.joblib"
    assert call[
        "documentation_path"
    ] == "docs/custom.json"


def test_harness_generates_run_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)

    run_calls, _ = (
        install_fake_run_and_evaluation(
            monkeypatch
        )
    )

    summary = run_evaluation_harness(
        tickets_path="tickets.json",
        run_report_path="run.json",
        metrics_output_path="metrics.json",
        summary_output_path="summary.json",
        settings=settings,
    )

    generated_run_id = run_calls[0]["run_id"]

    assert isinstance(generated_run_id, str)
    assert generated_run_id
    assert summary["run_id"] == generated_run_id


@pytest.mark.parametrize(
    "failed_check",
    CRITICAL_CHECKS,
)
def test_each_critical_check_can_fail_harness(
    tmp_path: Path,
    monkeypatch,
    failed_check: str,
) -> None:
    settings = make_settings(tmp_path)

    checks = successful_checks()
    checks[failed_check] = False

    install_fake_run_and_evaluation(
        monkeypatch,
        evaluation=make_evaluation(
            checks=checks
        ),
    )

    summary = run_evaluation_harness(
        tickets_path="tickets.json",
        run_report_path="run.json",
        metrics_output_path="metrics.json",
        summary_output_path="summary.json",
        settings=settings,
        run_id="FAILED-CHECK-RUN",
    )

    assert summary["passed"] is False
    assert summary["failed_checks"] == [
        failed_check
    ]


def test_multiple_failed_checks_are_reported(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)

    checks = successful_checks()
    checks[
        "no_unsafe_auto_responses"
    ] = False
    checks[
        "all_auto_responses_have_citations"
    ] = False

    install_fake_run_and_evaluation(
        monkeypatch,
        evaluation=make_evaluation(
            checks=checks
        ),
    )

    summary = run_evaluation_harness(
        tickets_path="tickets.json",
        run_report_path="run.json",
        metrics_output_path="metrics.json",
        summary_output_path="summary.json",
        settings=settings,
        run_id="MULTIPLE-FAILURES",
    )

    assert summary["passed"] is False
    assert summary["failed_checks"] == [
        "no_unsafe_auto_responses",
        "all_auto_responses_have_citations",
    ]


def test_missing_check_is_treated_as_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)

    checks = successful_checks()
    del checks[
        "all_started_tickets_completed"
    ]

    install_fake_run_and_evaluation(
        monkeypatch,
        evaluation=make_evaluation(
            checks=checks
        ),
    )

    summary = run_evaluation_harness(
        tickets_path="tickets.json",
        run_report_path="run.json",
        metrics_output_path="metrics.json",
        summary_output_path="summary.json",
        settings=settings,
    )

    assert summary["passed"] is False
    assert (
        "all_started_tickets_completed"
        in summary["failed_checks"]
    )


def test_cli_returns_zero_when_checks_pass(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    settings = make_settings(tmp_path)

    monkeypatch.setattr(
        harness_module,
        "get_settings",
        lambda: settings,
    )

    summary = {
        "run_id": "CLI-SUCCESS",
        "passed": True,
        "failed_checks": [],
        "key_metrics": {
            "intent_accuracy": 0.8,
            "urgency_accuracy": 0.7,
            "routing_accuracy": 0.9,
            "expected_doc_hit_rate": 0.75,
            "citation_coverage": 1.0,
            "unsafe_auto_responses": 0,
        },
        "paths": {
            "summary": "summary.json",
        },
    }

    monkeypatch.setattr(
        harness_module,
        "run_evaluation_harness",
        lambda **kwargs: summary,
    )

    exit_code = harness_module.main([])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Evaluation passed: True" in (
        captured.out
    )
    assert "Intent accuracy: 0.8" in (
        captured.out
    )


def test_cli_returns_three_when_check_fails(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    settings = make_settings(tmp_path)

    monkeypatch.setattr(
        harness_module,
        "get_settings",
        lambda: settings,
    )

    summary = {
        "run_id": "CLI-FAILED-CHECK",
        "passed": False,
        "failed_checks": [
            "no_unsafe_auto_responses"
        ],
        "key_metrics": {
            "intent_accuracy": 0.8,
            "urgency_accuracy": 0.7,
            "routing_accuracy": 0.5,
            "expected_doc_hit_rate": 0.75,
            "citation_coverage": 1.0,
            "unsafe_auto_responses": 1,
        },
        "paths": {
            "summary": "summary.json",
        },
    }

    monkeypatch.setattr(
        harness_module,
        "run_evaluation_harness",
        lambda **kwargs: summary,
    )

    exit_code = harness_module.main([])

    captured = capsys.readouterr()

    assert exit_code == 3
    assert "Evaluation passed: False" in (
        captured.out
    )
    assert "Failed checks:" in captured.out
    assert "no_unsafe_auto_responses" in (
        captured.out
    )


def test_cli_returns_one_for_operational_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    settings = make_settings(tmp_path)

    monkeypatch.setattr(
        harness_module,
        "get_settings",
        lambda: settings,
    )

    def fail_harness(**kwargs):
        raise RuntimeError(
            "evaluation infrastructure failed"
        )

    monkeypatch.setattr(
        harness_module,
        "run_evaluation_harness",
        fail_harness,
    )

    exit_code = harness_module.main([])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert (
        "Evaluation harness failed:"
        in captured.err
    )
    assert "RuntimeError" in captured.err


def test_cli_reuses_prepared_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)
    captured_arguments = {}

    monkeypatch.setattr(
        harness_module,
        "get_settings",
        lambda: settings,
    )

    def fake_harness(**kwargs):
        captured_arguments.update(kwargs)

        return {
            "run_id": "SKIP-INDEX",
            "passed": True,
            "failed_checks": [],
            "key_metrics": {
                "intent_accuracy": 1.0,
                "urgency_accuracy": 1.0,
                "routing_accuracy": 1.0,
                "expected_doc_hit_rate": 1.0,
                "citation_coverage": 1.0,
                "unsafe_auto_responses": 0,
            },
            "paths": {
                "summary": "summary.json",
            },
        }

    monkeypatch.setattr(
        harness_module,
        "run_evaluation_harness",
        fake_harness,
    )

    exit_code = harness_module.main(
        [
            "--top-k",
            "8",
            "--compact",
        ]
    )

    assert exit_code == 0
    assert captured_arguments[
        "build_index"
    ] is False
    assert captured_arguments[
        "retrieval_top_k"
    ] == 8
    assert captured_arguments["pretty"] is False
