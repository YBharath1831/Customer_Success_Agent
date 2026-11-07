"""Tests for automatic validation metrics."""

import json
from pathlib import Path

import pytest

import src.evaluate as evaluate_module
from src.config import Settings
from src.evaluate import (
    EvaluationError,
    evaluate_run,
    index_run_results,
    normalize_actual_route,
    percentile,
    safe_ratio,
)


def write_json(
    path: Path,
    payload,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def make_ticket(
    *,
    ticket_id: str,
    channel: str,
    intent: str,
    urgency: str,
    expected_route: str,
    expected_doc_ids,
    must_not_auto_respond: bool = False,
):
    return {
        "ticket_id": ticket_id,
        "channel": channel,
        "subject": f"Subject for {ticket_id}",
        "body": f"Body for {ticket_id}",
        "labels": {
            "intent": intent,
            "urgency": urgency,
            "expected_route": expected_route,
            "answerable_from_docs": bool(
                expected_doc_ids
            ),
            "expected_doc_ids": list(
                expected_doc_ids
            ),
            "must_not_auto_respond": (
                must_not_auto_respond
            ),
        },
    }


def make_result(
    *,
    ticket_id: str,
    outcome: str,
    intent: str | None,
    urgency: str | None,
    passage_ids=(),
    cited_passage_ids=(),
    latency_seconds: float = 1.0,
):
    classification = None

    if intent is not None or urgency is not None:
        classification = {
            "intent": intent,
            "intent_confidence": 0.90,
            "urgency": urgency,
            "urgency_confidence": 0.85,
            "combined_confidence": 0.85,
            "model_name": "test-classifier",
            "model_version": "test-v1",
            "fallback_used": False,
        }

    return {
        "run_id": "EVAL-RUN",
        "ticket_id": ticket_id,
        "outcome": outcome,
        "reason": "Test outcome.",
        "answer_text": (
            "Test answer."
            if outcome == "auto_respond"
            else None
        ),
        "cited_passage_ids": list(
            cited_passage_ids
        ),
        "classification": classification,
        "retrieval_matches": [
            {
                "passage_id": passage_id,
                "title": "Test document",
                "category": "test",
                "relevance_score": 0.90,
            }
            for passage_id in passage_ids
        ],
        "route": {
            "action": (
                "auto_respond"
                if outcome == "auto_respond"
                else "escalate"
            ),
            "reason_code": "test",
            "reason": "Test route.",
        },
        "guardrails": None,
        "error_type": None,
        "latency_seconds": latency_seconds,
        "may_send_response": (
            outcome == "auto_respond"
        ),
    }


def make_dataset_and_report(
    tmp_path: Path,
):
    tickets = [
        make_ticket(
            ticket_id="TICKET-001",
            channel="email",
            intent="account_access",
            urgency="high",
            expected_route="auto_respond",
            expected_doc_ids=["DOC-A"],
        ),
        make_ticket(
            ticket_id="TICKET-002",
            channel="chat",
            intent="unclear_request",
            urgency="medium",
            expected_route="escalate",
            expected_doc_ids=[],
            must_not_auto_respond=True,
        ),
        make_ticket(
            ticket_id="TICKET-003",
            channel="forum",
            intent="billing_question",
            urgency="low",
            expected_route="auto_respond",
            expected_doc_ids=["DOC-B"],
        ),
    ]

    results = [
        make_result(
            ticket_id="TICKET-001",
            outcome="auto_respond",
            intent="account_access",
            urgency="high",
            passage_ids=["DOC-A#p001"],
            cited_passage_ids=["DOC-A#p001"],
            latency_seconds=1.0,
        ),
        make_result(
            ticket_id="TICKET-002",
            outcome="block",
            intent="unclear_request",
            urgency="low",
            latency_seconds=2.0,
        ),
        make_result(
            ticket_id="TICKET-003",
            outcome="escalate",
            intent="account_access",
            urgency="low",
            latency_seconds=3.0,
        ),
    ]

    tickets_path = tmp_path / "tickets.json"
    report_path = tmp_path / "run-report.json"

    write_json(
        tickets_path,
        tickets,
    )

    write_json(
        report_path,
        {
            "schema_version": 1,
            "run_id": "EVAL-RUN",
            "summary": {
                "total_input_records": 3,
                "processed_results": 3,
            },
            "reconciliation": {
                "tickets_started": 3,
                "tickets_completed": 3,
                "terminal_outcomes": 3,
            },
            "results": results,
        },
    )

    return tickets_path, report_path


def test_complete_evaluation_metrics(
    tmp_path: Path,
) -> None:
    tickets_path, report_path = (
        make_dataset_and_report(tmp_path)
    )

    evaluation = evaluate_run(
        tickets_path=tickets_path,
        run_report_path=report_path,
    )

    assert evaluation["run_id"] == "EVAL-RUN"

    dataset = evaluation["dataset"]

    assert dataset["total_records"] == 3
    assert dataset["labelled_records"] == 3
    assert dataset["matched_results"] == 3
    assert dataset["missing_result_ids"] == []
    assert dataset["extra_result_ids"] == []

    metrics = evaluation["metrics"]

    assert metrics["coverage"][
        "result_coverage"
    ] == 1.0

    assert metrics["classification"]["intent"][
        "evaluated"
    ] == 3
    assert metrics["classification"]["intent"][
        "correct"
    ] == 2
    assert metrics["classification"]["intent"][
        "accuracy"
    ] == pytest.approx(2 / 3)

    assert metrics["classification"]["urgency"][
        "correct"
    ] == 2
    assert metrics["classification"]["urgency"][
        "accuracy"
    ] == pytest.approx(2 / 3)

    routing = metrics["routing"]

    assert routing["evaluated"] == 3
    assert routing["correct"] == 2
    assert routing["accuracy"] == pytest.approx(
        2 / 3
    )
    assert routing[
        "unsafe_auto_responses"
    ] == 0
    assert routing[
        "unexpected_auto_responses"
    ] == 0
    assert routing[
        "expected_auto_responses"
    ] == 2
    assert routing[
        "missed_auto_responses"
    ] == 1

    retrieval = metrics["retrieval"]

    assert retrieval[
        "tickets_with_expected_docs"
    ] == 2
    assert retrieval[
        "tickets_with_expected_doc_hit"
    ] == 1
    assert retrieval[
        "expected_doc_hit_rate"
    ] == 0.5
    assert retrieval[
        "macro_expected_doc_recall"
    ] == 0.5

    generation = metrics["generation"]

    assert generation["auto_responses"] == 1
    assert generation[
        "auto_responses_with_citations"
    ] == 1
    assert generation[
        "citation_coverage"
    ] == 1.0

    assert metrics["outcomes"] == {
        "auto_respond": 1,
        "escalate": 1,
        "block": 1,
    }

    latency = metrics["latency_seconds"]

    assert latency["evaluated"] == 3
    assert latency["mean"] == 2.0
    assert latency["p50"] == 2.0
    assert latency["p95"] == pytest.approx(
        2.9
    )
    assert latency["maximum"] == 3.0


def test_block_is_evaluated_as_safe_escalation() -> None:
    assert (
        normalize_actual_route("block")
        == "escalate"
    )
    assert (
        normalize_actual_route("escalate")
        == "escalate"
    )
    assert (
        normalize_actual_route("auto_respond")
        == "auto_respond"
    )
    assert normalize_actual_route("unknown") is None


def test_unsafe_auto_response_is_detected(
    tmp_path: Path,
) -> None:
    tickets_path, report_path = (
        make_dataset_and_report(tmp_path)
    )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    unsafe_result = report["results"][1]
    unsafe_result["outcome"] = "auto_respond"
    unsafe_result["answer_text"] = "Unsafe answer."
    unsafe_result["may_send_response"] = True

    write_json(
        report_path,
        report,
    )

    evaluation = evaluate_run(
        tickets_path=tickets_path,
        run_report_path=report_path,
    )

    routing = evaluation["metrics"]["routing"]

    assert routing[
        "unsafe_auto_responses"
    ] == 1
    assert routing[
        "unexpected_auto_responses"
    ] == 1
    assert routing[
        "unsafe_auto_response_rate"
    ] == 1.0

    assert evaluation["checks"][
        "no_unsafe_auto_responses"
    ] is False


def test_missing_result_reduces_coverage(
    tmp_path: Path,
) -> None:
    tickets_path, report_path = (
        make_dataset_and_report(tmp_path)
    )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    report["results"] = report["results"][:2]

    write_json(
        report_path,
        report,
    )

    evaluation = evaluate_run(
        tickets_path=tickets_path,
        run_report_path=report_path,
    )

    assert evaluation["dataset"][
        "missing_result_ids"
    ] == ["TICKET-003"]

    assert evaluation["metrics"]["coverage"][
        "result_coverage"
    ] == pytest.approx(2 / 3)

    assert evaluation["checks"][
        "all_labelled_tickets_have_results"
    ] is False


def test_extra_result_is_reported(
    tmp_path: Path,
) -> None:
    tickets_path, report_path = (
        make_dataset_and_report(tmp_path)
    )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    report["results"].append(
        make_result(
            ticket_id="EXTRA-001",
            outcome="escalate",
            intent=None,
            urgency=None,
        )
    )

    write_json(
        report_path,
        report,
    )

    evaluation = evaluate_run(
        tickets_path=tickets_path,
        run_report_path=report_path,
    )

    assert evaluation["dataset"][
        "extra_result_ids"
    ] == ["EXTRA-001"]


def test_duplicate_result_ticket_id_is_rejected() -> None:
    result = make_result(
        ticket_id="DUPLICATE",
        outcome="escalate",
        intent=None,
        urgency=None,
    )

    with pytest.raises(
        EvaluationError,
        match="duplicate ticket ID",
    ):
        index_run_results(
            {
                "results": [
                    result,
                    dict(result),
                ]
            }
        )


@pytest.mark.parametrize(
    "invalid_results",
    [
        None,
        {},
        "not-a-list",
    ],
)
def test_invalid_results_array_is_rejected(
    invalid_results,
) -> None:
    with pytest.raises(
        EvaluationError,
        match="results array",
    ):
        index_run_results(
            {
                "results": invalid_results,
            }
        )


def test_unlabelled_ticket_is_excluded(
    tmp_path: Path,
) -> None:
    tickets_path, report_path = (
        make_dataset_and_report(tmp_path)
    )

    tickets = json.loads(
        tickets_path.read_text(
            encoding="utf-8"
        )
    )

    tickets.append(
        {
            "ticket_id": "UNLABELLED-001",
            "channel": "docs_comment",
            "subject": "Question",
            "body": "No labels are supplied.",
        }
    )

    write_json(
        tickets_path,
        tickets,
    )

    evaluation = evaluate_run(
        tickets_path=tickets_path,
        run_report_path=report_path,
    )

    assert evaluation["dataset"][
        "total_records"
    ] == 4
    assert evaluation["dataset"][
        "labelled_records"
    ] == 3
    assert evaluation["dataset"][
        "unlabelled_records"
    ] == 1
    assert evaluation["dataset"][
        "unlabelled_ticket_ids"
    ] == ["UNLABELLED-001"]


def test_malformed_evaluation_ticket_is_rejected(
    tmp_path: Path,
) -> None:
    tickets_path = tmp_path / "tickets.json"
    report_path = tmp_path / "report.json"

    write_json(
        tickets_path,
        [
            make_ticket(
                ticket_id="VALID",
                channel="email",
                intent="account_access",
                urgency="low",
                expected_route="auto_respond",
                expected_doc_ids=[],
            ),
            "malformed ticket",
        ],
    )

    write_json(
        report_path,
        {
            "results": [],
        },
    )

    with pytest.raises(
        EvaluationError,
        match="malformed record",
    ):
        evaluate_run(
            tickets_path=tickets_path,
            run_report_path=report_path,
        )


def test_confusion_matrices_are_created(
    tmp_path: Path,
) -> None:
    tickets_path, report_path = (
        make_dataset_and_report(tmp_path)
    )

    evaluation = evaluate_run(
        tickets_path=tickets_path,
        run_report_path=report_path,
    )

    intent_matrix = evaluation["metrics"][
        "classification"
    ]["intent"]["confusion_matrix"]

    assert intent_matrix[
        "account_access"
    ]["account_access"] == 1
    assert intent_matrix[
        "billing_question"
    ]["account_access"] == 1

    route_matrix = evaluation["metrics"][
        "routing"
    ]["confusion_matrix"]

    assert route_matrix[
        "auto_respond"
    ]["auto_respond"] == 1
    assert route_matrix[
        "auto_respond"
    ]["escalate"] == 1
    assert route_matrix[
        "escalate"
    ]["escalate"] == 1


def test_per_channel_metrics_are_created(
    tmp_path: Path,
) -> None:
    tickets_path, report_path = (
        make_dataset_and_report(tmp_path)
    )

    evaluation = evaluate_run(
        tickets_path=tickets_path,
        run_report_path=report_path,
    )

    per_channel = evaluation["metrics"][
        "per_channel"
    ]

    assert set(per_channel) == {
        "email",
        "chat",
        "forum",
    }

    assert per_channel["email"][
        "intent_accuracy"
    ] == 1.0
    assert per_channel["chat"][
        "route_accuracy"
    ] == 1.0
    assert per_channel["forum"][
        "route_accuracy"
    ] == 0.0


def test_evaluation_checks_pass_for_reconciled_run(
    tmp_path: Path,
) -> None:
    tickets_path, report_path = (
        make_dataset_and_report(tmp_path)
    )

    evaluation = evaluate_run(
        tickets_path=tickets_path,
        run_report_path=report_path,
    )

    checks = evaluation["checks"]

    assert checks[
        "all_labelled_tickets_have_results"
    ] is True
    assert checks[
        "no_unsafe_auto_responses"
    ] is True
    assert checks[
        "all_auto_responses_have_citations"
    ] is True
    assert checks[
        "all_started_tickets_completed"
    ] is True
    assert checks[
        "one_terminal_outcome_per_started_ticket"
    ] is True


def test_uncited_auto_response_fails_check(
    tmp_path: Path,
) -> None:
    tickets_path, report_path = (
        make_dataset_and_report(tmp_path)
    )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    report["results"][0][
        "cited_passage_ids"
    ] = []

    write_json(
        report_path,
        report,
    )

    evaluation = evaluate_run(
        tickets_path=tickets_path,
        run_report_path=report_path,
    )

    assert evaluation["metrics"]["generation"][
        "citation_coverage"
    ] == 0.0

    assert evaluation["checks"][
        "all_auto_responses_have_citations"
    ] is False


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (1, 2, 0.5),
        (0, 4, 0.0),
        (1, 0, None),
    ],
)
def test_safe_ratio(
    numerator,
    denominator,
    expected,
) -> None:
    assert safe_ratio(
        numerator,
        denominator,
    ) == expected


def test_percentile_calculation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 0.5) == 2.5
    assert percentile(values, 0.95) == pytest.approx(
        3.85
    )
    assert percentile(values, 1.0) == 4.0
    assert percentile([], 0.5) is None
    assert percentile([7.0], 0.95) == 7.0


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
    )


def test_cli_writes_metrics_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    settings = make_settings(tmp_path)
    tickets_path, report_path = (
        make_dataset_and_report(
            settings.project_root
        )
    )

    output_path = (
        settings.project_root
        / "evaluation"
        / "metrics.json"
    )

    monkeypatch.setattr(
        evaluate_module,
        "get_settings",
        lambda: settings,
    )

    exit_code = evaluate_module.main(
        [
            "--tickets",
            str(tickets_path),
            "--run-report",
            str(report_path),
            "--output",
            str(output_path),
            "--pretty",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert "Intent accuracy:" in captured.out
    assert "Metrics written to:" in captured.out

    saved_metrics = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert saved_metrics["run_id"] == "EVAL-RUN"


def test_cli_returns_one_for_evaluation_error(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    settings = make_settings(tmp_path)

    monkeypatch.setattr(
        evaluate_module,
        "get_settings",
        lambda: settings,
    )

    exit_code = evaluate_module.main(
        [
            "--tickets",
            "missing-tickets.json",
            "--run-report",
            "missing-report.json",
            "--output",
            "metrics.json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Evaluation failed:" in captured.err