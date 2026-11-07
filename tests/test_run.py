"""Tests for the unattended command-line runner."""

import json
from pathlib import Path

import pytest

import src.run as run_module
from src.config import Settings
from src.ingest import IngestionFailure
from src.logging_store import (
    DecisionRecord,
    DecisionStore,
)
from src.run import (
    main,
    resolve_project_path,
    run_ticket_file,
    unique_failure_ticket_id,
    write_json_report,
)
from src.workflow import (
    WorkflowOutcome,
    WorkflowResult,
)


class FakeWorkflow:
    """Workflow that records deterministic successful outcomes."""

    def __init__(
        self,
        decision_store: DecisionStore,
    ) -> None:
        self.decision_store = decision_store
        self.calls = []

    def process_batch(
        self,
        tickets,
        *,
        run_id=None,
    ):
        self.calls.append(
            {
                "tickets": list(tickets),
                "run_id": run_id,
            }
        )

        results = []

        for ticket in tickets:
            self.decision_store.start_ticket(
                run_id=run_id,
                ticket_id=ticket.ticket_id,
                channel=ticket.channel.value,
            )

            self.decision_store.log_decision(
                DecisionRecord(
                    run_id=run_id,
                    ticket_id=ticket.ticket_id,
                    stage="ingestion",
                    action_taken="normalized",
                    reason="Fake workflow accepted ticket.",
                )
            )

            self.decision_store.complete_ticket(
                run_id=run_id,
                ticket_id=ticket.ticket_id,
                terminal_outcome="auto_respond",
                latency_seconds=0.01,
            )

            results.append(
                WorkflowResult(
                    run_id=run_id,
                    ticket_id=ticket.ticket_id,
                    outcome=(
                        WorkflowOutcome.AUTO_RESPOND
                    ),
                    reason="Fake workflow success.",
                    answer_text=(
                        "Test response [DOC001#p001]."
                    ),
                    cited_passage_ids=(
                        "DOC001#p001",
                    ),
                    latency_seconds=0.01,
                )
            )

        return results


class FakeApplication:
    """Minimal application object used by runner tests."""

    def __init__(
        self,
        decision_store: DecisionStore,
    ) -> None:
        self.decision_store = decision_store
        self.workflow = FakeWorkflow(
            decision_store
        )

    def startup_summary(self):
        return {
            "classifier_artifact": (
                "fake-classifier.joblib"
            ),
            "documentation_passage_count": 1,
            "configuration": {
                "openrouter_api_key_configured": True,
            },
        }


def make_settings(
    tmp_path: Path,
) -> Settings:
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    storage_dir = project_root / "storage"
    evaluation_dir = (
        project_root / "evaluation" / "results"
    )
    database_path = (
        storage_dir / "decisions.db"
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        storage_dir=storage_dir,
        evaluation_results_dir=evaluation_dir,
        chroma_path=storage_dir / "chroma",
        database_url=(
            f"sqlite:///{database_path}"
        ),
        database_path=database_path,
        openrouter_api_key="test-key",
    )


def install_fake_application(
    monkeypatch,
    settings: Settings,
):
    decision_store = DecisionStore(
        settings.database_path
    )
    decision_store.initialize()

    application = FakeApplication(
        decision_store
    )
    build_calls = []

    def fake_build_application(**kwargs):
        build_calls.append(kwargs)
        return application

    monkeypatch.setattr(
        run_module,
        "build_application",
        fake_build_application,
    )

    return application, build_calls


def write_ticket_file(
    path: Path,
    records,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(records),
        encoding="utf-8",
    )


def valid_ticket(
    ticket_id: str = "RUN-001",
):
    return {
        "ticket_id": ticket_id,
        "channel": "email",
        "subject": "Login help",
        "body": "I cannot log in.",
    }


def test_runner_processes_file_and_writes_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)
    application, build_calls = (
        install_fake_application(
            monkeypatch,
            settings,
        )
    )

    input_path = (
        settings.data_dir / "tickets.json"
    )
    output_path = (
        settings.evaluation_results_dir
        / "results.json"
    )

    write_ticket_file(
        input_path,
        [
            valid_ticket("RUN-001"),
            valid_ticket("RUN-002"),
        ],
    )

    report = run_ticket_file(
        input_path=input_path,
        output_path=output_path,
        settings=settings,
        run_id="TEST-RUN",
        pretty=True,
    )

    assert output_path.exists()
    assert report["run_id"] == "TEST-RUN"

    assert report["summary"] == {
        "total_input_records": 2,
        "normalized_records": 2,
        "ingestion_failures": 0,
        "processed_results": 2,
        "auto_responded": 2,
        "escalated": 0,
        "blocked": 0,
    }

    saved_report = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert saved_report["run_id"] == "TEST-RUN"
    assert len(saved_report["results"]) == 2
    assert saved_report["results"][0][
        "may_send_response"
    ] is True

    assert len(
        application.workflow.calls
    ) == 1

    assert len(build_calls) == 1
    assert build_calls[0]["settings"] is settings


def test_malformed_record_becomes_escalation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)
    application, _ = install_fake_application(
        monkeypatch,
        settings,
    )

    input_path = (
        settings.data_dir / "mixed.json"
    )
    output_path = (
        settings.evaluation_results_dir
        / "mixed-results.json"
    )

    write_ticket_file(
        input_path,
        [
            valid_ticket("VALID-001"),
            "this is not a ticket object",
            {
                "ticket_id": "",
                "channel": "email",
                "body": "Missing valid ID",
            },
        ],
    )

    report = run_ticket_file(
        input_path=input_path,
        output_path=output_path,
        settings=settings,
        run_id="MIXED-RUN",
    )

    summary = report["summary"]

    assert summary["total_input_records"] == 3
    assert summary["normalized_records"] == 1
    assert summary["ingestion_failures"] == 2
    assert summary["processed_results"] == 3
    assert summary["auto_responded"] == 1
    assert summary["escalated"] == 2

    malformed_results = [
        result
        for result in report["results"]
        if result["outcome"] == "escalate"
    ]

    assert len(malformed_results) == 2

    assert all(
        result["may_send_response"] is False
        for result in malformed_results
    )

    reconciliation = (
        application.decision_store
        .reconciliation_summary(
            run_id="MIXED-RUN"
        )
    )

    assert reconciliation[
        "tickets_started"
    ] == 3
    assert reconciliation[
        "tickets_completed"
    ] == 3
    assert reconciliation[
        "terminal_outcomes"
    ] == 3


def test_relative_paths_resolve_from_project_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)
    install_fake_application(
        monkeypatch,
        settings,
    )

    input_path = (
        settings.data_dir / "relative.json"
    )

    write_ticket_file(
        input_path,
        [valid_ticket()],
    )

    report = run_ticket_file(
        input_path="data/relative.json",
        output_path=(
            "evaluation/results/relative-output.json"
        ),
        settings=settings,
        run_id="RELATIVE-RUN",
    )

    expected_output = (
        settings.project_root
        / "evaluation"
        / "results"
        / "relative-output.json"
    )

    assert expected_output.exists()
    assert report["input_path"] == str(
        input_path.resolve()
    )


def test_output_directory_is_created(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path
        / "missing"
        / "nested"
        / "report.json"
    )

    assert not output_path.parent.exists()

    write_json_report(
        output_path=output_path,
        report={"status": "ok"},
        pretty=True,
    )

    assert output_path.exists()
    assert json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    ) == {
        "status": "ok"
    }


def test_atomic_writer_removes_temporary_file(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "report.json"

    write_json_report(
        output_path=output_path,
        report={"complete": True},
    )

    temporary_path = output_path.with_suffix(
        ".json.tmp"
    )

    assert output_path.exists()
    assert not temporary_path.exists()


def test_existing_output_is_replaced(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "report.json"

    output_path.write_text(
        '{"old": true}',
        encoding="utf-8",
    )

    write_json_report(
        output_path=output_path,
        report={"new": True},
    )

    assert json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    ) == {
        "new": True
    }


def test_unique_failure_id_avoids_collision() -> None:
    failure = IngestionFailure(
        record_index=7,
        ticket_id="DUPLICATE",
        error="Invalid record.",
    )

    result = unique_failure_ticket_id(
        failure,
        used_ticket_ids={"DUPLICATE"},
    )

    assert result == (
        "DUPLICATE__record_7"
    )


def test_unique_failure_id_handles_second_collision() -> None:
    failure = IngestionFailure(
        record_index=7,
        ticket_id="DUPLICATE",
        error="Invalid record.",
    )

    result = unique_failure_ticket_id(
        failure,
        used_ticket_ids={
            "DUPLICATE",
            "DUPLICATE__record_7",
        },
    )

    assert result == (
        "DUPLICATE__record_7_2"
    )


def test_resolve_project_path_handles_relative_path(
    tmp_path: Path,
) -> None:
    resolved = resolve_project_path(
        "data/tickets.json",
        project_root=tmp_path,
    )

    assert resolved == (
        tmp_path / "data" / "tickets.json"
    ).resolve()


def test_resolve_project_path_preserves_absolute_path(
    tmp_path: Path,
) -> None:
    absolute_path = (
        tmp_path / "external.json"
    ).resolve()

    resolved = resolve_project_path(
        absolute_path,
        project_root=tmp_path / "unused",
    )

    assert resolved == absolute_path


def test_invalid_top_k_is_rejected(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    with pytest.raises(
        ValueError,
        match="retrieval_top_k",
    ):
        run_ticket_file(
            input_path="tickets.json",
            output_path="results.json",
            settings=settings,
            retrieval_top_k=0,
        )


def test_cli_returns_zero_for_success(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    settings = make_settings(tmp_path)

    monkeypatch.setattr(
        run_module,
        "get_settings",
        lambda: settings,
    )

    fake_report = {
        "run_id": "CLI-RUN",
        "summary": {
            "total_input_records": 1,
            "auto_responded": 1,
            "escalated": 0,
            "blocked": 0,
        },
    }

    monkeypatch.setattr(
        run_module,
        "run_ticket_file",
        lambda **kwargs: fake_report,
    )

    exit_code = main(
        [
            "--input",
            "data/tickets.json",
            "--output",
            "evaluation/results/report.json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Run ID: CLI-RUN" in captured.out
    assert "Auto-responded: 1" in captured.out


def test_cli_returns_one_for_expected_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    settings = make_settings(tmp_path)

    monkeypatch.setattr(
        run_module,
        "get_settings",
        lambda: settings,
    )

    def fail_run(**kwargs):
        raise ValueError(
            "invalid runner configuration"
        )

    monkeypatch.setattr(
        run_module,
        "run_ticket_file",
        fail_run,
    )

    exit_code = main(
        [
            "--input",
            "data/tickets.json",
            "--output",
            "evaluation/results/report.json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Run failed:" in captured.err
    assert (
        "invalid runner configuration"
        in captured.err
    )


def test_cli_returns_two_for_unexpected_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    settings = make_settings(tmp_path)

    monkeypatch.setattr(
        run_module,
        "get_settings",
        lambda: settings,
    )

    def fail_unexpectedly(**kwargs):
        raise RuntimeError(
            "unexpected infrastructure problem"
        )

    monkeypatch.setattr(
        run_module,
        "run_ticket_file",
        fail_unexpectedly,
    )

    exit_code = main(
        [
            "--input",
            "data/tickets.json",
            "--output",
            "evaluation/results/report.json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Run failed unexpectedly" in captured.err
    assert "RuntimeError" in captured.err