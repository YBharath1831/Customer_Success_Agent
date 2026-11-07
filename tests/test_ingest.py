"""Tests for ticket loading and normalisation."""

import json

import pytest

from src.ingest import (
    TicketFileError,
    build_combined_text,
    fallback_ticket_id,
    load_and_normalize_tickets,
    normalize_records,
    normalize_ticket,
    read_ticket_file,
)
from src.models import Channel


@pytest.mark.parametrize(
    ("channel", "subject", "body"),
    [
        ("email", "Login problem", "I cannot sign in."),
        ("chat", "", "My deployment failed."),
        (
            "docs_comment",
            "API documentation",
            "The authentication example is unclear.",
        ),
        (
            "forum",
            "Rate limits",
            "How should exponential backoff work?",
        ),
    ],
)
def test_normalize_ticket_supports_all_channels(
    channel: str,
    subject: str,
    body: str,
) -> None:
    raw_ticket = {
        "ticket_id": f"TEST-{channel}",
        "channel": channel,
        "subject": subject,
        "body": body,
        "received_at": "2026-10-31T10:00:00Z",
    }

    ticket = normalize_ticket(raw_ticket)

    assert ticket.ticket_id == f"TEST-{channel}"
    assert ticket.channel.value == channel
    assert ticket.subject == subject
    assert ticket.body == body

    if subject:
        assert f"Subject: {subject}" in ticket.text
    else:
        assert "Subject:" not in ticket.text

    assert f"Message: {body}" in ticket.text


def test_normalize_ticket_returns_channel_enum() -> None:
    ticket = normalize_ticket(
        {
            "ticket_id": "EMAIL-001",
            "channel": "email",
            "subject": "Question",
            "body": "How does this work?",
        }
    )

    assert ticket.channel is Channel.EMAIL


def test_original_subject_and_body_are_preserved() -> None:
    ticket = normalize_ticket(
        {
            "ticket_id": "SPACE-001",
            "channel": "email",
            "subject": "  Login problem  ",
            "body": "  I cannot log in.  ",
        }
    )

    # Original values remain available for auditing.
    assert ticket.subject == "  Login problem  "
    assert ticket.body == "  I cannot log in.  "

    # The derived text is cleaned for classification and retrieval.
    assert ticket.text == (
        "Subject: Login problem\n\n"
        "Message: I cannot log in."
    )


def test_missing_subject_is_allowed() -> None:
    ticket = normalize_ticket(
        {
            "ticket_id": "CHAT-001",
            "channel": "chat",
            "body": "I need help.",
        }
    )

    assert ticket.subject == ""
    assert ticket.body == "I need help."
    assert ticket.text == "Message: I need help."
    assert ticket.metadata["empty_subject"] is True


def test_null_subject_and_body_are_converted_to_empty_strings() -> None:
    ticket = normalize_ticket(
        {
            "ticket_id": "EMPTY-001",
            "channel": "chat",
            "subject": None,
            "body": None,
        }
    )

    assert ticket.subject == ""
    assert ticket.body == ""
    assert ticket.text == ""
    assert ticket.metadata["empty_subject"] is True
    assert ticket.metadata["empty_body"] is True


def test_labels_and_history_are_preserved() -> None:
    ticket = normalize_ticket(
        {
            "ticket_id": "LABEL-001",
            "channel": "email",
            "subject": "Cannot sign in",
            "body": "My credentials are rejected.",
            "labels": {
                "intent": "authentication_failure",
                "urgency": "high",
                "expected_route": "auto_respond",
                "answerable_from_docs": True,
                "expected_doc_ids": ["DOC-AUTH-001"],
                "must_not_auto_respond": False,
            },
            "history": {
                "first_contact_resolution": True,
                "resolution_time_minutes": 12,
                "csat_rating": 4,
                "escalated": False,
                "repeat_contact": False,
            },
        }
    )

    assert ticket.labels is not None
    assert ticket.labels.intent == "authentication_failure"
    assert ticket.labels.expected_doc_ids == ["DOC-AUTH-001"]

    assert ticket.history is not None
    assert ticket.history.first_contact_resolution is True
    assert ticket.history.csat_rating == 4


def test_unknown_channel_is_reported_as_failure() -> None:
    result = normalize_records(
        [
            {
                "ticket_id": "BAD-CHANNEL",
                "channel": "telephone",
                "body": "Please help.",
            }
        ]
    )

    assert result.successful_records == 0
    assert result.failed_records == 1
    assert result.failures[0].ticket_id == "BAD-CHANNEL"
    assert "channel" in result.failures[0].error


def test_non_object_record_is_reported_as_failure() -> None:
    result = normalize_records(["this is not a ticket"])

    assert result.total_records == 1
    assert result.successful_records == 0
    assert result.failed_records == 1
    assert result.failures[0].ticket_id == "MALFORMED-0001"
    assert result.failures[0].error == (
        "Ticket record must be a JSON object."
    )


def test_batch_continues_after_malformed_record() -> None:
    records = [
        {
            "ticket_id": "GOOD-001",
            "channel": "chat",
            "body": "A valid ticket.",
        },
        {
            "ticket_id": "BAD-001",
            "channel": "unknown",
            "body": "An invalid ticket.",
        },
        {
            "ticket_id": "GOOD-002",
            "channel": "forum",
            "body": "Another valid ticket.",
        },
    ]

    result = normalize_records(records)

    assert result.total_records == 3
    assert result.successful_records == 2
    assert result.failed_records == 1

    assert [ticket.ticket_id for ticket in result.tickets] == [
        "GOOD-001",
        "GOOD-002",
    ]
    assert result.failures[0].ticket_id == "BAD-001"


def test_fallback_ticket_id_uses_supplied_id() -> None:
    assert fallback_ticket_id(
        {"ticket_id": "  EXISTING-001  "},
        record_index=7,
    ) == "EXISTING-001"


def test_fallback_ticket_id_uses_record_index() -> None:
    assert fallback_ticket_id(
        {"channel": "chat"},
        record_index=7,
    ) == "MALFORMED-0007"


def test_build_combined_text_handles_empty_values() -> None:
    assert build_combined_text("", "") == ""
    assert build_combined_text("", "Help") == "Message: Help"
    assert build_combined_text("Problem", "") == "Subject: Problem"


def test_read_ticket_file_reads_json_array(tmp_path) -> None:
    input_path = tmp_path / "tickets.json"
    records = [
        {
            "ticket_id": "FILE-001",
            "channel": "email",
            "body": "Loaded from a file.",
        }
    ]

    input_path.write_text(
        json.dumps(records),
        encoding="utf-8",
    )

    loaded_records = read_ticket_file(input_path)

    assert loaded_records == records


def test_read_ticket_file_rejects_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(TicketFileError, match="does not exist"):
        read_ticket_file(missing_path)


def test_read_ticket_file_rejects_invalid_json(tmp_path) -> None:
    input_path = tmp_path / "invalid.json"
    input_path.write_text(
        "{not valid JSON",
        encoding="utf-8",
    )

    with pytest.raises(TicketFileError, match="invalid JSON"):
        read_ticket_file(input_path)


def test_read_ticket_file_rejects_non_array_json(tmp_path) -> None:
    input_path = tmp_path / "object.json"
    input_path.write_text(
        json.dumps({"ticket_id": "ONE-TICKET"}),
        encoding="utf-8",
    )

    with pytest.raises(TicketFileError, match="JSON array"):
        read_ticket_file(input_path)


def test_load_and_normalize_tickets(tmp_path) -> None:
    input_path = tmp_path / "mixed-tickets.json"
    records = [
        {
            "ticket_id": "GOOD-001",
            "channel": "email",
            "subject": "Login",
            "body": "Cannot log in.",
        },
        {
            "ticket_id": "BAD-001",
            "channel": "telephone",
            "body": "Unsupported channel.",
        },
    ]

    input_path.write_text(
        json.dumps(records),
        encoding="utf-8",
    )

    result = load_and_normalize_tickets(input_path)

    assert result.total_records == 2
    assert result.successful_records == 1
    assert result.failed_records == 1
    assert result.tickets[0].ticket_id == "GOOD-001"
    assert result.failures[0].ticket_id == "BAD-001"