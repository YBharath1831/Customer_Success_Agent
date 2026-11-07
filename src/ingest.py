"""Ticket loading and normalisation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.models import IncomingTicket, NormalizedTicket


class TicketFileError(Exception):
    """Raised when an entire ticket file cannot be loaded."""


class TicketIngestionError(Exception):
    """Raised when one ticket cannot be normalised."""

    def __init__(self, ticket_id: str, message: str) -> None:
        super().__init__(message)
        self.ticket_id = ticket_id
        self.message = message


@dataclass(frozen=True)
class IngestionFailure:
    """A ticket that could not be normalised."""

    record_index: int
    ticket_id: str
    error: str


@dataclass
class BatchIngestionResult:
    """Result of normalising a collection of ticket records."""

    tickets: list[NormalizedTicket] = field(default_factory=list)
    failures: list[IngestionFailure] = field(default_factory=list)

    @property
    def total_records(self) -> int:
        return len(self.tickets) + len(self.failures)

    @property
    def successful_records(self) -> int:
        return len(self.tickets)

    @property
    def failed_records(self) -> int:
        return len(self.failures)


def read_ticket_file(input_path: str | Path) -> list[Any]:
    """
    Read ticket records from a JSON file.

    File-level errors are fatal because the evaluation harness cannot continue
    if the input file itself is missing, unreadable, or invalid JSON.
    """

    path = Path(input_path).expanduser().resolve()

    if not path.exists():
        raise TicketFileError(f"Ticket file does not exist: {path}")

    if not path.is_file():
        raise TicketFileError(f"Ticket path is not a file: {path}")

    try:
        # utf-8-sig also accepts ordinary UTF-8 while handling a possible BOM.
        raw_text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise TicketFileError(
            f"Could not read ticket file: {path}"
        ) from exc

    try:
        records = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise TicketFileError(
            f"Ticket file contains invalid JSON at "
            f"line {exc.lineno}, column {exc.colno}: {path}"
        ) from exc

    if not isinstance(records, list):
        raise TicketFileError(
            "The ticket file must contain a JSON array of ticket objects."
        )

    return records


def fallback_ticket_id(
    raw_ticket: Any,
    record_index: int | None = None,
) -> str:
    """
    Produce an identifier for malformed records.

    This identifier allows a malformed ticket to be logged rather than
    silently discarded.
    """

    if isinstance(raw_ticket, Mapping):
        supplied_id = raw_ticket.get("ticket_id")

        if isinstance(supplied_id, str) and supplied_id.strip():
            return supplied_id.strip()

    if record_index is not None:
        return f"MALFORMED-{record_index:04d}"

    return "MALFORMED-UNKNOWN"


def build_combined_text(subject: str, body: str) -> str:
    """
    Build classifier/retrieval text while retaining original fields separately.

    Chat tickets commonly have no subject, so empty sections are omitted.
    """

    parts: list[str] = []

    if subject.strip():
        parts.append(f"Subject: {subject.strip()}")

    if body.strip():
        parts.append(f"Message: {body.strip()}")

    return "\n\n".join(parts)


def normalize_ticket(
    raw_ticket: Mapping[str, Any] | IncomingTicket,
    *,
    record_index: int | None = None,
) -> NormalizedTicket:
    """
    Convert an incoming ticket into the shared internal representation.

    The same function handles email, chat, documentation comments, and forum
    tickets because the supplied dataset already exposes a common outer
    schema. Channel-specific differences, such as chat having no subject,
    are normalised here.
    """

    if isinstance(raw_ticket, IncomingTicket):
        validated = raw_ticket
    else:
        if not isinstance(raw_ticket, Mapping):
            ticket_id = fallback_ticket_id(raw_ticket, record_index)

            raise TicketIngestionError(
                ticket_id=ticket_id,
                message="Ticket record must be a JSON object.",
            )

        candidate = dict(raw_ticket)

        # Missing or null subject/body values are recoverable.
        if candidate.get("subject") is None:
            candidate["subject"] = ""

        if candidate.get("body") is None:
            candidate["body"] = ""

        try:
            validated = IncomingTicket.model_validate(candidate)
        except ValidationError as exc:
            ticket_id = fallback_ticket_id(candidate, record_index)

            raise TicketIngestionError(
                ticket_id=ticket_id,
                message=_format_validation_error(exc),
            ) from exc

    ticket_id = validated.ticket_id.strip()

    if not ticket_id:
        ticket_id = fallback_ticket_id(
            validated.model_dump(),
            record_index,
        )

        raise TicketIngestionError(
            ticket_id=ticket_id,
            message="ticket_id cannot be empty or whitespace.",
        )

    combined_text = build_combined_text(
        subject=validated.subject,
        body=validated.body,
    )

    return NormalizedTicket(
        ticket_id=ticket_id,
        channel=validated.channel,
        subject=validated.subject,
        body=validated.body,
        text=combined_text,
        received_at=validated.received_at,
        customer_id=validated.customer_id,
        customer_name=validated.customer_name,
        customer_tier=validated.customer_tier,
        customer_region=validated.customer_region,
        language_fluency=validated.language_fluency,
        labels=validated.labels,
        history=validated.history,
        metadata={
            "empty_subject": not bool(validated.subject.strip()),
            "empty_body": not bool(validated.body.strip()),
        },
    )


def normalize_records(records: Iterable[Any]) -> BatchIngestionResult:
    """
    Normalise multiple records without stopping at the first bad ticket.

    Failures are returned explicitly. The evaluation harness will later turn
    each failure into a logged escalation, ensuring no ticket is dropped.
    """

    result = BatchIngestionResult()

    for record_index, record in enumerate(records, start=1):
        try:
            ticket = normalize_ticket(
                record,
                record_index=record_index,
            )
        except TicketIngestionError as exc:
            result.failures.append(
                IngestionFailure(
                    record_index=record_index,
                    ticket_id=exc.ticket_id,
                    error=exc.message,
                )
            )
        else:
            result.tickets.append(ticket)

    return result


def load_and_normalize_tickets(
    input_path: str | Path,
) -> BatchIngestionResult:
    """Load and normalise all records from a ticket JSON file."""

    records = read_ticket_file(input_path)
    return normalize_records(records)


def _format_validation_error(error: ValidationError) -> str:
    """Convert Pydantic errors into a concise, non-sensitive message."""

    messages: list[str] = []

    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        message = item["msg"]
        messages.append(f"{location}: {message}")

    return "; ".join(messages)