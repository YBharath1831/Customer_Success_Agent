"""Shared data models for the customer-success pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Channel(str, Enum):
    """Supported ticket channels from the supplied dataset."""

    EMAIL = "email"
    CHAT = "chat"
    DOCS_COMMENT = "docs_comment"
    FORUM = "forum"


class Urgency(str, Enum):
    """Supported urgency labels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExpectedRoute(str, Enum):
    """Expected routing values available in labelled datasets."""

    AUTO_RESPOND = "auto_respond"
    ESCALATE = "escalate"


class TicketLabels(BaseModel):
    """
    Ground-truth labels supplied with development and evaluation data.

    These fields are used for training and evaluation. Production tickets
    may not contain a labels object.
    """

    model_config = ConfigDict(extra="allow")

    intent: str
    urgency: Urgency
    expected_route: ExpectedRoute
    answerable_from_docs: bool
    expected_doc_ids: list[str] = Field(default_factory=list)
    must_not_auto_respond: bool = False


class TicketHistory(BaseModel):
    """
    Historical human-support outcomes.

    This is baseline information, not something the application should try
    to reproduce.
    """

    model_config = ConfigDict(extra="allow")

    first_contact_resolution: bool | None = None
    resolution_time_minutes: float | None = Field(default=None, ge=0)
    csat_rating: float | None = Field(default=None, ge=0, le=5)
    escalated: bool | None = None
    repeat_contact: bool | None = None


class IncomingTicket(BaseModel):
    """
    Ticket structure accepted from JSON files or the API.

    Labels and history are optional so the same model works for labelled
    evaluation tickets and unlabelled real-world tickets.
    """

    model_config = ConfigDict(extra="allow")

    ticket_id: str = Field(min_length=1)
    channel: Channel
    subject: str = ""
    body: str = ""
    received_at: datetime | None = None

    customer_id: str | None = None
    customer_name: str | None = None
    customer_tier: str | None = None
    customer_region: str | None = None
    language_fluency: str | None = None

    labels: TicketLabels | None = None
    history: TicketHistory | None = None


class NormalizedTicket(BaseModel):
    """
    Internal ticket representation used by downstream components.
    All four input channels are converted to this structure by ingest.py.
    """

    model_config = ConfigDict(extra="allow")

    ticket_id: str
    channel: Channel
    subject: str
    body: str
    text: str
    received_at: datetime | None = None

    customer_id: str | None = None
    customer_name: str | None = None
    customer_tier: str | None = None
    customer_region: str | None = None
    language_fluency: str | None = None

    labels: TicketLabels | None = None
    history: TicketHistory | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)