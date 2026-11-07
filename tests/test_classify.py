"""Tests for local ticket classification."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.classify import (
    CLASSIFIER_ARTIFACT_VERSION,
    ClassificationAlternative,
    ClassifierError,
    LabelPrediction,
    TicketClassifier,
    fallback_classification,
    train_classifier,
    train_classifier_from_file,
)
from src.ingest import normalize_ticket
from src.models import NormalizedTicket


class FakeProbabilityModel:
    """Small deterministic replacement for a scikit-learn model."""

    def __init__(
        self,
        classes: list[str],
        probabilities: list[float],
    ) -> None:
        self.classes_ = np.asarray(classes)
        self._probabilities = np.asarray(
            [probabilities],
            dtype=float,
        )

    def predict_proba(
        self,
        texts: list[str],
    ) -> np.ndarray:
        assert len(texts) == 1
        return self._probabilities


class FailingProbabilityModel:
    """Model used to verify safe inference fallback."""

    classes_ = np.asarray(["one", "two"])

    def predict_proba(
        self,
        texts: list[str],
    ) -> np.ndarray:
        raise RuntimeError("Simulated model failure")


class EmptyProbabilityModel:
    """Model that incorrectly returns no classes."""

    classes_ = np.asarray([])

    def predict_proba(
        self,
        texts: list[str],
    ) -> np.ndarray:
        return np.asarray([[]])


def make_ticket(
    *,
    text: str = (
        "Subject: Login problem\n\n"
        "Message: I cannot sign in."
    ),
    labels: dict | None = None,
) -> NormalizedTicket:
    """Create a normalized ticket for classifier tests."""

    return NormalizedTicket(
        ticket_id="TEST-001",
        channel="email",
        subject="Login problem",
        body="I cannot sign in.",
        text=text,
        labels=labels,
    )


def make_fake_classifier() -> TicketClassifier:
    """Create a deterministic classifier."""

    intent_model = FakeProbabilityModel(
        classes=[
            "account_access",
            "authentication_failure",
            "billing_query",
        ],
        probabilities=[
            0.10,
            0.80,
            0.10,
        ],
    )

    urgency_model = FakeProbabilityModel(
        classes=[
            "high",
            "low",
            "medium",
        ],
        probabilities=[
            0.70,
            0.10,
            0.20,
        ],
    )

    return TicketClassifier(
        intent_model=intent_model,
        urgency_model=urgency_model,
        model_version="test-version",
    )


def make_training_tickets() -> list[NormalizedTicket]:
    """
    Create enough labelled examples for two-fold calibration.

    This is intentionally small and is not intended to measure quality.
    """

    tickets: list[NormalizedTicket] = []

    for index in range(6):
        tickets.append(
            normalize_ticket(
                {
                    "ticket_id": f"AUTH-{index}",
                    "channel": "email",
                    "subject": "Cannot sign in",
                    "body": (
                        "My password and credentials are "
                        "rejected when I log in."
                    ),
                    "labels": {
                        "intent": (
                            "authentication_failure"
                        ),
                        "urgency": "high",
                        "expected_route": (
                            "auto_respond"
                        ),
                        "answerable_from_docs": True,
                        "expected_doc_ids": [
                            "DOC-AUTH-001"
                        ],
                        "must_not_auto_respond": False,
                    },
                }
            )
        )

    for index in range(6):
        tickets.append(
            normalize_ticket(
                {
                    "ticket_id": f"BILL-{index}",
                    "channel": "forum",
                    "subject": "Billing question",
                    "body": (
                        "I have a question about the amount "
                        "shown on my monthly invoice."
                    ),
                    "labels": {
                        "intent": "billing_query",
                        "urgency": "low",
                        "expected_route": (
                            "auto_respond"
                        ),
                        "answerable_from_docs": True,
                        "expected_doc_ids": [
                            "DOC-BILL-001"
                        ],
                        "must_not_auto_respond": False,
                    },
                }
            )
        )

    return tickets


def test_classifier_returns_intent_and_urgency() -> None:
    classifier = make_fake_classifier()

    result = classifier.classify(
        make_ticket(),
        alternative_count=2,
    )

    assert result.intent.value == (
        "authentication_failure"
    )
    assert result.intent.confidence == 0.80

    assert result.urgency.value == "high"
    assert result.urgency.confidence == 0.70

    assert result.model_name == (
        "tfidf-logistic-regression"
    )
    assert result.model_version == "test-version"
    assert result.fallback_used is False
    assert result.failure_reason is None


def test_alternatives_are_ranked() -> None:
    classifier = make_fake_classifier()

    result = classifier.classify(
        make_ticket(),
        alternative_count=2,
    )

    assert result.intent.alternatives == (
        ClassificationAlternative(
            value="account_access",
            confidence=0.10,
        ),
        ClassificationAlternative(
            value="billing_query",
            confidence=0.10,
        ),
    )

    assert result.urgency.alternatives == (
        ClassificationAlternative(
            value="medium",
            confidence=0.20,
        ),
        ClassificationAlternative(
            value="low",
            confidence=0.10,
        ),
    )


def test_combined_confidence_uses_weaker_prediction() -> None:
    classifier = make_fake_classifier()
    result = classifier.classify(make_ticket())

    assert result.intent.confidence == 0.80
    assert result.urgency.confidence == 0.70
    assert result.confidence == 0.70


def test_zero_alternatives_are_supported() -> None:
    classifier = make_fake_classifier()

    result = classifier.classify(
        make_ticket(),
        alternative_count=0,
    )

    assert result.intent.alternatives == ()
    assert result.urgency.alternatives == ()


def test_negative_alternative_count_is_rejected() -> None:
    classifier = make_fake_classifier()

    with pytest.raises(
        ValueError,
        match="alternative_count",
    ):
        classifier.classify(
            make_ticket(),
            alternative_count=-1,
        )


def test_empty_ticket_uses_fallback() -> None:
    classifier = make_fake_classifier()

    result = classifier.classify(
        make_ticket(text="   ")
    )

    assert result.fallback_used is True
    assert result.intent.value == "unknown"
    assert result.intent.confidence == 0.0
    assert result.urgency.value == "medium"
    assert result.urgency.confidence == 0.0
    assert result.confidence == 0.0
    assert "no classifiable text" in (
        result.failure_reason or ""
    )


def test_inference_failure_uses_fallback() -> None:
    classifier = TicketClassifier(
        intent_model=FailingProbabilityModel(),
        urgency_model=FailingProbabilityModel(),
        model_version="broken-test",
    )

    result = classifier.classify(make_ticket())

    assert result.fallback_used is True
    assert result.confidence == 0.0
    assert result.intent.value == "unknown"
    assert "RuntimeError" in (
        result.failure_reason or ""
    )


def test_empty_model_prediction_uses_fallback() -> None:
    classifier = TicketClassifier(
        intent_model=EmptyProbabilityModel(),
        urgency_model=EmptyProbabilityModel(),
        model_version="empty-test",
    )

    result = classifier.classify(make_ticket())

    assert result.fallback_used is True
    assert result.confidence == 0.0


def test_invalid_probability_uses_fallback() -> None:
    classifier = TicketClassifier(
        intent_model=FakeProbabilityModel(
            classes=["one", "two"],
            probabilities=[1.2, -0.2],
        ),
        urgency_model=FakeProbabilityModel(
            classes=["high", "low"],
            probabilities=[0.5, 0.5],
        ),
        model_version="invalid-test",
    )

    result = classifier.classify(make_ticket())

    assert result.fallback_used is True
    assert result.confidence == 0.0


def test_classification_is_deterministic() -> None:
    classifier = make_fake_classifier()
    ticket = make_ticket()

    first = classifier.classify(ticket)
    second = classifier.classify(ticket)

    assert first == second


def test_prediction_does_not_read_ticket_labels() -> None:
    classifier = make_fake_classifier()

    contradictory_labels = {
        "intent": "security_incident",
        "urgency": "low",
        "expected_route": "escalate",
        "answerable_from_docs": False,
        "expected_doc_ids": [],
        "must_not_auto_respond": True,
    }

    result = classifier.classify(
        make_ticket(
            labels=contradictory_labels
        )
    )

    # The fake model predicts authentication_failure/high,
    # proving that ground-truth labels were not copied.
    assert result.intent.value == (
        "authentication_failure"
    )
    assert result.urgency.value == "high"


def test_fallback_classification_is_explicit() -> None:
    result = fallback_classification(
        model_version="test-v1",
        reason="Simulated failure",
    )

    assert result == result
    assert result.model_name == (
        "classification-fallback"
    )
    assert result.model_version == "test-v1"
    assert result.fallback_used is True
    assert result.failure_reason == "Simulated failure"


def test_training_rejects_unlabelled_tickets(
    tmp_path,
) -> None:
    ticket = make_ticket(labels=None)

    with pytest.raises(
        ClassifierError,
        match="No labelled",
    ):
        train_classifier(
            [ticket],
            artifact_path=(
                tmp_path / "classifier.joblib"
            ),
            calibration_folds=2,
        )


def test_training_requires_two_calibration_folds(
    tmp_path,
) -> None:
    with pytest.raises(
        ClassifierError,
        match="at least 2",
    ):
        train_classifier(
            make_training_tickets(),
            artifact_path=(
                tmp_path / "classifier.joblib"
            ),
            calibration_folds=1,
        )


def test_training_rejects_undersized_classes(
    tmp_path,
) -> None:
    tickets = make_training_tickets()[:7]

    # Six authentication examples and only one billing
    # example cannot support two-fold calibration.
    with pytest.raises(
        ClassifierError,
        match="need at least 2 records",
    ):
        train_classifier(
            tickets,
            artifact_path=(
                tmp_path / "classifier.joblib"
            ),
            calibration_folds=2,
        )


def test_train_save_load_and_predict(
    tmp_path,
) -> None:
    artifact_path = (
        tmp_path
        / "artifacts"
        / "classifier.joblib"
    )

    report = train_classifier(
        make_training_tickets(),
        artifact_path=artifact_path,
        calibration_folds=2,
    )

    assert artifact_path.exists()
    assert report.training_records == 12
    assert report.intent_classes == 2
    assert report.urgency_classes == 2
    assert report.artifact_path == str(
        artifact_path.resolve()
    )

    classifier = TicketClassifier.load(
        artifact_path
    )

    result = classifier.classify(
        make_ticket()
    )

    assert result.fallback_used is False
    assert result.intent.value in {
        "authentication_failure",
        "billing_query",
    }
    assert result.urgency.value in {
        "high",
        "low",
    }
    assert 0.0 <= result.intent.confidence <= 1.0
    assert 0.0 <= result.urgency.confidence <= 1.0


def test_load_rejects_missing_artifact(
    tmp_path,
) -> None:
    with pytest.raises(
        ClassifierError,
        match="does not exist",
    ):
        TicketClassifier.load(
            tmp_path / "missing.joblib"
        )


def test_load_rejects_non_dictionary_artifact(
    tmp_path,
) -> None:
    artifact_path = tmp_path / "invalid.joblib"
    joblib.dump(
        ["not", "a", "dictionary"],
        artifact_path,
    )

    with pytest.raises(
        ClassifierError,
        match="dictionary",
    ):
        TicketClassifier.load(artifact_path)


def test_load_rejects_missing_artifact_keys(
    tmp_path,
) -> None:
    artifact_path = tmp_path / "missing-keys.joblib"

    joblib.dump(
        {
            "artifact_version": (
                CLASSIFIER_ARTIFACT_VERSION
            )
        },
        artifact_path,
    )

    with pytest.raises(
        ClassifierError,
        match="missing",
    ):
        TicketClassifier.load(artifact_path)


def test_load_rejects_wrong_artifact_version(
    tmp_path,
) -> None:
    artifact_path = tmp_path / "wrong-version.joblib"

    joblib.dump(
        {
            "artifact_version": 999,
            "model_version": "test",
            "intent_model": FakeProbabilityModel(
                ["one", "two"],
                [0.5, 0.5],
            ),
            "urgency_model": FakeProbabilityModel(
                ["high", "low"],
                [0.5, 0.5],
            ),
        },
        artifact_path,
    )

    with pytest.raises(
        ClassifierError,
        match="Unsupported",
    ):
        TicketClassifier.load(artifact_path)


def test_training_from_file_rejects_malformed_data(
    tmp_path,
) -> None:
    input_path = tmp_path / "tickets.json"

    input_path.write_text(
        json.dumps(
            [
                {
                    "ticket_id": "BAD-001",
                    "channel": "telephone",
                    "body": "Invalid channel",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ClassifierError,
        match="malformed",
    ):
        train_classifier_from_file(
            input_path,
            artifact_path=(
                tmp_path / "classifier.joblib"
            ),
            calibration_folds=2,
        )