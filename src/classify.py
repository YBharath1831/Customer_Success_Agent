"""Local intent and urgency classification."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
import numpy as np

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.ingest import load_and_normalize_tickets
from src.models import NormalizedTicket


CLASSIFIER_ARTIFACT_VERSION = 1
DEFAULT_RANDOM_STATE = 42


class ClassifierError(Exception):
    """Raised when classifier training or loading fails."""


@dataclass(frozen=True)
class ClassificationAlternative:
    """One alternative classification considered by the model."""

    value: str
    confidence: float


@dataclass(frozen=True)
class LabelPrediction:
    """Prediction for one classification target."""

    value: str
    confidence: float
    alternatives: tuple[
        ClassificationAlternative,
        ...,
    ] = ()


@dataclass(frozen=True)
class ClassificationResult:
    """Intent and urgency predictions for one ticket."""

    intent: LabelPrediction
    urgency: LabelPrediction

    model_name: str
    model_version: str

    fallback_used: bool = False
    failure_reason: str | None = None

    @property
    def confidence(self) -> float:
        """
        Conservative combined confidence.

        Routing should trust the weaker of the intent and urgency predictions,
        not only the stronger prediction.
        """

        return min(
            self.intent.confidence,
            self.urgency.confidence,
        )


@dataclass(frozen=True)
class ClassifierTrainingReport:
    """Summary returned after training."""

    training_records: int
    intent_classes: int
    urgency_classes: int
    intent_distribution: dict[str, int]
    urgency_distribution: dict[str, int]
    artifact_path: str
    trained_at: str


class TicketClassifier:
    """Loaded local classifier used during ticket processing."""

    def __init__(
        self,
        *,
        intent_model: Any,
        urgency_model: Any,
        model_version: str,
    ) -> None:
        self.intent_model = intent_model
        self.urgency_model = urgency_model
        self.model_version = model_version

    @classmethod
    def load(
        cls,
        artifact_path: str | Path,
    ) -> "TicketClassifier":
        """
        Load a trusted classifier artifact.

        Joblib files can execute Python during loading. Only load an artifact
        created by this project from a trusted location.
        """

        path = Path(
            artifact_path
        ).expanduser().resolve()

        if not path.exists():
            raise ClassifierError(
                f"Classifier artifact does not exist: {path}"
            )

        if not path.is_file():
            raise ClassifierError(
                f"Classifier artifact path is not a file: {path}"
            )

        try:
            artifact = joblib.load(path)
        except Exception as exc:
            raise ClassifierError(
                f"Could not load classifier artifact: {path}"
            ) from exc

        if not isinstance(artifact, dict):
            raise ClassifierError(
                "Classifier artifact must contain a dictionary."
            )

        required_keys = {
            "artifact_version",
            "model_version",
            "intent_model",
            "urgency_model",
        }

        missing_keys = required_keys - set(artifact)

        if missing_keys:
            raise ClassifierError(
                "Classifier artifact is missing: "
                + ", ".join(sorted(missing_keys))
            )

        if (
            artifact["artifact_version"]
            != CLASSIFIER_ARTIFACT_VERSION
        ):
            raise ClassifierError(
                "Unsupported classifier artifact version: "
                f"{artifact['artifact_version']!r}"
            )

        return cls(
            intent_model=artifact["intent_model"],
            urgency_model=artifact["urgency_model"],
            model_version=artifact["model_version"],
        )

    def classify(
        self,
        ticket: NormalizedTicket,
        *,
        alternative_count: int = 3,
    ) -> ClassificationResult:
        """
        Classify a ticket without using evaluation labels.

        Prediction errors return a defined low-confidence fallback instead of
        crashing the complete evaluation run.
        """

        if alternative_count < 0:
            raise ValueError(
                "alternative_count cannot be negative."
            )

        if not ticket.text.strip():
            return fallback_classification(
                model_version=self.model_version,
                reason="Ticket contains no classifiable text.",
            )

        try:
            intent = _predict_label(
                model=self.intent_model,
                text=ticket.text,
                alternative_count=alternative_count,
            )

            urgency = _predict_label(
                model=self.urgency_model,
                text=ticket.text,
                alternative_count=alternative_count,
            )
        except Exception as exc:
            return fallback_classification(
                model_version=self.model_version,
                reason=(
                    "Classifier inference failed: "
                    f"{type(exc).__name__}"
                ),
            )

        return ClassificationResult(
            intent=intent,
            urgency=urgency,
            model_name="tfidf-logistic-regression",
            model_version=self.model_version,
        )


def train_classifier(
    tickets: Sequence[NormalizedTicket],
    *,
    artifact_path: str | Path,
    calibration_folds: int = 3,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ClassifierTrainingReport:
    """
    Train calibrated intent and urgency models.

    Only labelled development tickets may be supplied to this function.
    """

    if calibration_folds < 2:
        raise ClassifierError(
            "calibration_folds must be at least 2."
        )

    training_tickets = [
        ticket
        for ticket in tickets
        if ticket.labels is not None
        and ticket.text.strip()
    ]

    if not training_tickets:
        raise ClassifierError(
            "No labelled, non-empty training tickets were supplied."
        )

    texts = [
        ticket.text
        for ticket in training_tickets
    ]

    intent_labels = [
        ticket.labels.intent
        for ticket in training_tickets
        if ticket.labels is not None
    ]

    urgency_labels = [
        ticket.labels.urgency.value
        for ticket in training_tickets
        if ticket.labels is not None
    ]

    _validate_training_labels(
        labels=intent_labels,
        target_name="intent",
        calibration_folds=calibration_folds,
    )

    _validate_training_labels(
        labels=urgency_labels,
        target_name="urgency",
        calibration_folds=calibration_folds,
    )

    intent_model = _build_calibrated_model(
        calibration_folds=calibration_folds,
        random_state=random_state,
    )

    urgency_model = _build_calibrated_model(
        calibration_folds=calibration_folds,
        random_state=random_state,
    )

    intent_targets = np.asarray(
        intent_labels,
        dtype=str,
    )

    urgency_targets = np.asarray(
        urgency_labels,
        dtype=str,
    )

    intent_model.fit(
        texts,
        intent_targets,
    )

    urgency_model.fit(
        texts,
        urgency_targets,
    )

    trained_at = datetime.now(
        timezone.utc
    ).isoformat()

    model_version = (
        "tfidf-logreg-"
        + trained_at.replace(
            ":",
            "",
        ).replace(
            "+00:00",
            "Z",
        )
    )

    artifact = {
        "artifact_version": (
            CLASSIFIER_ARTIFACT_VERSION
        ),
        "model_version": model_version,
        "trained_at": trained_at,
        "training_records": len(training_tickets),
        "random_state": random_state,
        "calibration_folds": calibration_folds,
        "intent_distribution": dict(
            sorted(Counter(intent_labels).items())
        ),
        "urgency_distribution": dict(
            sorted(Counter(urgency_labels).items())
        ),
        "intent_model": intent_model,
        "urgency_model": urgency_model,
    }

    output_path = Path(
        artifact_path
    ).expanduser().resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        joblib.dump(
            artifact,
            temporary_path,
        )
        temporary_path.replace(output_path)
    except Exception as exc:
        if temporary_path.exists():
            temporary_path.unlink()

        raise ClassifierError(
            f"Could not save classifier artifact: {output_path}"
        ) from exc

    return ClassifierTrainingReport(
        training_records=len(training_tickets),
        intent_classes=len(
            set(intent_labels)
        ),
        urgency_classes=len(
            set(urgency_labels)
        ),
        intent_distribution=dict(
            sorted(Counter(intent_labels).items())
        ),
        urgency_distribution=dict(
            sorted(Counter(urgency_labels).items())
        ),
        artifact_path=str(output_path),
        trained_at=trained_at,
    )


def train_classifier_from_file(
    input_path: str | Path,
    *,
    artifact_path: str | Path,
    calibration_folds: int = 3,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ClassifierTrainingReport:
    """Load labelled tickets and train both classifiers."""

    ingestion_result = load_and_normalize_tickets(
        input_path
    )

    if ingestion_result.failures:
        first_failure = ingestion_result.failures[0]

        raise ClassifierError(
            "Training data contains malformed tickets. "
            f"First failure: record "
            f"{first_failure.record_index}, "
            f"ticket {first_failure.ticket_id}: "
            f"{first_failure.error}"
        )

    return train_classifier(
        ingestion_result.tickets,
        artifact_path=artifact_path,
        calibration_folds=calibration_folds,
        random_state=random_state,
    )


def fallback_classification(
    *,
    model_version: str,
    reason: str,
) -> ClassificationResult:
    """
    Return a defined result when classification is impossible.

    Urgency defaults to medium for queue prioritisation, but confidence remains
    zero so routing must escalate.
    """

    return ClassificationResult(
        intent=LabelPrediction(
            value="unknown",
            confidence=0.0,
        ),
        urgency=LabelPrediction(
            value="medium",
            confidence=0.0,
        ),
        model_name="classification-fallback",
        model_version=model_version,
        fallback_used=True,
        failure_reason=reason,
    )


def _build_calibrated_model(
    *,
    calibration_folds: int,
    random_state: int,
) -> CalibratedClassifierCV:
    """Build the deterministic text-classification pipeline."""

    features = FeatureUnion(
        [
            (
                "word_features",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    lowercase=True,
                    sublinear_tf=True,
                    max_features=12_000,
                ),
            ),
            (
                "character_features",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    lowercase=True,
                    sublinear_tf=True,
                    max_features=15_000,
                ),
            ),
        ]
    )

    base_classifier = LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
        random_state=random_state,
    )

    pipeline = Pipeline(
        [
            ("features", features),
            ("classifier", base_classifier),
        ]
    )

    return CalibratedClassifierCV(
        estimator=pipeline,
        method="sigmoid",
        cv=calibration_folds,
    )


def _predict_label(
    *,
    model: Any,
    text: str,
    alternative_count: int,
) -> LabelPrediction:
    """Predict one target and retain ranked alternatives."""

    probabilities = model.predict_proba(
        [text]
    )[0]

    classes = model.classes_

    if len(probabilities) != len(classes):
        raise ClassifierError(
            "Classifier returned mismatched classes "
            "and probabilities."
        )

    ranked = sorted(
        (
            (
                str(label),
                float(probability),
            )
            for label, probability in zip(
                classes,
                probabilities,
                strict=True,
            )
        ),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    if not ranked:
        raise ClassifierError(
            "Classifier returned no predictions."
        )

    for _, probability in ranked:
        if (
            not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise ClassifierError(
                "Classifier returned an invalid probability."
            )

    predicted_value, confidence = ranked[0]

    alternatives = tuple(
        ClassificationAlternative(
            value=value,
            confidence=probability,
        )
        for value, probability in ranked[
            1 : alternative_count + 1
        ]
    )

    return LabelPrediction(
        value=predicted_value,
        confidence=confidence,
        alternatives=alternatives,
    )


def _validate_training_labels(
    *,
    labels: list[str],
    target_name: str,
    calibration_folds: int,
) -> None:
    """Ensure every class has enough samples for calibration."""

    distribution = Counter(labels)

    if len(distribution) < 2:
        raise ClassifierError(
            f"{target_name} training requires at "
            "least two classes."
        )

    undersized = {
        label: count
        for label, count in distribution.items()
        if count < calibration_folds
    }

    if undersized:
        formatted = ", ".join(
            f"{label}={count}"
            for label, count in sorted(
                undersized.items()
            )
        )

        raise ClassifierError(
            f"{target_name} classes need at least "
            f"{calibration_folds} records each: "
            f"{formatted}"
        )


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the training command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the local ticket classifier."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to labelled development tickets."
        ),
    )

    parser.add_argument(
        "--output",
        default="./artifacts/ticket_classifier.joblib",
        help=(
            "Path for the trained classifier artifact."
        ),
    )

    parser.add_argument(
        "--calibration-folds",
        type=int,
        default=3,
        help="Number of probability-calibration folds.",
    )

    return parser


def main() -> None:
    """Run classifier training from the command line."""

    parser = _build_argument_parser()
    arguments = parser.parse_args()

    report = train_classifier_from_file(
        arguments.input,
        artifact_path=arguments.output,
        calibration_folds=(
            arguments.calibration_folds
        ),
    )

    print(f"Training records: {report.training_records}")
    print(f"Intent classes: {report.intent_classes}")
    print(f"Urgency classes: {report.urgency_classes}")
    print(f"Artifact: {report.artifact_path}")


if __name__ == "__main__":
    main()