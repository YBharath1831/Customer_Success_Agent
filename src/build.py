"""One-time preparation command for application artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .classify import (
    TicketClassifier,
    train_classifier_from_file,
)
from .config import (
    Settings,
    ensure_runtime_directories,
    get_settings,
)
from .logging_store import DecisionStore
from .retrieve import (
    ChromaRetriever,
    load_and_chunk_corpus,
)


DEFAULT_CLASSIFIER_ARTIFACT = (
    Path("artifacts") / "ticket_classifier.joblib"
)
DEFAULT_DOCUMENTATION_FILE = (
    Path("data") / "documentation.json"
)
DEFAULT_TRAINING_FILE = (
    Path("data") / "development_tickets.json"
)
DEFAULT_MANIFEST_FILE = (
    Path("storage") / "index_manifest.json"
)


class BuildError(RuntimeError):
    """Raised when one-time application preparation fails."""


def utc_now() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def resolve_project_path(
    supplied_path: str | Path | None,
    *,
    default: Path,
    project_root: Path,
) -> Path:
    """Resolve an optional path relative to the project root."""

    path = default if supplied_path is None else Path(supplied_path)
    path = path.expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def prepare_application(
    *,
    settings: Settings | None = None,
    documentation_path: str | Path | None = None,
    training_data_path: str | Path | None = None,
    classifier_artifact_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    rebuild_classifier: bool = False,
    rebuild_index: bool = False,
) -> dict[str, Any]:
    """Validate artifacts and prepare the persistent Chroma index."""

    current_settings = settings or get_settings()
    resolved_documentation_path = resolve_project_path(
        documentation_path,
        default=DEFAULT_DOCUMENTATION_FILE,
        project_root=current_settings.project_root,
    )
    resolved_classifier_path = resolve_project_path(
        classifier_artifact_path,
        default=DEFAULT_CLASSIFIER_ARTIFACT,
        project_root=current_settings.project_root,
    )
    resolved_training_path = resolve_project_path(
        training_data_path,
        default=DEFAULT_TRAINING_FILE,
        project_root=current_settings.project_root,
    )
    resolved_manifest_path = resolve_project_path(
        manifest_path,
        default=DEFAULT_MANIFEST_FILE,
        project_root=current_settings.project_root,
    )

    if not resolved_documentation_path.is_file():
        raise BuildError(
            "Documentation corpus does not exist: "
            f"{resolved_documentation_path}"
        )

    ensure_runtime_directories(current_settings)

    try:
        decision_store = DecisionStore(
            current_settings.database_path
        )
        decision_store.initialize()
    except Exception as exc:
        raise BuildError(
            "Decision database could not be initialized: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    classifier_rebuilt = (
        rebuild_classifier
        or not resolved_classifier_path.is_file()
    )
    training_report = None

    if classifier_rebuilt:
        if not resolved_training_path.is_file():
            raise BuildError(
                "Classifier training data does not exist: "
                f"{resolved_training_path}"
            )

        try:
            training_report = train_classifier_from_file(
                resolved_training_path,
                artifact_path=resolved_classifier_path,
            )
        except Exception as exc:
            raise BuildError(
                "Classifier could not be trained: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    try:
        classifier = TicketClassifier.load(
            resolved_classifier_path
        )
    except Exception as exc:
        raise BuildError(
            "Classifier artifact could not be loaded: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        passages = load_and_chunk_corpus(
            resolved_documentation_path,
            max_chars=(
                current_settings.chunk_max_characters
            ),
            overlap_chars=(
                current_settings.chunk_overlap_characters
            ),
        )
    except Exception as exc:
        raise BuildError(
            "Documentation could not be chunked: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not passages:
        raise BuildError(
            "Documentation produced no retrievable passages."
        )

    if rebuild_index:
        _remove_generated_index(
            chroma_path=current_settings.chroma_path,
            project_root=current_settings.project_root,
        )
        current_settings.chroma_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    retriever = ChromaRetriever(
        chroma_path=current_settings.chroma_path,
        embedding_model_name=(
            current_settings.embedding_model
        ),
    )

    try:
        index_report = retriever.build_index(passages)
    except Exception as exc:
        raise BuildError(
            "Documentation index could not be built: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    manifest = {
        "schema_version": 1,
        "built_at": utc_now(),
        "documentation_path": str(
            resolved_documentation_path
        ),
        "classifier_artifact": str(
            resolved_classifier_path
        ),
        "classifier_training_data": str(
            resolved_training_path
        ),
        "classifier_rebuilt": classifier_rebuilt,
        "classifier_training_records": (
            training_report.training_records
            if training_report is not None
            else None
        ),
        "classifier_model_name": getattr(
            classifier,
            "model_name",
            type(classifier).__name__,
        ),
        "embedding_model": (
            current_settings.embedding_model
        ),
        "chunk_max_characters": (
            current_settings.chunk_max_characters
        ),
        "chunk_overlap_characters": (
            current_settings.chunk_overlap_characters
        ),
        "collection_name": index_report.collection_name,
        "passage_count": index_report.passage_count,
        "corpus_fingerprint": (
            index_report.corpus_fingerprint
        ),
        "index_rebuilt": index_report.rebuilt,
        "chroma_path": index_report.chroma_path,
        "decision_database": str(
            current_settings.database_path
        ),
    }

    resolved_manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    resolved_manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        **manifest,
        "manifest_path": str(resolved_manifest_path),
    }


def _remove_generated_index(
    *,
    chroma_path: Path,
    project_root: Path,
) -> None:
    """Safely remove only the configured generated Chroma directory."""

    resolved_chroma = chroma_path.resolve()
    resolved_project = project_root.resolve()

    if resolved_chroma == resolved_project:
        raise BuildError(
            "CHROMA_PATH cannot be the project root."
        )

    try:
        resolved_chroma.relative_to(resolved_project)
    except ValueError as exc:
        raise BuildError(
            "Refusing to delete CHROMA_PATH outside the project: "
            f"{resolved_chroma}"
        ) from exc

    if resolved_chroma.exists():
        shutil.rmtree(resolved_chroma)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the preparation command parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare reusable classifier and Chroma artifacts. "
            "Run this only when source documents or build settings change."
        )
    )
    parser.add_argument(
        "--documentation",
        default=None,
        help="Optional documentation JSON path.",
    )
    parser.add_argument(
        "--classifier-artifact",
        default=None,
        help="Optional trained classifier artifact path.",
    )
    parser.add_argument(
        "--training-data",
        default=None,
        help="Optional labelled classifier training-data path.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional destination for the build manifest.",
    )
    parser.add_argument(
        "--rebuild-classifier",
        action="store_true",
        help="Retrain and replace the classifier artifact.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Delete and completely rebuild the generated Chroma index.",
    )
    parser.add_argument(
        "--rebuild-all",
        action="store_true",
        help="Retrain the classifier and completely rebuild the index.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run application preparation from the command line."""

    arguments = build_argument_parser().parse_args(argv)

    try:
        report = prepare_application(
            documentation_path=arguments.documentation,
            training_data_path=arguments.training_data,
            classifier_artifact_path=(
                arguments.classifier_artifact
            ),
            manifest_path=arguments.manifest,
            rebuild_classifier=(
                arguments.rebuild_classifier
                or arguments.rebuild_all
            ),
            rebuild_index=(
                arguments.rebuild_index
                or arguments.rebuild_all
            ),
        )
    except Exception as exc:
        print(
            f"Build failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
