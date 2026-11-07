"""Application construction and dependency composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .classify import TicketClassifier
from .config import (
    Settings,
    ensure_runtime_directories,
    get_settings,
)
from .generate import (
    ResponseGenerator,
    create_openrouter_generator,
)
from .guardrails import GuardrailEngine
from .ingest import normalize_ticket
from .logging_store import DecisionStore
from .models import NormalizedTicket
from .retrieve import (
    ChromaRetriever,
    IndexBuildReport,
    Passage,
    load_and_chunk_corpus,
)
from .route import RoutingPolicy
from .workflow import (
    CustomerSuccessWorkflow,
    WorkflowResult,
)


DEFAULT_CLASSIFIER_ARTIFACT = (
    Path("artifacts") / "ticket_classifier.joblib"
)

DEFAULT_DOCUMENTATION_FILE = (
    Path("data") / "documentation.json"
)


class ApplicationBuildError(RuntimeError):
    """Raised when the complete application cannot be constructed."""


@dataclass(frozen=True)
class ApplicationComponents:
    """Constructed application and its reusable dependencies."""

    settings: Settings
    workflow: CustomerSuccessWorkflow

    classifier: TicketClassifier
    retriever: ChromaRetriever
    generator: ResponseGenerator
    guardrail_engine: GuardrailEngine
    routing_policy: RoutingPolicy
    decision_store: DecisionStore

    documentation_passages: tuple[Passage, ...]
    index_report: IndexBuildReport | None
    index_status: Mapping[str, Any]

    classifier_artifact_path: Path
    documentation_path: Path

    def process_ticket(
        self,
        ticket: NormalizedTicket,
        *,
        run_id: str | None = None,
    ) -> WorkflowResult:
        """Process an already-normalized ticket."""

        return self.workflow.process(
            ticket,
            run_id=run_id,
        )

    def process_raw_ticket(
        self,
        record: Mapping[str, Any],
        *,
        run_id: str | None = None,
    ) -> WorkflowResult:
        """Normalize and process one raw ticket dictionary."""

        ticket = normalize_ticket(record)

        return self.process_ticket(
            ticket,
            run_id=run_id,
        )

    def process_batch(
        self,
        tickets: Sequence[NormalizedTicket],
        *,
        run_id: str | None = None,
    ) -> list[WorkflowResult]:
        """Process an already-normalized ticket batch."""

        return self.workflow.process_batch(
            tickets,
            run_id=run_id,
        )

    def startup_summary(self) -> dict[str, Any]:
        """
        Return startup information that is safe to print.

        The OpenRouter API key is never included.
        """

        index_summary = dict(self.index_status)

        if self.index_report is not None:
            index_summary["rebuilt"] = (
                self.index_report.rebuilt
            )

        return {
            "classifier_artifact": str(
                self.classifier_artifact_path
            ),
            "documentation_path": str(
                self.documentation_path
            ),
            "documentation_passage_count": int(
                self.index_status.get(
                    "passage_count",
                    len(self.documentation_passages),
                )
            ),
            "index": index_summary,
            "configuration": (
                self.settings.safe_summary()
            ),
        }


def build_application(
    *,
    settings: Settings | None = None,
    classifier_artifact_path: str | Path | None = None,
    documentation_path: str | Path | None = None,
    generator: ResponseGenerator | None = None,
    build_index: bool = False,
    retrieval_top_k: int | None = None,
    chunk_max_characters: int | None = None,
    chunk_overlap_characters: int | None = None,
    minimum_grounding_overlap: float | None = None,
) -> ApplicationComponents:
    """
    Construct the production application.

    Supplying a generator is useful for offline tests. When generator is not
    supplied, a real OpenRouter generator is created from Settings.
    """

    current_settings = settings or get_settings()

    resolved_retrieval_top_k = (
        current_settings.retrieval_top_k
        if retrieval_top_k is None
        else retrieval_top_k
    )
    resolved_chunk_max_characters = (
        current_settings.chunk_max_characters
        if chunk_max_characters is None
        else chunk_max_characters
    )
    resolved_chunk_overlap_characters = (
        current_settings.chunk_overlap_characters
        if chunk_overlap_characters is None
        else chunk_overlap_characters
    )
    resolved_minimum_grounding_overlap = (
        current_settings.minimum_grounding_overlap
        if minimum_grounding_overlap is None
        else minimum_grounding_overlap
    )

    resolved_classifier_path = _resolve_path(
        classifier_artifact_path,
        default=(
            current_settings.project_root
            / DEFAULT_CLASSIFIER_ARTIFACT
        ),
        project_root=current_settings.project_root,
    )

    resolved_documentation_path = _resolve_path(
        documentation_path,
        default=(
            current_settings.project_root
            / DEFAULT_DOCUMENTATION_FILE
        ),
        project_root=current_settings.project_root,
    )

    ensure_runtime_directories(
        current_settings
    )

    if not resolved_classifier_path.exists():
        raise ApplicationBuildError(
            (
                "Classifier artifact does not exist: "
                f"{resolved_classifier_path}. "
                "Train it first with "
                "'python -m src.classify'."
            )
        )

    try:
        classifier = TicketClassifier.load(
            resolved_classifier_path
        )
    except Exception as exc:
        raise ApplicationBuildError(
            (
                "Could not load the classifier artifact: "
                f"{resolved_classifier_path}. "
                f"Cause: {type(exc).__name__}"
            )
        ) from exc

    retriever = ChromaRetriever(
        chroma_path=current_settings.chroma_path,
        embedding_model_name=(
            current_settings.embedding_model
        ),
    )

    passages: list[Passage] = []
    index_report: IndexBuildReport | None = None

    if build_index:
        if not resolved_documentation_path.exists():
            raise ApplicationBuildError(
                (
                    "Documentation corpus does not exist: "
                    f"{resolved_documentation_path}."
                )
            )

        try:
            passages = load_and_chunk_corpus(
                resolved_documentation_path,
                max_chars=resolved_chunk_max_characters,
                overlap_chars=(
                    resolved_chunk_overlap_characters
                ),
            )
        except Exception as exc:
            raise ApplicationBuildError(
                (
                    "Could not load and chunk the documentation "
                    f"corpus: {resolved_documentation_path}. "
                    f"Cause: {type(exc).__name__}"
                )
            ) from exc

        if not passages:
            raise ApplicationBuildError(
                "Documentation produced no retrievable passages."
            )

        try:
            index_report = retriever.build_index(
                passages
            )
        except Exception as exc:
            raise ApplicationBuildError(
                (
                    "Could not build the documentation vector "
                    "index. "
                    f"Cause: {type(exc).__name__}: {exc}"
                )
            ) from exc

    try:
        index_status = retriever.index_status()
    except Exception as exc:
        raise ApplicationBuildError(
            "Could not inspect the documentation index. "
            f"Cause: {type(exc).__name__}: {exc}"
        ) from exc

    if not index_status.get("ready"):
        raise ApplicationBuildError(
            "Documentation index is not ready. Run "
            "'python -m src.build --rebuild-index' first."
        )

    resolved_generator = generator

    if resolved_generator is None:
        try:
            api_key = (
                current_settings.require_api_key()
            )

            resolved_generator = (
                create_openrouter_generator(
                    api_key=api_key,
                    model=current_settings.model_name,
                    timeout_seconds=(
                        current_settings
                        .request_timeout_seconds
                    ),
                    max_retries=(
                        current_settings.max_retries
                    ),
                )
            )
        except Exception as exc:
            raise ApplicationBuildError(
                (
                    "Could not configure response generation. "
                    f"Cause: {type(exc).__name__}: {exc}"
                )
            ) from exc

    guardrail_engine = GuardrailEngine(
        minimum_grounding_overlap=(
            resolved_minimum_grounding_overlap
        )
    )

    routing_policy = RoutingPolicy(
        confidence_threshold=(
            current_settings.confidence_threshold
        ),
        retrieval_threshold=(
            current_settings.retrieval_threshold
        ),
        kill_switch=current_settings.kill_switch,
    )

    decision_store = DecisionStore(
        current_settings.database_path
    )

    workflow = CustomerSuccessWorkflow(
        classifier=classifier,
        retriever=retriever,
        generator=resolved_generator,
        guardrail_engine=guardrail_engine,
        routing_policy=routing_policy,
        decision_store=decision_store,
        retrieval_top_k=resolved_retrieval_top_k,
    )

    return ApplicationComponents(
        settings=current_settings,
        workflow=workflow,
        classifier=classifier,
        retriever=retriever,
        generator=resolved_generator,
        guardrail_engine=guardrail_engine,
        routing_policy=routing_policy,
        decision_store=decision_store,
        documentation_passages=tuple(passages),
        index_report=index_report,
        index_status=index_status,
        classifier_artifact_path=(
            resolved_classifier_path
        ),
        documentation_path=(
            resolved_documentation_path
        ),
    )


def _resolve_path(
    supplied_path: str | Path | None,
    *,
    default: Path,
    project_root: Path,
) -> Path:
    """Resolve optional relative paths against the project root."""

    if supplied_path is None:
        path = default
    else:
        path = Path(supplied_path).expanduser()

        if not path.is_absolute():
            path = project_root / path

    return path.resolve()
