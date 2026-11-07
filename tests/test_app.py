"""Tests for application construction and dependency composition."""

from pathlib import Path

import pytest

import src.app as app_module
from src.app import (
    ApplicationBuildError,
    build_application,
)
from src.config import Settings
from src.retrieve import (
    IndexBuildReport,
    Passage,
)


class FakeClassifier:
    """Classifier returned by the fake artifact loader."""

    def classify(self, ticket):
        raise AssertionError(
            "Classification is not expected in app construction tests."
        )


class FakeTicketClassifierLoader:
    """Replacement for TicketClassifier during tests."""

    loaded_paths: list[Path] = []
    error: Exception | None = None

    @classmethod
    def load(cls, path):
        cls.loaded_paths.append(Path(path))

        if cls.error is not None:
            raise cls.error

        return FakeClassifier()


class FakeRetriever:
    """Replacement for ChromaRetriever that does no embedding work."""

    instances = []
    build_error: Exception | None = None
    index_ready: bool = True

    def __init__(
        self,
        *,
        chroma_path,
        embedding_model_name,
        collection_name="cloudserve_docs",
    ):
        self.chroma_path = Path(chroma_path)
        self.embedding_model_name = (
            embedding_model_name
        )
        self.collection_name = collection_name
        self.build_calls = []
        FakeRetriever.instances.append(self)

    def build_index(
        self,
        passages,
        *,
        batch_size=64,
    ):
        self.build_calls.append(
            {
                "passages": list(passages),
                "batch_size": batch_size,
            }
        )

        if self.build_error is not None:
            raise self.build_error

        return IndexBuildReport(
            collection_name=self.collection_name,
            passage_count=len(passages),
            corpus_fingerprint="fake-fingerprint",
            rebuilt=True,
            chroma_path=str(self.chroma_path),
        )

    def search(
        self,
        query_text,
        *,
        top_k=5,
        minimum_relevance=0.0,
    ):
        return []

    def index_status(self):
        return {
            "ready": self.index_ready,
            "collection_name": self.collection_name,
            "passage_count": 1 if self.index_ready else 0,
            "corpus_fingerprint": (
                "fake-fingerprint"
                if self.index_ready
                else None
            ),
            "embedding_model": self.embedding_model_name,
            "chroma_path": str(self.chroma_path),
        }


class FakeGenerator:
    """Offline generator supplied to build_application."""

    def generate(self, **kwargs):
        raise AssertionError(
            "Generation is not expected in app construction tests."
        )


def make_settings(
    tmp_path: Path,
    *,
    api_key: str | None = None,
    kill_switch: bool = False,
    chunk_max_characters: int = 1200,
    chunk_overlap_characters: int = 160,
    retrieval_top_k: int = 5,
    minimum_grounding_overlap: float = 0.15,
) -> Settings:
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    storage_dir = project_root / "storage"
    evaluation_dir = project_root / "evaluation" / "results"
    chroma_path = storage_dir / "chroma"
    database_path = storage_dir / "decisions.db"

    project_root.mkdir(
        parents=True,
        exist_ok=True,
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
        chroma_path=chroma_path,
        database_url=(
            f"sqlite:///{database_path}"
        ),
        database_path=database_path,
        openrouter_api_key=api_key,
        model_name="fake-openrouter-model",
        embedding_model="fake-embedding-model",
        chunk_max_characters=chunk_max_characters,
        chunk_overlap_characters=(
            chunk_overlap_characters
        ),
        retrieval_top_k=retrieval_top_k,
        minimum_grounding_overlap=(
            minimum_grounding_overlap
        ),
        confidence_threshold=0.82,
        retrieval_threshold=0.43,
        request_timeout_seconds=17.0,
        max_retries=4,
        kill_switch=kill_switch,
    )


def create_required_files(
    settings: Settings,
) -> tuple[Path, Path]:
    artifact_path = (
        settings.project_root
        / "artifacts"
        / "ticket_classifier.joblib"
    )
    documentation_path = (
        settings.project_root
        / "data"
        / "documentation.json"
    )

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_path.write_bytes(
        b"fake classifier artifact"
    )
    documentation_path.write_text(
        "[]",
        encoding="utf-8",
    )

    return artifact_path, documentation_path


def make_passage() -> Passage:
    return Passage(
        passage_id="DOC001#p001",
        doc_id="DOC001",
        title="Password reset",
        category="account_access",
        applies_to="all_users",
        chunk_index=1,
        text=(
            "Users can reset passwords from account settings."
        ),
        content_hash="fake-content-hash",
    )


@pytest.fixture
def fake_dependencies(monkeypatch):
    FakeTicketClassifierLoader.loaded_paths = []
    FakeTicketClassifierLoader.error = None

    FakeRetriever.instances = []
    FakeRetriever.build_error = None
    FakeRetriever.index_ready = True

    corpus_calls = []

    def fake_load_and_chunk_corpus(
        path,
        *,
        max_chars=1200,
        overlap_chars=160,
    ):
        corpus_calls.append(
            {
                "path": Path(path),
                "max_chars": max_chars,
                "overlap_chars": overlap_chars,
            }
        )
        return [make_passage()]

    monkeypatch.setattr(
        app_module,
        "TicketClassifier",
        FakeTicketClassifierLoader,
    )
    monkeypatch.setattr(
        app_module,
        "ChromaRetriever",
        FakeRetriever,
    )
    monkeypatch.setattr(
        app_module,
        "load_and_chunk_corpus",
        fake_load_and_chunk_corpus,
    )

    return {
        "corpus_calls": corpus_calls,
        "corpus_loader": (
            fake_load_and_chunk_corpus
        ),
    }


def test_build_application_with_injected_generator(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    settings = make_settings(tmp_path)
    artifact_path, documentation_path = (
        create_required_files(settings)
    )
    generator = FakeGenerator()

    components = build_application(
        settings=settings,
        generator=generator,
        build_index=True,
    )

    assert components.settings is settings
    assert components.generator is generator
    assert isinstance(
        components.classifier,
        FakeClassifier,
    )
    assert isinstance(
        components.retriever,
        FakeRetriever,
    )

    assert (
        components.classifier_artifact_path
        == artifact_path.resolve()
    )
    assert (
        components.documentation_path
        == documentation_path.resolve()
    )

    assert len(
        components.documentation_passages
    ) == 1

    assert components.index_report is not None
    assert (
        components.index_report.passage_count
        == 1
    )

    assert (
        FakeTicketClassifierLoader.loaded_paths
        == [artifact_path.resolve()]
    )

    assert len(FakeRetriever.instances) == 1

    retriever = FakeRetriever.instances[0]

    assert (
        retriever.chroma_path
        == settings.chroma_path
    )
    assert (
        retriever.embedding_model_name
        == "fake-embedding-model"
    )
    assert len(retriever.build_calls) == 1


def test_application_build_creates_runtime_directories(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    settings = make_settings(tmp_path)
    create_required_files(settings)

    assert not settings.storage_dir.exists()
    assert not settings.chroma_path.exists()
    assert not settings.evaluation_results_dir.exists()

    build_application(
        settings=settings,
        generator=FakeGenerator(),
    )

    assert settings.storage_dir.is_dir()
    assert settings.chroma_path.is_dir()
    assert settings.database_path.parent.is_dir()
    assert settings.evaluation_results_dir.is_dir()


def test_custom_chunk_settings_are_forwarded(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    settings = make_settings(tmp_path)
    create_required_files(settings)

    build_application(
        settings=settings,
        generator=FakeGenerator(),
        chunk_max_characters=700,
        chunk_overlap_characters=75,
        build_index=True,
    )

    corpus_call = (
        fake_dependencies["corpus_calls"][0]
    )

    assert corpus_call["max_chars"] == 700
    assert corpus_call["overlap_chars"] == 75


def test_retrieval_settings_are_read_from_settings(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    settings = make_settings(
        tmp_path,
        chunk_max_characters=650,
        chunk_overlap_characters=60,
        retrieval_top_k=3,
        minimum_grounding_overlap=0.25,
    )
    create_required_files(settings)

    components = build_application(
        settings=settings,
        generator=FakeGenerator(),
        build_index=True,
    )

    corpus_call = fake_dependencies["corpus_calls"][0]

    assert corpus_call["max_chars"] == 650
    assert corpus_call["overlap_chars"] == 60
    assert components.workflow.retrieval_top_k == 3
    assert (
        components.guardrail_engine
        .minimum_grounding_overlap
        == 0.25
    )


def test_build_index_can_be_skipped(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    settings = make_settings(tmp_path)
    create_required_files(settings)

    components = build_application(
        settings=settings,
        generator=FakeGenerator(),
        build_index=False,
    )

    assert components.index_report is None
    assert len(FakeRetriever.instances) == 1
    assert (
        FakeRetriever.instances[0].build_calls
        == []
    )


def test_relative_custom_paths_use_project_root(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    settings = make_settings(tmp_path)

    artifact_path = (
        settings.project_root
        / "models"
        / "custom.joblib"
    )
    documentation_path = (
        settings.project_root
        / "knowledge"
        / "custom-docs.json"
    )

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    documentation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_path.write_bytes(b"artifact")
    documentation_path.write_text(
        "[]",
        encoding="utf-8",
    )

    components = build_application(
        settings=settings,
        classifier_artifact_path=(
            "models/custom.joblib"
        ),
        documentation_path=(
            "knowledge/custom-docs.json"
        ),
        generator=FakeGenerator(),
        build_index=True,
    )

    assert (
        components.classifier_artifact_path
        == artifact_path.resolve()
    )
    assert (
        components.documentation_path
        == documentation_path.resolve()
    )


def test_absolute_custom_paths_are_preserved(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    settings = make_settings(tmp_path)

    external_directory = tmp_path / "external"
    external_directory.mkdir()

    artifact_path = (
        external_directory / "classifier.joblib"
    )
    documentation_path = (
        external_directory / "documentation.json"
    )

    artifact_path.write_bytes(b"artifact")
    documentation_path.write_text(
        "[]",
        encoding="utf-8",
    )

    components = build_application(
        settings=settings,
        classifier_artifact_path=artifact_path,
        documentation_path=documentation_path,
        generator=FakeGenerator(),
    )

    assert (
        components.classifier_artifact_path
        == artifact_path.resolve()
    )
    assert (
        components.documentation_path
        == documentation_path.resolve()
    )


def test_missing_classifier_artifact_has_clear_error(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    settings = make_settings(tmp_path)

    documentation_path = (
        settings.data_dir
        / "documentation.json"
    )
    documentation_path.write_text(
        "[]",
        encoding="utf-8",
    )

    with pytest.raises(
        ApplicationBuildError,
        match="Classifier artifact does not exist",
    ):
        build_application(
            settings=settings,
            generator=FakeGenerator(),
            build_index=True,
        )


def test_missing_documentation_has_clear_error(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    settings = make_settings(tmp_path)

    artifact_path = (
        settings.project_root
        / "artifacts"
        / "ticket_classifier.joblib"
    )
    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    artifact_path.write_bytes(b"artifact")

    with pytest.raises(
        ApplicationBuildError,
        match="Documentation corpus does not exist",
    ):
        build_application(
            settings=settings,
            generator=FakeGenerator(),
            build_index=True,
        )


def test_classifier_load_failure_is_wrapped(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    settings = make_settings(tmp_path)
    create_required_files(settings)

    FakeTicketClassifierLoader.error = RuntimeError(
        "corrupt artifact"
    )

    with pytest.raises(
        ApplicationBuildError,
        match="Could not load the classifier artifact",
    ) as error:
        build_application(
            settings=settings,
            generator=FakeGenerator(),
            build_index=True,
        )

    assert "RuntimeError" in str(error.value)


def test_corpus_loading_failure_is_wrapped(
    tmp_path: Path,
    fake_dependencies,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)
    create_required_files(settings)

    def failing_corpus_loader(*args, **kwargs):
        raise ValueError("invalid corpus")

    monkeypatch.setattr(
        app_module,
        "load_and_chunk_corpus",
        failing_corpus_loader,
    )

    with pytest.raises(
        ApplicationBuildError,
        match="Could not load and chunk",
    ) as error:
        build_application(
            settings=settings,
            generator=FakeGenerator(),
            build_index=True,
        )

    assert "ValueError" in str(error.value)


def test_empty_corpus_passages_are_rejected(
    tmp_path: Path,
    fake_dependencies,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)
    create_required_files(settings)

    monkeypatch.setattr(
        app_module,
        "load_and_chunk_corpus",
        lambda *args, **kwargs: [],
    )

    with pytest.raises(
        ApplicationBuildError,
        match="no retrievable passages",
    ):
        build_application(
            settings=settings,
            generator=FakeGenerator(),
            build_index=True,
        )


def test_index_build_failure_is_wrapped(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    settings = make_settings(tmp_path)
    create_required_files(settings)

    FakeRetriever.build_error = RuntimeError(
        "embedding model unavailable"
    )

    with pytest.raises(
        ApplicationBuildError,
        match="Could not build",
    ) as error:
        build_application(
            settings=settings,
            generator=FakeGenerator(),
            build_index=True,
        )

    assert "embedding model unavailable" in str(
        error.value
    )


def test_real_generator_factory_uses_settings(
    tmp_path: Path,
    fake_dependencies,
    monkeypatch,
) -> None:
    settings = make_settings(
        tmp_path,
        api_key="test-openrouter-key",
    )
    create_required_files(settings)

    generator = FakeGenerator()
    factory_calls = []

    def fake_generator_factory(**kwargs):
        factory_calls.append(kwargs)
        return generator

    monkeypatch.setattr(
        app_module,
        "create_openrouter_generator",
        fake_generator_factory,
    )

    components = build_application(
        settings=settings
    )

    assert components.generator is generator
    assert factory_calls == [
        {
            "api_key": "test-openrouter-key",
            "model": "fake-openrouter-model",
            "timeout_seconds": 17.0,
            "max_retries": 4,
        }
    ]


def test_missing_api_key_has_clear_build_error(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    settings = make_settings(
        tmp_path,
        api_key=None,
    )
    create_required_files(settings)

    with pytest.raises(
        ApplicationBuildError,
        match="Could not configure response generation",
    ) as error:
        build_application(
            settings=settings
        )

    assert "OPENROUTER_API_KEY" in str(
        error.value
    )


def test_routing_policy_uses_settings(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    settings = make_settings(
        tmp_path,
        kill_switch=True,
    )
    create_required_files(settings)

    components = build_application(
        settings=settings,
        generator=FakeGenerator(),
    )

    policy = components.routing_policy

    assert policy.confidence_threshold == 0.82
    assert policy.retrieval_threshold == 0.43
    assert policy.kill_switch is True


def test_startup_summary_is_safe(
    tmp_path: Path,
    fake_dependencies,
) -> None:
    secret = "super-secret-openrouter-key"

    settings = make_settings(
        tmp_path,
        api_key=secret,
    )
    create_required_files(settings)

    components = build_application(
        settings=settings,
        generator=FakeGenerator(),
    )

    summary = components.startup_summary()
    rendered_summary = str(summary)

    assert summary[
        "documentation_passage_count"
    ] == 1
    assert summary["index"]["passage_count"] == 1

    assert secret not in rendered_summary
    assert "openrouter_api_key" not in summary[
        "configuration"
    ]
    assert summary["configuration"][
        "openrouter_api_key_configured"
    ] is True


def test_process_raw_ticket_normalizes_before_workflow(
    tmp_path: Path,
    fake_dependencies,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)
    create_required_files(settings)

    components = build_application(
        settings=settings,
        generator=FakeGenerator(),
    )

    captured = {}

    def fake_process(ticket, *, run_id=None):
        captured["ticket"] = ticket
        captured["run_id"] = run_id
        return "processed"

    monkeypatch.setattr(
        components.workflow,
        "process",
        fake_process,
    )

    result = components.process_raw_ticket(
        {
            "ticket_id": "RAW-001",
            "channel": "chat",
            "subject": "  Login help  ",
            "body": "  I cannot log in.  ",
        },
        run_id="RAW-RUN",
    )

    assert result == "processed"
    assert captured["run_id"] == "RAW-RUN"
    assert captured["ticket"].ticket_id == "RAW-001"

    # Original values remain available.
    assert captured["ticket"].subject == (
        "  Login help  "
    )

    # Retrieval/classification text is normalized.
    assert captured["ticket"].text == (
        "Subject: Login help\n\n"
        "Message: I cannot log in."
    )
