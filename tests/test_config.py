"""Tests for application configuration."""

from pathlib import Path

import pytest

from src.config import (
    PROJECT_ROOT,
    ConfigurationError,
    clear_settings_cache,
    ensure_runtime_directories,
    get_settings,
)


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    """
    Provide predictable settings for every test.

    These values override the developer's local .env without modifying it.
    """

    test_environment = {
        "OPENROUTER_API_KEY": "test-key-not-real",
        "OPENROUTER_BASE_URL": (
            "https://openrouter.ai/api/v1"
        ),
        "MODEL_NAME": "test-model",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "CHUNK_MAX_CHARACTERS": "1200",
        "CHUNK_OVERLAP_CHARACTERS": "160",
        "RETRIEVAL_TOP_K": "5",
        "MINIMUM_GROUNDING_OVERLAP": "0.15",
        "DATA_DIR": "./data",
        "STORAGE_DIR": "./storage",
        "EVALUATION_RESULTS_DIR": "./evaluation/results",
        "CHROMA_PATH": "./storage/chroma",
        "DATABASE_URL": (
            "sqlite:///./storage/decisions.db"
        ),
        "CONFIDENCE_THRESHOLD": "0.80",
        "RETRIEVAL_THRESHOLD": "0.40",
        "REQUEST_TIMEOUT_SECONDS": "30",
        "MAX_RETRIES": "3",
        "LOG_LEVEL": "INFO",
        "KILL_SWITCH": "false",
    }

    for name, value in test_environment.items():
        monkeypatch.setenv(name, value)

    clear_settings_cache()
    yield
    clear_settings_cache()


def test_project_root_is_derived_from_source_file() -> None:
    settings = get_settings()

    assert settings.project_root == PROJECT_ROOT
    assert settings.project_root.name == (
        "Customer_Success_Agent"
    )


def test_relative_paths_resolve_from_project_root(
    tmp_path,
    monkeypatch,
) -> None:
    """
    Changing the terminal directory must not change project paths.
    """

    monkeypatch.chdir(tmp_path)
    clear_settings_cache()

    settings = get_settings()

    assert settings.data_dir == (
        PROJECT_ROOT / "data"
    ).resolve()

    assert settings.storage_dir == (
        PROJECT_ROOT / "storage"
    ).resolve()

    assert settings.chroma_path == (
        PROJECT_ROOT / "storage" / "chroma"
    ).resolve()

    assert settings.database_path == (
        PROJECT_ROOT / "storage" / "decisions.db"
    ).resolve()


def test_environment_values_are_loaded(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "CONFIDENCE_THRESHOLD",
        "0.73",
    )
    monkeypatch.setenv(
        "RETRIEVAL_THRESHOLD",
        "0.51",
    )
    monkeypatch.setenv(
        "REQUEST_TIMEOUT_SECONDS",
        "15.5",
    )
    monkeypatch.setenv(
        "MAX_RETRIES",
        "5",
    )
    monkeypatch.setenv(
        "LOG_LEVEL",
        "warning",
    )
    monkeypatch.setenv(
        "KILL_SWITCH",
        "yes",
    )
    monkeypatch.setenv(
        "CHUNK_MAX_CHARACTERS",
        "700",
    )
    monkeypatch.setenv(
        "CHUNK_OVERLAP_CHARACTERS",
        "75",
    )
    monkeypatch.setenv(
        "RETRIEVAL_TOP_K",
        "3",
    )
    monkeypatch.setenv(
        "RETRIEVAL_CANDIDATE_MULTIPLIER",
        "4",
    )
    monkeypatch.setenv(
        "MINIMUM_GROUNDING_OVERLAP",
        "0.25",
    )

    clear_settings_cache()
    settings = get_settings()

    assert settings.confidence_threshold == 0.73
    assert settings.retrieval_threshold == 0.51
    assert settings.request_timeout_seconds == 15.5
    assert settings.max_retries == 5
    assert settings.log_level == "WARNING"
    assert settings.kill_switch is True
    assert settings.chunk_max_characters == 700
    assert settings.chunk_overlap_characters == 75
    assert settings.retrieval_top_k == 3
    assert settings.retrieval_candidate_multiplier == 4
    assert settings.minimum_grounding_overlap == 0.25


@pytest.mark.parametrize(
    ("variable_name", "invalid_value"),
    [
        ("CHUNK_MAX_CHARACTERS", "199"),
        ("CHUNK_OVERLAP_CHARACTERS", "-1"),
        ("RETRIEVAL_TOP_K", "0"),
        ("RETRIEVAL_CANDIDATE_MULTIPLIER", "0"),
        ("MINIMUM_GROUNDING_OVERLAP", "1.01"),
    ],
)
def test_invalid_retrieval_configuration_is_rejected(
    variable_name,
    invalid_value,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        variable_name,
        invalid_value,
    )
    clear_settings_cache()

    with pytest.raises(
        ConfigurationError,
        match=variable_name,
    ):
        get_settings()


def test_chunk_overlap_must_be_smaller_than_chunk_size(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "CHUNK_MAX_CHARACTERS",
        "500",
    )
    monkeypatch.setenv(
        "CHUNK_OVERLAP_CHARACTERS",
        "500",
    )
    clear_settings_cache()

    with pytest.raises(
        ConfigurationError,
        match="CHUNK_OVERLAP_CHARACTERS",
    ):
        get_settings()


@pytest.mark.parametrize(
    "environment_value",
    ["true", "TRUE", "1", "yes", "on"],
)
def test_truthy_kill_switch_values(
    environment_value,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "KILL_SWITCH",
        environment_value,
    )
    clear_settings_cache()

    assert get_settings().kill_switch is True


@pytest.mark.parametrize(
    "environment_value",
    ["false", "FALSE", "0", "no", "off"],
)
def test_falsy_kill_switch_values(
    environment_value,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "KILL_SWITCH",
        environment_value,
    )
    clear_settings_cache()

    assert get_settings().kill_switch is False


def test_invalid_boolean_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "KILL_SWITCH",
        "sometimes",
    )
    clear_settings_cache()

    with pytest.raises(
        ConfigurationError,
        match="KILL_SWITCH",
    ):
        get_settings()


@pytest.mark.parametrize(
    ("variable_name", "invalid_value"),
    [
        ("CONFIDENCE_THRESHOLD", "-0.01"),
        ("CONFIDENCE_THRESHOLD", "1.01"),
        ("RETRIEVAL_THRESHOLD", "-1"),
        ("RETRIEVAL_THRESHOLD", "2"),
    ],
)
def test_invalid_probability_is_rejected(
    variable_name,
    invalid_value,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        variable_name,
        invalid_value,
    )
    clear_settings_cache()

    with pytest.raises(
        ConfigurationError,
        match=variable_name,
    ):
        get_settings()


def test_non_numeric_threshold_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "CONFIDENCE_THRESHOLD",
        "high",
    )
    clear_settings_cache()

    with pytest.raises(
        ConfigurationError,
        match="must be a number",
    ):
        get_settings()


def test_negative_retry_count_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "MAX_RETRIES",
        "-1",
    )
    clear_settings_cache()

    with pytest.raises(
        ConfigurationError,
        match="MAX_RETRIES",
    ):
        get_settings()


def test_invalid_log_level_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "LOG_LEVEL",
        "EVERYTHING",
    )
    clear_settings_cache()

    with pytest.raises(
        ConfigurationError,
        match="LOG_LEVEL",
    ):
        get_settings()


def test_unsupported_database_url_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://localhost/example",
    )
    clear_settings_cache()

    with pytest.raises(
        ConfigurationError,
        match="Only sqlite",
    ):
        get_settings()


def test_require_api_key_returns_configured_key(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "configured-test-key",
    )
    clear_settings_cache()

    settings = get_settings()

    assert (
        settings.require_api_key()
        == "configured-test-key"
    )


@pytest.mark.parametrize(
    "placeholder",
    [
        "",
        "your_key_here",
        "replace_me",
        "placeholder",
    ],
)
def test_placeholder_api_key_is_rejected(
    placeholder,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        placeholder,
    )
    clear_settings_cache()

    settings = get_settings()

    with pytest.raises(
        ConfigurationError,
        match="OPENROUTER_API_KEY",
    ):
        settings.require_api_key()


def test_safe_summary_never_contains_api_key(
    monkeypatch,
) -> None:
    secret = "this-must-not-appear"

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        secret,
    )
    clear_settings_cache()

    settings = get_settings()
    summary = settings.safe_summary()
    rendered_summary = str(summary)

    assert secret not in rendered_summary
    assert "openrouter_api_key" not in summary
    assert (
        summary["openrouter_api_key_configured"]
        is True
    )


def test_settings_are_cached() -> None:
    first = get_settings()
    second = get_settings()

    assert first is second


def test_clearing_cache_reloads_environment(
    monkeypatch,
) -> None:
    original = get_settings()
    assert original.max_retries == 3

    monkeypatch.setenv(
        "MAX_RETRIES",
        "7",
    )
    clear_settings_cache()

    updated = get_settings()

    assert updated.max_retries == 7
    assert updated is not original


def test_runtime_directories_are_created(
    tmp_path,
    monkeypatch,
) -> None:
    storage_dir = tmp_path / "runtime-storage"
    chroma_dir = storage_dir / "chroma"
    result_dir = tmp_path / "run-results"
    database_path = storage_dir / "database" / "test.db"

    monkeypatch.setenv(
        "STORAGE_DIR",
        str(storage_dir),
    )
    monkeypatch.setenv(
        "CHROMA_PATH",
        str(chroma_dir),
    )
    monkeypatch.setenv(
        "EVALUATION_RESULTS_DIR",
        str(result_dir),
    )
    monkeypatch.setenv(
        "DATABASE_URL",
        f"sqlite:///{database_path}",
    )

    clear_settings_cache()
    settings = get_settings()

    ensure_runtime_directories(settings)

    assert storage_dir.is_dir()
    assert chroma_dir.is_dir()
    assert result_dir.is_dir()
    assert database_path.parent.is_dir()

    # Configuration creates the parent directory, but it should not create
    # an empty database before the database layer requests one.
    assert not database_path.exists()
