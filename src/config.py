"""Application configuration and project path handling."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True)
class Settings:
    """Validated application settings."""

    project_root: Path

    data_dir: Path
    storage_dir: Path
    evaluation_results_dir: Path

    chroma_path: Path
    database_url: str
    database_path: Path

    openrouter_api_key: str | None = field(
        default=None,
        repr=False,
    )
    openrouter_base_url: str = (
        "https://openrouter.ai/api/v1"
    )

    model_name: str = (
        "meta-llama/llama-3.1-8b-instruct"
    )
    embedding_model: str = "all-MiniLM-L6-v2"

    chunk_max_characters: int = 1200
    chunk_overlap_characters: int = 160
    retrieval_top_k: int = 5
    minimum_grounding_overlap: float = 0.15

    confidence_threshold: float = 0.80
    retrieval_threshold: float = 0.40

    request_timeout_seconds: float = 30.0
    max_retries: int = 3
    log_level: str = "INFO"
    kill_switch: bool = False

    def __post_init__(self) -> None:
        """Validate values after the dataclass is created."""

        _validate_probability(
            "CONFIDENCE_THRESHOLD",
            self.confidence_threshold,
        )
        _validate_probability(
            "RETRIEVAL_THRESHOLD",
            self.retrieval_threshold,
        )
        _validate_probability(
            "MINIMUM_GROUNDING_OVERLAP",
            self.minimum_grounding_overlap,
        )

        if self.chunk_max_characters < 200:
            raise ConfigurationError(
                "CHUNK_MAX_CHARACTERS must be at least 200."
            )

        if self.chunk_overlap_characters < 0:
            raise ConfigurationError(
                "CHUNK_OVERLAP_CHARACTERS cannot be negative."
            )

        if (
            self.chunk_overlap_characters
            >= self.chunk_max_characters
        ):
            raise ConfigurationError(
                "CHUNK_OVERLAP_CHARACTERS must be smaller than "
                "CHUNK_MAX_CHARACTERS."
            )

        if self.retrieval_top_k <= 0:
            raise ConfigurationError(
                "RETRIEVAL_TOP_K must be greater than zero."
            )

        if self.request_timeout_seconds <= 0:
            raise ConfigurationError(
                "REQUEST_TIMEOUT_SECONDS must be greater than zero."
            )

        if self.max_retries < 0:
            raise ConfigurationError(
                "MAX_RETRIES cannot be negative."
            )

        allowed_log_levels = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if self.log_level not in allowed_log_levels:
            raise ConfigurationError(
                "LOG_LEVEL must be one of: "
                + ", ".join(sorted(allowed_log_levels))
            )

    def require_api_key(self) -> str:
        """
        Return the model-provider key or fail clearly.

        The key is not required for local ingestion, retrieval, tests, or
        classification. It is checked only when generation needs the provider.
        """

        key = self.openrouter_api_key

        invalid_placeholders = {
            "",
            "your_key_here",
            "replace_me",
            "placeholder",
        }

        if key is None or key.lower() in invalid_placeholders:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is required for answer generation. "
                "Create .env from .env.example and provide a valid key."
            )

        return key

    def safe_summary(self) -> dict[str, Any]:
        """
        Return configuration that is safe to print.

        The API key is never included.
        """

        return {
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "storage_dir": str(self.storage_dir),
            "evaluation_results_dir": str(
                self.evaluation_results_dir
            ),
            "chroma_path": str(self.chroma_path),
            "database_url": _redact_database_url(
                self.database_url
            ),
            "database_path": str(self.database_path),
            "openrouter_api_key_configured": bool(
                self.openrouter_api_key
            ),
            "openrouter_base_url": self.openrouter_base_url,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "chunk_max_characters": (
                self.chunk_max_characters
            ),
            "chunk_overlap_characters": (
                self.chunk_overlap_characters
            ),
            "retrieval_top_k": self.retrieval_top_k,
            "minimum_grounding_overlap": (
                self.minimum_grounding_overlap
            ),
            "confidence_threshold": self.confidence_threshold,
            "retrieval_threshold": self.retrieval_threshold,
            "request_timeout_seconds": (
                self.request_timeout_seconds
            ),
            "max_retries": self.max_retries,
            "log_level": self.log_level,
            "kill_switch": self.kill_switch,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load and cache application settings.

    Paths are resolved from the project directory, not the terminal's current
    working directory.
    """

    load_dotenv(
        dotenv_path=DEFAULT_ENV_FILE,
        override=False,
    )

    data_dir = _resolve_project_path(
        os.getenv("DATA_DIR", "./data")
    )
    storage_dir = _resolve_project_path(
        os.getenv("STORAGE_DIR", "./storage")
    )
    evaluation_results_dir = _resolve_project_path(
        os.getenv(
            "EVALUATION_RESULTS_DIR",
            "./evaluation/results",
        )
    )

    chroma_path = _resolve_project_path(
        os.getenv(
            "CHROMA_PATH",
            "./storage/chroma",
        )
    )

    database_url = os.getenv(
        "DATABASE_URL",
        "sqlite:///./storage/decisions.db",
    ).strip()

    database_path = _database_path_from_url(
        database_url
    )

    api_key = _optional_environment_value(
        "OPENROUTER_API_KEY"
    )

    log_level = os.getenv(
        "LOG_LEVEL",
        "INFO",
    ).strip().upper()

    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        storage_dir=storage_dir,
        evaluation_results_dir=evaluation_results_dir,
        chroma_path=chroma_path,
        database_url=database_url,
        database_path=database_path,
        openrouter_api_key=api_key,
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ).strip().rstrip("/"),
        model_name=os.getenv(
            "MODEL_NAME",
            "meta-llama/llama-3.1-8b-instruct",
        ).strip(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "all-MiniLM-L6-v2",
        ).strip(),
        chunk_max_characters=_read_integer(
            "CHUNK_MAX_CHARACTERS",
            1200,
        ),
        chunk_overlap_characters=_read_integer(
            "CHUNK_OVERLAP_CHARACTERS",
            160,
        ),
        retrieval_top_k=_read_integer(
            "RETRIEVAL_TOP_K",
            5,
        ),
        minimum_grounding_overlap=_read_float(
            "MINIMUM_GROUNDING_OVERLAP",
            0.15,
        ),
        confidence_threshold=_read_float(
            "CONFIDENCE_THRESHOLD",
            0.80,
        ),
        retrieval_threshold=_read_float(
            "RETRIEVAL_THRESHOLD",
            0.40,
        ),
        request_timeout_seconds=_read_float(
            "REQUEST_TIMEOUT_SECONDS",
            30.0,
        ),
        max_retries=_read_integer(
            "MAX_RETRIES",
            3,
        ),
        log_level=log_level,
        kill_switch=_read_boolean(
            "KILL_SWITCH",
            False,
        ),
    )


def ensure_runtime_directories(
    settings: Settings | None = None,
) -> None:
    """
    Create directories generated during application execution.

    This function is intentionally separate from get_settings so importing
    configuration does not unexpectedly modify the filesystem.
    """

    current = settings or get_settings()

    directories = {
        current.storage_dir,
        current.chroma_path,
        current.database_path.parent,
        current.evaluation_results_dir,
    }

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def clear_settings_cache() -> None:
    """
    Clear cached settings.

    This is mainly useful in tests that temporarily change environment
    variables.
    """

    get_settings.cache_clear()


def _resolve_project_path(value: str) -> Path:
    """Resolve relative configuration paths from the project root."""

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def _database_path_from_url(database_url: str) -> Path:
    """
    Extract a local file path from a SQLite database URL.

    This project currently uses sqlite3 directly. Supporting PostgreSQL later
    would require a separate database adapter rather than pretending it is a
    local file.
    """

    prefix = "sqlite:///"

    if not database_url.startswith(prefix):
        raise ConfigurationError(
            "Only sqlite:/// database URLs are currently supported."
        )

    raw_path = database_url[len(prefix):].strip()

    if not raw_path:
        raise ConfigurationError(
            "DATABASE_URL must include a SQLite file path."
        )

    return _resolve_project_path(raw_path)


def _optional_environment_value(name: str) -> str | None:
    """Read an optional environment variable."""

    value = os.getenv(name)

    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def _read_float(name: str, default: float) -> float:
    """Read a floating-point environment variable."""

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be a number, received {raw_value!r}."
        ) from exc


def _read_integer(name: str, default: int) -> int:
    """Read an integer environment variable."""

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be an integer, received {raw_value!r}."
        ) from exc


def _read_boolean(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()

    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}

    if normalized in truthy:
        return True

    if normalized in falsy:
        return False

    raise ConfigurationError(
        f"{name} must be one of "
        f"{sorted(truthy | falsy)}, received {raw_value!r}."
    )


def _validate_probability(
    name: str,
    value: float,
) -> None:
    """Validate a value expected to be between zero and one."""

    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(
            f"{name} must be between 0 and 1, received {value}."
        )


def _redact_database_url(database_url: str) -> str:
    """
    Redact credentials if a future database URL contains them.

    SQLite URLs normally contain no credentials.
    """

    if "@" not in database_url:
        return database_url

    scheme_separator = "://"

    if scheme_separator not in database_url:
        return "[REDACTED]"

    scheme, remainder = database_url.split(
        scheme_separator,
        maxsplit=1,
    )

    host_part = remainder.rsplit("@", maxsplit=1)[-1]
    return f"{scheme}{scheme_separator}[REDACTED]@{host_part}"
