"""Knowledge-base loading and passage chunking."""

from __future__ import annotations

import hashlib
import json
import re
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class CorpusError(Exception):
    """Raised when the documentation corpus is invalid."""


@dataclass(frozen=True)
class KnowledgeDocument:
    """One source article from documentation.json."""

    doc_id: str
    title: str
    category: str
    applies_to: str
    content: str
    related_docs: tuple[str, ...]
    last_reviewed_days_ago: int | None


@dataclass(frozen=True)
class Passage:
    """A traceable passage produced from a knowledge document."""

    passage_id: str
    doc_id: str
    title: str
    category: str
    applies_to: str
    chunk_index: int
    text: str
    content_hash: str

    def chroma_metadata(self) -> dict[str, str | int]:
        """
        Return scalar metadata suitable for Chroma.

        Chroma metadata should contain simple scalar values rather than lists
        or nested dictionaries.
        """

        return {
            "passage_id": self.passage_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "category": self.category,
            "applies_to": self.applies_to,
            "chunk_index": self.chunk_index,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class MarkdownSection:
    """A heading and the body belonging to that heading."""

    heading: str
    body: str

    @property
    def rendered_text(self) -> str:
        if self.heading and self.body:
            return f"{self.heading}\n\n{self.body}"

        return self.heading or self.body


def load_documents( corpus_path: str | Path) -> list[KnowledgeDocument]:
    """
    Load and validate the supplied documentation corpus.

    Unlike malformed customer tickets, a malformed corpus is a run-level
    problem because retrieval cannot work safely without trusted documents.
    """

    path = Path(corpus_path).expanduser().resolve()

    if not path.exists():
        raise CorpusError(
            f"Documentation corpus does not exist: {path}"
        )

    if not path.is_file():
        raise CorpusError(
            f"Documentation corpus path is not a file: {path}"
        )

    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise CorpusError(
            f"Could not read documentation corpus: {path}"
        ) from exc

    try:
        records = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise CorpusError(
            "Documentation corpus contains invalid JSON at "
            f"line {exc.lineno}, column {exc.colno}."
        ) from exc

    if not isinstance(records, list):
        raise CorpusError(
            "Documentation corpus must contain a JSON array."
        )

    documents: list[KnowledgeDocument] = []
    seen_doc_ids: set[str] = set()

    for record_index, record in enumerate(records, start=1):
        document = _parse_document_record(
            record,
            record_index=record_index,
        )

        if document.doc_id in seen_doc_ids:
            raise CorpusError(
                f"Duplicate document ID: {document.doc_id}"
            )

        seen_doc_ids.add(document.doc_id)
        documents.append(document)

    if not documents:
        raise CorpusError(
            "Documentation corpus contains no documents."
        )

    return documents


def chunk_documents(documents: list[KnowledgeDocument],*,max_chars: int = 1200,overlap_chars: int = 160) -> list[Passage]:
    """
    Split documents into deterministic, traceable passages.

    Markdown headings are used as natural boundaries. If a section is too
    large, it is divided at paragraph, newline, sentence, or word boundaries.
    The section heading is repeated so each passage retains context.
    """

    _validate_chunk_settings(
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    passages: list[Passage] = []

    for document in documents:
        passage_texts = chunk_markdown(
            document.content,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        for chunk_index, passage_text in enumerate(
            passage_texts,
            start=1,
        ):
            passage_id = (
                f"{document.doc_id}#p{chunk_index:03d}"
            )

            passages.append(
                Passage(
                    passage_id=passage_id,
                    doc_id=document.doc_id,
                    title=document.title,
                    category=document.category,
                    applies_to=document.applies_to,
                    chunk_index=chunk_index,
                    text=passage_text,
                    content_hash=_sha256(passage_text),
                )
            )

    return passages


def load_and_chunk_corpus(corpus_path: str | Path, *, max_chars: int = 1200, overlap_chars: int = 160) -> list[Passage]:
    """Load the corpus and create all retrievable passages."""

    documents = load_documents(corpus_path)

    return chunk_documents(
        documents,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )


def chunk_markdown( content: str, *, max_chars: int = 1200, overlap_chars: int = 160) -> list[str]:
    """Split one Markdown article into section-aware passages."""

    _validate_chunk_settings(
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    normalized_content = _normalize_newlines(content).strip()

    if not normalized_content:
        return []

    sections = split_markdown_sections(normalized_content)

    chunks: list[str] = []

    for section in sections:
        rendered = section.rendered_text.strip()

        if not rendered:
            continue

        if len(rendered) <= max_chars:
            chunks.append(rendered)
            continue

        chunks.extend(
            _split_large_section(
                section,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )

    return chunks


def split_markdown_sections(content: str) -> list[MarkdownSection]:
    """
    Split Markdown whenever a heading begins.

    Heading lines are kept because they provide useful retrieval context.
    """

    normalized = _normalize_newlines(content)
    lines = normalized.splitlines()

    sections: list[MarkdownSection] = []
    current_heading = ""
    current_body_lines: list[str] = []

    for line in lines:
        if _is_markdown_heading(line):
            _append_section(
                sections,
                heading=current_heading,
                body_lines=current_body_lines,
            )
            current_heading = line.strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)

    _append_section(
        sections,
        heading=current_heading,
        body_lines=current_body_lines,
    )

    return sections


def corpus_fingerprint(
    passages: list[Passage],
) -> str:
    """
    Calculate a deterministic fingerprint for a complete passage set.

    Bootstrap can later compare this value with the indexed value to decide
    whether Chroma needs rebuilding.
    """

    digest = hashlib.sha256()

    for passage in passages:
        digest.update(
            passage.passage_id.encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(
            passage.content_hash.encode("ascii")
        )
        digest.update(b"\0")

    return digest.hexdigest()


def _parse_document_record(
    record: Any,
    *,
    record_index: int,
) -> KnowledgeDocument:
    """Validate and convert one corpus record."""

    if not isinstance(record, Mapping):
        raise CorpusError(
            f"Document {record_index} must be a JSON object."
        )

    required_string_fields = (
        "doc_id",
        "title",
        "category",
        "applies_to",
        "content",
    )

    values: dict[str, str] = {}

    for field_name in required_string_fields:
        value = record.get(field_name)

        if not isinstance(value, str):
            raise CorpusError(
                f"Document {record_index} field "
                f"{field_name!r} must be a string."
            )

        if not value.strip():
            raise CorpusError(
                f"Document {record_index} field "
                f"{field_name!r} cannot be empty."
            )

        values[field_name] = value

    related_docs_value = record.get(
        "related_docs",
        [],
    )

    if not isinstance(related_docs_value, list):
        raise CorpusError(
            f"Document {record_index} field "
            "'related_docs' must be an array."
        )

    related_docs: list[str] = []

    for related_doc in related_docs_value:
        if not isinstance(related_doc, str):
            raise CorpusError(
                f"Document {record_index} contains a "
                "non-string related document ID."
            )

        if related_doc.strip():
            related_docs.append(related_doc.strip())

    reviewed_value = record.get(
        "last_reviewed_days_ago"
    )

    if reviewed_value is not None:
        if (
            not isinstance(reviewed_value, int)
            or isinstance(reviewed_value, bool)
            or reviewed_value < 0
        ):
            raise CorpusError(
                f"Document {record_index} field "
                "'last_reviewed_days_ago' must be a "
                "non-negative integer or null."
            )

    return KnowledgeDocument(
        doc_id=values["doc_id"].strip(),
        title=values["title"].strip(),
        category=values["category"].strip(),
        applies_to=values["applies_to"].strip(),
        content=values["content"],
        related_docs=tuple(related_docs),
        last_reviewed_days_ago=reviewed_value,
    )


def _split_large_section(
    section: MarkdownSection,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split an oversized section while retaining its heading."""

    heading = section.heading.strip()
    body = section.body.strip()

    if not body:
        return _split_text_window(
            heading,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

    prefix = f"{heading}\n\n" if heading else ""

    available_body_chars = max_chars - len(prefix)

    # If an unusually long heading consumes the window, split everything
    # normally rather than producing a negative body size.
    if available_body_chars < 200:
        return _split_text_window(
            section.rendered_text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

    body_overlap = min(
        overlap_chars,
        max(0, available_body_chars // 3),
    )

    body_chunks = _split_text_window(
        body,
        max_chars=available_body_chars,
        overlap_chars=body_overlap,
    )

    return [
        f"{prefix}{body_chunk}".strip()
        for body_chunk in body_chunks
    ]


def _split_text_window(text: str,*,max_chars: int,overlap_chars: int) -> list[str]:
    """
    Split text using a sliding window with preferred natural boundaries.
    """

    cleaned = text.strip()

    if not cleaned:
        return []

    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        tentative_end = min(
            start + max_chars,
            text_length,
        )

        if tentative_end == text_length:
            final_chunk = cleaned[start:].strip()

            if final_chunk:
                chunks.append(final_chunk)

            break

        split_at = _find_split_boundary(
            cleaned,
            start=start,
            tentative_end=tentative_end,
        )

        if split_at <= start:
            split_at = tentative_end

        chunk = cleaned[start:split_at].strip()

        if chunk:
            chunks.append(chunk)

        next_start = max(
            split_at - overlap_chars,
            start + 1,
        )

        # Move forward to a whitespace boundary so the next chunk does not
        # begin in the middle of a word.
        while (next_start < split_at and not cleaned[next_start].isspace()):
            next_start += 1

        while (
            next_start < text_length
            and cleaned[next_start].isspace()
        ):
            next_start += 1

        start = next_start

    return chunks


def _find_split_boundary(text: str,*,start: int,tentative_end: int) -> int:
    """Find the best natural split near the end of a window."""

    minimum_boundary = start + int(
        (tentative_end - start) * 0.60
    )

    boundary_patterns = (
        "\n\n",
        "\n",
        ". ",
        "; ",
        ", ",
        " ",
    )

    for pattern in boundary_patterns:
        position = text.rfind(
            pattern,
            minimum_boundary,
            tentative_end,
        )

        if position != -1:
            return position + len(pattern)

    return tentative_end


def _append_section(sections: list[MarkdownSection],*,heading: str,body_lines: list[str]) -> None:
    """Append a section when it contains text."""

    body = "\n".join(body_lines).strip()

    if heading.strip() or body:
        sections.append(
            MarkdownSection(
                heading=heading.strip(),
                body=body,
            )
        )


def _is_markdown_heading(line: str) -> bool:
    """Return whether a line is a Markdown heading."""

    return bool(
        re.match(r"^\s{0,3}#{1,6}\s+\S", line)
    )


def _normalize_newlines(text: str) -> str:
    """Normalise Windows and old Mac line endings."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def _validate_chunk_settings(*,max_chars: int,overlap_chars: int) -> None:
    """Validate chunking configuration."""

    if max_chars < 200:
        raise ValueError(
            "max_chars must be at least 200."
        )

    if overlap_chars < 0:
        raise ValueError(
            "overlap_chars cannot be negative."
        )

    if overlap_chars >= max_chars:
        raise ValueError(
            "overlap_chars must be smaller than max_chars."
        )


def _sha256(value: str) -> str:
    """Return a SHA-256 hexadecimal digest."""

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

@dataclass(frozen=True)
class RetrievedMatch:
    """One ranked retrieval result."""

    passage_id: str
    doc_id: str
    title: str
    category: str
    chunk_index: int
    text: str

    relevance_score: float
    distance: float


@dataclass(frozen=True)
class IndexBuildReport:
    """Summary of a Chroma indexing operation."""

    collection_name: str
    passage_count: int
    corpus_fingerprint: str
    rebuilt: bool
    chroma_path: str


class ChromaRetriever:
    """
    Persistent semantic retrieval over documentation passages.

    Embeddings are calculated directly with Sentence Transformers. This keeps
    retrieval independent from LangChain's integration package.
    """

    def __init__(
        self,
        *,
        chroma_path: str | Path,
        embedding_model_name: str,
        collection_name: str = "cloudserve_docs",
    ) -> None:
        self.chroma_path = (
            Path(chroma_path)
            .expanduser()
            .resolve()
        )
        self.embedding_model_name = (
            embedding_model_name
        )
        self.collection_name = collection_name

        self._client: Any | None = None
        self._embedding_model: Any | None = None

    def build_index(
        self,
        passages: Sequence[Passage],
        *,
        batch_size: int = 64,
    ) -> IndexBuildReport:
        """
        Build or reuse the persistent Chroma collection.

        If the corpus fingerprint changes, the generated collection is
        replaced so old and new passages cannot be mixed.
        """

        if not passages:
            raise CorpusError(
                "Cannot build an index without passages."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        passage_ids = [
            passage.passage_id
            for passage in passages
        ]

        if len(set(passage_ids)) != len(passage_ids):
            raise CorpusError(
                "Passage IDs must be unique."
            )

        fingerprint = corpus_fingerprint(
            list(passages)
        )

        client = self._get_client()
        existing = self._get_collection_if_exists()

        if existing is not None:
            metadata = existing.metadata or {}

            existing_fingerprint = metadata.get(
                "corpus_fingerprint"
            )

            existing_count = existing.count()

            if (
                existing_fingerprint == fingerprint
                and existing_count == len(passages)
            ):
                return IndexBuildReport(
                    collection_name=(
                        self.collection_name
                    ),
                    passage_count=existing_count,
                    corpus_fingerprint=fingerprint,
                    rebuilt=False,
                    chroma_path=str(self.chroma_path),
                )

            # Chroma is generated runtime storage. A collection built from a
            # different corpus must be replaced rather than mixed.
            client.delete_collection(
                name=self.collection_name
            )

        collection = client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "corpus_fingerprint": fingerprint,
                "embedding_model": (
                    self.embedding_model_name
                ),
            },
        )

        for start in range(
            0,
            len(passages),
            batch_size,
        ):
            batch = list(
                passages[start : start + batch_size]
            )

            texts = [
                passage.text
                for passage in batch
            ]

            embeddings = self._encode(texts)

            collection.add(
                ids=[
                    passage.passage_id
                    for passage in batch
                ],
                documents=texts,
                metadatas=[
                    passage.chroma_metadata()
                    for passage in batch
                ],
                embeddings=embeddings,
            )

        self._persist_if_supported()

        return IndexBuildReport(
            collection_name=self.collection_name,
            passage_count=collection.count(),
            corpus_fingerprint=fingerprint,
            rebuilt=True,
            chroma_path=str(self.chroma_path),
        )

    def search(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        minimum_relevance: float = 0.0,
    ) -> list[RetrievedMatch]:
        """
        Retrieve ranked passages.

        Chroma cosine distance is converted to relevance using:

            relevance = 1 - cosine_distance

        Higher relevance is better. Results below minimum_relevance are
        discarded.
        """

        cleaned_query = query_text.strip()

        if not cleaned_query:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        _validate_relevance(
            "minimum_relevance",
            minimum_relevance,
        )

        collection = self._require_collection()

        query_embedding = self._encode(
            [cleaned_query]
        )[0]

        available_passages = collection.count()

        if available_passages == 0:
            return []

        requested_results = min(
            top_k,
            available_passages,
        )

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=requested_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = _first_result_list(
            result.get("ids")
        )
        documents = _first_result_list(
            result.get("documents")
        )
        metadatas = _first_result_list(
            result.get("metadatas")
        )
        distances = _first_result_list(
            result.get("distances")
        )

        lengths = {
            len(ids),
            len(documents),
            len(metadatas),
            len(distances),
        }

        if len(lengths) != 1:
            raise CorpusError(
                "Chroma returned inconsistent result lengths."
            )

        matches: list[RetrievedMatch] = []

        for (
            passage_id,
            document_text,
            metadata,
            raw_distance,
        ) in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            distance = float(raw_distance)

            if not math.isfinite(distance):
                continue

            relevance = max(
                0.0,
                min(
                    1.0,
                    1.0 - distance,
                ),
            )

            if relevance < minimum_relevance:
                continue

            metadata = metadata or {}

            matches.append(
                RetrievedMatch(
                    passage_id=str(passage_id),
                    doc_id=str(
                        metadata.get(
                            "doc_id",
                            "",
                        )
                    ),
                    title=str(
                        metadata.get(
                            "title",
                            "",
                        )
                    ),
                    category=str(
                        metadata.get(
                            "category",
                            "",
                        )
                    ),
                    chunk_index=int(
                        metadata.get(
                            "chunk_index",
                            0,
                        )
                    ),
                    text=str(document_text),
                    relevance_score=relevance,
                    distance=distance,
                )
            )

        return matches

    def resolve_passage(
        self,
        passage_id: str,
    ) -> Passage | None:
        """
        Resolve a citation identifier to its exact indexed passage.

        This is the mechanism used to prove that generated citations refer to
        real passages rather than plausible-looking invented references.
        """

        cleaned_id = passage_id.strip()

        if not cleaned_id:
            return None

        collection = self._require_collection()

        result = collection.get(
            ids=[cleaned_id],
            include=[
                "documents",
                "metadatas",
            ],
        )

        ids = result.get("ids") or []

        if not ids:
            return None

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        if not documents or not metadatas:
            raise CorpusError(
                "Indexed passage is missing content or metadata."
            )

        metadata = metadatas[0] or {}
        text = str(documents[0])

        return Passage(
            passage_id=str(ids[0]),
            doc_id=str(
                metadata.get("doc_id", "")
            ),
            title=str(
                metadata.get("title", "")
            ),
            category=str(
                metadata.get("category", "")
            ),
            applies_to=str(
                metadata.get("applies_to", "")
            ),
            chunk_index=int(
                metadata.get("chunk_index", 0)
            ),
            text=text,
            content_hash=str(
                metadata.get(
                    "content_hash",
                    _sha256(text),
                )
            ),
        )

    def index_status(self) -> dict[str, Any]:
        """Return collection status without exposing embeddings."""

        collection = self._get_collection_if_exists()

        if collection is None:
            return {
                "ready": False,
                "collection_name": (
                    self.collection_name
                ),
                "passage_count": 0,
                "corpus_fingerprint": None,
                "embedding_model": (
                    self.embedding_model_name
                ),
                "chroma_path": str(
                    self.chroma_path
                ),
            }

        metadata = collection.metadata or {}

        return {
            "ready": collection.count() > 0,
            "collection_name": (
                self.collection_name
            ),
            "passage_count": collection.count(),
            "corpus_fingerprint": metadata.get(
                "corpus_fingerprint"
            ),
            "embedding_model": metadata.get(
                "embedding_model",
                self.embedding_model_name,
            ),
            "chroma_path": str(
                self.chroma_path
            ),
        }

    def _encode(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Create normalized embedding vectors."""

        if not texts:
            return []

        model = self._get_embedding_model()

        try:
            vectors = model.encode(
                list(texts),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except TypeError:
            # Compatibility fallback for older Sentence Transformer versions.
            vectors = model.encode(
                list(texts),
                show_progress_bar=False,
            )

            vectors = _normalize_vectors(vectors)

        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()

        output = [
            [
                float(component)
                for component in vector
            ]
            for vector in vectors
        ]

        if len(output) != len(texts):
            raise CorpusError(
                "Embedding model returned an unexpected "
                "number of vectors."
            )

        return output

    def _get_embedding_model(self) -> Any:
        """Load the embedding model only when it is first needed."""

        if self._embedding_model is not None:
            return self._embedding_model

        try:
            from sentence_transformers import (
                SentenceTransformer,
            )
        except ImportError as exc:
            raise CorpusError(
                "sentence-transformers is not installed. "
                "Install the retrieval dependencies first."
            ) from exc

        try:
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name
            )
        except Exception as exc:
            raise CorpusError(
                "Could not load embedding model "
                f"{self.embedding_model_name!r}."
            ) from exc

        return self._embedding_model

    def _get_client(self) -> Any:
        """Create a persistent client across Chroma API versions."""

        if self._client is not None:
            return self._client

        self.chroma_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            import chromadb
        except ImportError as exc:
            raise CorpusError(
                "chromadb is not installed. Install the "
                "retrieval dependencies first."
            ) from exc

        try:
            if hasattr(chromadb, "PersistentClient"):
                self._client = chromadb.PersistentClient(
                    path=str(self.chroma_path)
                )
            else:
                # Compatibility with Chroma 0.3.x.
                from chromadb.config import Settings

                self._client = chromadb.Client(
                    Settings(
                        chroma_db_impl=(
                            "duckdb+parquet"
                        ),
                        persist_directory=str(
                            self.chroma_path
                        ),
                        anonymized_telemetry=False,
                    )
                )
        except Exception as exc:
            raise CorpusError(
                "Could not initialize persistent Chroma "
                f"storage at {self.chroma_path}."
            ) from exc

        return self._client

    def _get_collection_if_exists(
        self,
    ) -> Any | None:
        """Return the configured collection if it exists."""

        client = self._get_client()

        try:
            return client.get_collection(
                name=self.collection_name
            )
        except Exception as exc:
            # Chroma exception types differ between releases.
            if _looks_like_missing_collection(exc):
                return None

            raise CorpusError(
                "Could not inspect the Chroma collection."
            ) from exc

    def _require_collection(self) -> Any:
        """Return the collection or report that bootstrap is required."""

        collection = self._get_collection_if_exists()

        if collection is None:
            raise CorpusError(
                "The documentation index has not been built. "
                "Run corpus bootstrap before retrieval."
            )

        return collection

    def _persist_if_supported(self) -> None:
        """Persist explicitly for Chroma versions that require it."""

        client = self._get_client()
        persist = getattr(
            client,
            "persist",
            None,
        )

        if callable(persist):
            persist()


def _first_result_list(
    value: Any,
) -> list[Any]:
    """Extract the first query's results from Chroma output."""

    if value is None:
        return []

    if not isinstance(value, list):
        raise CorpusError(
            "Chroma returned an unexpected result structure."
        )

    if not value:
        return []

    first = value[0]

    if not isinstance(first, list):
        raise CorpusError(
            "Chroma returned an unexpected nested structure."
        )

    return first


def _normalize_vectors(
    vectors: Any,
) -> Any:
    """Normalize embedding vectors for older model APIs."""

    import numpy as np

    array = np.asarray(
        vectors,
        dtype=float,
    )

    if array.ndim != 2:
        raise CorpusError(
            "Embedding model returned invalid vectors."
        )

    norms = np.linalg.norm(
        array,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0
    return array / norms


def _looks_like_missing_collection(
    error: Exception,
) -> bool:
    """Handle missing-collection errors across Chroma releases."""

    name = type(error).__name__.lower()
    message = str(error).lower()

    return (
        "notfound" in name
        or "not found" in message
        or "does not exist" in message
        or "unknown collection" in message
    )


def _validate_relevance(
    name: str,
    value: float,
) -> None:
    """Validate a normalized relevance threshold."""

    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a number between zero and one."
        )

    if (
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(
            f"{name} must be a number between zero and one."
        )