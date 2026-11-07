"""Tests for persistent vector retrieval."""

import math

import numpy as np
import pytest

from src.retrieve import ( ChromaRetriever, CorpusError, Passage)

class FakeEmbeddingModel:
    """Deterministic embedding model used in unit tests."""

    def encode( self, texts, *, show_progress_bar=False, normalize_embeddings=False, **kwargs):
        vectors = []

        for text in texts:
            vector = np.asarray(
                [
                    float(len(text)),
                    float(
                        sum(ord(char) for char in text)
                        % 100
                    ),
                ],
                dtype=float,
            )

            if normalize_embeddings:
                norm = np.linalg.norm(vector)

                if norm:
                    vector = vector / norm

            vectors.append(vector)

        return np.asarray(vectors)


class LegacyFakeEmbeddingModel:
    """Simulates an older Sentence Transformers encode API."""

    def encode( self, texts, *, show_progress_bar=False, **kwargs):
        if "normalize_embeddings" in kwargs:
            raise TypeError(
                "normalize_embeddings is unsupported"
            )

        return np.asarray(
            [
                [float(len(text)), 2.0]
                for text in texts
            ],
            dtype=float,
        )


class FakeCollection:
    """Small in-memory substitute for a Chroma collection."""

    def __init__( self, *, name: str, metadata: dict) -> None:
        self.name = name
        self.metadata = dict(metadata)
        self.records: dict[str, dict] = {}
        self.query_distances: list[float] = []

    def count(self) -> int:
        return len(self.records)

    def add( self, *, ids, documents, metadatas, embeddings) -> None:
        assert (
            len(ids)
            == len(documents)
            == len(metadatas)
            == len(embeddings)
        )

        for ( passage_id, document, metadata, embedding) in zip(
            ids,
            documents,
            metadatas,
            embeddings,
            strict=True,
        ):
            self.records[passage_id] = {
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def query( self, *, query_embeddings, n_results, include) -> dict:
        assert len(query_embeddings) == 1

        selected = list(
            self.records.items()
        )[:n_results]

        distances = list(
            self.query_distances[:n_results]
        )

        while len(distances) < len(selected):
            distances.append(
                0.10 + 0.10 * len(distances)
            )

        return {
            "ids": [
                [
                    passage_id
                    for passage_id, _ in selected
                ]
            ],
            "documents": [
                [
                    record["document"]
                    for _, record in selected
                ]
            ],
            "metadatas": [
                [
                    record["metadata"]
                    for _, record in selected
                ]
            ],
            "distances": [distances],
        }

    def get( self, *, ids, include) -> dict:
        found_ids = []
        documents = []
        metadatas = []

        for passage_id in ids:
            record = self.records.get(passage_id)

            if record is None:
                continue

            found_ids.append(passage_id)
            documents.append(record["document"])
            metadatas.append(record["metadata"])

        return {
            "ids": found_ids,
            "documents": documents,
            "metadatas": metadatas,
        }


class FakeClient:
    """Small in-memory substitute for a Chroma client."""

    def __init__(self) -> None:
        self.collections: dict[
            str,
            FakeCollection,
        ] = {}

        self.persist_called = False
        self.delete_count = 0
        self.create_count = 0

    def get_collection( self, *, name: str) -> FakeCollection:
        if name not in self.collections:
            raise RuntimeError(
                f"Collection {name} does not exist"
            )

        return self.collections[name]

    def create_collection( self, *, name: str, metadata: dict) -> FakeCollection:
        collection = FakeCollection(
            name=name,
            metadata=metadata,
        )

        self.collections[name] = collection
        self.create_count += 1

        return collection

    def delete_collection( self, *, name: str) -> None:
        del self.collections[name]
        self.delete_count += 1

    def persist(self) -> None:
        self.persist_called = True


def make_passage(
    *,
    passage_id: str = "DOC-AUTH-001#p001",
    doc_id: str = "DOC-AUTH-001",
    chunk_index: int = 1,
    text: str = "Reset invalid login credentials.",
) -> Passage:
    """Create one passage for retrieval tests."""

    return Passage(
        passage_id=passage_id,
        doc_id=doc_id,
        title="Authentication help",
        category="authentication",
        applies_to="Console",
        chunk_index=chunk_index,
        text=text,
        content_hash=(
            f"hash-{passage_id}-{len(text)}"
        ),
    )


def make_passages() -> list[Passage]:
    """Create multiple passages."""

    return [
        make_passage(),
        make_passage(
            passage_id="DOC-API-001#p001",
            doc_id="DOC-API-001",
            text="Use exponential backoff for rate limits.",
        ),
        make_passage(
            passage_id="DOC-BILL-001#p001",
            doc_id="DOC-BILL-001",
            text="Review invoice and billing details.",
        ),
    ]


def make_retriever( tmp_path, *, legacy_embeddings: bool = False) -> tuple[ChromaRetriever, FakeClient]:
    """Create a retriever with fake dependencies."""

    client = FakeClient()

    retriever = ChromaRetriever(
        chroma_path=tmp_path / "chroma",
        embedding_model_name="fake-model",
        collection_name="test_collection",
    )

    retriever._client = client

    if legacy_embeddings:
        retriever._embedding_model = (
            LegacyFakeEmbeddingModel()
        )
    else:
        retriever._embedding_model = (
            FakeEmbeddingModel()
        )

    return retriever, client


def test_build_index_creates_collection(tmp_path) -> None:
    retriever, client = make_retriever( tmp_path)

    passages = make_passages()
    report = retriever.build_index(passages)

    assert report.rebuilt is True
    assert report.passage_count == 3
    assert report.collection_name == (
        "test_collection"
    )
    assert len(report.corpus_fingerprint) == 64

    assert client.create_count == 1
    assert client.delete_count == 0
    assert client.persist_called is True

    collection = client.collections[
        "test_collection"
    ]

    assert collection.count() == 3
    assert collection.metadata[
        "hnsw:space"
    ] == "cosine"

    assert collection.metadata[
        "embedding_model"
    ] == "fake-model"


def test_build_index_is_idempotent(tmp_path) -> None:
    retriever, client = make_retriever(
        tmp_path
    )

    passages = make_passages()

    first = retriever.build_index(passages)
    second = retriever.build_index(passages)

    assert first.rebuilt is True
    assert second.rebuilt is False

    assert first.corpus_fingerprint == (
        second.corpus_fingerprint
    )

    assert client.create_count == 1
    assert client.delete_count == 0


def test_changed_corpus_rebuilds_collection(tmp_path) -> None:
    retriever, client = make_retriever(
        tmp_path
    )

    original = make_passages()
    retriever.build_index(original)

    changed = make_passages()
    changed[0] = make_passage(
        text=(
            "Updated instructions for resetting "
            "invalid login credentials."
        )
    )

    report = retriever.build_index(changed)

    assert report.rebuilt is True
    assert client.delete_count == 1
    assert client.create_count == 2


def test_duplicate_passage_ids_are_rejected(tmp_path) -> None:
    retriever, _ = make_retriever(
        tmp_path
    )

    duplicate = make_passage()

    with pytest.raises(
        CorpusError,
        match="unique",
    ):
        retriever.build_index(
            [duplicate, duplicate]
        )


def test_empty_passage_list_is_rejected(tmp_path) -> None:
    retriever, _ = make_retriever( tmp_path )

    with pytest.raises(
        CorpusError,
        match="without passages",
    ):
        retriever.build_index([])


def test_invalid_batch_size_is_rejected(tmp_path) -> None:
    retriever, _ = make_retriever(tmp_path)

    with pytest.raises(ValueError, match="batch_size"):
        retriever.build_index(
            make_passages(),
            batch_size=0,
        )


def test_search_converts_cosine_distance_to_relevance(tmp_path) -> None:
    retriever, client = make_retriever(tmp_path)

    retriever.build_index(make_passages())

    collection = client.collections[
        "test_collection"
    ]

    collection.query_distances = [
        0.10,
        0.45,
        0.90,
    ]

    results = retriever.search(
        "How do I sign in?",
        top_k=3,
        minimum_relevance=0.50,
    )

    assert len(results) == 2

    assert results[0].passage_id == (
        "DOC-AUTH-001#p001"
    )
    assert results[0].distance == pytest.approx(
        0.10
    )
    assert results[0].relevance_score == (
        pytest.approx(0.90)
    )

    assert results[1].passage_id == (
        "DOC-API-001#p001"
    )
    assert results[1].relevance_score == (
        pytest.approx(0.55)
    )


def test_search_returns_empty_for_blank_query(tmp_path) -> None:
    retriever, _ = make_retriever(tmp_path)

    assert retriever.search("   ") == []


def test_search_requires_built_index(tmp_path) -> None:
    retriever, _ = make_retriever(
        tmp_path
    )

    with pytest.raises(
        CorpusError,
        match="has not been built",
    ):
        retriever.search("login failure")


@pytest.mark.parametrize(
    "top_k",
    [0, -1],
)
def test_search_rejects_invalid_top_k( tmp_path, top_k) -> None:
    retriever, _ = make_retriever( tmp_path)

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        retriever.search(
            "login",
            top_k=top_k,
        )


@pytest.mark.parametrize( "threshold", [ -0.01, 1.01, math.nan, math.inf, True])

def test_search_rejects_invalid_threshold( tmp_path, threshold) -> None:
    retriever, _ = make_retriever(tmp_path)

    with pytest.raises(
        ValueError,
        match="minimum_relevance",
    ):
        retriever.search(
            "login",
            minimum_relevance=threshold,
        )


def test_search_never_requests_more_than_collection(
    tmp_path,
) -> None:
    retriever, _ = make_retriever(
        tmp_path
    )

    retriever.build_index(
        [make_passage()]
    )

    results = retriever.search(
        "login",
        top_k=100,
    )

    assert len(results) == 1


def test_resolve_passage_returns_exact_text(
    tmp_path,
) -> None:
    retriever, _ = make_retriever(
        tmp_path
    )

    original = make_passage()
    retriever.build_index([original])

    resolved = retriever.resolve_passage(
        "DOC-AUTH-001#p001"
    )

    assert resolved is not None
    assert resolved.passage_id == (
        original.passage_id
    )
    assert resolved.doc_id == original.doc_id
    assert resolved.text == original.text
    assert resolved.content_hash == (
        original.content_hash
    )


def test_resolve_unknown_passage_returns_none(
    tmp_path,
) -> None:
    retriever, _ = make_retriever(
        tmp_path
    )

    retriever.build_index(
        [make_passage()]
    )

    assert (
        retriever.resolve_passage(
            "DOC-UNKNOWN#p999"
        )
        is None
    )


def test_empty_passage_id_returns_none(
    tmp_path,
) -> None:
    retriever, _ = make_retriever(
        tmp_path
    )

    retriever.build_index(
        [make_passage()]
    )

    assert retriever.resolve_passage("   ") is None


def test_index_status_before_build(
    tmp_path,
) -> None:
    retriever, _ = make_retriever(
        tmp_path
    )

    status = retriever.index_status()

    assert status["ready"] is False
    assert status["passage_count"] == 0
    assert status["corpus_fingerprint"] is None


def test_index_status_after_build(
    tmp_path,
) -> None:
    retriever, _ = make_retriever(
        tmp_path
    )

    retriever.build_index(make_passages())
    status = retriever.index_status()

    assert status["ready"] is True
    assert status["passage_count"] == 3
    assert len(status["corpus_fingerprint"]) == 64
    assert status["embedding_model"] == "fake-model"


def test_embeddings_are_normalized(
    tmp_path,
) -> None:
    retriever, _ = make_retriever(
        tmp_path
    )

    vectors = retriever._encode(
        ["first text", "second text"]
    )

    assert len(vectors) == 2

    for vector in vectors:
        assert np.linalg.norm(vector) == (
            pytest.approx(1.0)
        )


def test_legacy_embeddings_are_normalized(
    tmp_path,
) -> None:
    retriever, _ = make_retriever(
        tmp_path,
        legacy_embeddings=True,
    )

    vectors = retriever._encode(
        ["first text", "second text"]
    )

    assert len(vectors) == 2

    for vector in vectors:
        assert np.linalg.norm(vector) == (
            pytest.approx(1.0)
        )