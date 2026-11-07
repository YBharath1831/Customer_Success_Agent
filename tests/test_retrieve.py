"""Tests for corpus loading and passage chunking."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.retrieve import (
    CorpusError,
    KnowledgeDocument,
    chunk_documents,
    chunk_markdown,
    corpus_fingerprint,
    load_and_chunk_corpus,
    load_documents,
    split_markdown_sections,
)


def make_document(
    *,
    doc_id: str = "DOC-TEST-001",
    title: str = "Test article",
    content: str = "# Overview\n\nExample content.",
) -> KnowledgeDocument:
    """Create a knowledge document for unit tests."""

    return KnowledgeDocument(
        doc_id=doc_id,
        title=title,
        category="testing",
        applies_to="Test environment",
        content=content,
        related_docs=(),
        last_reviewed_days_ago=0,
    )


def write_corpus(
    path: Path,
    records: list,
) -> None:
    """Write corpus records as UTF-8 JSON."""

    path.write_text(
        json.dumps(records),
        encoding="utf-8",
    )


def valid_corpus_record(
    *,
    doc_id: str = "DOC-TEST-001",
    content: str = "# Overview\n\nExample content.",
) -> dict:
    """Return one valid documentation record."""

    return {
        "doc_id": doc_id,
        "title": "Test article",
        "category": "testing",
        "applies_to": "Test environment",
        "content": content,
        "related_docs": ["DOC-TEST-002"],
        "last_reviewed_days_ago": 5,
    }


def test_load_documents_reads_valid_corpus(
    tmp_path,
) -> None:
    corpus_path = tmp_path / "documentation.json"
    write_corpus(
        corpus_path,
        [valid_corpus_record()],
    )

    documents = load_documents(corpus_path)

    assert len(documents) == 1

    document = documents[0]

    assert document.doc_id == "DOC-TEST-001"
    assert document.title == "Test article"
    assert document.category == "testing"
    assert document.related_docs == (
        "DOC-TEST-002",
    )
    assert document.last_reviewed_days_ago == 5


def test_load_documents_rejects_missing_file(
    tmp_path,
) -> None:
    corpus_path = tmp_path / "missing.json"

    with pytest.raises(
        CorpusError,
        match="does not exist",
    ):
        load_documents(corpus_path)


def test_load_documents_rejects_invalid_json(
    tmp_path,
) -> None:
    corpus_path = tmp_path / "invalid.json"
    corpus_path.write_text(
        "{not valid JSON",
        encoding="utf-8",
    )

    with pytest.raises(
        CorpusError,
        match="invalid JSON",
    ):
        load_documents(corpus_path)


def test_load_documents_rejects_non_array(
    tmp_path,
) -> None:
    corpus_path = tmp_path / "object.json"
    corpus_path.write_text(
        json.dumps(valid_corpus_record()),
        encoding="utf-8",
    )

    with pytest.raises(
        CorpusError,
        match="JSON array",
    ):
        load_documents(corpus_path)


def test_load_documents_rejects_empty_corpus(
    tmp_path,
) -> None:
    corpus_path = tmp_path / "empty.json"
    write_corpus(corpus_path, [])

    with pytest.raises(
        CorpusError,
        match="no documents",
    ):
        load_documents(corpus_path)


def test_load_documents_rejects_duplicate_ids(
    tmp_path,
) -> None:
    corpus_path = tmp_path / "duplicates.json"
    write_corpus(
        corpus_path,
        [
            valid_corpus_record(
                doc_id="DOC-DUPLICATE"
            ),
            valid_corpus_record(
                doc_id="DOC-DUPLICATE"
            ),
        ],
    )

    with pytest.raises(
        CorpusError,
        match="Duplicate document ID",
    ):
        load_documents(corpus_path)


@pytest.mark.parametrize(
    "missing_field",
    [
        "doc_id",
        "title",
        "category",
        "applies_to",
        "content",
    ],
)
def test_required_document_fields_are_enforced(
    tmp_path,
    missing_field,
) -> None:
    corpus_path = tmp_path / "missing-field.json"
    record = valid_corpus_record()
    del record[missing_field]

    write_corpus(corpus_path, [record])

    with pytest.raises(
        CorpusError,
        match=missing_field,
    ):
        load_documents(corpus_path)


def test_related_docs_must_be_an_array(
    tmp_path,
) -> None:
    corpus_path = tmp_path / "related-docs.json"
    record = valid_corpus_record()
    record["related_docs"] = "DOC-OTHER"

    write_corpus(corpus_path, [record])

    with pytest.raises(
        CorpusError,
        match="related_docs",
    ):
        load_documents(corpus_path)


@pytest.mark.parametrize(
    "invalid_value",
    [-1, 1.5, "five", True],
)
def test_last_reviewed_days_must_be_valid(
    tmp_path,
    invalid_value,
) -> None:
    corpus_path = tmp_path / "reviewed.json"
    record = valid_corpus_record()
    record["last_reviewed_days_ago"] = (
        invalid_value
    )

    write_corpus(corpus_path, [record])

    with pytest.raises(
        CorpusError,
        match="last_reviewed_days_ago",
    ):
        load_documents(corpus_path)


def test_markdown_is_split_at_headings() -> None:
    content = """
Introductory text.

# Symptoms

The deployment fails.

## Resolution

Restart the deployment.

# Notes

Contact support if the problem continues.
""".strip()

    sections = split_markdown_sections(content)

    assert len(sections) == 4

    assert sections[0].heading == ""
    assert sections[0].body == "Introductory text."

    assert sections[1].heading == "# Symptoms"
    assert sections[1].body == (
        "The deployment fails."
    )

    assert sections[2].heading == "## Resolution"
    assert sections[2].body == (
        "Restart the deployment."
    )

    assert sections[3].heading == "# Notes"


def test_small_sections_remain_whole() -> None:
    content = """
# Symptoms

Invalid credentials are shown.

# Resolution

1. Check the username.
2. Reset the password.
3. Try signing in again.
""".strip()

    chunks = chunk_markdown(
        content,
        max_chars=400,
        overlap_chars=50,
    )

    assert len(chunks) == 2

    assert chunks[0] == (
        "# Symptoms\n\n"
        "Invalid credentials are shown."
    )

    assert chunks[1].startswith("# Resolution")
    assert "1. Check the username." in chunks[1]
    assert "3. Try signing in again." in chunks[1]


def test_large_section_is_split() -> None:
    paragraphs = [
        (
            f"Paragraph {number}. "
            "This is documentation text containing enough "
            "words to require deterministic chunking."
        )
        for number in range(1, 15)
    ]

    content = (
        "# Long section\n\n"
        + "\n\n".join(paragraphs)
    )

    chunks = chunk_markdown(
        content,
        max_chars=300,
        overlap_chars=40,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.startswith("# Long section")
        assert len(chunk) <= 300


def test_chunking_is_deterministic() -> None:
    content = (
        "# Resolution\n\n"
        + "Use this repeatable resolution step. " * 50
    )

    first = chunk_markdown(
        content,
        max_chars=300,
        overlap_chars=50,
    )

    second = chunk_markdown(
        content,
        max_chars=300,
        overlap_chars=50,
    )

    assert first == second


def test_empty_content_returns_no_chunks() -> None:
    assert chunk_markdown(
        "   \n\n   ",
        max_chars=300,
        overlap_chars=20,
    ) == []


@pytest.mark.parametrize(
    ("max_chars", "overlap_chars"),
    [
        (199, 20),
        (300, -1),
        (300, 300),
        (300, 301),
    ],
)
def test_invalid_chunk_settings_are_rejected(
    max_chars,
    overlap_chars,
) -> None:
    with pytest.raises(ValueError):
        chunk_markdown(
            "Example text",
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )


def test_passage_ids_are_stable() -> None:
    document = make_document(
        content=(
            "# Overview\n\n"
            + "Stable documentation content. " * 40
        )
    )

    first = chunk_documents(
        [document],
        max_chars=300,
        overlap_chars=40,
    )

    second = chunk_documents(
        [document],
        max_chars=300,
        overlap_chars=40,
    )

    assert [p.passage_id for p in first] == [
        p.passage_id for p in second
    ]

    assert first[0].passage_id == (
        "DOC-TEST-001#p001"
    )


def test_passage_hashes_are_stable() -> None:
    document = make_document()

    first = chunk_documents([document])
    second = chunk_documents([document])

    assert first[0].content_hash == (
        second[0].content_hash
    )
    assert len(first[0].content_hash) == 64


def test_corpus_fingerprint_is_stable() -> None:
    document = make_document()

    first_passages = chunk_documents([document])
    second_passages = chunk_documents([document])

    assert corpus_fingerprint(
        first_passages
    ) == corpus_fingerprint(second_passages)


def test_corpus_fingerprint_changes_with_content() -> None:
    original = make_document(
        content="# Overview\n\nOriginal content."
    )

    updated = replace(
        original,
        content="# Overview\n\nUpdated content.",
    )

    original_fingerprint = corpus_fingerprint(
        chunk_documents([original])
    )

    updated_fingerprint = corpus_fingerprint(
        chunk_documents([updated])
    )

    assert (
        original_fingerprint
        != updated_fingerprint
    )


def test_chroma_metadata_contains_scalar_values() -> None:
    passage = chunk_documents(
        [make_document()]
    )[0]

    metadata = passage.chroma_metadata()

    assert metadata["passage_id"] == (
        "DOC-TEST-001#p001"
    )
    assert metadata["doc_id"] == "DOC-TEST-001"
    assert metadata["chunk_index"] == 1

    assert all(
        isinstance(value, (str, int, float, bool))
        for value in metadata.values()
    )


def test_load_and_chunk_corpus(
    tmp_path,
) -> None:
    corpus_path = tmp_path / "documentation.json"

    write_corpus(
        corpus_path,
        [
            valid_corpus_record(
                doc_id="DOC-001",
                content=(
                    "# Overview\n\nFirst article."
                ),
            ),
            valid_corpus_record(
                doc_id="DOC-002",
                content=(
                    "# Overview\n\nSecond article."
                ),
            ),
        ],
    )

    passages = load_and_chunk_corpus(
        corpus_path,
        max_chars=400,
        overlap_chars=40,
    )

    assert len(passages) == 2
    assert passages[0].passage_id == (
        "DOC-001#p001"
    )
    assert passages[1].passage_id == (
        "DOC-002#p001"
    )