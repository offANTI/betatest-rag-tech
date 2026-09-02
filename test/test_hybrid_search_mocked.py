import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid import hybrid_search, chunk_ranks


def _fake_chunks():
    return [
        {"text": "isinstance checks if object is an instance of a class"},
        {"text": "type() returns the type of an object"},
        {"text": "list is a mutable sequence type"},
        {"text": "dict is a mapping type with key-value pairs"},
        {"text": "str represents immutable text sequences"},
    ]


def test_hybrid_search_returns_top_k():
    chunks = _fake_chunks()
    n = len(chunks)


    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])


    dense_embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
    ])


    fake_bm25 = MagicMock()
    fake_bm25.get_scores.return_value = np.array([5.0, 1.0, 0.5, 0.3, 0.1])

    results = hybrid_search(
        query="check instance type",
        model=fake_model,
        chunks=chunks,
        dense_embeddings=dense_embeddings,
        bm25=fake_bm25,
        top_k=3,
    )

    assert len(results) == 3

    assert results[0]["chunk_idx"] == 0
    assert results[0]["dense_rank"] == 0
    assert results[0]["bm25_rank"] == 0


def test_hybrid_search_calls_model_and_bm25_correctly():
    chunks = _fake_chunks()

    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])

    dense_embeddings = np.random.rand(5, 3)

    fake_bm25 = MagicMock()
    fake_bm25.get_scores.return_value = np.random.rand(5)

    hybrid_search(
        query="my test query",
        model=fake_model,
        chunks=chunks,
        dense_embeddings=dense_embeddings,
        bm25=fake_bm25,
        top_k=2,
    )


    fake_model.encode.assert_called_once_with(["my test query"])

    fake_bm25.get_scores.assert_called_once()


def test_hybrid_search_result_structure():
    chunks = _fake_chunks()
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
    dense_embeddings = np.random.rand(5, 3)
    fake_bm25 = MagicMock()
    fake_bm25.get_scores.return_value = np.random.rand(5)

    results = hybrid_search(
        "query", fake_model, chunks, dense_embeddings, fake_bm25, top_k=1
    )

    result = results[0]
    assert set(result.keys()) == {"chunk_idx", "text", "rrf_score", "dense_rank", "bm25_rank"}
    assert isinstance(result["text"], str)
    assert isinstance(result["rrf_score"], float)