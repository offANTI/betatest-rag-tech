import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.dense_search import cosine_similarity


def test_cosine_similarity_identical_direction():
    query = np.array([3.0, 4.0])
    vecs = np.array([[6.0, 8.0]])

    result = cosine_similarity(query, vecs)
    assert np.isclose(result[0], 1.0)


def test_cosine_similarity_orthogonal():
    query = np.array([1.0, 0.0])
    vecs = np.array([[0.0, 1.0]])

    result = cosine_similarity(query, vecs)
    assert np.isclose(result[0], 0.0)


def test_cosine_similarity_opposite():
    query = np.array([1.0, 1.0])
    vecs = np.array([[-1.0, -1.0]])

    result = cosine_similarity(query, vecs)
    assert np.isclose(result[0], -1.0)


def test_cosine_similarity_multiple_vectors():
    query = np.array([1.0, 0.0])
    vecs = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])

    result = cosine_similarity(query, vecs)
    assert result.shape == (3,)
    assert np.isclose(result[0], 1.0)
    assert np.isclose(result[1], 0.0)
    assert np.isclose(result[2], -1.0)