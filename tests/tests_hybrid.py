import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid import chunk_ranks


def test_chunk_ranks_basic():
    scores = np.array([0.65, 0.89, 0.45, 0.20, 0.99])
    ranks = chunk_ranks(scores)


    assert ranks[4] == 0

    assert ranks[3] == 4


def test_chunk_ranks_are_unique_and_complete():
    scores = np.random.rand(50)
    ranks = chunk_ranks(scores)

    assert sorted(ranks.tolist()) == list(range(50))


def test_chunk_ranks_higher_score_gets_lower_rank():
    scores = np.array([1.0, 5.0, 3.0])
    ranks = chunk_ranks(scores)


    assert ranks[1] < ranks[2] < ranks[0]