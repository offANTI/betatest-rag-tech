import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.bm25 import tokenize


def test_tokenize_basic():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_removes_punctuation():
    result = tokenize("list, tuple, and dict.")
    assert "," not in "".join(result)
    assert "." not in "".join(result)
    assert result == ["list", "tuple", "and", "dict"]


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_tokenize_consistent_for_matching():

    chunk_text = "Return True if object is an instance of classinfo."
    query_text = "check object instance classinfo"

    chunk_tokens = tokenize(chunk_text)
    query_tokens = tokenize(query_text)


    assert "object" in chunk_tokens, query_tokens
    assert "instance" in chunk_tokens, query_tokens
    assert "classinfo" in chunk_tokens, query_tokens

def test_tokenize_splits_on_punctuation_without_space():

    result = tokenize("isinstance(object, classinfo)")
    assert result == ["isinstanceobject", "classinfo"]