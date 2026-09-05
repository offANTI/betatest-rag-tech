import json
import re
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from utils.logger import get_project_logger

logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

DEFAULT_SOURCE = "python_docs"
TOP_K = 5


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    text_lower = text.lower()
    cleaned_text = re.sub(r"[^\w\s]", "", text_lower)
    tokens = cleaned_text.split()
    return tokens


def chunks_path_for(source_name: str) -> Path:
    return PROJECT_ROOT / "data" / "chunks" / f"{source_name}.json"


def batch_chunks(source_name: str = DEFAULT_SOURCE) -> list[dict]:
    chunks_file = chunks_path_for(source_name)
    logger.info(f"Loading chunks from {chunks_file}")
    text = chunks_file.read_text(encoding="utf-8")
    chunks = json.loads(text)
    return chunks


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized_texts = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_texts)



def bm25_search(query: str, bm25: BM25Okapi, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    results = []
    for i in top_indices:
        chunk = chunks[i]
        results.append(
            {
                "score": float(scores[i]),
                "heading": chunk["heading"],
                "text": chunk["text"],
                "source_file": chunk["source_file"],
            }
        )
    return results
  

