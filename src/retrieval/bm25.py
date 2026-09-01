import json
import re
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from utils.logger import get_project_logger

logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "python_docs.json"
TOP_K = 5


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    text_lower = text.lower()
    cleaned_text = re.sub(r"[^\w\s]", "", text_lower)
    tokens = cleaned_text.split()
    return tokens


def batch_chunks() -> list[dict]:
    logger.info(f"Loading chunks from {CHUNKS_FILE}")
    text = CHUNKS_FILE.read_text(encoding="utf-8")
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
  


if __name__ == "__main__":
    chunks = batch_chunks()
    bm25 = build_bm25_index(chunks)

    query = "how to check if a variable is a list"
    results = bm25_search(query, bm25, chunks)

    for r in results:
        logger.info(f"[{r['score']:.4f}] [{r['heading']}] {r['text'][:150]}...")