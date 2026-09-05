import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from utils.logger import get_project_logger
from .dense_search import cosine_similarity, load_data
from .bm25 import build_bm25_index, tokenize, batch_chunks
from .embed import clean_for_embedding

logger = get_project_logger(__name__)

RRF_K = 60
TOP_K = 5


def chunk_ranks(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError("scores must be a 1-D array")
    return np.argsort(np.argsort(-scores))


def hybrid_search(
    query,
    model,
    chunks,
    dense_embeddings,
    bm25,
    top_k=TOP_K,
    method: str = "rrf",
    w_dense: float = 0.7,
    w_bm25: float = 1.3,
    rrf_k: int = RRF_K,
    candidate_pool: int = 1000,
    reranker=None,
    rerank_top: int = 20,
):
    logger.info(
        "Hybrid search: query=%r, corpus size=%d, top_k=%d, method=%s, w_dense=%.2f, w_bm25=%.2f",
        query,
        len(chunks) if chunks is not None else 0,
        top_k,
        method,
        w_dense,
        w_bm25,
    )

    dense_embeddings = np.asarray(dense_embeddings)
    if dense_embeddings.ndim == 0 or dense_embeddings.size == 0:
        logger.warning("Empty dense_embeddings provided")
        return []

    n = dense_embeddings.shape[0]

    if not chunks or len(chunks) != n:
        raise ValueError(f"length mismatch: chunks ({len(chunks) if chunks else 'None'}) vs dense_embeddings ({n})")

    try:
        top_k = int(top_k)
    except Exception:
        top_k = TOP_K
    if top_k <= 0:
        logger.info("top_k <= 0, returning empty result")
        return []
    top_k = min(top_k, n)

    try:
        query_embedding = model.encode([clean_for_embedding(query)])[0]
    except Exception as e:
        logger.exception("Failed to encode query: %s", e)
        raise

    dense_scores = cosine_similarity(query_embedding, dense_embeddings)

    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    if len(bm25_scores) != n:
        raise ValueError("BM25 scores length != number of dense embeddings")

    nd = min(candidate_pool // 2, n)
    nb = min(candidate_pool - nd, n)
    dense_top = np.argsort(-dense_scores)[:nd]
    bm25_top = np.argsort(-bm25_scores)[:nb]

    candidate_ids = []
    seen = set()
    for arr in (dense_top, bm25_top):
        for i in arr:
            if int(i) not in seen:
                seen.add(int(i))
                candidate_ids.append(int(i))

    if not candidate_ids:
        logger.info("No candidates found")
        return []

    eps = 1e-8
    scores = np.zeros(n, dtype=float)

    if method == "sum":
        dmin, dmax = float(dense_scores.min()), float(dense_scores.max())
        bmin, bmax = float(bm25_scores.min()), float(bm25_scores.max())
        d_range = dmax - dmin if dmax - dmin > eps else eps
        b_range = bmax - bmin if bmax - bmin > eps else eps
        dense_norm = (dense_scores - dmin) / d_range
        bm25_norm = (bm25_scores - bmin) / b_range
        scores = w_dense * dense_norm + w_bm25 * bm25_norm
    else:
        dense_ranks = chunk_ranks(dense_scores)
        bm25_ranks = chunk_ranks(bm25_scores)
        scores = w_dense / (rrf_k + dense_ranks) + w_bm25 / (rrf_k + bm25_ranks)

    cand_scores = np.array([scores[i] for i in candidate_ids])
    k = min(top_k, len(candidate_ids))
    if k < len(candidate_ids):
        part = np.argpartition(-cand_scores, k - 1)[:k]
        top_unordered = [candidate_ids[i] for i in part]
        top_sorted = sorted(top_unordered, key=lambda i: -scores[i])
    else:
        top_sorted = sorted(candidate_ids, key=lambda i: -scores[i])[:k]

    rerank_scores_map = {}
    if reranker is not None and len(top_sorted) > 0:
        rerank_candidates = top_sorted[:rerank_top]
        pairs = [[query, chunks[i]["text"][:1024]] for i in rerank_candidates]
        try:
            rerank_scores = reranker.predict(pairs)
            rerank_scores_map = dict(zip(rerank_candidates, rerank_scores))
            ranked = sorted(zip(rerank_candidates, rerank_scores), key=lambda x: -x[1])[:top_k]
            top_sorted = [i for i, s in ranked]
        except Exception:
            logger.exception("Reranker failed, falling back to combined scores")

    final_results = []
    dense_ranks = chunk_ranks(dense_scores)
    bm25_ranks = chunk_ranks(bm25_scores)
    for idx in top_sorted[:top_k]:
        final_results.append(
            {
                "chunk_idx": int(idx),
                "text": chunks[idx].get("text", ""),
                "score": float(scores[idx]),
                "rerank_score": float(rerank_scores_map[idx]) if idx in rerank_scores_map else None,
                "dense_score": float(dense_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "dense_rank": int(dense_ranks[idx]),
                "bm25_rank": int(bm25_ranks[idx]),
            }
        )

    logger.info("Hybrid search returned %d results (top_k=%d)", len(final_results), top_k)
    return final_results



TEST_QUERIES = {
    "python_docs": "How do I use GIL or asyncio event loops in Python?",
    "dbt_docs": "How do I use ref() to reference another model?",
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run hybrid search for a specified source.")
    parser.add_argument("source", nargs="?", default="python_docs", help="Source name (e.g., dbt_docs, python_docs)")
    parser.add_argument("--query", default=None, help="Override the default test query for this source")
    args = parser.parse_args()

    source_name = args.source
    query = args.query or TEST_QUERIES.get(source_name, "How does this work?")

    logger.info("Running search for source: %s", source_name)

    chunks = batch_chunks(source_name)
    bm25 = build_bm25_index(chunks)

    dense_chunks, dense_embeddings = load_data(source_name)

    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    results = hybrid_search(
        query=query,
        model=model,
        chunks=chunks,
        dense_embeddings=dense_embeddings,
        bm25=bm25,
        reranker=reranker,
        rerank_top=20,
    )

    for r in results:
        rerank_str = f"{r['rerank_score']:.4f}" if r["rerank_score"] is not None else "None"
        logger.info(
            f"[score={r['score']:.5f}] [rerank={rerank_str}] [dense_rank={r['dense_rank']} bm25_rank={r['bm25_rank']}] {r['text'][:150]}..."
        )