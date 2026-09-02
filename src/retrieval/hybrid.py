import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logger import get_project_logger
from .dense_search import  cosine_similarity, load_data
from .bm25 import build_bm25_index, tokenize, batch_chunks
logger = get_project_logger(__name__)

RRF_K = 60
TOP_K = 5

def chunk_ranks(scores: np.ndarray) -> np.ndarray:
    sorted_indices = np.argsort(scores)[::-1]
    ranks = np.argsort(sorted_indices)
    return ranks


def hybrid_search(query, model, chunks, dense_embeddings, bm25, top_k=TOP_K):
    logger.info(
        f"Hybrid search: query='{query}', corpus size={len(chunks)}, top_k={top_k}"
    )
    if isinstance(dense_embeddings, list):
        dense_embeddings = np.asarray(dense_embeddings)
    n = len(dense_embeddings)
    top_k = min(top_k, n)
    query_embedding = model.encode([query])[0]
    dense_scores = cosine_similarity(query_embedding, dense_embeddings)
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    if len(bm25_scores) != n:
        raise ValueError("BM25 scores length != number of dense embeddings")
    dense_ranks = chunk_ranks(dense_scores)
    bm25_ranks = chunk_ranks(bm25_scores)
    rrf_scores = 1.0 / (RRF_K + dense_ranks) + 1.0 / (RRF_K + bm25_ranks)
    top_k_unordered = np.argpartition(-rrf_scores, top_k)[:top_k]
    top_k_indices = top_k_unordered[np.argsort(-rrf_scores[top_k_unordered])]
    final_results = []
    for idx in top_k_indices:
        final_results.append(
            {
                "chunk_idx": int(idx),
                "text": chunks[idx]["text"],
                "rrf_score": float(rrf_scores[idx]),
                "dense_rank": int(dense_ranks[idx]),
                "bm25_rank": int(bm25_ranks[idx]),
            }
        )

    logger.info(f"Hybrid search returned {len(final_results)} results")
    return final_results




if __name__ == "__main__":
    chunks = batch_chunks()
    bm25 = build_bm25_index(chunks)

    dense_chunks, dense_embeddings = load_data()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query = "check object type python"
    results = hybrid_search(query, model, chunks, dense_embeddings, bm25)

    for r in results:
        logger.info(
            f"[RRF={r['rrf_score']:.5f}] [dense_rank={r['dense_rank']} bm25_rank={r['bm25_rank']}] {r['text'][:150]}..."
        )