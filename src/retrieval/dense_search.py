import json
import numpy as np
from utils.logger import get_project_logger
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import Optional, Tuple, List

logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent


CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "python_docs.json"
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "chunks" / "python_docs_embeddings.npy"

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
TOP_K = 5
_EPS = 1e-8


def chunks_path_for(source_name: str) -> Path:
    return PROJECT_ROOT / "data" / "chunks" / f"{source_name}.json"


def embeddings_path_for(source_name: str) -> Path:
    return PROJECT_ROOT / "data" / "chunks" / f"{source_name}_embeddings.npy"


def load_data(
    source_name: Optional[str] = None,
    chunks_path: Optional[Path] = None,
    embeddings_path: Optional[Path] = None,
    mmap_mode: Optional[str] = None,
) -> Tuple[List[dict], Optional[np.ndarray]]:

    if chunks_path is None:
        chunks_path = chunks_path_for(source_name) if source_name else CHUNKS_FILE
    if embeddings_path is None:
        embeddings_path = embeddings_path_for(source_name) if source_name else EMBEDDINGS_FILE

    if not chunks_path.exists():
        logger.error("Chunks file not found: %s", chunks_path)
        return [], None

    try:
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.exception("Failed to load chunks from %s: %s", chunks_path, e)
        return [], None

    if not embeddings_path.exists():
        logger.warning("Embeddings file not found: %s", embeddings_path)
        return chunks, None

    try:
        if mmap_mode:
            embeddings = np.load(embeddings_path, mmap_mode=mmap_mode)
        else:
            embeddings = np.load(embeddings_path)
        embeddings = np.asarray(embeddings, dtype=np.float32)
    except Exception as e:
        logger.exception("Failed to load embeddings from %s: %s", embeddings_path, e)
        return chunks, None

    logger.info("Loaded %d chunks and embeddings shape=%s", len(chunks), embeddings.shape)
    return chunks, embeddings

def cosine_similarity(query_vec: np.ndarray, all_vecs: np.ndarray) -> np.ndarray:

    query_vec = np.asarray(query_vec, dtype=np.float32).ravel()
    all_vecs = np.asarray(all_vecs, dtype=np.float32)
    if all_vecs.ndim != 2 or query_vec.ndim != 1:
        raise ValueError("all_vecs must be 2-D and query_vec 1-D")


    numerator = all_vecs.dot(query_vec)


    q_norm = np.linalg.norm(query_vec)
    vecs_norms = np.linalg.norm(all_vecs, axis=1)
    denom = vecs_norms * (q_norm if q_norm > 0 else _EPS)
    denom = np.where(denom > 0, denom, _EPS)

    similarities = numerator / denom

    similarities = np.clip(similarities, -1.0, 1.0)
    return similarities


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:

    n = scores.shape[0]
    if n == 0 or k <= 0:
        return np.array([], dtype=int)
    k = min(k, n)
    if k == n:
        return np.argsort(-scores)
    idx_part = np.argpartition(-scores, k - 1)[:k]
    return idx_part[np.argsort(-scores[idx_part])]


def search(query: str, model: SentenceTransformer, chunks: list[dict], embeddings: np.ndarray, top_k: int = TOP_K) -> list[dict]:

    if embeddings is None:
        raise ValueError("embeddings is None — compute embeddings first")


    try:
        q_enc = model.encode([query])
    except TypeError:
        q_enc = model.encode([query])

    q_vec = np.asarray(q_enc)[0]

    scores = cosine_similarity(q_vec, embeddings)
    indices = _topk_indices(scores, top_k)

    results = []
    for i in indices:
        chunk = chunks[i]
        results.append({
            "score": float(scores[i]),
            "heading": chunk.get("heading"),
            "text": chunk.get("text"),
            "source_file": chunk.get("source_file"),
        })
    return results


