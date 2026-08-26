import json
import numpy as np
from utils.logger import get_project_logger
from sentence_transformers import SentenceTransformer
from pathlib import Path
logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "python_docs.json"
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "chunks" / "python_docs_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def load_data():
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_FILE)
    return chunks, embeddings


def cosine_similarity(query_vec: np.ndarray, all_vecs: np.ndarray) -> np.ndarray:
    numerator = all_vecs @ query_vec
    query_norm = np.linalg.norm(query_vec)
    vecs_norms = np.linalg.norm(all_vecs, axis=1)
    denominator = vecs_norms * query_norm
    similarities = numerator / denominator
    return similarities

def search(query: str, model: SentenceTransformer, chunks: list[dict], embeddings: np.ndarray) -> list[dict]:
    query_vec = model.encode([query])[0]
    scores = cosine_similarity(query_vec, embeddings)
    top_indices = np.argsort(scores)[-TOP_K:][::-1]
    results = []
    for i in top_indices:
        chunk = chunks[i]
        results.append({
            "score": float(scores[i]),
            "heading": chunk["heading"],
            "text": chunk["text"],
            "source_file": chunk["source_file"],
        })
    return results




if __name__ == "__main__":
    chunks, embeddings = load_data()
    model = SentenceTransformer(MODEL_NAME)

    query = "how to check if a variable is a list"
    results = search(query, model, chunks, embeddings)

    for r in results:
        logger.info(f"[{r['heading']}] {r['text'][:150]}...")