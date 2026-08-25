from pathlib import Path
import json
import numpy as np
from utils.logger import get_project_logger

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


