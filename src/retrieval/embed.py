from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logger import get_project_logger

logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "python_docs.json"
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "chunks" / "python_docs_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    logger.info(f"Loading chunks from {CHUNKS_FILE}")
    text = CHUNKS_FILE.read_text(encoding="utf-8")
    chunks = json.loads(text)
    return chunks


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    texts = [chunk["text"] for chunk in chunks]
    logger.info(f"Encoding {len(chunks)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def save_embeddings(embeddings: np.ndarray) -> None:
    np.save(EMBEDDINGS_FILE, embeddings)
    logger.info(f"Saved embeddings to {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    chunks = load_chunks()
    logger.info(f"Loaded {len(chunks)} chunks")
    embeddings = embed_chunks(chunks)
    logger.info(f"Computed embeddings, shape: {embeddings.shape}")
    save_embeddings(embeddings)
    logger.info(f"Saved embeddings to {EMBEDDINGS_FILE}")