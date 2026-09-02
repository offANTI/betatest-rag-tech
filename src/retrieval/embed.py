import json
import argparse
from typing import List, Optional
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logger import get_project_logger

logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

DEFAULT_SOURCE = "python_docs"
DEFAULT_CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / f"{DEFAULT_SOURCE}.json"
DEFAULT_EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "chunks" / f"{DEFAULT_SOURCE}_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"


def chunks_path_for(source_name: str) -> Path:
    return PROJECT_ROOT / "data" / "chunks" / f"{source_name}.json"


def embeddings_path_for(source_name: str) -> Path:
    return PROJECT_ROOT / "data" / "chunks" / f"{source_name}_embeddings.npy"


def load_chunks(chunks_path: Optional[Path] = None, source_name: Optional[str] = None) -> List[dict]:

    if chunks_path is None:
        chunks_path = chunks_path_for(source_name) if source_name else DEFAULT_CHUNKS_FILE

    if not chunks_path.exists():
        logger.error("Chunks file not found: %s", chunks_path)
        return []

    try:
        text = chunks_path.read_text(encoding="utf-8")
        chunks = json.loads(text)
        logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
        return chunks
    except Exception as e:
        logger.exception("Failed to load chunks from %s: %s", chunks_path, e)
        return []


class EmbeddingManager:

    _model: Optional[SentenceTransformer] = None
    _model_name: Optional[str] = None

    @classmethod
    def load_model(cls, model_name: str = MODEL_NAME, device: Optional[str] = None) -> SentenceTransformer:
        if cls._model is not None and cls._model_name == model_name:
            return cls._model
        logger.info("Loading SentenceTransformer model: %s", model_name)
        if device:
            model = SentenceTransformer(model_name, device=device)
        else:
            model = SentenceTransformer(model_name)
        cls._model = model
        cls._model_name = model_name
        return model


def embed_chunks(
    chunks: List[dict],
    model: Optional[SentenceTransformer] = None,
    model_name: str = MODEL_NAME,
    device: Optional[str] = None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:

    if not chunks:
        logger.warning("No chunks to embed")
        return np.zeros((0,), dtype=np.float32)

    if model is None:
        model = EmbeddingManager.load_model(model_name=model_name, device=device)

    texts = [chunk.get("text", "") for chunk in chunks]
    logger.info(
        "Encoding %d chunks with model %s (batch_size=%d, device=%s)",
        len(texts),
        model_name,
        batch_size,
        device or "default",
    )

    
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
    except TypeError:
        # fallback if convert_to_numpy isn't supported in this SentenceTransformer version
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    logger.info("Computed embeddings shape=%s dtype=%s", embeddings.shape, embeddings.dtype)
    return embeddings


def save_embeddings(embeddings: np.ndarray, out_path: Optional[Path] = None, source_name: Optional[str] = None) -> None:

    if out_path is None:
        out_path = embeddings_path_for(source_name) if source_name else DEFAULT_EMBEDDINGS_FILE

    out_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_to_save = np.asarray(embeddings, dtype=np.float32)
    np.save(out_path, embeddings_to_save)
    logger.info(
        "Saved embeddings to %s (dtype=%s, size=%.2f MB)",
        out_path,
        embeddings_to_save.dtype,
        out_path.stat().st_size / (1024 * 1024),
    )


def load_embeddings(emb_path: Optional[Path] = None, source_name: Optional[str] = None) -> Optional[np.ndarray]:

    if emb_path is None:
        emb_path = embeddings_path_for(source_name) if source_name else DEFAULT_EMBEDDINGS_FILE

    if not emb_path.exists():
        logger.warning("Embeddings file not found: %s", emb_path)
        return None

    try:
        arr = np.load(emb_path)
        logger.info("Loaded embeddings from %s shape=%s dtype=%s", emb_path, arr.shape, arr.dtype)
        return arr
    except Exception as e:
        logger.exception("Failed to load embeddings from %s: %s", emb_path, e)
        return None


def embed_source(
    source_name: str,
    model_name: str = MODEL_NAME,
    device: Optional[str] = None,
    batch_size: int = 64,
    show_progress: bool = True,
    out_path: Optional[Path] = None,
) -> Optional[Path]:

    chunks_file = chunks_path_for(source_name)
    if not chunks_file.exists():
        logger.error("Chunks file for source not found: %s", chunks_file)
        return None

    chunks = load_chunks(chunks_file)
    if not chunks:
        logger.error("No chunks loaded for source %s, aborting embed", source_name)
        return None

    model = EmbeddingManager.load_model(model_name=model_name, device=device)
    embeddings = embed_chunks(
        chunks,
        model=model,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    if out_path is None:
        out_path = embeddings_path_for(source_name)
    save_embeddings(embeddings, out_path)
    return out_path


def main(argv: Optional[List[str]] = None) -> Optional[Path]:
    parser = argparse.ArgumentParser(description="Compute and save embeddings for chunks")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Source name (used to find data/chunks/<source>.json)")
    parser.add_argument("--chunks", type=Path, default=None, help="Path to chunks.json (overrides --source)")
    parser.add_argument("--out", type=Path, default=None, help="Output .npy file (overrides default)")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="SentenceTransformer model name")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. 'cpu' or 'cuda'")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for encoding")
    parser.add_argument("--no-progress", action="store_true", help="Hide progress bar")
    args = parser.parse_args(argv)

    # Determine files
    if args.chunks:
        chunks_path = args.chunks
    else:
        chunks_path = chunks_path_for(args.source)

    out_path = args.out if args.out else embeddings_path_for(args.source)

    chunks = load_chunks(chunks_path)
    if not chunks:
        logger.error("No chunks loaded, exiting")
        return None

    model = EmbeddingManager.load_model(model_name=args.model, device=args.device)
    embeddings = embed_chunks(
        chunks, model=model, device=args.device, batch_size=args.batch, show_progress=not args.no_progress
    )
    save_embeddings(embeddings, out_path)
    return out_path


if __name__ == "__main__":
    main()