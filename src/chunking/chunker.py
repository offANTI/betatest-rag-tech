import json
import re
from pathlib import Path
from itertools import zip_longest

from utils.logger import get_project_logger

logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "python_docs"
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "python_docs.json"

MAX_CHUNK_SIZE = 1000


def split_headings(markdown_text: str) -> list[tuple[str, str]]:
    parts = re.split(r'^(?:##|####)\s+(.+)$', markdown_text, flags=re.M)
    headings = parts[1::2]
    contents = parts[2::2]

    sections = [("Introduction", parts[0].strip())] if parts[0].strip() else []
    sections += [(h.strip(), c.strip() if c else "")
                 for h, c in zip_longest(headings, contents, fillvalue="")]
    return sections

def split_long_paragraph(paragraph: str, max_size: int) -> list[str]:
    paragraph = (paragraph or "").strip()
    if not paragraph:
        return []

    if len(paragraph) <= max_size:
        return [paragraph]

    words = paragraph.split(" ")
    chunks: list[str] = []
    current = ""

    for w in words:
        if not w:
            continue

        if current:
            if len(current) + 1 + len(w) <= max_size:
                current += " " + w
                continue
            else:
                chunks.append(current)

        if len(w) <= max_size:
            current = w
        else:
            start = 0
            while start < len(w):
                part = w[start:start + max_size]
                chunks.append(part)
                start += max_size
            current = ""

    if current:
        chunks.append(current)

    return chunks


def split_by_size(text: str, max_size: int) -> list[str]:
    text = text or ""
    if not text.strip():
        return []

    if len(text) <= max_size:
        return [text.strip()]

    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []

    for paragraph in raw_paragraphs:
        pieces = split_long_paragraph(paragraph, max_size)
        for piece in pieces:
            if not piece:
                continue
            if chunks and len(chunks[-1]) + 2 + len(piece) <= max_size:
                chunks[-1] = chunks[-1] + "\n\n" + piece
            else:
                chunks.append(piece)

    return chunks

def chunk_one_file(md_path: Path) -> list[dict]:
    text = md_path.read_text(encoding="utf-8")
    sections = split_headings(text)
    logger.debug(f"{md_path.name}: found {len(sections)} sections")

    chunks = []
    for i, (heading, section_text) in enumerate(sections):
        pieces = split_by_size(section_text, MAX_CHUNK_SIZE)
        for j, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{md_path.stem}_{i:03d}_{j:03d}",
                "source_file": md_path.name,
                "heading": heading,
                "text": piece.strip(),
            })

    if not chunks:
        logger.warning(f"{md_path.name}: produced 0 chunks")
    else:
        logger.info(f"{md_path.name}: {len(chunks)} chunks")

    return chunks


def chunk_all():
    all_chunks = []
    md_files = list(PROCESSED_DIR.glob("*.md"))
    logger.info(f"Chunking {len(md_files)} files")

    for md_path in md_files:
        all_chunks.extend(chunk_one_file(md_path))

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(all_chunks)} chunks from {len(md_files)} files to {CHUNKS_FILE}")


if __name__ == "__main__":
    chunk_all()

# if __name__ == "__main__":
#     script_dir = Path(__file__).parent.parent.parent
#     file_path = script_dir / "data" / "processed" / "python_docs" / "3_library_intro.md"
#
#     try:
#         with open(file_path, "r", encoding="utf-8") as file:
#             test_text = file.read()
#
#
#         parsed_sections = split_headings(test_text)
#
#
#         print(f"Test file: {file_path}")
#         print(f"Found chunks: {len(parsed_sections)}\n")
#         for title, content in parsed_sections:
#             print(f"=== Title: {title} ===")
#             print(content)
#             print("-" * 30)
#     except FileNotFoundError:
#         print(f"Error: file '{file_path}'not found.")