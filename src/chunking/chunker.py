import json
import re
from pathlib import Path
from itertools import zip_longest
from common.config import load_source_config
from utils.logger import get_project_logger

logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

DEFAULT_SOURCE = "python_docs"
PROCESSED_DIR_FOR = lambda src: PROJECT_ROOT / "data" / "processed" / src
CHUNKS_FILE_FOR = lambda src: PROJECT_ROOT / "data" / "chunks" / f"{src}.json"

RAW_DIR_FOR = lambda src: PROJECT_ROOT / "data" / "raw" / src
PROCESSED_DIR_FOR = lambda src: PROJECT_ROOT / "data" / "processed" / src
CHUNKS_FILE_FOR = lambda src: PROJECT_ROOT / "data" / "chunks" / f"{src}.json"

MAX_CHUNK_SIZE = 1000


def split_headings_legacy(markdown_text: str) -> list[tuple[str, str]]:
    parts = re.split(r'^(?:##|####)\s+(.+)$', markdown_text, flags=re.M)
    headings = parts[1::2]
    contents = parts[2::2]

    sections = [("Introduction", parts[0].strip())] if parts[0].strip() else []
    sections += [(h.strip(), c.strip() if c else "")
                 for h, c in zip_longest(headings, contents, fillvalue="")]
    return sections


def split_headings_new(markdown_text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r'^(#{1,6})\s+(.*)$', flags=re.M)
    matches = list(pattern.finditer(markdown_text))
    if not matches:
        return [("Introduction", markdown_text.strip())] if markdown_text.strip() else []
    sections = []
    if matches[0].start() > 0:
        pre = markdown_text[:matches[0].start()].strip()
        if pre:
            sections.append(("Introduction", pre))
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        content = markdown_text[start:end].strip()
        sections.append((f"{'#' * level} {title}", content))
    return sections


def split_headings(markdown_text: str, mode: str = "legacy") -> list[tuple[str, str]]:
    if mode == "legacy":
        return split_headings_legacy(markdown_text)
    elif mode == "new":
        return split_headings_new(markdown_text)
    else:
        raise ValueError("mode must be 'legacy' or 'new'")


def split_long_paragraph(paragraph: str, max_size: int) -> list[str]:
    paragraph = (paragraph or "").strip()
    if not paragraph:
        return []

    if len(paragraph) <= max_size:
        return [paragraph]

    words = paragraph.split()
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

    raw_paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', text) if p.strip()]
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


def chunk_one_file(md_path: Path, max_chunk_size: int = MAX_CHUNK_SIZE, mode: str = "legacy") -> list[dict]:
    text = md_path.read_text(encoding="utf-8")
    sections = split_headings(text, mode=mode)
    logger.debug(f"{md_path.name}: found {len(sections)} sections")

    chunks = []
    for i, (heading, section_text) in enumerate(sections):
        pieces = split_by_size(section_text, max_chunk_size)
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


def chunk_all(
    source_name: str = DEFAULT_SOURCE,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    mode: str = "legacy",
):
    config = load_source_config(source_name)
    raw_markdown = config.get("raw_markdown", False)


    if raw_markdown:
        input_dir = RAW_DIR_FOR(source_name)
    else:
        input_dir = PROCESSED_DIR_FOR(source_name)

    chunks_file = CHUNKS_FILE_FOR(source_name)

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        return


    md_files = sorted(input_dir.rglob("*.md"))

    logger.info(
        f"Chunking {len(md_files)} files for source '{source_name}' (raw_markdown={raw_markdown})"
    )

    if not md_files:
        logger.warning(f"No .md files found in {input_dir}")
        return

    all_chunks = []
    for md_path in md_files:
        all_chunks.extend(
            chunk_one_file(md_path, max_chunk_size=max_chunk_size, mode=mode)
        )

    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    chunks_file.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        f"Saved {len(all_chunks)} chunks from {len(md_files)} files to {chunks_file}"
    )

