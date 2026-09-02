from pathlib import Path
import re
import json
import argparse
from collections import Counter, defaultdict

from utils.logger import get_project_logger

logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent


def split_headings_new(markdown_text: str):
    if not (markdown_text or "").strip():
        return []

    pattern = re.compile(r'^(#{1,6})\s+(.*)$', flags=re.M)
    matches = list(pattern.finditer(markdown_text))
    if not matches:
        # весь текст как введение
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


def split_long_paragraph(paragraph: str, max_size: int):
    paragraph = (paragraph or "").strip()
    if not paragraph:
        return []

    if len(paragraph) <= max_size:
        return [paragraph]

    words = paragraph.split()
    chunks = []
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


def split_by_size(text: str, max_size: int):
    text = text or ""
    if not text.strip():
        return []

    if len(text) <= max_size:
        return [text.strip()]


    raw_paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', text) if p.strip()]
    chunks = []

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


def chunk_one_file_v2(md_path: Path, max_chunk_size: int):
    text = md_path.read_text(encoding="utf-8")
    sections = split_headings_new(text)
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
    return chunks


def build_chunks_v2(processed_dir: Path, max_chunk_size: int):
    md_files = sorted(processed_dir.glob("*.md"))
    logger.info("Found %d markdown files in %s", len(md_files), processed_dir)
    all_chunks = []
    per_file_counts = {}
    for md_path in md_files:
        chunks = chunk_one_file_v2(md_path, max_chunk_size)
        all_chunks.extend(chunks)
        per_file_counts[md_path.name] = len(chunks)
        logger.debug("%s -> %d chunks", md_path.name, len(chunks))

    return all_chunks, per_file_counts


def load_old_chunks(chunks_file: Path):
    if not chunks_file.exists():
        logger.warning("Old chunks file not found: %s", chunks_file)
        return []
    try:
        text = chunks_file.read_text(encoding="utf-8")
        return json.loads(text)
    except Exception as e:
        logger.exception("Failed to load old chunks file: %s", e)
        return []


def compare_and_report(old_chunks, new_chunks, old_per_file_counts, new_per_file_counts, show_samples=5):
    old_total = len(old_chunks)
    new_total = len(new_chunks)
    print("\nSummary:")
    print(f"  Old chunks total: {old_total}")
    print(f"  New chunks total: {new_total}")
    print(f"  Delta: {new_total - old_total} ({(new_total - old_total)/max(1,old_total):+.2%})")


    old_headings = Counter(c.get("heading", "UNDEF") for c in old_chunks)
    new_headings = Counter(c.get("heading", "UNDEF") for c in new_chunks)

    print("\nTop 10 headings (old):")
    for h, cnt in old_headings.most_common(10):
        print(f"  {cnt:5d}  {h}")

    print("\nTop 10 headings (new):")
    for h, cnt in new_headings.most_common(10):
        print(f"  {cnt:5d}  {h}")


    all_files = sorted(set(list(old_per_file_counts.keys()) + list(new_per_file_counts.keys())))
    diffs = []
    for f in all_files:
        old_c = old_per_file_counts.get(f, 0)
        new_c = new_per_file_counts.get(f, 0)
        if old_c != new_c:
            diffs.append((f, old_c, new_c))
    if diffs:
        print("\nFiles with different chunk counts (old -> new):")
        for f, o, n in diffs:
            print(f"  {o:4d} -> {n:4d}   {f}")
    else:
        print("\nNo per-file chunk count differences detected.")


    print("\nSample new chunks (up to %d):" % show_samples)
    for i, c in enumerate(new_chunks[:show_samples]):
        print(f"  - [{c['chunk_id']}] {c['source_file']} | heading: {c['heading'][:80]!s} | text_snippet: {c['text'][:60]!s}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="python_docs", help="Source name under data/processed and data/chunks")
    parser.add_argument("--out", type=Path, default=None, help="Output chunks_v2.json path (defaults to data/chunks/<source>_v2.json)")
    parser.add_argument("--max-chunk-size", type=int, default=1000)
    parser.add_argument("--preview", action="store_true", help="Only show comparison/preview, do not save")
    parser.add_argument("--force", action="store_true", help="Overwrite output file if exists")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample chunks to show in preview")
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / "data" / "processed" / args.source
    chunks_dir = PROJECT_ROOT / "data" / "chunks"
    old_chunks_file = chunks_dir / f"{args.source}.json"
    out_file = args.out if args.out else chunks_dir / f"{args.source}_v2.json"

    if not processed_dir.exists():
        logger.error("Processed directory not found: %s", processed_dir)
        return

    new_chunks, new_per_file_counts = build_chunks_v2(processed_dir, args.max_chunk_size)


    old_chunks = load_old_chunks(old_chunks_file)
    old_per_file_counts = defaultdict(int)
    for c in old_chunks:
        old_per_file_counts[c.get("source_file", "UNKNOWN")] += 1


    compare_and_report(old_chunks, new_chunks, old_per_file_counts, new_per_file_counts, show_samples=args.samples)

    if args.preview:
        print("\nPreview mode: no file written. Use without --preview to save, or --force to overwrite.")
        return


    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and not args.force:
        logger.error("Output file %s already exists. Use --force to overwrite.", out_file)
        return

    out_file.write_text(json.dumps(new_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved new chunks to %s (total %d chunks)", out_file, len(new_chunks))
    print(f"\nWrote: {out_file}  (total chunks: {len(new_chunks)})")
    print("Note: old chunks file was not modified:", old_chunks_file if old_chunks_file.exists() else "(not found)")

if __name__ == "__main__":
    main()