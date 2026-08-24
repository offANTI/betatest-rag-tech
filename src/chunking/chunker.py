import re
from pathlib import Path

from utils.logger import get_project_logger

logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "python_docs"
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "python_docs.json"

MAX_CHUNK_SIZE = 1000

def split_headings (markdown_text: str) -> list[tuple[str,str]]:
    parts = re.split(r'^##\s+(.+)$', markdown_text, flags=re.M)
    result = []

    intro = parts[0].strip()
    if intro:
        result.append(("Introduction",intro))

    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        result.append((title, content))

    return result


if __name__ == "__main__":
    script_dir = Path(__file__).parent.parent.parent
    file_path = script_dir / "data" / "processed" / "python_docs" / "3_library_intro.md"

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            test_text = file.read()


        parsed_sections = split_headings(test_text)


        print(f"Test file: {file_path}")
        print(f"Found chunks: {len(parsed_sections)}\n")
        for title, content in parsed_sections:
            print(f"=== Title: {title} ===")
            print(content)
            print("-" * 30)
    except FileNotFoundError:
        print(f"Error: file '{file_path}'not found.")