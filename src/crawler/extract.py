from pathlib import Path
from bs4 import BeautifulSoup
from utils.logger import get_project_logger

logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "python_docs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "python_docs"

def html_to_markdown(main_content) -> str:
    lines = []
    tags = main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "pre"])
    for tag in tags:
        if tag.name.startswith("h"):
            level = int(tag.name[1])
            for headerlink in tag.find_all("a", class_="headerlink"):
                headerlink.decompose()
            lines.append("#" * level + " " + tag.get_text())
        elif tag.name == "p":
            lines.append(tag.get_text())
        elif tag.name == "pre":
            lines.append("```\n" + tag.get_text() + "\n```")
    return "\n\n".join(lines)

def extract_one_file(html_path: Path) -> str | None:
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    main_content = soup.find("div", {"role": "main"})
    if main_content is None:
        logger.warning(f"No main content found in {html_path.name}")
        return None

    return html_to_markdown(main_content)

def save_markdown(html_path: Path, markdown_text: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / html_path.with_suffix(".md").name
    output_path.write_text(markdown_text, encoding="utf-8")

def extract_all():
    html_files = list(RAW_DIR.glob("*.html"))
    logger.info(f"Found {len(html_files)} HTML files to process")

    for html_path in html_files:
        markdown_text = extract_one_file(html_path)
        if markdown_text is None:
            continue
        save_markdown(html_path, markdown_text)
        logger.info(f"Processed: {html_path.name}")


if __name__ == "__main__":
    extract_all()