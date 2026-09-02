import argparse
from pathlib import Path
from bs4 import BeautifulSoup
from utils.logger import get_project_logger
from common.config import load_source_config

logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent


def html_to_markdown(main_content) -> str:
    lines = []
    tags = main_content.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "dt", "dd"]
    )
    for tag in tags:
        if tag.name == "p" and tag.find_parent("dd"):
            continue

        if tag.name.startswith("h"):
            level = int(tag.name[1])
            for headerlink in tag.find_all("a", class_="headerlink"):
                headerlink.decompose()
            lines.append("#" * level + " " + tag.get_text())
        elif tag.name == "dt":
            for headerlink in tag.find_all("a", class_="headerlink"):
                headerlink.decompose()
            lines.append("#### " + tag.get_text())
        elif tag.name == "dd":
            lines.append(tag.get_text())
        elif tag.name == "p":
            lines.append(tag.get_text())
        elif tag.name == "pre":
            lines.append("```\n" + tag.get_text() + "\n```")

    return "\n\n".join(lines)


def extract_one_file(html_path: Path, selector: dict) -> str | None:
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    main_content = soup.find(selector["tag"], selector["attrs"])
    if main_content is None:
        logger.warning(f"No main content found in {html_path.name}")
        return None

    return html_to_markdown(main_content)


def save_markdown(html_path: Path, markdown_text: str, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / html_path.with_suffix(".md").name
    output_path.write_text(markdown_text, encoding="utf-8")


def extract_all(source_name: str):
    config = load_source_config(source_name)
    selector = config["main_content_selector"]

    raw_dir = PROJECT_ROOT / "data" / "raw" / source_name
    processed_dir = PROJECT_ROOT / "data" / "processed" / source_name

    html_files = list(raw_dir.glob("*.html"))
    logger.info(f"[{source_name}] Found {len(html_files)} HTML files to process")

    for html_path in html_files:
        markdown_text = extract_one_file(html_path, selector)
        if markdown_text is None:
            continue
        save_markdown(html_path, markdown_text, processed_dir)
        logger.info(f"[{source_name}] Processed: {html_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract markdown from a crawled documentation source")
    parser.add_argument("source", help="Source name from config/sources.yaml, e.g. python_docs")
    args = parser.parse_args()

    extract_all(args.source)