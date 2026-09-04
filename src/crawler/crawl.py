import time
import os
import requests

import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from pathlib import Path
from utils.logger import get_project_logger
from common.config import load_source_config

logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

MAX_PAGES = 200
REQUEST_DELAY = 0.5


def valid_url(url: str, allowed_prefix: str, blacklist: list, visited: set, queue: list) -> bool:
    if not url.startswith(allowed_prefix):
        return False
    if url in visited or url in queue:
        return False
    if any(bad in url for bad in blacklist):
        return False
    return True

def extract_markdown_links(markdown_text: str, base_url: str) -> list[str]:
    pattern = r"(?<!\!)\[.*?\]\(([\s\S]*?)\)|<([a-zA-Z0-9+.-]+://[^\s>]+)>|^\s*\[[^\]]+\]:\s*(\S+)"

    raw_links = []
    for match in re.finditer(pattern, markdown_text, re.MULTILINE):
        url = match.group(1) or match.group(2) or match.group(3)
        if url:

            raw_links.append(url.strip().split()[0])

    normalized_links = []
    for link in raw_links:

        if link.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue


        full_url = urljoin(base_url, link)
        parsed = urlparse(full_url)


        clean_url = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                "",
            )
        )

        normalized_links.append(clean_url)
    return  normalized_links


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        full_url = urljoin(base_url, tag["href"])
        full_url = full_url.split("#")[0]
        links.append(full_url)
    return links


def fetch_page(url: str) -> requests.Response | None:
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url}: {e}")
        return None

    response.encoding = "utf-8"

    if response.status_code != 200:
        logger.warning(f"Bad status {response.status_code} for {url}")
        return None

    return response


def fetch_or_cache(
    url: str, source_name: str, raw_markdown: bool = False
) -> str | None:
    if raw_markdown:
        fetch_url = url if url.endswith(".md") else f"{url}.md"
    else:
        fetch_url = url


    parsed = urlparse(url)
    clean_path = parsed.path

    while clean_path.endswith(".md"):
        clean_path = clean_path[:-3]

    ext = ".md" if raw_markdown else ".html"
    filename = (clean_path.strip("/").replace("/", "_") or "index") + ext
    filepath = PROJECT_ROOT / "data" / "raw" / source_name / filename

    if filepath.exists():
        return filepath.read_text(encoding="utf-8")

    response = fetch_page(fetch_url)
    if response is None:
        return None

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(response.text, encoding="utf-8")
    time.sleep(REQUEST_DELAY)
    return response.text

def save_html(url: str, html: str, source_name: str) -> None:
    parsed = urlparse(url)
    filename = parsed.path[1:].replace("/", "_")

    output_dir = PROJECT_ROOT / "data" / "raw" / source_name
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)


def crawl(source_name: str):
    config = load_source_config(source_name)
    start_url = config["start_url"]
    allowed_prefix = config["allowed_prefix"]
    blacklist = config["blacklist"]
    raw_markdown = config.get("raw_markdown", False)

    visited = set()
    queue = [start_url]
    logger.info(f"[{source_name}] Starting crawl from {start_url}")

    while queue and len(visited) < MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue

        html = fetch_or_cache(url, source_name, raw_markdown=raw_markdown)
        if html is None:
            continue

        visited.add(url)

        if raw_markdown:
            links = extract_markdown_links(html, start_url)
        else:
            links = extract_links(html, start_url)

        for link in links:
            if valid_url(link, allowed_prefix, blacklist, visited, queue):
                queue.append(link)

        logger.info(f"[{source_name}] [{len(visited)}/{MAX_PAGES}] Visited: {url}")

    return visited
