import argparse
import time
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
from utils.logger import get_project_logger
from common.config import load_source_config

logger = get_project_logger(__name__)

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

MAX_PAGES = 500
REQUEST_DELAY = 0.5


def valid_url(url: str, allowed_prefix: str, blacklist: list, visited: set, queue: list) -> bool:
    if not url.startswith(allowed_prefix):
        return False
    if url in visited or url in queue:
        return False
    if any(bad in url for bad in blacklist):
        return False
    return True


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

    visited = set()
    queue = [start_url]
    logger.info(f"[{source_name}] Starting crawl from {start_url}")

    while queue and len(visited) < MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue

        response = fetch_page(url)
        if response is None:
            continue

        visited.add(url)
        save_html(url, response.text, source_name)

        for link in extract_links(response.text, start_url):
            if valid_url(link, allowed_prefix, blacklist, visited, queue):
                queue.append(link)

        logger.info(f"[{source_name}] [{len(visited)}/{MAX_PAGES}] Visited: {url}")
        time.sleep(REQUEST_DELAY)

    return visited


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl a documentation source")
    parser.add_argument("source", help="Source name from config/sources.yaml, e.g. python_docs")
    args = parser.parse_args()

    crawl(args.source)