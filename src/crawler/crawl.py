import time
import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from utils.logger import get_project_logger
from urllib.parse import urlparse
from pathlib import Path

CURRENT_FILE = Path(__file__)

PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

logger = get_project_logger(__name__)


START_URL = "https://docs.python.org/3/library/index.html"
ALLOWED_PREFIX = "https://docs.python.org/3/"
BLACKLIST = ["genindex", "py-modindex", "bugs.html", "search.html", "copyright.html"]
MAX_PAGES = 30
REQUEST_DELAY = 0.5

def valid_url(url: str, visited: set, queue: list) -> bool:
    if not url.startswith(ALLOWED_PREFIX):
        return False
    if url in visited or url in queue:
        return False
    if any(bad in url for bad in BLACKLIST):
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
        response.encoding = "utf-8"
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url}: {e}")
        return None

    if response.status_code != 200:
        logger.warning(f"Bad status {response.status_code} for {url}")
        return None

    return response

def save_html(url: str, html: str) -> None:
    parsed = urlparse(url)
    filename = parsed.path[1:].replace("/", "_")

    output_dir = PROJECT_ROOT / "data" / "raw" / "python_docs"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)


def crawl():
    visited = set()
    queue = [START_URL]
    logger.info(f"Starting crawl from {START_URL}")

    while queue and len(visited) < MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue

        response = fetch_page(url)
        if response is None:
            continue

        visited.add(url)
        save_html(url, response.text)

        for link in extract_links(response.text, START_URL):
            if valid_url(link, visited, queue):
                queue.append(link)

        logger.info(f"[{len(visited)}/{MAX_PAGES}] Visited: {url}")
        time.sleep(REQUEST_DELAY)

    return visited


if __name__ == "__main__":
    crawl()