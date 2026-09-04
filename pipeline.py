import argparse
from utils.logger import get_project_logger
from src.crawler.crawl import crawl
from src.crawler.extract import extract_all
from src.chunking.chunker import chunk_all
from src.retrieval.embed import embed_source
from common.config import load_source_config
logger = get_project_logger(__name__)

def run_pipeline(source_name: str, skip_crawl: bool = False):
    logger.info("Pipeline start (requested source: %s)", source_name)
    config = load_source_config(source_name)
    raw_markdown = config.get("raw_markdown", False)
    if skip_crawl:
        logger.info("[%s] Skipping crawl step (--skip-crawl)", source_name)
    else:
        logger.info("[%s] Step 1/4: crawl", source_name)
        crawl(source_name)
    if raw_markdown:
        logger.info("[%s] Skipping extract step (raw_markdown=True)", source_name)
    else:
        logger.info("[%s] Step 2/4: extract HTML", source_name)
        extract_all(source_name)

    logger.info("[%s] Step 3/4: chunk", source_name)
    chunk_all(source_name)

    logger.info("[%s] Step 4/4: embed", source_name)
    embed_source(source_name)

    logger.info("[%s] Pipeline complete", source_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full pipeline for a documentation source"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="python_docs",
        help="Source name from config/sources.yaml",
    )
    parser.add_argument(
        "--skip-crawl", action="store_true", help="Skip crawling step"
    )
    args = parser.parse_args()

    run_pipeline(args.source, skip_crawl=args.skip_crawl)