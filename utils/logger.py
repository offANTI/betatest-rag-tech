import logging
import os


def get_project_logger(module_name: str) -> logging.Logger:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s",
    )

    return logging.getLogger(module_name)