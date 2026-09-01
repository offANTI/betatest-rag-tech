from pathlib import Path
import yaml

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent

SOURCES_FILE = PROJECT_ROOT / "config" / "sources.yaml"


def load_source_config(source_name: str) -> dict:
    all_sources = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8"))

    if source_name not in all_sources:
        available = ", ".join(all_sources.keys())
        raise ValueError(f"Unknown source '{source_name}'. Available: {available}")

    return all_sources[source_name]