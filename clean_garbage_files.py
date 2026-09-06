import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RAW_DIR_FOR = lambda src: PROJECT_ROOT / "data" / "raw" / src


def is_garbage_text(text: str, threshold: float = 0.05) -> bool:
    if not text.strip():
        return True
    bad = text.count("\ufffd")
    return (bad / len(text)) > threshold


def clean_garbage_files(source_name: str, dry_run: bool = True) -> list[Path]:
    raw_dir = RAW_DIR_FOR(source_name)
    if not raw_dir.exists():
        print(f"Raw dir not found: {raw_dir}")
        return []

    garbage_files = []
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            garbage_files.append(path)
            continue

        if is_garbage_text(text):
            garbage_files.append(path)

    print(f"Found {len(garbage_files)} garbage files in {raw_dir}")
    for f in garbage_files:
        print(f"  {f.name}")

    if not dry_run:
        for f in garbage_files:
            f.unlink()
        print(f"Deleted {len(garbage_files)} garbage files")
    else:
        print("Dry run — nothing deleted. Run with --delete to actually remove them.")

    return garbage_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and remove garbage (binary-content) raw files.")
    parser.add_argument("source", help="Source name, e.g. dbt_docs")
    parser.add_argument("--delete", action="store_true", help="Actually delete files (default is dry run)")
    args = parser.parse_args()

    clean_garbage_files(args.source, dry_run=not args.delete)