from pathlib import Path

CURRENT_FILE = Path(__file__)
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

processed_dir = PROJECT_ROOT / "data" / "processed" / "python_docs"
suspicious = ["¶", "Â", "headerlink"]

for md_file in processed_dir.glob("*.md"):
    content = md_file.read_text(encoding="utf-8")
    for pattern in suspicious:
        if pattern in content:
            print(f"{md_file.name}: found '{pattern}'")