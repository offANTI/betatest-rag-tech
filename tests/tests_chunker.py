import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.chunker import split_headings, split_by_size, split_long_paragraph


def test_split_headings_basic():
    text = "# Title\n\nIntro text.\n\n## Section One\n\nContent one.\n\n## Section Two\n\nContent two."
    sections = split_headings(text)

    assert len(sections) == 3
    assert sections[0][0] == "Introduction"
    assert "Intro text" in sections[0][1]
    assert sections[1][0] == "Section One"
    assert sections[2][0] == "Section Two"


def test_split_headings_no_headings():
    text = "Just plain text, no headings at all."
    sections = split_headings(text)

    assert len(sections) == 1
    assert sections[0][0] == "Introduction"


def test_split_headings_empty_intro():
    text = "## Only Section\n\nSome content."
    sections = split_headings(text)

    assert len(sections) == 1
    assert sections[0][0] == "Only Section"


def test_split_headings_dt_style():
    text = "# Built-in Functions\n\n#### abs(x)\n\nReturn absolute value.\n\n#### isinstance(obj)\n\nCheck instance."
    sections = split_headings(text)

    headings = [h for h, _ in sections]
    assert "abs(x)" in headings
    assert "isinstance(obj)" in headings


def test_split_headings_h3_not_split():
    text = "## Parent\n\nParent text.\n\n### Nested\n\nNested text."
    sections = split_headings(text)

    assert len(sections) == 1
    assert "### Nested" in sections[0][1]


def test_split_by_size_short_text():
    result = split_by_size("Short text.", max_size=100)
    assert result == ["Short text."]


def test_split_by_size_empty():
    assert split_by_size("", max_size=100) == []
    assert split_by_size("   ", max_size=100) == []


def test_split_by_size_respects_limit():
    text = "\n\n".join([f"Paragraph number {i} with some words." for i in range(20)])
    result = split_by_size(text, max_size=50)

    for chunk in result:
        assert len(chunk) <= 50


def test_split_long_paragraph_single_long_word():
    paragraph = "A" * 120
    result = split_long_paragraph(paragraph, max_size=50)

    assert len(result) == 3
    for piece in result:
        assert len(piece) <= 50


def test_split_long_paragraph_breaks_on_word_boundary():
    paragraph = "one two three four five six seven eight nine ten"
    result = split_long_paragraph(paragraph, max_size=20)

    for piece in result:
        assert len(piece) <= 20
        assert not piece.startswith(" ")