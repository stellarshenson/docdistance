"""Build report.md of the relevance fixture (E17) from the two source articles.

The report is one document: a new title line, then each article as a part. Every heading of an
article is moved one level down, its own title line and the provenance comment of the second
article are left out, and no other line is changed. Run from the repository root:

    python3 data/processed/relevance-fixture/build_report.py
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
PARTS = (
    ("Part 1 - Enterprise adoption of AI", ROOT / "data/interim/exec-summaries/ibm-ai-adoption/source/source-article.md"),
    ("Part 2 - Impact of artificial intelligence on society", ROOT / "data/interim/ai-society-wergeland/source/source-article.md"),
)
TITLE = "# Market report - enterprise adoption of AI and AI in society (constructed fixture)"


def part_lines(path: Path) -> list[str]:
    """The lines of one article without its comment block and title line, headings one level down."""
    text = re.sub(r"<!--.*?-->\s*", "", path.read_text(encoding="utf-8"), flags=re.S)
    lines = text.strip().splitlines()
    assert lines[0].startswith("# "), path
    return ["#" + line if line.startswith("#") else line for line in lines[1:]]


def main() -> None:
    out = [TITLE, ""]
    for heading, path in PARTS:
        out += [f"## {heading}", *part_lines(path), ""]
    (Path(__file__).parent / "report.md").write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
