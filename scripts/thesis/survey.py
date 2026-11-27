"""The published-pipeline survey, read for the book's tables.

The survey in ``docs/related_pipeline_workflows.md`` is the authority on
which published pipeline implements which capability. The book's
comparison tables read it through this module rather than restating
it, so a table cannot claim a capability the survey does not record,
and a change to the survey changes the tables with it.

This is the same authority the architectural figures read
(``scripts/plot_architectural_win.py``), so the figure and the table
cannot disagree.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SURVEY = ROOT / "docs" / "related_pipeline_workflows.md"

# The six capabilities, in the order the survey lists them.
CAPABILITIES = [
    ("C1", "Retrieval-grounded detection"),
    ("C2", "Multi-stage pipeline"),
    ("C3", "Claim-level verification"),
    ("C4", "Multi-signal fusion"),
    ("C5", "Post-hoc repair"),
    ("C6", "Selective prediction"),
]

# The dimensions of the head-to-head table that are answered with a tick.
# The other rows of that table are free text describing what a system
# does, and a cell that is neither a tick nor a cross would read as
# "does not implement it" when it means "does it differently".
TICK_DIMENSIONS = [
    "Mitigates",
    "Re-verifies own edit",
    "Abstains",
]

# The rows of the head-to-head table that are free text, kept for the
# dimension-by-dimension comparison table.
TEXT_DIMENSIONS = [
    "Detection unit",
    "Detection signal",
    "Fusion",
    "Retrieval",
    "Verifier",
    "Cost",
]


def _read() -> list[str]:
    if not SURVEY.exists():
        return []
    return SURVEY.read_text(encoding="utf-8").splitlines()


def _section(lines: list[str], prefix: str) -> list[str]:
    """The lines of a section, up to the next heading of the same level."""
    start = None
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            start = i
            break
    if start is None:
        return []
    out = []
    for line in lines[start + 1:]:
        if line.startswith("## ") or line.startswith("### "):
            break
        out.append(line)
    return out


def _table_rows(lines: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """The header and the body of the first markdown table in ``lines``."""
    header: list[str] | None = None
    rows: dict[str, list[str]] = {}
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            if cells and cells[0] == "Dimension":
                header = [c.replace("*", "").strip() for c in cells[1:]]
            continue
        if len(cells) < 2:
            continue
        rows[cells[0]] = cells[1:]
    if header is None:
        return [], {}
    return header, rows


def capability_grid() -> list[tuple[str, list[bool]]]:
    """The survey's landscape grid: each pipeline and its six capabilities."""
    lines = _read()
    in_grid = False
    out: list[tuple[str, list[bool]]] = []
    for line in lines:
        if line.startswith("## 2. The landscape at a glance"):
            in_grid = True
            continue
        if in_grid and line.startswith("## "):
            break
        if not in_grid or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        name = cells[0].replace("*", "").strip()
        if name in ("Paper", "") or set(name) <= set("-: "):
            continue
        caps = [c.strip() == "✓" for c in cells[1:7]]
        out.append((name, caps))
    return out


def head_to_head() -> tuple[list[str], dict[str, list[str]]]:
    """The survey's head-to-head table: the systems and their dimensions."""
    lines = _read()
    section = _section(lines, "## 3.7 Head-to-head")
    return _table_rows(section)


def coverage_counts() -> list[int]:
    """How many surveyed pipelines implement each capability."""
    grid = capability_grid()
    return [sum(caps[i] for _, caps in grid) for i in range(len(CAPABILITIES))]
