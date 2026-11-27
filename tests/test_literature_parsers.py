"""The literature parsers must read the consolidated comparison document.

``src/evaluation/literature.py`` used to read three separate papers --
``why_we_win.md``, ``five_position_comparison.md`` and ``comparison.md``
-- and those files were folded into one document. The parsers were
pointed at the new path, but a parser that reads the right file can
still return nothing, or return a stale slice of it, without any test
noticing: the frontend renders an empty chart and nobody can say why.

These tests assert that each parser finds its section in the document
that is actually committed, and that what it extracts agrees with the
document's own tables.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.evaluation import literature as L


MD_PATH = Path("Paper/comparison.md")
CSV_PATH = Path("Paper/comparison_table.csv")


@pytest.fixture(scope="module")
def md_text():
    if not MD_PATH.exists():
        pytest.skip("comparison.md not present")
    return MD_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The parsers point at the consolidated document.
# ---------------------------------------------------------------------------


def test_the_parsers_read_the_consolidated_document():
    """Both files the module reads are the consolidated ones.

    The module resolved its paths from the repository root, so this is
    really a check that the paths still point at files that exist: a
    parser reading a deleted paper would return an empty payload with
    no error.
    """
    assert L._MD_PATH.name == "comparison.md", (
        f"the markdown parser reads {L._MD_PATH.name}, not the "
        "consolidated comparison.md"
    )
    assert L._CSV_PATH.name == "comparison_table.csv"
    assert L._MD_PATH.exists(), f"{L._MD_PATH} does not exist"
    assert L._CSV_PATH.exists(), f"{L._CSV_PATH} does not exist"


def test_no_parser_still_points_at_a_deleted_paper():
    """None of the folded-in papers is read any more.

    ``why_we_win.md``, ``five_position_comparison.md``,
    ``wins_summary.md`` and ``comparison_detailed.md`` were folded into
    ``comparison.md``. A parser left pointing at one of them would
    silently return nothing.
    """
    source = Path(L.__file__).read_text(encoding="utf-8")
    for gone in ("why_we_win.md", "five_position_comparison.md",
                 "wins_summary.md", "comparison_detailed.md"):
        assert f'"{gone}"' not in source, (
            f"literature.py still names {gone}; the parsers must read "
            "the consolidated comparison.md"
        )


# ---------------------------------------------------------------------------
# Each parser finds its section.
# ---------------------------------------------------------------------------


def test_the_verdict_table_parser_finds_every_matrix_row(md_text):
    """The parser returns the matrix rows, with a verdict on each one.

    The rows are counted by column position, the same way the paper
    counts them, so a row added to the document cannot be dropped by the
    parser without the counts disagreeing.

    One row -- Detection AUPR -- carries a dash instead of a verdict,
    because no published paper reports AUPR on HaluEval, so there is
    nothing for it to be compared against. The parser returns that row
    too; it is counted separately here so a second dash row would fail.
    """
    rows = L._parse_verdict_table(md_text)
    assert rows, "the verdict parser found no rows in comparison.md"

    expected = []
    undecided = []
    for line in md_text.splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 8:
            continue
        # cells[0] is the empty string before the row's first pipe, so
        # the row number is cells[1].
        if cells[6] not in ("✓", "✗", "≈"):
            undecided.append((cells[1], cells[6]))
            continue
        expected.append((cells[1], cells[6]))

    assert len(undecided) == 1, (
        f"the document has {len(undecided)} rows with no verdict "
        f"({undecided}); exactly one is expected, because no published "
        "paper reports AUPR on HaluEval"
    )

    decided = [r for r in rows if r["winner"] in ("✓", "✗", "≈")]
    assert len(decided) == len(expected), (
        f"the parser returned {len(decided)} decided rows but the document "
        f"has {len(expected)}"
    )
    for row, (num, mark) in zip(decided, expected):
        assert row["num"] == num, (
            f"the parser read row {row['num']} where the document has {num}"
        )
        assert row["winner"] == mark, (
            f"row {num} ({row['dimension']}): the parser read the verdict "
            f"{row['winner']!r} where the document has {mark!r}"
        )


def test_the_tally_parser_agrees_with_the_matrix_it_sits_below(md_text):
    """The tally is the count of the matrix, not a hand-written number."""
    rows = L._parse_verdict_table(md_text)
    assert rows, "the verdict parser found no rows"

    counts = {"win": 0, "tie": 0, "loss": 0}
    for row in rows:
        # Detection AUPR carries a dash, not a verdict: no published
        # paper reports AUPR on HaluEval, so it is in none of the three
        # counts and the tally must not try to place it.
        if row["winner"] not in ("✓", "✗", "≈"):
            continue
        counts[{"✓": "win", "≈": "tie", "✗": "loss"}[row["winner"]]] += 1

    tally = L._parse_verdict_tally(md_text)
    assert tally["win"] == counts["win"], (
        f"the tally says {tally['win']} wins but the matrix has "
        f"{counts['win']}"
    )
    assert tally["tie"] == counts["tie"], (
        f"the tally says {tally['tie']} ties but the matrix has "
        f"{counts['tie']}"
    )
    assert tally["loss"] == counts["loss"], (
        f"the tally says {tally['loss']} losses but the matrix has "
        f"{counts['loss']}"
    )


def test_the_visual_headline_parser_finds_the_block(md_text):
    """The ASCII headline block is parsed into one row per metric."""
    rows = L._parse_visual_headline(md_text)
    assert len(rows) >= 3, (
        "the visual headline parser found "
        f"{len(rows)} rows; the frontend's headline chart would be empty"
    )
    for row in rows:
        assert row["metric"], "a headline row has no metric name"
        assert row["tag"], f"{row['metric']} has no verdict tag"
        assert isinstance(row["ours"], float)


def test_the_five_positions_parser_finds_all_five(md_text):
    """The five positions are read from the consolidated document."""
    positions = L._parse_five_positions(md_text)
    assert len(positions) == 5, (
        f"the position parser found {len(positions)} positions, expected 5"
    )
    for number, entries in positions.items():
        assert len(entries) == 1
        entry = entries[0]
        assert entry["title"], f"position {number} has no title"
        assert entry["verdict"], f"position {number} has no verdict"


# ---------------------------------------------------------------------------
# What the parsers extract agrees with the document's own tables.
# ---------------------------------------------------------------------------


def test_the_parsed_verdicts_match_the_metric_table(md_text):
    """The parser's verdicts are the ones the paper's table prints.

    The verdict matrix and the metric table are two views of the same
    measurements. The frontend renders both, so a disagreement between
    them is visible on the page.
    """
    block = re.search(
        r"\*\*The full metric set on HaluEval:\*\*.*?\n\s*\n"
        r"\| Metric \|.*?\n(?:\|\|?.*\n)*",
        md_text,
        re.DOTALL,
    )
    assert block, "the document has no HaluEval metric table"

    paper = {}
    for line in block.group(0).splitlines():
        if not line.startswith("| ") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 5 and cells[0] != "Metric":
            paper[cells[0]] = cells

    # The matrix names a dimension differently from the metric table:
    # the matrix says "Sentence accuracy" where the table says
    # "Detection accuracy", and the matrix says "Detection ROC-AUC"
    # where the table says "ROC-AUC".
    aliases = {
        "Claim-level F1": "Claim-level F1",
        "Detection ROC-AUC": "ROC-AUC",
        "Detection AUPR": "AUPR",
        "Sentence accuracy": "Detection accuracy",
        "VSR": "VSR",
        "HRR": "HRR",
        "P@R90": "P@R90",
        "Spearman ρ": "Spearman ρ",
    }
    checked = 0
    for row in L._parse_verdict_table(md_text):
        dimension = row["dimension"].strip("*").strip()
        table_name = aliases.get(dimension)
        if table_name is None or table_name not in paper:
            continue
        assert row["winner"] == paper[table_name][4], (
            f"{dimension}: the verdict matrix says "
            f"{row['winner']!r} and the metric table says "
            f"{paper[table_name][4]!r}"
        )
        checked += 1

    assert checked >= 3, (
        f"only {checked} dimensions were cross-checked between the "
        "verdict matrix and the metric table"
    )


def test_the_parsed_headline_numbers_match_the_metric_table(md_text):
    """The headline block's numbers are the paper's numbers."""
    block = re.search(
        r"\*\*The full metric set on HaluEval:\*\*.*?\n\s*\n"
        r"\| Metric \|.*?\n(?:\|\|?.*\n)*",
        md_text,
        re.DOTALL,
    )
    assert block, "the document has no HaluEval metric table"

    paper = {}
    for line in block.group(0).splitlines():
        if not line.startswith("| ") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 5 and cells[0] != "Metric":
            paper[cells[0]] = cells

    # The headline block names its metrics differently from the table.
    aliases = {
        "Claim F1": "Claim-level F1",
        "Detection AUPR": "AUPR",
        "Detection ROC-AUC": "ROC-AUC",
        "HRR": "HRR",
        "VSR": "VSR",
    }

    checked = 0
    for row in L._parse_visual_headline(md_text):
        table_name = aliases.get(row["metric"])
        if table_name is None or table_name not in paper:
            continue
        paper_value = float(paper[table_name][1])
        assert abs(row["ours"] - paper_value) < 5e-4, (
            f"{row['metric']}: the headline block says {row['ours']} and "
            f"the metric table says {paper_value}"
        )
        checked += 1

    assert checked >= 3, (
        f"only {checked} headline numbers were cross-checked against the "
        "metric table"
    )
