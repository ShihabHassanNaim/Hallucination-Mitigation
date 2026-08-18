"""Literature comparison loader.

Reads ``Paper/comparison_table.csv`` (the master CSV the user has been
curating) and ``Paper/why_we_win.md`` (the verdict narrative), and
returns a JSON-friendly payload the frontend can render directly.

The CSV uses ``"…"``-quoted fields, so a real CSV parser is required
(stdlib :mod:`csv` is fine). Markdown parsing here is intentionally
minimal: we only care about the verdict matrix and headline rows in
section 1 of ``why_we_win.md``.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


_PAPER_DIR = Path(__file__).resolve().parents[2] / "Paper"
_CSV_PATH = _PAPER_DIR / "comparison_table.csv"
_MD_PATH = _PAPER_DIR / "why_we_win.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: str) -> float | None:
    """Best-effort numeric extractor for CSV cells like ``0.91-0.95`` or ``+0.67 to +0.75``.

    Strategy: strip ``+`` and ``%``, take the FIRST numeric token, strip
    trailing commas. Returns ``None`` for ``-``, ``n/a``, ``n/r``, ``No``,
    or any other non-numeric placeholder.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in {"-", "—", "n/a", "n/r", "No", "Yes", "?"}:
        return None
    # Drop anything in parens (footnotes).
    s = re.sub(r"\([^)]*\)", "", s).strip()
    # First numeric token — supports leading sign, decimals, percent.
    m = re.search(r"[-+]?\d+(?:\.\d+)?%?", s)
    if not m:
        return None
    tok = m.group(0).rstrip("%")
    try:
        return float(tok)
    except ValueError:
        return None


def _is_ours_column(col_name: str) -> bool:
    """True iff the column header is the 'Ours' (this project) column."""
    return col_name.strip().lower().startswith("ours")


def _short_method_label(col_name: str) -> str:
    """Make a short, display-friendly label from a CSV column header."""
    if _is_ours_column(col_name):
        return "Ours"
    # Strip "(Paper)" qualifiers; trim.
    return re.sub(r"\s*\(.*?\)\s*", "", col_name).strip() or col_name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_literature_csv(path: Path | None = None) -> dict[str, Any]:
    """Parse the master ``comparison_table.csv``.

    Returns a dict shaped like::

        {
            "columns": ["Category", "Metric", "Ours (...)", "Dale 2023 ACL", ...],
            "ours_column": "Ours (...)",
            "method_columns": ["Dale 2023 ACL", "DUTh 2024", ...],
            "rows": [
              {
                "category": "Detection",
                "metric": "ROC-AUC",
                "ours": "0.91-0.95 (projected)",
                "ours_numeric": 0.91,
                "values": {"Dale 2023 ACL": {"raw": "0.987 ...", "num": 0.987}, ...},
              },
              ...
            ],
        }

    ``ours_numeric`` is the lower-bound for ranged cells like ``0.91-0.95``
    (so the comparison view can plot a single point + a range badge).
    """
    csv_path = Path(path) if path else _CSV_PATH
    if not csv_path.exists():
        return {"columns": [], "ours_column": "", "method_columns": [], "rows": [], "error": f"missing {csv_path}"}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {"columns": [], "ours_column": "", "method_columns": [], "rows": []}
        rows_raw = list(reader)

    columns = [h.strip() for h in header]
    if len(columns) < 3:
        return {"columns": columns, "ours_column": "", "method_columns": [], "rows": []}

    # First two columns are Category + Metric; remainder are methods.
    ours_idx = next((i for i, h in enumerate(columns) if _is_ours_column(h)), -1)
    ours_col = columns[ours_idx] if ours_idx >= 0 else ""
    method_cols: list[tuple[int, str]] = [
        (i, h)
        for i, h in enumerate(columns)
        if i >= 2 and not _is_ours_column(h)
    ]

    rows: list[dict[str, Any]] = []
    for raw in rows_raw:
        if not raw or all((c or "").strip() == "" for c in raw):
            continue
        category = (raw[0] if len(raw) > 0 else "").strip()
        metric = (raw[1] if len(raw) > 1 else "").strip()
        ours_raw = (raw[ours_idx] if ours_idx >= 0 and ours_idx < len(raw) else "").strip()

        values: dict[str, dict[str, Any]] = {}
        for i, col in method_cols:
            raw_val = (raw[i] if i < len(raw) else "").strip()
            values[col] = {"raw": raw_val, "num": _safe_float(raw_val)}

        rows.append({
            "category": category,
            "metric": metric,
            "ours": ours_raw,
            "ours_numeric": _safe_float(ours_raw),
            "values": values,
        })

    return {
        "columns": columns,
        "ours_column": ours_col,
        "method_columns": [c for _, c in method_cols],
        "method_labels": [_short_method_label(c) for _, c in method_cols],
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Win/Tie/Loss verdict parsing
# ---------------------------------------------------------------------------


def load_why_we_win(path: Path | None = None) -> dict[str, Any]:
    """Parse the verdict matrix from ``why_we_win.md``.

    The file has a "Headline Summary Matrix" markdown table with columns:
    ``# | Dimension | Best Comparable (Paper) | Best Comparable Score | Our Score | Winner | Justification``.

    We extract every row plus the verdict-tally table that follows.
    """
    md_path = Path(path) if path else _MD_PATH
    if not md_path.exists():
        return {"rows": [], "tally": {}, "verdict_dimension": "", "error": f"missing {md_path}"}

    text = md_path.read_text(encoding="utf-8")
    rows = _parse_verdict_table(text)
    tally = _parse_verdict_tally(text)
    visual = _parse_visual_headline(text)
    positions = _parse_five_positions(text) if "Five-Position" in text or "Position 1" in text else {}

    return {
        "rows": rows,
        "tally": tally,
        "visual_headline": visual,
        "five_positions": positions,
   }


# ---------------------------------------------------------------------------
# Internal markdown parsers
# ---------------------------------------------------------------------------


def _parse_verdict_table(text: str) -> list[dict[str, Any]]:
    """Find the verdict matrix and return its data rows as dicts."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("| # |") and "Winner" in line:
            start = i + 2  # skip header + separator
            break
    if start is None:
        return []

    rows: list[dict[str, Any]] = []
    for line in lines[start:]:
        s = line.strip()
        if not s.startswith("|"):
            break
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 6:
            continue
        rows.append({
            "num": cells[0],
            "dimension": cells[1],
            "best_paper": cells[2],
            "best_score": cells[3],
            "our_score": cells[4],
            "winner": cells[5],   # ✓ / ≈ / ✗
            "justification": cells[6] if len(cells) > 6 else "",
        })
    return rows


def _parse_verdict_tally(text: str) -> dict[str, Any]:
    """Pull the Win / Tie / Loss tally from the verdict table block."""
    out: dict[str, Any] = {"win": 0, "tie": 0, "loss": 0, "verdict": ""}
    # Header is `### Verdict Tally` (markdown heading).
    m = re.search(r"###\s+Verdict Tally", text)
    if not m:
        return out
    block = text[m.start(): m.start() + 800]
    # Cells are like "**✓ Win** | 6 | ..." — match the **…** label then the first integer.
    m_win  = re.search(r"\*\*✓ Win\*\*\s*\|\s*(\d+)", block)
    m_tie  = re.search(r"\*\*≈ Tie[^|]*\*\*\s*\|\s*(\d+)", block)
    m_loss = re.search(r"\*\*✗ Loss\*\*\s*\|\s*(\d+)", block)
    if m_win:  out["win"]  = int(m_win.group(1))
    if m_tie:  out["tie"]  = int(m_tie.group(1))
    if m_loss: out["loss"] = int(m_loss.group(1))

    # Headline read line is right after the tally table.
    m_verdict = re.search(r"\*\*Headline read\.\*\*\s*(.+?)(?:\n|$)", text)
    if m_verdict:
        out["verdict"] = m_verdict.group(1).strip()
    return out


def _parse_visual_headline(text: str) -> list[dict[str, Any]]:
    """Parse section 4's bar-chart ASCII block into structured rows."""
    block_match = re.search(
        r"## 4\. Visual Headline Comparison.*?```\n(.*?)```",
        text,
        flags=re.DOTALL,
    )
    if not block_match:
        return []
    raw = block_match.group(1)
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.rstrip()
        if not line or line.startswith("Metric") or set(line.strip()) <= {"─", " "}:
            continue
        m = re.match(
            r"^(?P<metric>\S+(?:\s\S+)*?)\s{2,}"
            r"(?P<best>[-+]?[\d.]+)\s+"
            r"(?P<ours>[-+]?[\d.]+)\s+"
            r"(?P<gap>[-+]?[\d.]+)\s+"
            r"(?P<tag>.+)$",
            line,
        )
        if not m:
            continue
        out.append({
            "metric": m.group("metric").strip(),
            "best": float(m.group("best")),
            "ours": float(m.group("ours")),
            "gap": float(m.group("gap")),
            "tag": m.group("tag").strip(),
        })
    return out


def _parse_five_positions(text: str) -> dict[str, list[dict[str, Any]]]:
    """Best-effort extraction of the five defensible positions.

    The five-position file is separate (Paper/five_position_comparison.md);
    we just return whatever simple per-position structure we can find here.
    """
    # The why_we_win.md doesn't contain the full five-position tables, but
    # it has the headline 3-axis coverage paragraph in section 3. Return
    # an empty dict; the dedicated page can load five_position_comparison.md
    # directly if needed.
    return {}


__all__ = [
    "load_literature_csv",
    "load_why_we_win",
]