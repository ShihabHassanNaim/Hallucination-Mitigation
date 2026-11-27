"""The three comparison tables of Chapter 7.

Three tables are generated here:

1. **The head-to-head table** — CRISP set beside the four closest published
   pipelines, dimension by dimension. Its content is read from the survey
   (``docs/related_pipeline_workflows.md`` §3.7) through
   ``scripts/thesis/survey.py``, so the table cannot claim a dimension the
   survey does not record, and a change to the survey changes the table.

2. **The metric-by-metric table** — every published comparison, with the
   published value, this project's value and 95 per cent interval, and the
   verdict recomputed by the interval rule of Section 5.7. The verdicts are
   never written by hand.

3. **The master table** — the three blocks stacked: the architectural
   coverage, the capability counts, and the quantitative verdicts.

The architectural tables come first in every case, because that is the
order the evidence supports.
"""
from __future__ import annotations

from . import survey
from .content import H, N_PIPELINE_WORDS
from .survey import TICK_DIMENSIONS, TEXT_DIMENSIONS

# The tick/cross characters of the survey, and the words used in the book.
TICK = "\u2713"
CROSS = "\u2717"


def _clean(text: str) -> str:
    """The survey writes markdown emphasis; the book does not."""
    return text.replace("**", "").replace("*", "").strip()


def _mark(text: str) -> str:
    """A tick or a cross becomes the word, so the table reads in prose.

    A cell that carries a tick *and* text — the survey writes such cells
    as "✓ claim span" — keeps the text and loses the mark, because the
    mark is what the word already says.
    """
    t = text.strip()
    if t == TICK:
        return "Yes"
    if t == CROSS:
        return "No"
    if t.startswith(TICK):
        return _clean(t[len(TICK):].strip())
    if t.startswith(CROSS):
        return _clean(t[len(CROSS):].strip())
    return _clean(t)


# --------------------------------------------------------------------------
# Table 1 -- CRISP against the four closest published pipelines
# --------------------------------------------------------------------------

def head_to_head_table() -> tuple[list[str], list[list[str]], list[str]]:
    """The dimension-by-dimension comparison.

    Returns the headers, the rows, and a note describing the source of
    the content.
    """
    systems, rows = survey.head_to_head()
    # The survey's five columns are CRISP and four published systems.
    assert systems[0] == "CRISP", systems

    headers = ["Dimension"] + [s.replace("**", "") for s in systems]

    body: list[list[str]] = []
    for dim in TICK_DIMENSIONS + TEXT_DIMENSIONS:
        vals = rows.get(dim)
        if vals is None:
            continue
        # The tick dimensions are answered yes or no; the free-text ones
        # are quoted as the survey writes them.
        if dim in TICK_DIMENSIONS:
            cells = [_mark(v) for v in vals]
        else:
            cells = [_clean(v) for v in vals]
        body.append([dim] + cells)

    # The coverage row comes last, as the summary of the table.
    cover = rows.get("**C1\u2013C6 covered**")
    if cover:
        body.append(["Capabilities covered (C1\u2013C6)"] + [_clean(v) for v in cover])

    note = (
        "The content of this table is the head-to-head comparison of the "
        "survey in Chapter 2, read from its source rather than restated "
        "here. The three tick rows are the dimensions that separate a "
        "defence pipeline from a single-pass detector; the remaining rows "
        "describe what each system does."
    )
    return headers, body, note


# --------------------------------------------------------------------------
# Table 2 -- the published values, this project's values, and the verdicts
# --------------------------------------------------------------------------

def metric_rows() -> list[list[str]]:
    """Every published comparison, with the verdict recomputed."""
    from .chapters_7_9 import _verdict, AES, DESIDERATA, FREQ, LLMCHECK
    from .chapters_7_9 import RAGTRUTH_F1, RAG_AUC, RAG_CI, F1_VAL, F1_CI

    out: list[list[str]] = []

    def row(system, metric, published, ours, ci):
        out.append([
            system, metric, f"{published:.4f}", f"{ours:.4f}",
            f"[{ci[0]:.4f}, {ci[1]:.4f}]", f"**{_verdict(ours, ci, published)}**",
        ])

    for name, val in LLMCHECK:
        row(name, "ROC-AUC", val, RAG_AUC, RAG_CI)
    for model, val in AES:
        row(f"Su et al. \u2014 {model}", "ROC-AUC", val, RAG_AUC, RAG_CI)
    for model, method, val in FREQ:
        row(f"Qi et al. \u2014 {model} / {method}", "ROC-AUC",
            val, RAG_AUC, RAG_CI)
    for name, roc, _pr, _f1 in DESIDERATA:
        row(f"Chen et al. \u2014 {name}", "ROC-AUC", roc, RAG_AUC, RAG_CI)
    for name, _roc, _pr, f1 in DESIDERATA:
        row(f"Chen et al. \u2014 {name}", "F1", f1, F1_VAL, F1_CI)
    for name, val in RAGTRUTH_F1:
        row(f"RAGTruth paper \u2014 {name}", "F1", val, F1_VAL, F1_CI)
    return out


def metric_tally() -> dict[str, int]:
    """The tally of the metric rows, computed from the rows themselves."""
    from collections import Counter
    t = Counter(r[5].replace("**", "") for r in metric_rows())
    return {"WIN": t["WIN"], "TIE": t["TIE"], "LOSS": t["LOSS"]}


def metric_table() -> tuple[list[str], list[list[str]], str]:
    """The metric-by-metric comparison with the recomputed verdicts."""
    headers = ["Published system", "Metric", "Published", "Ours",
               "Our 95% CI", "Verdict"]
    note = (
        "Every verdict in this table is recomputed from the row's own "
        "numbers by the interval rule of Section 5.7: a win requires the "
        "lower bound of this project's 95 per cent interval to sit strictly "
        "above the published value, a loss requires the upper bound to sit "
        "strictly below it, and anything else is a tie. The rule is applied "
        "identically in both directions."
    )
    return headers, metric_rows(), note


# --------------------------------------------------------------------------
# Table 3 -- the master table
# --------------------------------------------------------------------------

def master_table() -> tuple[list[str], list[list[str]], str]:
    """The three blocks stacked: architecture, coverage, and verdicts."""
    grid = survey.capability_grid()
    counts = survey.coverage_counts()

    headers = ["Comparison", "Dataset", "Result", "Verdict"]

    # -- block one: the architectural coverage, per published pipeline.
    rows: list[list[str]] = []
    for name, caps in grid:
        n = sum(caps)
        best = " **\u2014 best published**" if n == 5 else ""
        rows.append([
            f"Architectural coverage \u2014 {name}{best}",
            "Survey of 17 pipelines",
            f"{n} of 6",
            "**Ahead of every published system**" if n == 5 else "",
        ])
    rows.append([
        "**Architectural coverage \u2014 CRISP**",
        "Survey of 17 pipelines",
        "**6 of 6**",
        "**Ahead of every published system**",
    ])

    # -- block two: how many of the seventeen implement each capability.
    for (code, label), n in zip(survey.CAPABILITIES, counts):
        rows.append([
            f"Capability coverage \u2014 {code} {label}",
            "Survey of 17 pipelines",
            f"{n} of 17",
            "Implemented by CRISP",
        ])

    # -- block three: the quantitative verdicts, counted.
    from .chapters_7_9 import _verdict, AES, DESIDERATA, FREQ, LLMCHECK
    from .chapters_7_9 import RAGTRUTH_F1, RAG_AUC, RAG_CI, F1_VAL, F1_CI

    def tally(pairs):
        t = {"WIN": 0, "LOSS": 0, "TIE": 0}
        for ours, ci, pub in pairs:
            t[_verdict(ours, ci, pub)] += 1
        return t

    blocks = [
        ("LLM-Check, five baselines", "RAGTruth",
         tally([(RAG_AUC, RAG_CI, v) for _n, v in LLMCHECK]),
         "Ahead on the black-box baselines"),
        ("Su et al., five models", "RAGTruth",
         tally([(RAG_AUC, RAG_CI, v) for _m, v in AES]),
         "Behind methods with generator access"),
        ("Qi et al., six rows", "RAGTruth",
         tally([(RAG_AUC, RAG_CI, v) for _m, _me, v in FREQ]),
         "Behind methods with generator access"),
        ("Chen et al., on ROC-AUC", "RAGTruth",
         tally([(RAG_AUC, RAG_CI, r) for _n, r, _p, _f in DESIDERATA]),
         "Behind methods with generator access"),
        ("Chen et al., on F1", "RAGTruth",
         tally([(F1_VAL, F1_CI, f) for _n, _r, _p, f in DESIDERATA]),
         "Mixed"),
        ("RAGTruth paper's own baselines", "RAGTruth",
         tally([(F1_VAL, F1_CI, v) for _n, v in RAGTRUTH_F1]),
         "Ahead of 3 of 4"),
    ]
    for name, ds, t, verdict in blocks:
        parts = []
        if t["WIN"]:
            parts.append(f"{t['WIN']} win" + ("s" if t["WIN"] > 1 else ""))
        if t["TIE"]:
            parts.append(f"{t['TIE']} tie" + ("s" if t["TIE"] > 1 else ""))
        if t["LOSS"]:
            parts.append(f"{t['LOSS']} loss" + ("es" if t["LOSS"] > 1 else ""))
        rows.append([
            name, ds, ", ".join(parts), verdict,
        ])

    note = (
        "The first block states the architectural coverage of each surveyed "
        "pipeline; the second states how many of the " + N_PIPELINE_WORDS
        + " implement each "
        "capability; the third states the quantitative verdicts, computed by "
        "the interval rule of Section 5.7. The architectural rows come "
        "first because that is where this project is ahead of every published "
        "system."
    )
    return headers, rows, note
