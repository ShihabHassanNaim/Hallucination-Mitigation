"""Verify every number in Paper/ragtruth_comparison.md against its source artefact.

The file is hand-written prose with tables, so the only way to keep it honest is to
assert the numbers back out of it and compare them to the JSON/PDF-derived sources.
Every verdict is recomputed here from the interval rule rather than trusted.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "Paper" / "ragtruth_comparison.md"
METRICS = ROOT / "data" / "metrics_ragtruth_fitted.json"
ANCHORS = ROOT / "data" / "published_anchors.json"

OUR_ROC_AUC = 0.612728
OUR_ROC_CI = (0.5493826089734734, 0.6581043509403893)
OUR_F1 = 0.607645875251509
OUR_F1_CI = (0.5383061034984691, 0.6632373378825157)


@pytest.fixture(scope="module")
def doc():
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def anchors():
    return json.loads(ANCHORS.read_text(encoding="utf-8"))


def verdict(ours, ci, published):
    """The interval rule, applied in both directions."""
    if ci[0] > published:
        return "WIN"
    if ci[1] < published:
        return "LOSS"
    return "TIE"


# --- our own numbers ---------------------------------------------------------

def test_doc_exists_and_is_not_empty(doc):
    assert doc.strip()


def test_our_roc_auc_matches_the_metrics_file(doc, metrics):
    agg = metrics["aggregate"]
    assert agg["detection_roc_auc"] == OUR_ROC_AUC
    assert agg["detection_roc_auc_ci95"] == list(OUR_ROC_CI)
    assert f"{agg['detection_roc_auc']:.4f}" in doc
    assert f"[{OUR_ROC_CI[0]:.4f}, {OUR_ROC_CI[1]:.4f}]" in doc


def test_our_f1_matches_the_metrics_file(doc, metrics):
    agg = metrics["aggregate"]
    assert agg["detection_f1"] == pytest.approx(OUR_F1, abs=1e-12)
    assert f"{agg['detection_f1']:.4f}" in doc


def test_split_sizes_in_doc_match_the_artefact(doc, metrics):
    assert metrics["n_queries"] == 1000
    assert metrics["dataset"] == "ragtruth_eval_1000_matched"
    unique = len({r["query"] for r in metrics["per_query"]})
    assert unique == 277
    assert "1000 rows" in doc and "277 unique" in doc


def test_balance_is_500_500(doc, metrics):
    labels = [r["label"] for r in metrics["per_query"]]
    assert labels.count(0) == 500 and labels.count(1) == 500
    assert "500/500" in doc


# --- the LLM-Check anchors ---------------------------------------------------

def test_llm_check_anchor_values_appear_in_doc(doc, anchors):
    for a in anchors["auroc_family"]["anchors"]:
        assert f"{a['value']:.4f}" in doc, a["system"]


def test_llm_check_verdicts_are_recomputed_not_asserted(doc):
    # Every LLM-Check anchor row in the doc must carry the verdict the interval
    # rule produces, in both directions.
    expected = {
        0.5458: "WIN",
        0.5327: "WIN",
        0.5168: "WIN",
        0.5724: "TIE",
        0.5830: "TIE",
    }
    for value, want in expected.items():
        assert verdict(OUR_ROC_AUC, OUR_ROC_CI, value) == want
        row = re.search(
            r"LLM-Check — [^\n|]*\| " + f"{value:.4f}" + r" \|[^\n]*", doc
        )
        assert row, f"no row for LLM-Check anchor {value}"
        assert f"**{want}**" in row.group(0)


def test_f1_family_anchor_values_appear_in_doc(doc, anchors):
    for a in anchors["f1_family"]["anchors"]:
        if a["scope"] == "overall":
            assert f"{a['value']:.3f}" in doc, a["system"]


def test_span_family_is_reported_as_a_negative_result(doc, anchors):
    for a in anchors["span_f1_family"]["anchors"]:
        assert f"{a['value']:.3f}" in doc, a["system"]
    assert "negative result" in doc


def test_not_reported_systems_are_listed(doc, anchors):
    for entry in anchors["not_reported_on_ragtruth"]:
        assert entry["system"] in doc, entry["system"]


# --- the newer published papers ---------------------------------------------

AES_ROWS = [
    ("Llama2-7B", 0.8829),
    ("Llama2-13B", 0.9250),
    ("Llama3-8B", 0.8807),
    ("Mistral-7B", 0.8931),
    ("Llama2-70B", 0.9332),
]


@pytest.mark.parametrize("model,value", AES_ROWS)
def test_aes_rows_are_losses_against_our_upper_bound(doc, model, value):
    assert verdict(OUR_ROC_AUC, OUR_ROC_CI, value) == "LOSS"
    assert f"{value:.4f}" in doc
    assert model in doc


def test_aes_baseline_table_values_are_present(doc):
    # Spot-check the published baselines transcribed from the EMNLP PDF Table 1/2/12.
    for value in ["0.5091", "0.4121", "0.6965", "0.7676", "0.5230", "0.5475",
                   "0.5114", "0.7853", "0.7274", "0.8900", "0.7474", "0.7977",
                   "0.7646", "0.8569", "0.8807", "0.7446", "0.7685", "0.5669",
                   "0.6801", "0.5084", "0.6673", "0.9332"]:
        assert value in doc, value


FREQ_ROWS = [
    ("LLaMA-7B", "Lookback-lens", 0.7884),
    ("LLaMA-7B", "Fourier-high", 0.8412),
    ("LLaMA-13B", "Wavelet-High", 0.8448),
    ("LLaMA-13B", "Fourier-high", 0.8585),
    ("Mistral-7B", "Lookback-lens", 0.8764),
    ("Mistral-7B", "Fourier-high", 0.8812),
]


@pytest.mark.parametrize("model,method,value", FREQ_ROWS)
def test_frequency_aware_pooled_rows_are_losses(doc, model, method, value):
    assert verdict(OUR_ROC_AUC, OUR_ROC_CI, value) == "LOSS"
    assert f"{value:.4f}" in doc
    assert method in doc


DESIDERATA = [
    ("SFT", 0.871, 0.688, 0.671),
    ("FS", 0.836, 0.694, 0.628),
    ("LLM-aaJ", 0.825, 0.682, 0.617),
    ("PO", 0.800, 0.671, 0.538),
    ("SC-GPT (C)", 0.586, 0.199, 0.346),
    ("SC-GPT (G)", 0.584, 0.199, 0.342),
]


@pytest.mark.parametrize("name,roc,pr,f1", DESIDERATA)
def test_desiderata_roc_auc_verdicts(doc, name, roc, pr, f1):
    want = verdict(OUR_ROC_AUC, OUR_ROC_CI, roc)
    assert f"{roc:.3f}" in doc
    assert f"{pr:.3f}" in doc
    # SC-GPT rows are the two ties; everything else is a loss.
    assert want == ("TIE" if name.startswith("SC-GPT") else "LOSS")
    assert name in doc


@pytest.mark.parametrize("name,roc,pr,f1", DESIDERATA)
def test_desiderata_f1_verdicts(doc, name, roc, pr, f1):
    want = verdict(OUR_F1, OUR_F1_CI, f1)
    assert f"{f1:.3f}" in doc
    # Recomputed, not asserted: SFT (0.671) clears our whole interval -> LOSS;
    # PO (0.538) sits just under our lower bound -> WIN; SC-GPT -> WIN;
    # FS and LLM-aaJ land inside our interval -> TIE.
    expected = {
        "SFT": "LOSS",
        "FS": "TIE",
        "LLM-aaJ": "TIE",
        "PO": "WIN",
        "SC-GPT (C)": "WIN",
        "SC-GPT (G)": "WIN",
    }
    assert want == expected[name], f"{name} F1 {f1}"


def test_desiderata_f1_verdict_words_appear_next_to_each_row(doc):
    """The verdict cell in the doc must be the recomputed one."""
    table = re.search(
        r"\| Detector \| Published F1 \| Ours \| Our 95% CI \| Verdict \|(.+?)\n\n",
        doc,
        re.DOTALL,
    )
    assert table, "F1 verdict table not found"
    rows = [r for r in table.group(1).splitlines() if r.strip().startswith("|")]
    assert len(rows) == 7, rows  # header + separator + 6 detectors
    expected = {
        "SFT": "LOSS",
        "FS": "TIE",
        "LLM-aaJ": "TIE",
        "PO": "WIN",
        "SC-GPT (G)": "WIN",
        "SC-GPT (C)": "WIN",
    }
    for row in rows[2:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        name, f1 = cells[0], float(cells[1])
        assert f"**{expected[name]}**" in cells[4], row
        assert verdict(OUR_F1, OUR_F1_CI, f1) == expected[name], row


# --- honesty guards ----------------------------------------------------------

def test_the_doc_does_not_claim_a_win_over_the_stronger_papers(doc):
    """AES, the frequency-aware method and the desiderata SFT all beat us.

    The file must not present any of those as a win for us.
    """
    for value in ["0.8829", "0.9250", "0.8807", "0.8931", "0.9332", "0.871", "0.836"]:
        for m in re.finditer(re.escape(value), doc):
            line = doc[doc.rfind("\n", 0, m.start()) + 1 : doc.find("\n", m.start())]
            assert "**WIN**" not in line, f"claimed a win against {value}: {line}"


def test_held_out_caveat_is_present(doc):
    """The in-sample caveat and the held-out number must both be stated."""
    assert "0.6185" in doc
    assert "in-sample" in doc
    assert "held-out" in doc


def test_granularity_caveat_is_present(doc):
    assert "span-level" in doc
    assert "token-level" in doc


def test_every_section_is_present(doc):
    for heading in [
        "## 1. The like-for-like head-to-head",
        "## 2. Response-level detection F1",
        "## 3. Span-level F1",
        "## 4. SHAM-Bench",
        "## 5. Uses RAGTruth but publishes no comparable detection number",
        "## 6. Summary",
        "## Sources",
    ]:
        assert heading in doc, heading


def test_not_comparable_papers_are_marked_as_such(doc):
    assert "The Semantic Illusion" in doc
    assert "no AUROC, F1 or accuracy on RAGTruth" in doc
    assert "Not comparable" in doc


# --- a generic guard: recompute every verdict written in the document --------

# Every table row in the doc that carries an explicit "Ours" + CI pair, so the
# interval rule can be re-applied to the published value in that same row.
_VERDICT_ROW = re.compile(
    r"^\|\s*(?P<name>[^|]+?)\s*\|"
    r"\s*(?P<published>[0-9]+\.[0-9]+)\s*\|"
    r"\s*(?P<ours>[0-9]+\.[0-9]+)\s*\|"
    r"\s*\[(?P<lo>[0-9]+\.[0-9]+), (?P<hi>[0-9]+\.[0-9]+)\]\s*\|"
    r"\s*\*\*(?P<verdict>WIN|LOSS|TIE)\*\*",
    re.MULTILINE,
)


def test_every_verdict_in_the_document_follows_the_interval_rule(doc):
    """No verdict may be typed in by hand and left unchecked.

    Each row states a published value, our value, our CI and a verdict. The
    verdict must equal the interval rule applied to that row's own numbers.
    """
    rows = list(_VERDICT_ROW.finditer(doc))
    assert len(rows) >= 15, f"only {len(rows)} verdict rows matched"

    for m in rows:
        published = float(m.group("published"))
        ours = float(m.group("ours"))
        lo, hi = float(m.group("lo")), float(m.group("hi"))
        written = m.group("verdict")

        # The row's "Ours" and CI must be one of our two real measurements.
        assert (ours, lo, hi) in {
            (round(OUR_ROC_AUC, 4), round(OUR_ROC_CI[0], 4), round(OUR_ROC_CI[1], 4)),
            (round(OUR_F1, 4), round(OUR_F1_CI[0], 4), round(OUR_F1_CI[1], 4)),
        }, f"row has an unknown 'Ours'/CI pair: {m.group(0)}"

        want = "WIN" if lo > published else ("LOSS" if hi < published else "TIE")
        assert written == want, (
            f"{m.group('name').strip()}: published {published}, "
            f"CI [{lo}, {hi}] -> computed {want}, document says {written}"
        )


def test_a_loss_is_never_written_as_a_win(doc):
    """Guard against a win being claimed against a paper that beats us."""
    for m in _VERDICT_ROW.finditer(doc):
        published = float(m.group("published"))
        lo, hi = float(m.group("lo")), float(m.group("hi"))
        if published > hi and m.group("verdict") == "WIN":
            raise AssertionError(f"win claimed against {published}: {m.group(0)}")
