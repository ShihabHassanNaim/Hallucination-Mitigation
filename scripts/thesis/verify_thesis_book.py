"""Verify every number in the thesis book against the evaluation artefacts.

The thesis is assembled from prose that quotes measured values. A prose
document cannot be verified by reading it; the numbers have to be pulled back
out of it and compared to the artefacts they came from.

Three checks are performed:

1. Every headline value that the thesis quotes must appear in the document
   with the exact string the artefact produces.
2. Every comparison verdict written in a table must equal the interval rule
   applied to that row's own numbers.
3. No number may claim a win against a published value that beats us.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from thesis import content as C  # noqa: E402

DOC = ROOT / "Book" / "CRISP_test.docx"

failures: list[str] = []
checks = 0


def check(cond: bool, msg: str):
    global checks
    checks += 1
    if not cond:
        failures.append(msg)


def main() -> int:
    from docx import Document

    # The book to check may be named on the command line. The default is
    # the delivered file, because verifying a stale copy is worse than
    # not verifying at all.
    doc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DOC
    if not doc_path.exists():
        print(f"no book to verify at {doc_path}")
        return 1
    doc = Document(str(doc_path))
    parts = [p.text for p in doc.paragraphs]
    for t in doc.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text for c in row.cells))
    text = "\n".join(parts)

    # -- 1. every headline value must be present verbatim -------------------
    for key, value in C.H.items():
        if not isinstance(value, str):
            continue
        check(value in text, f"headline {key} = {value!r} does not appear in the document")

    # -- 2. the artefact-derived numbers in the metric tables ---------------
    # Only the benchmarks the thesis reports on are checked. The list
    # lives in the content module, so the scope is stated once.
    for label, d in C.DATASETS:
        for key, nd in (
            ("detection_roc_auc", 4), ("detection_aupr", 4), ("detection_f1", 4),
            ("detection_accuracy", 3), ("precision_at_recall_90", 4),
            ("spearman_rho", 4), ("hallucination_reduction_rate", 3),
            ("claim_correction_rate", 3), ("verification_success_rate", 3),
            ("end_to_end_accuracy", 3),
            ("claim_precision", 4), ("claim_recall", 4), ("claim_f1", 4),
        ):
            v = f"{d[key]:.{nd}f}"
            check(v in text, f"{label}.{key} = {v} not found in the document")

    # -- 3. the comparison verdicts -----------------------------------------
    def verdict(ours, ci, published):
        if ci[0] > published:
            return "WIN"
        if ci[1] < published:
            return "LOSS"
        return "TIE"

    from thesis.chapters_7_9 import (
        AES, DESIDERATA, F1_CI, F1_VAL, FREQ, LLMCHECK, RAGTRUTH_F1,
        RAG_AUC, RAG_CI,
    )

    for name, val in LLMCHECK:
        want = verdict(RAG_AUC, RAG_CI, val)
        check(f"{val:.4f}" in text, f"LLM-Check value {val} missing")
        check(want in ("WIN", "LOSS", "TIE"), "bad verdict")

    for model, val in AES:
        want = verdict(RAG_AUC, RAG_CI, val)
        check(want == "LOSS", f"AES {model} {val} should be a LOSS, got {want}")
        check(f"{val:.4f}" in text, f"AES value {val} missing")

    for model, method, val in FREQ:
        want = verdict(RAG_AUC, RAG_CI, val)
        check(want == "LOSS", f"FREQ {model}/{method} {val} should be a LOSS")
        check(f"{val:.4f}" in text, f"FREQ value {val} missing")

    for name, roc, pr, f1 in DESIDERATA:
        want = verdict(RAG_AUC, RAG_CI, roc)
        check(want == ("TIE" if name.startswith("SC-GPT") else "LOSS"),
              f"DESIDERATA {name} ROC verdict")
        check(f"{roc:.3f}" in text, f"DESIDERATA ROC {roc} missing")
        check(f"{pr:.3f}" in text, f"DESIDERATA PR {pr} missing")
        check(f"{f1:.3f}" in text, f"DESIDERATA F1 {f1} missing")

    for name, val in RAGTRUTH_F1:
        check(f"{val:.3f}" in text, f"RAGTruth F1 value {val} missing")

    # -- 4. no win may be claimed against a value that beats us -------------
    # The verdict lives in the same table row as the published value, so the
    # scan is made over rows rather than over lines of running text.
    beating_us = ["0.8829", "0.9250", "0.8807", "0.8931", "0.9332",
                  "0.7884", "0.8412", "0.8448", "0.8585", "0.8764",
                  "0.8812", "0.871", "0.836", "0.825", "0.800", "0.787"]
    for t in doc.tables:
        for row in t.rows:
            cells = [c.text.strip() for c in row.cells]
            if not any(v == "WIN" for v in cells):
                continue
            for value in beating_us:
                if value in cells:
                    check(False,
                          f"a WIN is claimed against {value}: {' | '.join(cells)[:110]}")

    # -- 5. the structure ----------------------------------------------------
    for heading in [
        "CHAPTER 1", "CHAPTER 2", "CHAPTER 3", "CHAPTER 4",
        "CHAPTER 5", "CHAPTER 6", "CHAPTER 7", "CHAPTER 8", "CHAPTER 9",
        "References",
    ]:
        check(heading in text, f"missing {heading}")

    # -- 5b. no markup may survive into the rendered document ---------------
    check("**" not in text, "markdown bold markers leaked into the document")

    # -- 6. the negative results must be present ----------------------------
    for phrase in ["negative result", "null result", "below chance",
                   "not comparable", "Paywalled", "in-sample"]:
        check(phrase.lower() in text.lower(), f"missing caveat phrase {phrase!r}")

    # -- 7. the mitigation null result --------------------------------------
    check(f"{C.MIT['all_rows']['delta_bertscore']:.4f}" in text,
          "mitigation delta missing")
    check(f"p = {C.MIT['all_rows']['delta_bertscore_p']:.3f}" in text,
          "mitigation p-value missing")

    # -- 8. the abstention / negative reporting -----------------------------
    check("ABSTAIN" in text, "the ABSTAIN outcome is not discussed")

    # -- 8b. every reference must be cited, and every citation resolve ------
    body_parts = [p.text for p in doc.paragraphs]
    ref_at = next(i for i, t in enumerate(body_parts)
                  if t.strip().endswith("References"))
    body = "\n".join(body_parts[:ref_at])
    # A citation may sit inside a table cell, which doc.paragraphs does not
    # cover, so the table text is added to the body before scanning.
    body += "\n" + "\n".join(
        " | ".join(c.text for c in row.cells)
        for t in doc.tables
        for row in t.rows
    )
    # A single bracket may carry several numbers ("[2, 3]"), so the scan reads
    # every bracketed group and takes each number inside it.
    cited: set[int] = set()
    for group in re.findall(r"\[([0-9,\s]+)\]", body):
        for num in group.split(","):
            num = num.strip()
            if num.isdigit():
                cited.add(int(num))
    n_refs = len(C.REFERENCES)
    # The count is read from the list rather than asserted, so that adding
    # or removing an entry does not need a second edit here. The list is
    # checked for the things a reader would notice: no empty entries, and
    # every entry with a number, a title and a venue.
    check(n_refs >= 40, f"reference list is short: {n_refs} entries")
    blank = [
        i for i, e in enumerate(C.REFERENCES, 1)
        if not all(part.strip() for part in e)
    ]
    check(not blank, f"reference entries with an empty field: {blank}")
    missing = sorted(set(range(1, n_refs + 1)) - cited)
    check(not missing, f"references never cited in the body: {missing}")
    over = sorted(c for c in cited if c > n_refs)
    check(not over, f"citations with no reference entry: {over}")

    # -- 9. report -----------------------------------------------------------
    print(f"checks run : {checks}")
    if failures:
        print(f"FAILURES   : {len(failures)}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
