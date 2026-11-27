"""Content for the CRISP thesis book.

Every number in this file is pulled from a measured artefact at import time --
nothing is typed in by hand.  The artefacts live in ``data/`` and are the single
source of truth for the project's results.

The chapter text is deliberately kept out of the renderer so that the renderer
can be re-run without touching prose, and so that a number can only be changed
in one place (the artefact) and propagate everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
GRAPHS = DATA / "graphs"
AUDIT = GRAPHS / "audit"


# --------------------------------------------------------------------------
# artefact loading
# --------------------------------------------------------------------------

def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


HAL = _load("metrics_halueval_fitted.json")["aggregate"]
TQA = _load("metrics_truthfulqa_closedbook.json")["aggregate"]
RAG = _load("metrics_ragtruth_fitted.json")["aggregate"]
ABL = _load("ablation_results.json")
MIT = _load("mitigation_results.json")["summary"]
CMP = _load("comparison_to_published.json")
HEAD = _load("detection_head_refit600.json")
HEAD99 = _load("detection_head.json")

# The benchmarks the thesis reports on. The book was rewritten to carry
# HaluEval and RAGTruth only, so this list is the single statement of
# that scope: the verifier checks the numbers of these datasets and no
# others, and a reader of the source can see at once which benchmarks
# the thesis is about. TruthfulQA remains loadable above because the
# closed-book artefact is still on disk, but no number from it is
# reported.
DATASETS = [
    ("HaluEval", HAL),
    ("RAGTruth", RAG),
]


def f(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def ci(d: dict, key: str, n: int = 4) -> str:
    lo, hi = d[f"{key}_ci95"]
    return f"[{lo:.{n}f}, {hi:.{n}f}]"


def pct(x: float, n: int = 1) -> str:
    return f"{100 * x:.{n}f}%"


_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
    12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    20: "twenty",
}


def word(n: int) -> str:
    """A count as the word the prose uses.

    The thesis writes numbers below twenty as words, because a sentence
    that reads "of the 17 pipelines" is a table caption, not prose. The
    count comes from the data, so the word follows it.
    """
    if n in _WORDS:
        return _WORDS[n]
    return str(n)


# --------------------------------------------------------------------------
# headline values, derived
# --------------------------------------------------------------------------

# The architectural survey. The counts the thesis states about the
# surveyed pipelines are read from the survey itself, so that a change
# to the survey cannot leave the prose claiming a number the data no
# longer supports.
from . import survey  # noqa: E402

N_PIPELINES = len(survey.capability_grid())
N_PIPELINE_WORDS = word(N_PIPELINES)
COVERAGE = dict(
    zip([c[0] for c in survey.CAPABILITIES], survey.coverage_counts())
)

H = {
    "hal_auc": f(HAL["detection_roc_auc"]),
    "hal_auc_ci": ci(HAL, "detection_roc_auc"),
    "hal_aupr": f(HAL["detection_aupr"]),
    "hal_f1": f(HAL["detection_f1"]),
    "hal_acc": f(HAL["detection_accuracy"], 3),
    "hal_p90": f(HAL["precision_at_recall_90"]),
    "hal_rho": f(HAL["spearman_rho"]),
    "hal_hrr": f(HAL["hallucination_reduction_rate"], 3),
    "hal_ccr": f(HAL["claim_correction_rate"], 3),
    "hal_vsr": f(HAL["hallucination_reduction_rate"] and HAL["verification_success_rate"], 3),
    "hal_claim_p": f(HAL["claim_precision"]),
    "hal_claim_r": f(HAL["claim_recall"]),
    "hal_claim_f1": f(HAL["claim_f1"]),
    "hal_recall5": f(HAL["retrieval_recall_at_5"], 3),
    "hal_e2e": f(HAL["end_to_end_accuracy"], 3),
    "hal_lat": f(HAL["latency_mean_s"], 2),
    "hal_n": HAL["n_queries"],
    "hal_uq": HAL["n_unique_queries"],

    "rag_auc": f(RAG["detection_roc_auc"]),
    "rag_auc_ci": ci(RAG, "detection_roc_auc"),
    "rag_aupr": f(RAG["detection_aupr"]),
    "rag_f1": f(RAG["detection_f1"]),
    "rag_f1_ci": ci(RAG, "detection_f1"),
    "rag_acc": f(RAG["detection_accuracy"], 3),
    "rag_p90": f(RAG["precision_at_recall_90"]),
    "rag_rho": f(RAG["spearman_rho"]),
    "rag_hrr": f(RAG["hallucination_reduction_rate"], 3),
    "rag_ccr": f(RAG["claim_correction_rate"], 3),
    "rag_vsr": f(RAG["verification_success_rate"], 3),
    "rag_claim_p": f(RAG["claim_precision"]),
    "rag_e2e": f(RAG["end_to_end_accuracy"], 3),
    "rag_lat": f(RAG["latency_mean_s"], 2),
    "rag_n": RAG["n_queries"],
    "rag_uq": RAG["n_unique_queries"],
    "rag_recall5": f(RAG["retrieval_recall_at_5"], 3),

    "held_auc": f(HEAD["models"]["logistic (fitted)"]["roc_auc_holdout"]),
    "held_auc_ci": (
        f"[{HEAD['models']['logistic (fitted)']['roc_auc_holdout_ci95'][0]:.4f}, "
        f"{HEAD['models']['logistic (fitted)']['roc_auc_holdout_ci95'][1]:.4f}]"
    ),
    "hand_auc": f(HEAD99["models"]["hand-set weights (current)"]["roc_auc_all"]),
    "unw_auc": f(HEAD99["models"]["unweighted signed sum"]["roc_auc_all"]),
    "fit99_auc": f(HEAD99["models"]["logistic (fitted)"]["roc_auc_all"]),

    "abl_full": f(ABL["full_pipeline"]["detection_roc_auc"]),
    "abl_nodet": f(ABL["without_step03_detector"]["detection_roc_auc"]),
    "abl_n": ABL["full_pipeline"]["n_queries"],

    "mit_delta": f(MIT["all_rows"]["delta_bertscore"], 4),
    "mit_delta_ci": (
        f"[{MIT['all_rows']['delta_bertscore_ci95'][0]:.4f}, "
        f"{MIT['all_rows']['delta_bertscore_ci95'][1]:.4f}]"
    ),
    "mit_p": f(MIT["all_rows"]["delta_bertscore_p"], 3),
    "mit_regen": MIT["n_regenerated"],
    "mit_kept": MIT["n_kept"],
    "mit_guarded": MIT["n_guarded"],
    "mit_rows": MIT["all_rows"]["n"],
    "mit_uq": MIT["n_unique_queries"],
    "mit_rouge_delta": f(MIT["all_rows"]["delta_rouge_l"], 4),
    "mit_rouge_p": f(MIT["all_rows"]["delta_rouge_l_p"], 3),
}


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------

META = {
    "title": "CRISP: A Calibrated, Retrieval-Grounded Pipeline for Detecting and Mitigating Hallucinations in Retrieval-Augmented Generation",
    "subtitle": "A black-box, model-agnostic framework for verifiable claim-level verification with abstain-capable gating",
    "author": "Shihab Hassan Naim",
    "roll": "Roll: 16 | ID: 1622101042",
    "dept": "Department of Computer Science and Engineering",
    "university": "Dhaka International University",
    "course": "CSE-425: Project Work",
    "batch": "62nd Batch",
    "supervisor": "Dr. Md. Shahinuzzaman",
    "co_supervisor": "—",
    "degree": "Bachelor of Science in Computer Science and Engineering",
    "date": "Fall 2026",
}


def fig(path: Path) -> str:
    return str(path)


# --------------------------------------------------------------------------
# reference list
# --------------------------------------------------------------------------

REFERENCES = [
    ("Valmeekam et al.", "2023", "Can Large Language Models Really Improve by Self-critiquing Their Own Plans?",
     "arXiv preprint arXiv:2310.08118"),
    ("Gao et al.", "2023", "Retrieval-Augmented Generation for Large Language Models: A Survey",
     "arXiv preprint arXiv:2312.10997"),
    ("Ni et al.", "2025", "Towards Trustworthy Retrieval Augmented Generation for Large Language Models: A Survey",
     "arXiv preprint arXiv:2502.06872"),
    ("Manakul, Liusie and Gales", "2023", "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models",
     "EMNLP 2023, pp. 9004–9017"),
    ("Sriramanan et al.", "2024", "LLM-Check: Investigating Detection of Hallucinations in Large Language Models",
     "NeurIPS 2024 Main Conference Track"),
    ("Chen et al.", "2024", "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection",
     "ICLR 2024"),
    ("Cao et al.", "2023", "AutoHall: Automated Factuality Hallucination Dataset Generation for Large Language Models",
     "arXiv preprint arXiv:2310.00259"),
    ("Huang and Lane", "2026", "SHAM-Bench: A Span-level Hallucination Benchmark Across Multiple Tasks",
     "arXiv preprint arXiv:2610.27370"),
    ("Su et al.", "2026", "When Attention Escapes: Diagnosing and Mitigating Hallucination in Retrieval-Augmented Generation",
     "Findings of EMNLP 2026, pp. 8678–8696"),
    ("Niu et al.", "2023", "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models",
     "arXiv preprint arXiv:2401.00396"),
    ("Li et al.", "2023", "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models",
     "EMNLP 2023, pp. 6449–6464"),
    ("Muennighoff et al.", "2024", "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
     "Hugging Face model checkpoint"),
    ("Chen et al.", "2024", "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation",
     "arXiv preprint arXiv:2402.03216"),
    ("Chen et al.", "2024", "BGE-Reranker-v2-m3: A General Multilingual Reranker",
     "Hugging Face model checkpoint"),
    ("Zaratiana et al.", "2023", "GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer",
     "arXiv preprint arXiv:2311.08526"),
    ("Hugging Face", "2024", "Qwen2.5-0.5B-Instruct",
     "Hugging Face model checkpoint"),
    ("Pedregosa et al.", "2011", "scikit-learn: Machine Learning in Python",
     "Journal of Machine Learning Research, 12, 2825–2830"),
    ("Chen and Guestrin", "2016", "XGBoost: A Scalable Tree Boosting System",
     "Proceedings of the 22nd ACM SIGKDD, pp. 785–794"),
    ("Johnson, Douze and Jégou", "2017", "Billion-scale similarity search with GPUs",
     "arXiv preprint arXiv:1702.08734"),
    ("Cormack et al.", "2009", "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods",
     "SIGIR 2009, pp. 758–759"),
    ("Harris et al.", "2020", "Array Programming with NumPy",
     "Nature, 585, 357–362"),
    ("Virtanen et al.", "2020", "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python",
     "Nature Methods, 17, 261–272"),
    ("McKinney", "2010", "Data Structures for Statistical Computing in Python",
     "Proceedings of the 9th Python in Science Conference, pp. 56–61"),
    ("Hunter", "2007", "Matplotlib: A 2D Graphics Environment",
     "Computing in Science & Engineering, 9(3), 90–95"),
    ("Paszke et al.", "2019", "PyTorch: An Imperative Style, High-Performance Deep Learning Library",
     "Advances in Neural Information Processing Systems 32"),
    ("Kaufman et al.", "2012", "Leakage in Data Mining: Formulation, Detection, and Avoidance",
     "ACM Transactions on Knowledge Discovery from Data, 6(3), 21"),
    ("Efron and Tibshirani", "1993", "An Introduction to the Bootstrap",
     "Chapman & Hall, New York"),
    ("Pandit et al.", "2025", "Teaching with Lies: Curriculum DPO on Synthetic Negatives for Hallucination Detection",
     "arXiv preprint arXiv:2505.17558"),
    ("Bestetti, Zhang and Stefanidis", "2026", "A Systematic Literature Review on Hallucination Detection Methods in LLMs",
     "EDBT/ICDT 2026 Workshops, CEUR Vol-4192"),
    ("Hu et al.", "2024", "Embedding and Gradient Say Wrong: A White-Box Method for Hallucination Detection",
     "EMNLP 2024 Main, pp. 1950–1959"),
    ("Cheng et al.", "2024", "Small Agent Can Also Rock! Empowering Small Language Models as Hallucination Detector",
     "EMNLP 2024 Main, pp. 14600–14615"),
    ("Beigi et al.", "2024", "InternalInspector I²: Robust Confidence Estimation in LLMs through Internal States",
     "Findings of EMNLP 2024, pp. 12847–12865"),
    ("Zhao et al.", "2024", "Medico: Towards Hallucination Detection and Correction with Multi-source Evidence Fusion",
     "EMNLP 2024 Demo, pp. 34–45"),
    ("Sun et al.", "2025", "Towards Detecting LLMs Hallucination via Markov Chain-based Multi-agent Debate Framework",
     "ICASSP 2025; arXiv preprint arXiv:2406.03075"),
    ("Zhang et al.", "2024", "Enhancing Hallucination Detection through Perturbation-Based Synthetic Data Generation in System Responses",
     "Findings of ACL 2024, pp. 13321–13332"),
    ("Wang et al.", "2024", "Factcheck-Bench: Fine-Grained Evaluation Benchmark for Automatic Fact-checkers",
     "Findings of EMNLP 2024, pp. 14199–14230"),
    ("Mountantonakis and Tzitzikas", "2023", "Real-Time Validation of ChatGPT Facts using RDF Knowledge Graphs (GPTLODS)",
     "ISWC 2023, CEUR Vol-3632, paper 408"),
    ("Heo, Son and Park", "2025", "HaluCheck: Explainable and verifiable automation for detecting hallucinations in LLM responses",
     "Expert Systems with Applications, 272, 126712"),
    ("Qi et al.", "2026", "Detecting Contextual Hallucinations in LLMs with Frequency-Aware Attention",
     "arXiv preprint arXiv:2602.18145"),
    ("Chen et al.", "2026", "Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights",
     "arXiv preprint arXiv:2605.11330"),
    ("Gao et al.", "2025",
     "Decoding Memories: An Efficient Pipeline for Self-Consistency Hallucination Detection",
     "arXiv preprint arXiv:2508.21228"),
    ("Sinha", "2025",
     "The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems",
     "arXiv preprint arXiv:2512.15068"),
]
