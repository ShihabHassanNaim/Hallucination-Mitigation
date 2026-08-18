"""Flask front-end for the 13-step hallucination pipeline.

Routes:
  GET  /              — chat-style UI (index + documents + query)
  GET  /structure     — full project tree (renderer)
  POST /api/index     — JSON: build/rebuild index from {"documents":[...]}
  POST /api/query     — JSON: run pipeline {"query":"..."}
  GET  /api/health    — JSON: liveness + index status
  GET  /api/metrics   — JSON: contents of data/metrics.json
  GET  /api/structure — JSON: full project tree (skipping junk dirs)

The Flask process is independent from the FastAPI process — they share
src/pipeline.py but each manages its own Pipeline singleton. The front-end
is intentionally minimal: vanilla HTML/CSS/JS, no build step.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Path bootstrap so ``python -m frontend.app`` from anywhere works.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask, jsonify, render_template, request  # noqa: E402

from src.config import load_config  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.utils.chunker import chunk_documents  # noqa: E402

logger = logging.getLogger("frontend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s)")


_SKIP_DIRS = {".venv", ".git", ".pytest_cache", "__pycache__", ".puku", "node_modules", "dist", "build"}
_SKIP_FILES = {".DS_Store"}
_MAX_DEPTH = 6


def _build_tree(root: Path, depth: int = 0, max_depth: int = _MAX_DEPTH) -> dict[str, Any]:
    """Recursively build a tree of {name, type, children?, size?}."""
    node: dict[str, Any] = {"name": root.name or str(root), "type": "dir"}
    try:
        children: list[dict[str, Any]] = []
        for entry in sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if entry.name in _SKIP_FILES:
                continue
            if entry.is_dir():
                if entry.name in _SKIP_DIRS:
                    continue
                if depth + 1 > max_depth:
                    continue
                children.append(_build_tree(entry, depth + 1, max_depth))
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                children.append({
                    "name": entry.name,
                    "type": "file",
                    "size": size,
                    "ext": entry.suffix.lstrip(".") or "",
                })
        node["children"] = children
        node["n_files"] = sum(1 for c in children if c["type"] == "file")
        node["n_dirs"] = sum(1 for c in children if c["type"] == "dir")
    except PermissionError:
        node["children"] = []
        node["error"] = "permission denied"
    return node


def create_app(index_dir: str | None = None) -> Flask:
    """Build the Flask app and bootstrap a single Pipeline instance.

    The Pipeline is loaded lazily on the first call to ``get_pipeline()``
    so the server can come up and serve the lightweight JSON endpoints
    (``/api/evaluation*``, ``/api/structure``, ``/api/health``)
    immediately while the heavy HF models download in the background.
    """
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    if index_dir is None:
        index_dir = os.environ.get("HALLUC_INDEX") or None

    cfg = load_config()
    app.extensions["pipeline"] = None  # set lazily
    app.extensions["pipeline_lock"] = False
    app.extensions["index_loaded"] = False
    app.extensions["n_documents"] = 0
    app.extensions["project_root"] = str(_ROOT)

    def get_pipeline() -> Pipeline:
        """Lazy pipeline bootstrap — loads models on first use."""
        if app.extensions.get("pipeline") is not None:
            return app.extensions["pipeline"]
        if app.extensions.get("pipeline_lock"):
            raise RuntimeError("pipeline is already being initialised")
        app.extensions["pipeline_lock"] = True
        try:
            pipe = Pipeline(cfg=cfg)
            app.extensions["pipeline"] = pipe
            if index_dir:
                try:
                    _try_load_index(pipe, index_dir)
                    app.extensions["index_loaded"] = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("failed to load index %s: %s", index_dir, exc)
            return pipe
        finally:
            app.extensions["pipeline_lock"] = False

    # ---- Pages ----------------------------------------------------------
    @app.get("/")
    def index_page():
        return render_template("index.html")

    @app.get("/structure")
    def structure_page():
        return render_template("structure.html")

    @app.get("/evaluation")
    def evaluation_page():
        return render_template("evaluation.html")

    @app.get("/literature")
    def literature_page():
        return render_template("literature.html")

    # ---- JSON API -------------------------------------------------------
    @app.get("/api/health")
    def api_health():
        return jsonify(
            {
                "status": "ok",
                "version": "0.2.0",
                "index_loaded": bool(app.extensions["index_loaded"]),
                "n_documents": int(app.extensions.get("n_documents", 0)),
            }
        )

    @app.post("/api/index")
    def api_index():
        body = request.get_json(silent=True) or {}
        docs = body.get("documents") or []
        if not isinstance(docs, list) or not docs:
            return jsonify({"error": "documents must be a non-empty list"}), 422
        if not all(isinstance(d, str) and d.strip() for d in docs):
            return jsonify({"error": "every document must be a non-empty string"}), 422
        try:
            pipeline = get_pipeline()
        except Exception as exc:  # noqa: BLE001
            logger.exception("pipeline init failed")
            return jsonify({"error": f"pipeline init failed: {exc}"}), 500
        if pipeline.embedder is None:
            return jsonify({"error": "embedder not loaded"}), 500
        chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=50)
        doc_ids = [c[0] for c in chunks]
        chunk_texts = [c[1] for c in chunks]
        try:
            vecs = pipeline.embedder.encode(chunk_texts)
            ret = HybridRetriever(dim=vecs.shape[1])
            ret.add(doc_ids, chunk_texts, vecs)
            pipeline.retriever = ret
        except Exception as exc:  # noqa: BLE001
            logger.exception("index build failed")
            return jsonify({"error": str(exc)}), 500
        app.extensions["index_loaded"] = True
        app.extensions["n_documents"] = ret.n_docs
        return jsonify({"status": "indexed", "n_documents": ret.n_docs})

    @app.post("/api/query")
    def api_query():
        body = request.get_json(silent=True) or {}
        query = (body.get("query") or "").strip()
        if not query:
            return jsonify({"error": "query must be a non-empty string"}), 422
        try:
            pipeline = get_pipeline()
        except Exception as exc:  # noqa: BLE001
            logger.exception("pipeline init failed")
            return jsonify({"error": f"pipeline init failed: {exc}"}), 500
        if not app.extensions["index_loaded"] or pipeline.retriever is None:
            return jsonify({"error": "no index loaded — POST /api/index first"}), 409
        try:
            result = pipeline.run(query)
            payload: dict[str, Any] = result.to_dict()
            payload["report_markdown"] = result.report_markdown
            return jsonify(payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("pipeline.run failed")
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/evaluation")
    def api_evaluation():
        """Return the evaluation report JSON.

        Reads ``data/metrics.json`` and computes aggregate stats on the
        fly. Always returns 200 — empty fields if the file is missing
        so the UI can render an empty state without special-casing.

        Query param ``dataset`` selects a registered dataset slug
        (see ``DATASET_REGISTRY``). Defaults to the first available
        dataset in the registry, falling back to ``data/metrics.json``
        if none are registered.
        """
        from src.evaluation.report import (  # noqa: WPS433
            DATASET_REGISTRY,
            load_multi_report,
            load_report,
        )

        slug = (request.args.get("dataset") or "").strip()
        if slug and slug in DATASET_REGISTRY:
            path = Path(DATASET_REGISTRY[slug]["path"])
            return jsonify(load_report(path))

        multi = load_multi_report()
        if multi["datasets"]:
            for d in multi["datasets"]:
                if d["available"]:
                    return jsonify(d["report"])
        return jsonify(load_report())

    @app.get("/api/evaluation/datasets")
    def api_evaluation_datasets():
        """List registered datasets + their load status."""
        from src.evaluation.report import load_multi_report  # noqa: WPS433

        return jsonify(load_multi_report())

    @app.get("/api/evaluation/comparison")
    def api_evaluation_comparison():
        """Side-by-side comparison table across all registered datasets.

        Shape::

            {
              "rows": [
                 { "metric": "Detection Acc.",
                   "halueval": 0.83, "truthfulqa": 0.91, "delta": -0.08 },
                 ...
              ],
              "datasets": [
                 {"slug":"halueval","label":"HaluEval QA",...},
                 ...
              ],
            }
        """
        from src.evaluation.report import DATASET_REGISTRY, load_multi_report  # noqa: WPS433

        multi = load_multi_report()
        slugs = list(DATASET_REGISTRY.keys())

        # Metrics to surface in the comparison table (in display order).
        comparison_keys = [
            ("Detection Acc.",         "detection_accuracy"),
            ("Detection Precision",    "detection_precision"),
            ("Detection Recall",       "detection_recall"),
            ("Detection F1",           "detection_f1"),
            ("Detection ROC-AUC",      "detection_roc_auc"),
            ("Hallucination Reduction","hallucination_reduction_rate"),
            ("Claim Correction Rate",  "claim_correction_rate"),
            ("End-to-End Accuracy",    "end_to_end_accuracy"),
            ("Claim F1",               "claim_f1"),
            ("Verification Success",   "verification_success_rate"),
            ("Retrieval R@5",          "retrieval_recall_at_5"),
            ("Retrieval P@5",          "retrieval_precision_at_5"),
            ("Retrieval MRR",          "retrieval_mrr"),
            ("Retrieval nDCG",         "retrieval_ndcg"),
            ("Mean Confidence",        "mean_confidence"),
        ]

        rows: list[dict[str, Any]] = []
        for label, key in comparison_keys:
            row: dict[str, Any] = {"metric": label, "key": key}
            values: list[float] = []
            for slug in slugs:
                ds = next((d for d in multi["datasets"] if d["slug"] == slug), None)
                if ds is None:
                    row[slug] = None
                    continue
                agg = ds["report"].get("aggregate") or {}
                val = agg.get(key)
                if val is None and not key.startswith("mean_"):
                    val = agg.get(f"mean_{key}")
                row[slug] = val
                if isinstance(val, (int, float)):
                    values.append(float(val))
            if len(values) >= 2:
                row["delta"] = round(values[-1] - values[0], 4)
            else:
                row["delta"] = None
            rows.append(row)

        return jsonify({
            "rows": rows,
            "datasets": [
                {"slug": d["slug"], "label": d["label"], "available": d["available"],
                 "n_queries": (d["report"].get("n_queries") or 0)}
                for d in multi["datasets"]
            ],
        })

    @app.get("/api/accuracy")
    def api_accuracy():
        """Return a focused accuracy dashboard payload.

        Reuses :func:`src.evaluation.report.load_report` and layers on:

        * ``headline``  — the four headline accuracy numbers as a flat
          dict (``detection_accuracy``, ``end_to_end_accuracy``,
          ``claim_f1``, ``verification_success_rate``), plus per-row
          ``mean_*`` aggregates when present.
        * ``breakdown`` — decision distribution (PASS / REGEN / ABSTAIN)
          and a confusion-style split of correct vs incorrect
          per-decision.
        * ``per_query`` — the raw rows (same as ``/api/evaluation``).
        * ``meta``      — n_queries, source, last_modified.

        Always returns 200 with empty fields when ``metrics.json`` is
        missing, so the UI can render an empty state cleanly.

        Query param ``dataset`` selects a registered dataset slug.
        """
        from src.evaluation.report import (  # noqa: WPS433
            DATASET_REGISTRY,
            load_multi_report,
            load_report,
        )

        slug = (request.args.get("dataset") or "").strip()
        if slug and slug in DATASET_REGISTRY:
            path = Path(DATASET_REGISTRY[slug]["path"])
            report = load_report(path)
        else:
            multi = load_multi_report()
            report = None
            if multi["datasets"]:
                for d in multi["datasets"]:
                    if d["available"]:
                        report = d["report"]
                        slug = d["slug"]
                        break
            if report is None:
                report = load_report()

        aggregate = report.get("aggregate") or {}
        rows = report.get("per_query") or []

        # ---- Headline accuracy numbers ----------------------------------
        headline: dict[str, Any] = {}
        for key in (
            "detection_accuracy",
            "end_to_end_accuracy",
            "verification_success_rate",
        ):
            if key in aggregate:
                headline[key] = aggregate[key]
        for key in (
            "claim_f1",
            "retrieval_recall_at_5",
            "retrieval_precision_at_5",
            "retrieval_mrr",
            "retrieval_ndcg",
            "confidence",
        ):
            mean_key = f"mean_{key}"
            if mean_key in aggregate:
                headline[mean_key] = aggregate[mean_key]

        # ---- Decision breakdown -----------------------------------------
        decisions = {"PASS": 0, "REGEN": 0, "ABSTAIN": 0}
        correct = 0
        incorrect = 0
        per_query_acc: list[dict[str, Any]] = []
        for row in rows:
            dec = (row.get("decision") or "").upper()
            if dec not in decisions:
                decisions[dec] = 0
            decisions[dec] += 1
            e2e = row.get("end_to_end_accuracy")
            is_correct = bool(e2e) if isinstance(e2e, (int, float)) else False
            if is_correct:
                correct += 1
            else:
                incorrect += 1
            per_query_acc.append({
                "query": row.get("query") or "",
                "decision": row.get("decision") or "",
                "correct": is_correct,
                "end_to_end_accuracy": e2e,
                "claim_f1": row.get("claim_f1"),
                "confidence": row.get("confidence"),
            })

        breakdown = {
            "decisions": decisions,
            "total": int(report.get("n_queries") or len(rows)),
            "correct": correct,
            "incorrect": incorrect,
        }

        return jsonify({
            "headline": headline,
            "breakdown": breakdown,
            "per_query": per_query_acc,
            "meta": {
                "n_queries": report.get("n_queries") or 0,
                "source": report.get("source") or "unknown",
                "source_file": report.get("source_file") or "",
                "last_modified": report.get("last_modified"),
                "dataset": slug,
            },
        })

    @app.get("/api/metrics")
    def api_metrics():
        from src.evaluation.report import (  # noqa: WPS433
            DATASET_REGISTRY,
            load_multi_report,
        )

        slug = (request.args.get("dataset") or "").strip()
        if slug and slug in DATASET_REGISTRY:
            p = Path(DATASET_REGISTRY[slug]["path"])
        else:
            multi = load_multi_report()
            p = None
            if multi["datasets"]:
                for d in multi["datasets"]:
                    if d["available"]:
                        p = Path(d["path"])
                        break
        if p is None or not p.exists():
            return jsonify({"rows": []})
        try:
            return jsonify(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"metrics unreadable: {exc}"}), 500

    @app.get("/models/status")
    def models_status():
        """Return runtime status of the 6 first-class pipeline components.

        Reuses the same probe logic as the FastAPI app so the UI shows
        the same data regardless of which backend is serving the page.
        """
        from src.models.registry import build_status_report  # noqa: WPS433

        return jsonify(build_status_report())

    @app.get("/api/literature/comparison")
    def api_literature_comparison():
        """Side-by-side comparison vs. 11 published hallucination systems.

        Reads ``Paper/comparison_table.csv`` (master CSV the user curates)
        and returns a JSON-friendly payload with one row per (category,
        metric) and one column per method. ``ours_numeric`` is the lower
        bound for ranged cells like ``"0.91-0.95 (projected)"`` so the
        frontend can plot a single point + a range badge.
        """
        from src.evaluation.literature import load_literature_csv  # noqa: WPS433

        return jsonify(load_literature_csv())

    @app.get("/api/literature/why-we-win")
    def api_literature_why_we_win():
        """Verdict matrix + headline bar chart from Paper/why_we_win.md.

        Shape::

            {
              "rows":      [ {dimension, best_paper, best_score,
                               our_score, winner, justification}, ... ],
              "tally":     {win, tie, loss, verdict},
              "visual_headline": [ {metric, best, ours, gap, tag}, ... ],
            }

        ``winner`` is the raw character from the paper (✓ / ≈ / ✗) so the
        frontend can colour-code without re-parsing the text.
        """
        from src.evaluation.literature import load_why_we_win  # noqa: WPS433

        return jsonify(load_why_we_win())

    @app.get("/api/literature/headline")
    def api_literature_headline():
        """One-shot payload for the literature card on the demo page.

        Combines the verdict tally + the wins/losses + the headline bar
        chart so the frontend can render the 'why my paper wins' card
        with a single fetch.
        """
        from src.evaluation.literature import (  # noqa: WPS433
            load_literature_csv,
            load_why_we_win,
        )

        csv_data = load_literature_csv()
        md_data = load_why_we_win()

        wins = [v for v in md_data.get("rows", []) if v.get("winner", "").startswith("✓")]
        losses = [v for v in md_data.get("rows", []) if v.get("winner", "").startswith("✗")]
        ties = [v for v in md_data.get("rows", []) if v.get("winner", "").startswith("≈")]

        # Per-row win/loss attached to the CSV rows where metric matches.
        # Best-effort — the verdict metric is free-form text, but the
        # visual_headline entries map cleanly to CSV rows.
        per_row_outcomes: list[dict[str, Any]] = []
        csv_lookup = {r["metric"].strip(): r for r in csv_data["rows"]}
        for v in md_data.get("visual_headline", []):
            csv_row = csv_lookup.get(v["metric"].strip())
            per_row_outcomes.append({
                "metric": v["metric"],
                "best": v["best"],
                "ours": v["ours"],
                "gap": v["gap"],
                "tag": v["tag"],
                "is_win": "Best" in v["tag"] or "best" in v["tag"].lower(),
            })

        return jsonify({
            "methods_compared": csv_data["method_columns"],
            "n_rows": len(csv_data["rows"]),
            "tally": md_data.get("tally", {}),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "visual_headline": md_data.get("visual_headline", []),
            "per_row_outcomes": per_row_outcomes,
        })

    @app.get("/api/structure")
    def api_structure():
        """Return the project tree under PROJECT_ROOT.

        Query params:
          depth: int (default 4, max 8) — recursion depth.
          include_hidden: "1" to include dotfiles/dirs (default off).
        """
        depth = int(request.args.get("depth", "4"))
        depth = max(1, min(8, depth))
        include_hidden = request.args.get("include_hidden", "0") == "1"

        # .venv is always skipped (huge, useless), .git is always skipped
        # (huge + slow), the rest only if user wants hidden.
        skip_dirs = {".venv", ".git", "__pycache__"}
        if not include_hidden:
            skip_dirs.update(_SKIP_DIRS - {".venv", ".git", "__pycache__"})

        original_skip = _SKIP_DIRS.copy()
        _SKIP_DIRS.clear()
        _SKIP_DIRS.update(skip_dirs)
        try:
            tree = _build_tree(_ROOT, depth=0, max_depth=depth)
        finally:
            _SKIP_DIRS.clear()
            _SKIP_DIRS.update(original_skip)

        total_files = 0
        total_size = 0

        def walk(n: dict[str, Any]) -> None:
            nonlocal total_files, total_size
            for c in n.get("children", []):
                if c["type"] == "file":
                    total_files += 1
                    total_size += int(c.get("size", 0))
                else:
                    walk(c)

        walk(tree)
        return jsonify({
            "root": str(_ROOT),
            "depth": depth,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "tree": tree,
        })

    return app


def _try_load_index(pipeline: Pipeline, index_dir: str) -> None:
    """Best-effort index load. The saved format mirrors scripts/build_index.py."""
    from src.retrieval.bm25_index import BM25Index

    root = Path(index_dir)
    doc_ids_path = root / "doc_ids.json"
    docs_path = root / "docs.json"
    dense_path = root / "dense.npy"
    if not (doc_ids_path.exists() and docs_path.exists() and dense_path.exists()):
        raise FileNotFoundError(f"index files missing under {root}")
    import numpy as np

    doc_ids = json.loads(doc_ids_path.read_text(encoding="utf-8"))
    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    vecs = np.load(dense_path)
    ret = HybridRetriever(dim=vecs.shape[1])
    ret.faiss.add(doc_ids, docs, vecs)
    ret.bm25.add(doc_ids, docs)
    pipeline.retriever = ret
    # Update the app extensions via a flag the route handler reads on first hit.
    # The caller knows to update app.extensions after this returns.


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=bool(int(os.environ.get("DEBUG", "0"))))