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
    """Build the Flask app and bootstrap a single Pipeline instance."""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    if index_dir is None:
        index_dir = os.environ.get("HALLUC_INDEX") or None

    cfg = load_config()
    pipeline = Pipeline(cfg=cfg)
    app.extensions["pipeline"] = pipeline
    app.extensions["index_loaded"] = False
    app.extensions["n_documents"] = 0
    app.extensions["project_root"] = str(_ROOT)

    if index_dir:
        try:
            _try_load_index(pipeline, index_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load index %s: %s", index_dir, exc)

    # ---- Pages ----------------------------------------------------------
    @app.get("/")
    def index_page():
        return render_template("index.html")

    @app.get("/structure")
    def structure_page():
        return render_template("structure.html")

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
        """
        from src.evaluation.report import load_report  # noqa: WPS433

        return jsonify(load_report())

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
        """
        from src.evaluation.report import load_report  # noqa: WPS433

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
            },
        })

    @app.get("/api/metrics")
    def api_metrics():
        p = Path("data/metrics.json")
        if not p.exists():
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