"""FastAPI wrapper around the CRISP pipeline.

Design choices
--------------
* **Singleton app state.** A single ``Pipeline`` lives in ``app.state`` for
  the lifetime of the process. Multi-worker deployments would need one
  pipeline per worker (uvicorn ``--workers N`` starts N processes).
* **Sync ``Pipeline.run``.** ``run()`` is a blocking method and calls into
  numpy / transformers which release the GIL mostly via CPU work. We run
  it in a worker thread via ``asyncio.to_thread`` so the FastAPI event
  loop stays responsive.
* **Pipeline state bootstrap.** Loaded eagerly on startup so that the
  first ``/query`` doesn't pay the index-load cost. Configurable via
  ``CRISP_INDEX`` (directory of a saved index). If unset, the service
  starts with an *empty* index and waits for ``POST /index`` to build.
* **Crash isolation.** A failing ``Pipeline.run`` returns a structured
  HTTP 500 (preserves the message but never leaks a stack trace).
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path bootstrap so ``python -m uvicorn api.app`` from anywhere works.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.config import load_config  # noqa: E402
from src.pipeline import Pipeline, RAGResult  # noqa: E402

logger = logging.getLogger("crisp.api")


# ---------------------------------------------------------------------------
# Request / response DTOs
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Question to send through the pipeline."""

    query: str = Field(min_length=1, max_length=4096)


class IndexRequest(BaseModel):
    """Rebuild the index from a list of documents."""

    documents: List[str] = Field(min_length=1, max_length=10_000)


class HealthResponse(BaseModel):
    """Liveness + index state probe."""

    status: str
    version: str
    index_loaded: bool
    n_documents: int
    mock_mode: bool


class IndexResponse(BaseModel):
    """Returned after a successful rebuild."""

    status: str
    n_documents: int
    built_at: str
    took_ms: float


def _rag_result_to_dict(result: RAGResult) -> Dict[str, Any]:
    """Strip ClaimVerdict objects into the same JSON shape the CLI uses.

    The pipeline already ships ``result.to_dict()`` but that doesn't
    serialise nested ``ClaimVerdict`` objects — it serialises them as
    plain dicts already (see ``RAGResult.to_dict``), so this is a thin
    wrapper that just emits ``to_dict()`` and renames nothing. Kept as
    its own function so we have a stable place to evolve the schema
    without touching callers.
    """
    return result.to_dict()


# ---------------------------------------------------------------------------
# Lifespan — bootstrap the Pipeline once per process.
# ---------------------------------------------------------------------------


def _build_pipeline() -> Pipeline:
    """Instantiate the pipeline. Honours env-var overrides (CRISP_MOCK, etc.)."""
    cfg = load_config()
    pipeline = Pipeline(config=cfg)
    logger.info(
        "pipeline built (mock=%s, mode=%s)",
        cfg.mock,
        cfg.retrieval.mode,
    )
    return pipeline


def _try_load_index(pipeline: Pipeline, index_dir: Optional[str]) -> bool:
    """Try ``pipeline.load_index(index_dir)`` — return True on success.

    A failing load is non-fatal: the service still comes up so the
    client can POST /index to build a fresh index.
    """
    if not index_dir:
        return False
    try:
        pipeline.load_index(index_dir)
        logger.info("loaded existing index from %s", index_dir)
        return True
    except Exception as exc:                              # pragma: no cover
        logger.warning("failed to load index at %s: %s", index_dir, exc)
        return False


async def _pipeline_lifespan(app: FastAPI, *, index_dir: Optional[str]) -> None:
    """Replace the synchronous ``Pipeline.run`` with no-op at startup."""
    pipeline = _build_pipeline()
    app.state.pipeline = pipeline
    app.state.index_loaded = False
    app.state.n_documents = 0
    app.state.index_loaded = _try_load_index(pipeline, index_dir)
    if app.state.index_loaded:
        # ``index_dir`` may be a directory whose contents reflect how many
        # chunks the loader produces; we don't know exactly without
        # reading the index files, so report 0 on startup unless later
        # indices report it.
        app.state.n_documents = 0
    yield  # type: ignore[unreachable]  # FastAPI turns this into an ASGI lifespan.


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(index_dir: Optional[str] = None) -> FastAPI:
    """Build a FastAPI app.

    Parameters
    ----------
    index_dir
        Optional directory containing a pre-built index
        (``CRISP_INDEX`` env var is read when this is omitted). If the
        directory doesn't exist or is empty, the service starts in
        *no-index* mode and ``POST /index`` must be called before
        ``POST /query`` returns useful answers.
    """
    if index_dir is None:
        import os
        index_dir = os.environ.get("CRISP_INDEX") or None

    lifespan_handler = _pipeline_lifespan
    # FastAPI lifespan can be a contextmanager OR (app, ...) callable in
    # modern FastAPI; the simpler form (decorator) keeps it short.
    app = FastAPI(
        title="CRISP — Hallucination-Aware RAG",
        version="0.5.0",
        description=(
            "CRISP exposes a small REST surface over the pipeline:\n\n"
            "* `POST /query`  — answer a question, with hallucination verdicts.\n"
            "* `POST /index`  — (re)build the index from supplied documents.\n"
            "* `GET /health`  — liveness + index status.\n"
        ),
        lifespan=_make_lifespan(index_dir),
    )

    # --- routes ---------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        pipe: Pipeline = request.app.state.pipeline
        cfg = pipe.config
        return HealthResponse(
            status="ok",
            version=app.version,
            index_loaded=bool(request.app.state.index_loaded),
            n_documents=int(getattr(request.app.state, "n_documents", 0)),
            mock_mode=bool(cfg.mock),
        )

    @app.post("/query", response_model=Dict[str, Any])
    async def query(request: Request, body: QueryRequest) -> Dict[str, Any]:
        pipe: Pipeline = request.app.state.pipeline
        if not request.app.state.index_loaded:
            raise HTTPException(
                status_code=409,
                detail="No index loaded — POST /index first.",
            )
        t0 = time.perf_counter()
        try:
            result: RAGResult = await asyncio.to_thread(pipe.run, body.query)
        except Exception as exc:
            logger.exception("pipeline.run failed for query=%r", body.query)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        took_ms = (time.perf_counter() - t0) * 1000.0
        payload = _rag_result_to_dict(result)
        payload["_meta"] = {"took_ms": took_ms, "query": body.query}
        return payload

    @app.post("/index", response_model=IndexResponse)
    async def build_index(request: Request, body: IndexRequest) -> IndexResponse:
        pipe: Pipeline = request.app.state.pipeline
        t0 = time.perf_counter()
        try:
            await asyncio.to_thread(pipe.build_index, list(body.documents))
        except Exception as exc:
            logger.exception("build_index failed for %d docs", len(body.documents))
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        took_ms = (time.perf_counter() - t0) * 1000.0
        request.app.state.index_loaded = True
        request.app.state.n_documents = len(body.documents)
        return IndexResponse(
            status="ready",
            n_documents=len(body.documents),
            built_at=datetime.now(timezone.utc).isoformat(),
            took_ms=round(took_ms, 3),
        )

    return app


def _make_lifespan(index_dir: Optional[str]):
    """Build the lifespan contextmanager for ``create_app``.

    Using a factory keeps ``create_app`` tiny and lets tests pass
    ``index_dir=None`` to opt out of any auto-load attempt.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pipeline = _build_pipeline()
        app.state.pipeline = pipeline
        app.state.index_loaded = False
        app.state.n_documents = 0
        if index_dir:
            ok = _try_load_index(pipeline, index_dir)
            app.state.index_loaded = ok
        yield
        # shutdown: nothing to release — Pipeline has no native close hook.

    return lifespan


# ---------------------------------------------------------------------------
# Module-level app for ``uvicorn api.app:app``
# ---------------------------------------------------------------------------

import os                                                              # noqa: E402

_DEFAULT_INDEX = os.environ.get("CRISP_INDEX") or None
app = create_app(index_dir=_DEFAULT_INDEX)
