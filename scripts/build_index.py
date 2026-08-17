"""Build a retriever index from a JSONL of {text: ...} rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.models import BGE_M3Embedder
from src.retrieval.hybrid import HybridRetriever
from src.utils.chunker import chunk_documents


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a FAISS+BM25 hybrid index.")
    ap.add_argument("--input", required=True, help="JSONL file with {text: ...} rows")
    ap.add_argument("--out", default="data/index")
    args = ap.parse_args()

    cfg = load_config()
    embedder = BGE_M3Embedder.from_config(cfg) if hasattr(BGE_M3Embedder, "from_config") else BGE_M3Embedder()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    docs = [r["text"] for r in rows]
    chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=50)
    doc_ids = [c[0] for c in chunks]
    docs = [c[1] for c in chunks]
    vecs = embedder.encode(docs)
    ret = HybridRetriever(dim=vecs.shape[1])
    ret.add(doc_ids, docs, vecs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "doc_ids.json").write_text(json.dumps(doc_ids), encoding="utf-8")
    (out / "docs.json").write_text(json.dumps(docs), encoding="utf-8")
    if ret.faiss.vecs is not None:
        import numpy as np
        np.save(out / "dense.npy", ret.faiss.vecs)
    print(f"indexed {ret.n_docs} chunks -> {out}")


if __name__ == "__main__":
    main()
