"""Regenerate the offline Tier-1 fixtures from the live local setup.

``make fixture`` (→ ``ledgion fixture``) runs this against the populated docker
Qdrant and the real bge model to produce the committed artifacts the CI gate and
the offline hybrid path replay:

* ``index.npz`` — every corpus chunk's dense vector plus parallel
  ``chunk_id``/``doc_id``/``page_num`` arrays. No chunk *text*: Tier 1 scores
  pages, so the text would only bloat the repo.
* ``query_vectors.npz`` — one pre-computed embedding per golden qid, produced by
  the *same* ``embed_query`` (hence the same bge revision and query instruction
  prefix) the live retriever applies, so the offline path queries with an
  identical vector.
* ``sparse_rankings.npz`` — per golden qid, the top-STORED_DEPTH sparse chunk_ids
  and their BM25 scores, computed over the *full corpus* by a real BM25 index.
  Depth is fixed (not tied to retrieval.top_k) so one fixture serves any top_k up
  to that depth; the offline sparse retriever truncates at runtime. No chunk text
  and no full BM25 index ship to CI — only these frozen per-query rankings.
* ``rerank_scores.npz`` — per golden qid, the cross-encoder score for every
  candidate chunk_id in that question's dense top-STORED_DEPTH pool, computed live
  by the real reranker. Lets a reranked run score offline (no model) and reorder
  identically to the live path; a candidate outside this frozen pool trips
  FixtureReranker's loud missing-score guard.
* ``manifest.json`` — provenance: the embedding model_id + revision, the chunk
  count, the golden qids, the source git SHA, a timestamp, a ``sparse`` block
  (backend, tokeniser settings, BM25 parameters, stored depth, corpus fingerprint),
  AND a ``reranker`` block (model_id, revision, candidate depth, corpus fingerprint)
  that the sparse/hybrid and rerank fixture guards check against the config.

Dense, sparse, and rerank artifacts are regenerated **together, atomically**
(computed from one corpus scroll, staged in a temp dir, then moved into place) so
they can never drift apart — the corpus fingerprint that guards the sparse and
rerank paths is only meaningful if all three artifacts describe the same corpus.

The ``.npz`` files are written deterministically (chunks sorted by chunk_id,
queries/qids sorted by qid) so regenerating on unchanged inputs yields identical
bytes; only the manifest carries a wall-clock timestamp.

This module is import-safe without torch (numpy + qdrant only at import); the bge
model and the BM25 stack are loaded lazily when ``generate_fixtures`` runs, so they
never burden the offline path that merely imports the package.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ledgion.config import Settings
from ledgion.ingest.index import DENSE_VECTOR
from ledgion.interfaces import Chunk


def _default_fixtures_dir() -> Path:
    # Single source of truth for the fixtures location (shared with the retriever).
    from ledgion.retrieve.fixture import FIXTURES_DIR

    return FIXTURES_DIR


def _scroll_corpus(cfg: Settings) -> list[dict]:
    """Pull every point's dense vector + full payload from the live collection, once,
    sorted by chunk_id for a deterministic dump. One scroll feeds both the dense index
    and the BM25 build, so the two artifacts describe exactly the same corpus."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=cfg.qdrant.url, prefer_grpc=cfg.qdrant.prefer_grpc)
    records: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=cfg.qdrant.collection_name,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=[DENSE_VECTOR],
        )
        for p in points:
            vector = p.vector[DENSE_VECTOR] if isinstance(p.vector, dict) else p.vector
            payload = p.payload or {}
            records.append(
                {
                    "chunk_id": payload["chunk_id"],
                    "doc_id": payload["doc_id"],
                    "page_num": int(payload["page_num"]),
                    "text": payload["text"],
                    "company": payload["company"],
                    "ticker": payload["ticker"],
                    "fiscal_year": payload["fiscal_year"],
                    "form_type": payload["form_type"],
                    "vector": np.asarray(vector, dtype=np.float32),
                }
            )
        if offset is None:
            break

    records.sort(key=lambda r: r["chunk_id"])
    return records


def _write_index(records: Sequence[dict], out_dir: Path) -> Path:
    path = out_dir / "index.npz"
    np.savez(
        path,
        vectors=np.stack([r["vector"] for r in records]),
        chunk_id=np.array([r["chunk_id"] for r in records]),
        doc_id=np.array([r["doc_id"] for r in records]),
        page_num=np.array([r["page_num"] for r in records], dtype=np.int64),
    )
    return path


def _write_query_vectors(cfg: Settings, golden: Sequence[dict], out_dir: Path) -> Path:
    """Embed each golden question with the real bge model (same prefix as retrieval)
    and save one vector per qid. Sorted by qid so the archive is deterministic."""
    from ledgion.ingest.embed import BGEEmbedder

    embedder = BGEEmbedder.from_config(cfg)
    rows = sorted(golden, key=lambda r: r["qid"])
    vectors = {
        row["qid"]: embedder.embed_query(row["question"]).astype(np.float32) for row in rows
    }
    path = out_dir / "query_vectors.npz"
    np.savez(path, **vectors)
    return path


def _write_sparse_rankings(
    cfg: Settings, records: Sequence[dict], golden: Sequence[dict], out_dir: Path
) -> tuple[Path, int]:
    """Build a real BM25 index over the full corpus and freeze the top-STORED_DEPTH
    (chunk_id, score) list per golden qid. Returns the path and the depth used."""
    from ledgion.retrieve.fixture import STORED_DEPTH
    from ledgion.retrieve.sparse import BM25Tokenizer, SparseRetriever

    chunks = [
        Chunk(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            page_num=r["page_num"],
            text=r["text"],
            company=r["company"],
            ticker=r["ticker"],
            fiscal_year=r["fiscal_year"],
            form_type=r["form_type"],
        )
        for r in records
    ]
    sparse = SparseRetriever(
        chunks=chunks,
        tokenizer=BM25Tokenizer.from_config(cfg),
        k1=cfg.sparse.k1,
        b=cfg.sparse.b,
        epsilon=cfg.sparse.epsilon,
    )
    depth = min(STORED_DEPTH, len(chunks))

    rows = sorted(golden, key=lambda r: r["qid"])
    qids = [row["qid"] for row in rows]
    chunk_ids: list[list[str]] = []
    scores: list[list[float]] = []
    for row in rows:
        retrieved = sparse.retrieve(row["question"], top_k=depth)
        chunk_ids.append([rc.chunk.chunk_id for rc in retrieved])
        scores.append([rc.score for rc in retrieved])

    path = out_dir / "sparse_rankings.npz"
    np.savez(
        path,
        qids=np.array(qids),
        chunk_ids=np.array(chunk_ids),
        scores=np.array(scores, dtype=np.float32),
    )
    return path, depth


def _write_rerank_scores(
    cfg: Settings, records: Sequence[dict], golden: Sequence[dict], out_dir: Path
) -> tuple[Path, int, list[dict]]:
    """Freeze the cross-encoder score of every candidate in each golden qid's pool, for
    the active reranker **and** every model in ``reranker.compare``.

    The pool is the **dense top-``depth``** set (depth = STORED_DEPTH), shared across
    all models: parity between the live and fixture dense retrievers is already
    established (Phase 5), and at ``sparse_weight <= 0.5`` the fused hybrid top-50 is a
    reordering of dense's top-50, so the dense top-100 pool contains every candidate any
    reranked config (dense@50, dense@100, hybrid@50) can surface. A chunk outside it
    trips FixtureReranker's missing-score guard — exactly the "run make fixture" signal.

    Scores are stored **keyed by reranker revision** ([n_models, n_qids, depth]) so
    several models coexist and none can borrow another's scores. Runs live (docker
    Qdrant + each cross-encoder — MiniLM ~80MB, bge-reranker-base ~1.1GB), scoring
    depth×|golden| pairs *per model* — the slow part of `make fixture`. Each qid's pool
    is sorted by chunk_id so the archive is byte-deterministic regardless of retrieval
    order (the reranker rebuilds a dict at load, so order never affects the result).

    Returns the path, the depth, and the list of frozen ``{model_id, revision}`` (for
    the manifest guard)."""
    from ledgion.retrieve.dense import DenseRetriever
    from ledgion.retrieve.fixture import STORED_DEPTH
    from ledgion.retrieve.rerank import CrossEncoderReranker

    depth = min(STORED_DEPTH, len(records))
    dense = DenseRetriever.from_config(cfg)

    # Models to freeze: the active reranker first, then each compare model, deduped by
    # (model_id, revision) so listing the active model again is harmless.
    refs: list[tuple[str, str | None]] = [(cfg.reranker.model_id, cfg.reranker.revision)]
    for r in cfg.reranker.compare:
        if (r.model_id, r.revision) not in refs:
            refs.append((r.model_id, r.revision))

    rows = sorted(golden, key=lambda r: r["qid"])
    qids = [row["qid"] for row in rows]

    # Build each qid's pool ONCE (dense top-depth), sorted by chunk_id — shared by all
    # models, so the stored chunk_ids are model-independent and deterministic.
    pool_ids: list[list[str]] = []
    pool_texts: list[list[str]] = []
    for row in rows:
        pool = dense.retrieve(row["question"], top_k=depth)
        pairs = sorted(((rc.chunk.chunk_id, rc.chunk.text) for rc in pool), key=lambda p: p[0])
        pool_ids.append([cid for cid, _ in pairs])
        pool_texts.append([text for _, text in pairs])

    # Score each model over every qid's pool; model_scores[m][i] aligns to pool_ids[i].
    model_scores: list[list[list[float]]] = []
    models: list[dict] = []
    for model_id, revision in refs:
        reranker = CrossEncoderReranker(
            model_id=model_id,
            revision=revision,
            device=cfg.reranker.device,
            batch_size=cfg.reranker.batch_size,
            num_threads=cfg.torch.num_threads,
        )
        model_scores.append(
            [
                reranker.score(row["question"], texts)
                for row, texts in zip(rows, pool_texts, strict=True)
            ]
        )
        models.append({"model_id": model_id, "revision": revision})

    path = out_dir / "rerank_scores.npz"
    np.savez(
        path,
        qids=np.array(qids),
        chunk_ids=np.array(pool_ids),                       # [n_qids, depth]
        revisions=np.array([rev for _, rev in refs]),       # [n_models]
        scores=np.array(model_scores, dtype=np.float32),    # [n_models, n_qids, depth]
    )
    return path, depth, models


def _write_manifest(
    cfg: Settings,
    golden: Sequence[dict],
    records: Sequence[dict],
    stored_depth: int,
    rerank_depth: int,
    rerank_models: list[dict],
    out_dir: Path,
) -> Path:
    from ledgion.eval.runner import git_sha
    from ledgion.retrieve.sparse import bm25_params, corpus_fingerprint, tokenizer_config

    fingerprint = corpus_fingerprint([r["chunk_id"] for r in records])
    manifest = {
        "embedding": {
            "model_id": cfg.embedding.model_id,
            "revision": cfg.embedding.revision,
        },
        "chunk_count": len(records),
        "golden_qids": sorted(row["qid"] for row in golden),
        # The sparse ablation's provenance — the sparse/hybrid fixture guard checks
        # every field here against the resolved config (and the corpus fingerprint
        # against index.npz), same pattern as the embedding-revision guard.
        "sparse": {
            "backend": cfg.sparse.backend,
            "tokenizer": tokenizer_config(cfg),
            "bm25": bm25_params(cfg),
            "stored_depth": stored_depth,
            "corpus_fingerprint": fingerprint,
        },
        # The reranker fixture's provenance — the rerank fixture guard checks that the
        # configured (model_id, revision) is among `models` (which keys the frozen
        # scores to a model version) and the corpus fingerprint. candidate_depth is the
        # pool depth scores were frozen at; models lists every frozen reranker.
        "reranker": {
            "candidate_depth": rerank_depth,
            "corpus_fingerprint": fingerprint,
            "models": rerank_models,
        },
        "source_git_sha": git_sha(),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def generate_fixtures(
    cfg: Settings, *, out_dir: Path | None = None, golden: Sequence[dict] | None = None
) -> dict:
    """Regenerate all fixtures atomically. Returns a summary (counts + byte sizes)."""
    out_dir = Path(out_dir) if out_dir is not None else _default_fixtures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if golden is None:
        from ledgion.eval.runner import load_golden

        golden = load_golden()

    # Compute the dense index first: an empty collection is a hard error before any
    # file is touched, so a failed run never leaves half-written fixtures.
    records = _scroll_corpus(cfg)
    if not records:
        raise SystemExit(
            f"collection {cfg.qdrant.collection_name!r} is empty; run `ledgion ingest` first."
        )

    # Stage every artifact in a temp dir, then move each into place. Dense and sparse
    # are written from the *same* scroll, so the fingerprint that ties them can't lie.
    staging = out_dir / ".fixture_tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_index(records, staging)
        _write_query_vectors(cfg, golden, staging)
        _, depth = _write_sparse_rankings(cfg, records, golden, staging)
        _, rerank_depth, rerank_models = _write_rerank_scores(cfg, records, golden, staging)
        _write_manifest(cfg, golden, records, depth, rerank_depth, rerank_models, staging)

        names = [
            "index.npz",
            "query_vectors.npz",
            "sparse_rankings.npz",
            "rerank_scores.npz",
            "manifest.json",
        ]
        for name in names:
            os.replace(staging / name, out_dir / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    sizes = {name: (out_dir / name).stat().st_size for name in names}
    return {
        "chunk_count": len(records),
        "query_count": len(golden),
        "stored_depth": depth,
        "rerank_depth": rerank_depth,
        "rerank_models": rerank_models,
        "out_dir": str(out_dir),
        "sizes": sizes,
        "total_bytes": sum(sizes.values()),
    }
