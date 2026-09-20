"""Regenerate the offline Tier-1 fixtures from the live local setup.

``make fixture`` (→ ``ledgion fixture``) runs this against the populated docker
Qdrant and the real bge model to produce the three committed artifacts the CI
gate replays:

* ``index.npz`` — every corpus chunk's dense vector plus parallel
  ``chunk_id``/``doc_id``/``page_num`` arrays. No chunk *text*: Tier 1 scores
  pages, so the text would only bloat the repo.
* ``query_vectors.npz`` — one pre-computed embedding per golden qid, produced by
  the *same* ``embed_query`` (hence the same bge revision and query instruction
  prefix) the live retriever applies, so the offline path queries with an
  identical vector.
* ``manifest.json`` — provenance: the embedding model_id + revision, the chunk
  count, the golden qids, the source git SHA, and a generation timestamp. The
  revision is what ``FixtureRetriever`` guards against a config mismatch.

The two ``.npz`` files are written deterministically (chunks sorted by chunk_id,
queries by qid) so regenerating on unchanged inputs yields identical bytes; only
the manifest carries a wall-clock timestamp.

This module is import-safe without torch (numpy + qdrant only at import); the bge
model is loaded lazily via ``BGEEmbedder`` when ``generate_fixtures`` actually
runs, so it never burdens the offline path that merely imports the package.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ledgion.config import Settings
from ledgion.ingest.index import DENSE_VECTOR


def _default_fixtures_dir() -> Path:
    # Single source of truth for the fixtures location (shared with the retriever).
    from ledgion.retrieve.fixture import FIXTURES_DIR

    return FIXTURES_DIR


def _scroll_index(cfg: Settings) -> list[dict]:
    """Pull every point's dense vector + (chunk_id, doc_id, page_num) from the live
    collection, sorted by chunk_id for a deterministic dump."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=cfg.qdrant.url, prefer_grpc=cfg.qdrant.prefer_grpc)
    records: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=cfg.qdrant.collection_name,
            limit=512,
            offset=offset,
            with_payload=["chunk_id", "doc_id", "page_num"],
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


def _write_manifest(
    cfg: Settings, golden: Sequence[dict], chunk_count: int, out_dir: Path
) -> Path:
    from ledgion.eval.runner import git_sha

    manifest = {
        "embedding": {
            "model_id": cfg.embedding.model_id,
            "revision": cfg.embedding.revision,
        },
        "chunk_count": chunk_count,
        "golden_qids": sorted(row["qid"] for row in golden),
        "source_git_sha": git_sha(),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def generate_fixtures(
    cfg: Settings, *, out_dir: Path | None = None, golden: Sequence[dict] | None = None
) -> dict:
    """Regenerate all three fixtures. Returns a small summary for the CLI to print."""
    out_dir = Path(out_dir) if out_dir is not None else _default_fixtures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if golden is None:
        from ledgion.eval.runner import load_golden

        golden = load_golden()

    records = _scroll_index(cfg)
    if not records:
        raise SystemExit(
            f"collection {cfg.qdrant.collection_name!r} is empty; run `ledgion ingest` first."
        )

    index_path = _write_index(records, out_dir)
    query_path = _write_query_vectors(cfg, golden, out_dir)
    manifest_path = _write_manifest(cfg, golden, len(records), out_dir)

    return {
        "chunk_count": len(records),
        "query_count": len(golden),
        "index": str(index_path),
        "query_vectors": str(query_path),
        "manifest": str(manifest_path),
    }
