"""The offline sparse fixture: staleness/depth guards and the CI hybrid path.

Everything here runs with no model, no network, and no committed fixture data —
tiny ``.npz``/manifest files are fabricated under tmp_path, exactly like
test_fixture_retriever.py. They pin that a stale sparse fixture fails loudly (a
drifted BM25 parameter, a drifted corpus fingerprint, or a top_k past the stored
depth), that a valid fixture serves rankings by qid, and that the full offline
hybrid path — live dense over fixture vectors → frozen sparse rankings → fusion →
scoring — runs end to end (with a fake fuse standing in for the user's function).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ledgion.config import load_config
from ledgion.retrieve.fixture import FixtureRetriever, FixtureSparseRetriever
from ledgion.retrieve.fusion import HybridRetriever
from ledgion.retrieve.sparse import bm25_params, corpus_fingerprint, tokenizer_config


def _write_index(fixtures_dir, *, chunk_ids, doc_ids, pages, vectors=None) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "chunk_id": np.array(chunk_ids),
        "doc_id": np.array(doc_ids),
        "page_num": np.array(pages, dtype=np.int64),
    }
    if vectors is not None:
        arrays["vectors"] = np.asarray(vectors, dtype=np.float32)
    np.savez(fixtures_dir / "index.npz", **arrays)


def _sparse_block(cfg, chunk_ids, *, stored_depth=100, fingerprint=None) -> dict:
    return {
        "backend": cfg.sparse.backend,
        "tokenizer": tokenizer_config(cfg),
        "bm25": bm25_params(cfg),
        "stored_depth": stored_depth,
        "corpus_fingerprint": fingerprint
        if fingerprint is not None
        else corpus_fingerprint(chunk_ids),
    }


def _write_manifest(fixtures_dir, *, sparse_block, revision) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    (fixtures_dir / "manifest.json").write_text(
        json.dumps(
            {"embedding": {"model_id": "toy", "revision": revision}, "sparse": sparse_block}
        ),
        encoding="utf-8",
    )


def _write_rankings(fixtures_dir, qid_to_pairs) -> None:
    """qid_to_pairs: {qid: [(chunk_id, score), ...]} (uniform depth per qid)."""
    qids = sorted(qid_to_pairs)
    chunk_ids = [[cid for cid, _ in qid_to_pairs[q]] for q in qids]
    scores = [[s for _, s in qid_to_pairs[q]] for q in qids]
    np.savez(
        fixtures_dir / "sparse_rankings.npz",
        qids=np.array(qids),
        chunk_ids=np.array(chunk_ids),
        scores=np.array(scores, dtype=np.float32),
    )


# --- staleness guards -------------------------------------------------------


def test_staleness_guard_fires_on_changed_bm25_param(tmp_path):
    cfg = load_config()
    ids = ["000000000000000a", "000000000000000b", "000000000000000c"]
    _write_index(tmp_path, chunk_ids=ids, doc_ids=["D"] * 3, pages=[1, 2, 3])
    block = _sparse_block(cfg, ids)
    block["bm25"] = {**block["bm25"], "k1": 9.9}  # drifted from the resolved config
    _write_manifest(tmp_path, sparse_block=block, revision=cfg.embedding.revision)

    with pytest.raises(SystemExit, match="run make fixture"):
        FixtureSparseRetriever.from_config(cfg, fixtures_dir=tmp_path, golden=[])


def test_staleness_guard_fires_on_changed_corpus_fingerprint(tmp_path):
    cfg = load_config()
    ids = ["000000000000000a", "000000000000000b", "000000000000000c"]
    _write_index(tmp_path, chunk_ids=ids, doc_ids=["D"] * 3, pages=[1, 2, 3])
    block = _sparse_block(cfg, ids, fingerprint="deadbeef")  # not the real corpus hash
    _write_manifest(tmp_path, sparse_block=block, revision=cfg.embedding.revision)

    with pytest.raises(SystemExit, match="run make fixture"):
        FixtureSparseRetriever.from_config(cfg, fixtures_dir=tmp_path, golden=[])


# --- depth guard ------------------------------------------------------------


def test_depth_guard_fires_when_top_k_exceeds_stored_depth(tmp_path):
    cfg = load_config()
    ids = ["000000000000000a", "000000000000000b", "000000000000000c"]
    _write_index(tmp_path, chunk_ids=ids, doc_ids=["D"] * 3, pages=[10, 20, 30])
    _write_manifest(
        tmp_path,
        sparse_block=_sparse_block(cfg, ids, stored_depth=2),
        revision=cfg.embedding.revision,
    )
    _write_rankings(tmp_path, {"q1": [(ids[2], 5.0), (ids[0], 2.0)]})
    golden = [{"qid": "q1", "question": "Q?", "evidence_pages": [30]}]

    retriever = FixtureSparseRetriever.from_config(cfg, fixtures_dir=tmp_path, golden=golden)
    with pytest.raises(SystemExit, match="exceeds the stored sparse depth"):
        retriever.retrieve("Q?", top_k=3)


# --- valid lookup -----------------------------------------------------------


def test_returns_valid_retrieved_chunks_by_qid(tmp_path):
    cfg = load_config()
    ids = ["000000000000000a", "000000000000000b", "000000000000000c"]
    _write_index(tmp_path, chunk_ids=ids, doc_ids=["DOC"] * 3, pages=[10, 20, 30])
    _write_manifest(
        tmp_path,
        sparse_block=_sparse_block(cfg, ids, stored_depth=2),
        revision=cfg.embedding.revision,
    )
    # Stored best-first: page 20 (score 3) then page 10 (score 1).
    _write_rankings(tmp_path, {"q1": [(ids[1], 3.0), (ids[0], 1.0)]})
    golden = [{"qid": "q1", "question": "Q?", "evidence_pages": [20]}]

    retriever = FixtureSparseRetriever.from_config(cfg, fixtures_dir=tmp_path, golden=golden)
    results = retriever.retrieve("Q?", top_k=2)

    assert [rc.chunk.page_num for rc in results] == [20, 10]
    assert [rc.chunk.chunk_id for rc in results] == [ids[1], ids[0]]
    assert [rc.score for rc in results] == [3.0, 1.0]
    assert all(rc.chunk.doc_id == "DOC" for rc in results)


# --- the offline hybrid path (what runs in CI) ------------------------------


def test_offline_hybrid_runs_end_to_end_with_fake_fuse(tmp_path):
    from ledgion.eval.tier1 import run_tier1

    cfg = load_config()
    ids = ["000000000000000a", "000000000000000b", "000000000000000c"]
    # Orthonormal vectors: chunk a/b/c -> pages 10/20/30.
    _write_index(
        tmp_path,
        chunk_ids=ids,
        doc_ids=["DOC"] * 3,
        pages=[10, 20, 30],
        vectors=np.eye(3),
    )
    # Query equals chunk b's vector, so dense surfaces page 20 first.
    np.savez(tmp_path / "query_vectors.npz", q1=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    _write_manifest(
        tmp_path,
        sparse_block=_sparse_block(cfg, ids, stored_depth=3),
        revision=cfg.embedding.revision,
    )
    # Sparse surfaces the evidence page 30 (which dense may bury behind a tie). Depth 3
    # == the whole 3-chunk corpus, so the top_k=3 budget cap drops nothing here (the
    # cap itself is exercised in test_fusion.test_hybrid_returns_exactly_top_k_chunks).
    _write_rankings(tmp_path, {"q1": [(ids[2], 5.0), (ids[0], 2.0), (ids[1], 1.0)]})
    golden = [
        {
            "qid": "q1",
            "question": "Q?",
            "doc_id": "DOC",
            "answer_type": "numeric",
            "evidence_pages": [30],
        }
    ]

    dense = FixtureRetriever.from_config(cfg, fixtures_dir=tmp_path, golden=golden)
    sparse = FixtureSparseRetriever.from_config(cfg, fixtures_dir=tmp_path, golden=golden)

    def fake_fuse(dense_ranked, sparse_ranked, k):
        # Stand-in for the user's RRF: dense first, then sparse extras, deduped by id.
        seen: set[str] = set()
        merged = []
        for rc in list(dense_ranked) + list(sparse_ranked):
            if rc.chunk.chunk_id not in seen:
                seen.add(rc.chunk.chunk_id)
                merged.append(rc)
        return merged

    hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=cfg.fusion.rrf_k, fuse=fake_fuse)

    # The fused ranking draws on both arms: dense's page 20 and sparse's page 30.
    pages = {rc.chunk.page_num for rc in hybrid.retrieve("Q?", top_k=3)}
    assert 20 in pages and 30 in pages

    result = run_tier1(
        golden,
        hybrid,
        top_k=3,
        metric_specs=["recall@k", "recall@10", "ndcg@10", "mrr", "hit@1"],
        default_k=3,
    )
    assert result["metrics"]["overall"]  # scoring ran over the fused ranking
    # Evidence page 30 came from the sparse arm; recall@10 sees it.
    assert result["results"][0]["metrics"]["recall@10"] == 1.0
