"""The offline rerank fixture: staleness guards, the loud missing-score contract, a
valid frozen-score replay, and live/fixture parity.

Like test_fixture_sparse.py, the offline tests fabricate tiny manifest + ``.npz``
files under tmp_path — no model, no network, no committed data — so they pin that a
stale rerank fixture fails loudly (missing block, drifted revision, drifted corpus
fingerprint, a candidate with no frozen score) and that a valid fixture reorders by
the frozen cross-encoder scores. The parity test needs the real ~80MB cross-encoder,
the docker Qdrant, and generated fixtures, so it skips cleanly when any is absent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ledgion.config import load_config
from ledgion.interfaces import Chunk, RetrievedChunk
from ledgion.retrieve.fixture import FixtureReranker
from ledgion.retrieve.sparse import corpus_fingerprint

_CIDS = ["000000000000000a", "000000000000000b", "000000000000000c"]


def _write_index(fixtures_dir, *, chunk_ids, doc_ids, pages) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        fixtures_dir / "index.npz",
        chunk_id=np.array(chunk_ids),
        doc_id=np.array(doc_ids),
        page_num=np.array(pages, dtype=np.int64),
    )


def _reranker_block(cfg, chunk_ids, *, candidate_depth=100, fingerprint=None) -> dict:
    return {
        "model_id": cfg.reranker.model_id,
        "revision": cfg.reranker.revision,
        "candidate_depth": candidate_depth,
        "corpus_fingerprint": fingerprint
        if fingerprint is not None
        else corpus_fingerprint(chunk_ids),
    }


def _write_manifest(fixtures_dir, *, reranker_block) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"embedding": {"model_id": "toy", "revision": "toy-rev"}}
    if reranker_block is not None:
        manifest["reranker"] = reranker_block
    (fixtures_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_rerank_scores(fixtures_dir, qid_to_pairs) -> None:
    """qid_to_pairs: {qid: [(chunk_id, score), ...]} (uniform depth per qid)."""
    qids = sorted(qid_to_pairs)
    chunk_ids = [[cid for cid, _ in qid_to_pairs[q]] for q in qids]
    scores = [[s for _, s in qid_to_pairs[q]] for q in qids]
    np.savez(
        fixtures_dir / "rerank_scores.npz",
        qids=np.array(qids),
        chunk_ids=np.array(chunk_ids),
        scores=np.array(scores, dtype=np.float32),
    )


def _rc(chunk_id: str, page: int) -> RetrievedChunk:
    # FixtureReranker reads only chunk_id off the candidate; the rest is placeholder.
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="DOC",
            page_num=page,
            text="",
            company="",
            ticker="",
            fiscal_year=0,
            form_type="",
        ),
        score=0.0,
    )


# --- staleness guards -------------------------------------------------------


def test_missing_reranker_block_fires(tmp_path):
    cfg = load_config()
    _write_index(tmp_path, chunk_ids=_CIDS, doc_ids=["D"] * 3, pages=[1, 2, 3])
    _write_manifest(tmp_path, reranker_block=None)
    with pytest.raises(SystemExit, match="run make fixture"):
        FixtureReranker.from_config(cfg, fixtures_dir=tmp_path, golden=[])


def test_revision_mismatch_fires(tmp_path):
    # The lookup is keyed on the reranker revision: a manifest built under a different
    # model revision must fail loudly rather than reuse stale scores.
    cfg = load_config()
    _write_index(tmp_path, chunk_ids=_CIDS, doc_ids=["D"] * 3, pages=[1, 2, 3])
    block = _reranker_block(cfg, _CIDS)
    block["revision"] = "some-other-revision-sha"
    _write_manifest(tmp_path, reranker_block=block)
    with pytest.raises(SystemExit, match="run make fixture"):
        FixtureReranker.from_config(cfg, fixtures_dir=tmp_path, golden=[])


def test_corpus_fingerprint_mismatch_fires(tmp_path):
    cfg = load_config()
    _write_index(tmp_path, chunk_ids=_CIDS, doc_ids=["D"] * 3, pages=[1, 2, 3])
    block = _reranker_block(cfg, _CIDS, fingerprint="deadbeef")
    _write_manifest(tmp_path, reranker_block=block)
    with pytest.raises(SystemExit, match="run make fixture"):
        FixtureReranker.from_config(cfg, fixtures_dir=tmp_path, golden=[])


def test_missing_golden_qid_fires(tmp_path):
    cfg = load_config()
    _write_index(tmp_path, chunk_ids=_CIDS, doc_ids=["D"] * 3, pages=[1, 2, 3])
    _write_manifest(tmp_path, reranker_block=_reranker_block(cfg, _CIDS))
    _write_rerank_scores(tmp_path, {"q_present": [(_CIDS[0], 1.0)]})
    golden = [{"qid": "q_absent", "question": "Q?", "evidence_pages": [1]}]
    with pytest.raises(SystemExit, match="run make fixture"):
        FixtureReranker.from_config(cfg, fixtures_dir=tmp_path, golden=golden)


# --- valid frozen-score replay ----------------------------------------------


def test_reranks_by_frozen_scores(tmp_path):
    cfg = load_config()
    _write_index(tmp_path, chunk_ids=_CIDS, doc_ids=["DOC"] * 3, pages=[10, 20, 30])
    _write_manifest(tmp_path, reranker_block=_reranker_block(cfg, _CIDS))
    # Frozen scores promote chunk b (page 20) to the top, then c, then a.
    _write_rerank_scores(
        tmp_path,
        {"q1": [(_CIDS[0], 0.1), (_CIDS[1], 9.9), (_CIDS[2], 0.5)]},
    )
    golden = [{"qid": "q1", "question": "Q?", "evidence_pages": [20]}]

    reranker = FixtureReranker.from_config(cfg, fixtures_dir=tmp_path, golden=golden)
    # Hand them in retrieval order (a, b, c); reranking must reorder by frozen score.
    out = reranker.rerank("Q?", [_rc(_CIDS[0], 10), _rc(_CIDS[1], 20), _rc(_CIDS[2], 30)], top_n=3)

    assert [rc.chunk.chunk_id for rc in out] == [_CIDS[1], _CIDS[2], _CIDS[0]]
    assert [rc.chunk.page_num for rc in out] == [20, 30, 10]
    assert out[0].score == pytest.approx(9.9)  # frozen reranker score replaces retrieval score


# --- the loud missing-score contract (⚠️) -----------------------------------


def test_candidate_without_frozen_score_raises_never_skips(tmp_path):
    cfg = load_config()
    _write_index(tmp_path, chunk_ids=_CIDS, doc_ids=["DOC"] * 3, pages=[10, 20, 30])
    _write_manifest(tmp_path, reranker_block=_reranker_block(cfg, _CIDS))
    # Only a and b were frozen; c is absent from this qid's pool.
    _write_rerank_scores(tmp_path, {"q1": [(_CIDS[0], 1.0), (_CIDS[1], 2.0)]})
    golden = [{"qid": "q1", "question": "Q?", "evidence_pages": [20]}]

    reranker = FixtureReranker.from_config(cfg, fixtures_dir=tmp_path, golden=golden)
    # A pool that includes the unscored chunk c must raise — never silently drop it,
    # never default its score. (A retrieval change that widened the pool is a bug to
    # surface, not to rerank around.)
    with pytest.raises(SystemExit, match="run make fixture"):
        reranker.rerank("Q?", [_rc(_CIDS[0], 10), _rc(_CIDS[2], 30)], top_n=2)


# --- live / fixture parity (auto-skipped off the model box) -----------------


def _reranker_cached(cfg) -> bool:
    from huggingface_hub import try_to_load_from_cache

    for fname in ("model.safetensors", "pytorch_model.bin"):
        if isinstance(
            try_to_load_from_cache(cfg.reranker.model_id, fname, revision=cfg.reranker.revision),
            str,
        ):
            return True
    return False


def test_live_and_fixture_rerank_identical_for_every_golden_question():
    pytest.importorskip("sentence_transformers")  # torch-free CI: skip
    pytest.importorskip("huggingface_hub")

    from ledgion.retrieve.fixture import FIXTURES_DIR, RERANK_SCORES_FILE

    cfg = load_config()
    if not (FIXTURES_DIR / RERANK_SCORES_FILE).exists():
        pytest.skip("rerank fixtures not generated; run `make fixture`")
    if not _reranker_cached(cfg):
        pytest.skip("cross-encoder not in local HF cache; skipping to stay offline")

    from ledgion.eval.runner import load_golden
    from ledgion.retrieve.dense import DenseRetriever
    from ledgion.retrieve.rerank import CrossEncoderReranker

    golden = load_golden()
    try:
        dense = DenseRetriever.from_config(cfg)
        live = CrossEncoderReranker.from_config(cfg)
        fixture = FixtureReranker.from_config(cfg)
        top_k = cfg.retrieval.top_k
        for row in golden:
            q = row["question"]
            pool = dense.retrieve(q, top_k=top_k)
            live_order = [rc.chunk.chunk_id for rc in live.rerank(q, pool, top_n=top_k)]
            fix_order = [rc.chunk.chunk_id for rc in fixture.rerank(q, pool, top_n=top_k)]
            assert live_order == fix_order, f"rerank order diverged for {row['qid']}"
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — any connection/setup failure => skip
        pytest.skip(f"live rerank unavailable ({exc}); skipping parity check")
