"""The fixture-backed retriever: offline guards, an offline end-to-end query, and
a live-parity check.

Split by what they need:

* **Offline, always run** — the two startup guards (revision mismatch, missing
  golden qid) and a full retrieve against a *fabricated* fixture set. These build
  an in-process Qdrant from tiny hand-made ``.npz`` files and never load a model
  or touch the network, so they run on the torch-free CI env.
* **Live parity, auto-skipped** — the committed fixtures must reproduce the *real*
  retriever's page rankings exactly. This needs the bge model, the docker Qdrant,
  and generated fixtures, so it skips cleanly when any is absent (e.g. on CI).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ledgion.config import load_config
from ledgion.retrieve.fixture import FixtureRetriever

# A 16-hex chunk_id is required (QdrantIndexer derives the uint64 point id from it).
_CIDS = ["000000000000000a", "000000000000000b", "000000000000000c"]


def _write_fixture_set(
    fixtures_dir, *, revision, index=None, query_vectors=None
) -> None:
    """Write a minimal manifest + (optional) index.npz + query_vectors.npz."""
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    (fixtures_dir / "manifest.json").write_text(
        json.dumps({"embedding": {"model_id": "toy", "revision": revision}}),
        encoding="utf-8",
    )
    if index is not None:
        np.savez(fixtures_dir / "index.npz", **index)
    if query_vectors is not None:
        np.savez(fixtures_dir / "query_vectors.npz", **query_vectors)


# --- offline: startup guards ------------------------------------------------


def test_revision_mismatch_fires_with_run_make_fixture(tmp_path):
    cfg = load_config()
    _write_fixture_set(tmp_path, revision="stale-revision-sha")
    with pytest.raises(SystemExit, match="run make fixture"):
        FixtureRetriever.from_config(cfg, fixtures_dir=tmp_path, golden=[])


def test_missing_golden_qid_fires_with_run_make_fixture(tmp_path):
    cfg = load_config()
    # Manifest revision matches, but the query vectors lack the golden qid.
    _write_fixture_set(
        tmp_path,
        revision=cfg.embedding.revision,
        query_vectors={"present_qid": np.ones(4, dtype=np.float32)},
    )
    golden = [{"qid": "absent_qid", "question": "Q?", "evidence_pages": [1]}]
    with pytest.raises(SystemExit, match="run make fixture"):
        FixtureRetriever.from_config(cfg, fixtures_dir=tmp_path, golden=golden)


# --- offline: full retrieve against a fabricated fixture set ----------------


def test_retrieve_ranks_by_similarity_no_model(tmp_path):
    cfg = load_config()
    # Three orthogonal unit vectors on pages 10/20/30; the query equals page 20's,
    # so page 20 must rank first — all without embedding anything.
    index = {
        "vectors": np.eye(3, dtype=np.float32),
        "chunk_id": np.array(_CIDS),
        "doc_id": np.array(["DOC", "DOC", "DOC"]),
        "page_num": np.array([10, 20, 30], dtype=np.int64),
    }
    query_vectors = {"q1": np.array([0.0, 1.0, 0.0], dtype=np.float32)}
    _write_fixture_set(
        tmp_path, revision=cfg.embedding.revision, index=index, query_vectors=query_vectors
    )
    golden = [{"qid": "q1", "question": "which page?", "evidence_pages": [20]}]

    retriever = FixtureRetriever.from_config(cfg, fixtures_dir=tmp_path, golden=golden)
    results = retriever.retrieve("which page?", top_k=3)

    assert [rc.chunk.page_num for rc in results][0] == 20
    assert {rc.chunk.page_num for rc in results} == {10, 20, 30}


# --- live parity: committed fixtures reproduce the real rankings ------------


def _bge_cached(cfg) -> bool:
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(
        cfg.embedding.model_id, "model.safetensors", revision=cfg.embedding.revision
    )
    return isinstance(path, str)


def test_fixture_and_live_return_identical_rankings():
    from ledgion.eval.tier1 import collapse_to_pages

    pytest.importorskip("sentence_transformers")  # torch-free CI: skip live parity
    pytest.importorskip("huggingface_hub")

    from ledgion.retrieve.dense import DenseRetriever
    from ledgion.retrieve.fixture import FIXTURES_DIR

    cfg = load_config()  # docker Qdrant + dense backend
    if not (FIXTURES_DIR / "index.npz").exists():
        pytest.skip("fixtures not generated; run `make fixture`")
    if not _bge_cached(cfg):
        pytest.skip("bge-base not in local HF cache; skipping to stay offline")

    from ledgion.eval.runner import load_golden

    golden = load_golden()
    try:
        live = DenseRetriever.from_config(cfg)
        fixture = FixtureRetriever.from_config(cfg)
        # Probe the live server once; skip (don't fail) if it isn't up.
        probe = golden[0]["question"]
        live_first = collapse_to_pages(live.retrieve(probe, top_k=cfg.retrieval.top_k))
    except Exception as exc:  # noqa: BLE001 — any connection/setup failure => skip
        pytest.skip(f"live retrieval unavailable ({exc}); skipping parity check")

    assert live_first == collapse_to_pages(fixture.retrieve(probe, top_k=cfg.retrieval.top_k))
    for row in golden:
        live_pages = collapse_to_pages(live.retrieve(row["question"], top_k=cfg.retrieval.top_k))
        fix_pages = collapse_to_pages(fixture.retrieve(row["question"], top_k=cfg.retrieval.top_k))
        assert fix_pages == live_pages, f"ranking diverged for {row['qid']}"
