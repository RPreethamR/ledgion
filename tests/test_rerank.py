"""Cross-encoder reranking — the reordering primitive, the revision guard, and the
``rerank`` composition, all offline.

No model is loaded here: ``rerank_by_scores`` is pure, and ``rerank`` is exercised by
stubbing ``score`` on the instance (so ``_ensure_model`` is never called). The live
model + fixture parity are covered in test_fixture_rerank.py, which auto-skips when
the ~80MB cross-encoder isn't cached. This keeps the reranker's ordering logic green
on the torch-free CI env.
"""

from __future__ import annotations

import pytest

from ledgion.interfaces import Chunk, RetrievedChunk
from ledgion.retrieve.rerank import CrossEncoderReranker, rerank_by_scores

_REV = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def _rc(idx: int, text: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"{idx:016x}",
            doc_id="AMD_2022_10K",
            page_num=idx + 1,
            text=text,
            company="AMD",
            ticker="AMD",
            fiscal_year=2022,
            form_type="10-K",
        ),
        score=score,
    )


# --- the pure reordering primitive ------------------------------------------


def test_rerank_by_scores_reorders_and_truncates():
    # Retrieval order (a, b, c) with the relevant chunk buried at rank 3; the
    # cross-encoder scores promote it to rank 1, and top_n trims to 2.
    candidates = [_rc(0, "filler one"), _rc(1, "filler two"), _rc(2, "the answer")]
    scores = [0.1, 0.2, 9.9]  # parallel to candidates

    out = rerank_by_scores(candidates, scores, top_n=2)

    assert [rc.chunk.chunk_id for rc in out] == ["0000000000000002", "0000000000000001"]
    # The reranker score replaces the upstream retrieval score.
    assert out[0].score == 9.9
    assert len(out) == 2


def test_rerank_by_scores_breaks_ties_by_chunk_id():
    # Equal scores must resolve deterministically by chunk_id ascending, not by input
    # order — so results/<hash>.json is reproducible regardless of float noise.
    candidates = [_rc(2, "c"), _rc(0, "a"), _rc(1, "b")]
    scores = [1.0, 1.0, 1.0]

    out = rerank_by_scores(candidates, scores, top_n=3)

    assert [rc.chunk.chunk_id for rc in out] == [
        "0000000000000000",
        "0000000000000001",
        "0000000000000002",
    ]


def test_rerank_by_scores_length_mismatch_raises():
    # A candidate scored against the wrong pool must fail loudly, not zip-truncate.
    with pytest.raises(ValueError, match="length mismatch"):
        rerank_by_scores([_rc(0, "a"), _rc(1, "b")], [1.0], top_n=2)


# --- the revision guard (mirrors the embedder) ------------------------------


def test_none_revision_refuses_to_construct():
    with pytest.raises(ValueError, match="reranker.revision is None"):
        CrossEncoderReranker(model_id="cross-encoder/x", revision=None)


# --- rerank() composition with a stubbed scorer (no model) ------------------


def test_rerank_promotes_obviously_relevant_chunk_first():
    reranker = CrossEncoderReranker(model_id="cross-encoder/x", revision=_REV, top_n=5)
    # Stub the model call: the chunk that actually mentions the query term scores high.
    reranker.score = lambda query, texts: [9.0 if "xilinx" in t else 0.0 for t in texts]

    candidates = [
        _rc(0, "net revenue increased"),
        _rc(1, "the xilinx acquisition closed"),
        _rc(2, "gross margin was flat"),
    ]
    out = reranker.rerank("when did the xilinx deal close?", candidates, top_n=5)

    assert out[0].chunk.chunk_id == "0000000000000001"
    assert out[0].chunk.text == "the xilinx acquisition closed"


def test_rerank_empty_pool_returns_empty():
    reranker = CrossEncoderReranker(model_id="cross-encoder/x", revision=_REV)
    # Must not touch the model for an empty candidate list.
    reranker.score = lambda query, texts: pytest.fail("score should not be called")
    assert reranker.rerank("q?", [], top_n=5) == []
