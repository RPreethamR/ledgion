"""HybridRetriever scaffolding — candidate gathering around the user's fusion stub.

The fusion maths (``reciprocal_rank_fusion``) is the user's to write and ships as a
``NotImplementedError`` stub. These tests exercise everything *around* it: that the
hybrid retriever pulls top_k from both arms and hands both rankings to the fusion,
that an injected fuse lets the wiring run before the stub is filled in, and that the
stub itself raises — so leaving it unimplemented never silently no-ops.
"""

from __future__ import annotations

import pytest

from ledgion.interfaces import Chunk, RetrievedChunk
from ledgion.retrieve.fusion import HybridRetriever, reciprocal_rank_fusion


def _chunk(cid: int, page: int) -> Chunk:
    return Chunk(
        chunk_id=f"{cid:016x}",
        doc_id="DOC",
        page_num=page,
        text="",
        company="",
        ticker="",
        fiscal_year=0,
        form_type="",
    )


def _ranked(pairs: list[tuple[int, int]]) -> list[RetrievedChunk]:
    # (chunk_id, page) pairs -> a best-first ranking with strictly decreasing scores.
    return [
        RetrievedChunk(chunk=_chunk(cid, page), score=1.0 - 0.1 * i)
        for i, (cid, page) in enumerate(pairs)
    ]


class _FakeRetriever:
    """Returns a fixed ranking and records the top_k it was asked for."""

    def __init__(self, ranking: list[RetrievedChunk]) -> None:
        self._ranking = ranking
        self.asked_top_k: int | None = None

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        self.asked_top_k = top_k
        return self._ranking[:top_k]


def test_hybrid_pulls_from_both_sources():
    dense = _FakeRetriever(_ranked([(1, 10), (2, 20), (3, 30)]))
    sparse = _FakeRetriever(_ranked([(3, 30), (4, 40), (5, 50)]))

    seen: dict = {}

    def spy_fuse(dense_ranked, sparse_ranked, k):
        seen["dense"] = list(dense_ranked)
        seen["sparse"] = list(sparse_ranked)
        seen["k"] = k
        return list(dense_ranked) + list(sparse_ranked)

    hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=60, fuse=spy_fuse)
    out = hybrid.retrieve("q", top_k=3)

    # Both arms were queried at the requested depth…
    assert dense.asked_top_k == 3
    assert sparse.asked_top_k == 3
    # …and both rankings reached the fusion, with the configured k.
    assert [rc.chunk.page_num for rc in seen["dense"]] == [10, 20, 30]
    assert [rc.chunk.page_num for rc in seen["sparse"]] == [30, 40, 50]
    assert seen["k"] == 60
    # The retriever returns the fused ranking capped at the top_k budget.
    assert out == (seen["dense"] + seen["sparse"])[:3]


def test_hybrid_returns_exactly_top_k_chunks():
    # Disjoint arms -> the fused union is 2*top_k before truncation. Hybrid must emit
    # exactly top_k, the SAME candidate budget as dense, so a recall gain can't be an
    # artefact of a pool twice dense's size.
    dense = _FakeRetriever(_ranked([(i, 100 + i) for i in range(50)]))
    sparse = _FakeRetriever(_ranked([(1000 + i, 200 + i) for i in range(50)]))

    def union_fuse(dense_ranked, sparse_ranked, k):
        # No dedup needed (disjoint); returns the full 100-chunk union, best-first.
        return list(dense_ranked) + list(sparse_ranked)

    hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=60, fuse=union_fuse)
    out = hybrid.retrieve("q", top_k=50)

    assert len(out) == 50  # exactly the budget, not the 100-chunk union
    assert len(out) == len(dense.retrieve("q", top_k=50))  # same budget as dense
    # it kept the best-first prefix of the fused ranking (here, dense's 50)
    assert out == (dense.retrieve("q", top_k=50) + sparse.retrieve("q", top_k=50))[:50]


def test_stub_fusion_raises_not_implemented():
    # The user's function is intentionally unwritten; it must raise, not no-op.
    with pytest.raises(NotImplementedError):
        reciprocal_rank_fusion([], [], 60)


def test_scaffolding_survives_the_unimplemented_stub():
    # Constructing the hybrid with the default (stub) fusion must not raise — the
    # wiring is intact; only calling through the stub surfaces NotImplementedError.
    dense = _FakeRetriever(_ranked([(1, 10)]))
    sparse = _FakeRetriever(_ranked([(2, 20)]))
    hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=60)
    assert hybrid.fuse is reciprocal_rank_fusion
    with pytest.raises(NotImplementedError):
        hybrid.retrieve("q", top_k=1)
