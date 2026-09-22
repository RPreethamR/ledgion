"""HybridRetriever scaffolding — candidate gathering + budget cap around the fusion.

The fusion maths (``reciprocal_rank_fusion``) is the user's; these tests exercise the
scaffolding around it: that the hybrid retriever pulls top_k from both arms and hands
both rankings (plus k and the arm weights) to the fusion, that the fused result is
capped to the top_k candidate budget, and — the pre-registered property — that a small
enough sparse_weight leaves dense's candidate set intact. The candidate-gathering and
cap tests inject a fake fuse; the property test uses the real reciprocal_rank_fusion.
"""

from __future__ import annotations

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

    def spy_fuse(dense_ranked, sparse_ranked, k, dense_weight=1.0, sparse_weight=1.0):
        seen["dense"] = list(dense_ranked)
        seen["sparse"] = list(sparse_ranked)
        seen["k"] = k
        seen["weights"] = (dense_weight, sparse_weight)
        return list(dense_ranked) + list(sparse_ranked)

    hybrid = HybridRetriever(
        dense=dense, sparse=sparse, rrf_k=60, dense_weight=1.0, sparse_weight=0.5, fuse=spy_fuse
    )
    out = hybrid.retrieve("q", top_k=3)

    # Both arms were queried at the requested depth…
    assert dense.asked_top_k == 3
    assert sparse.asked_top_k == 3
    # …and both rankings, k, AND the configured arm weights reached the fusion.
    assert [rc.chunk.page_num for rc in seen["dense"]] == [10, 20, 30]
    assert [rc.chunk.page_num for rc in seen["sparse"]] == [30, 40, 50]
    assert seen["k"] == 60
    assert seen["weights"] == (1.0, 0.5)
    # The retriever returns the fused ranking capped at the top_k budget.
    assert out == (seen["dense"] + seen["sparse"])[:3]


def test_hybrid_returns_exactly_top_k_chunks():
    # Disjoint arms -> the fused union is 2*top_k before truncation. Hybrid must emit
    # exactly top_k, the SAME candidate budget as dense, so a recall gain can't be an
    # artefact of a pool twice dense's size.
    dense = _FakeRetriever(_ranked([(i, 100 + i) for i in range(50)]))
    sparse = _FakeRetriever(_ranked([(1000 + i, 200 + i) for i in range(50)]))

    def union_fuse(dense_ranked, sparse_ranked, k, dense_weight=1.0, sparse_weight=1.0):
        # No dedup needed (disjoint); returns the full 100-chunk union, best-first.
        return list(dense_ranked) + list(sparse_ranked)

    hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=60, fuse=union_fuse)
    out = hybrid.retrieve("q", top_k=50)

    assert len(out) == 50  # exactly the budget, not the 100-chunk union
    assert len(out) == len(dense.retrieve("q", top_k=50))  # same budget as dense
    # it kept the best-first prefix of the fused ranking (here, dense's 50)
    assert out == (dense.retrieve("q", top_k=50) + sparse.retrieve("q", top_k=50))[:50]


def test_default_fuse_is_the_real_fusion_and_runs():
    # The default fuse is the implemented reciprocal_rank_fusion; the hybrid path runs
    # end to end and returns a budget-capped fused list.
    dense = _FakeRetriever(_ranked([(1, 10), (2, 20)]))
    sparse = _FakeRetriever(_ranked([(2, 20), (3, 30)]))
    hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=60)

    assert hybrid.fuse is reciprocal_rank_fusion
    out = hybrid.retrieve("q", top_k=2)
    assert len(out) == 2
    assert all(isinstance(rc, RetrievedChunk) for rc in out)
    # chunk 2 is in both arms, so it fuses highest and leads.
    assert out[0].chunk.chunk_id == f"{2:016x}"


def test_low_sparse_weight_preserves_dense_candidate_set():
    # The pre-registered property, at unit scale, through the REAL fusion + budget cap:
    # with dense_weight 1.0 and sparse_weight 0.5, no sparse-only chunk can outscore any
    # dense chunk (0.5/(60+1) < 1/(60+50)), so the top-K fused SET equals dense's — which
    # is exactly why recall@K (a set-at-depth-K metric) is unchanged by these weights.
    k = 50
    dense = _FakeRetriever(_ranked([(i, 100 + i) for i in range(k)]))
    sparse = _FakeRetriever(_ranked([(1000 + i, 900 + i) for i in range(k)]))  # disjoint
    hybrid = HybridRetriever(
        dense=dense, sparse=sparse, rrf_k=60, dense_weight=1.0, sparse_weight=0.5
    )

    out = hybrid.retrieve("q", top_k=k)
    dense_ids = {rc.chunk.chunk_id for rc in dense.retrieve("q", top_k=k)}
    assert {rc.chunk.chunk_id for rc in out} == dense_ids  # sparse-only never displaces
