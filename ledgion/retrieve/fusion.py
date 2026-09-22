"""Hybrid retrieval — combine the dense and sparse arms into one ranking.

The ``HybridRetriever`` pulls ``retrieval.top_k`` candidates from each of the dense
(embedding) and sparse (BM25) retrievers and hands both ranked lists to a fusion
function, which merges them into a single ranking. The default fusion is Reciprocal
Rank Fusion (RRF), smoothed by ``fusion.rrf_k``.

Division of labour (see CLAUDE.md working agreement): **candidate gathering, config
wiring, and the Retriever interface here are Claude's; the fusion maths in
``reciprocal_rank_fusion`` is the user's** — it is the number that has to be
explained in an interview, so it is implemented explicitly here. To keep the
surrounding scaffolding independently testable, ``HybridRetriever`` takes the fusion
as an injectable ``fuse`` callable (defaulting to ``reciprocal_rank_fusion``):
tests can pass a fake fuse to verify candidate gathering separately from the real
fusion maths.

The fused ranking is truncated back to ``retrieval.top_k`` chunks, so hybrid is
scored over the **same candidate budget as dense** (top_k in from each arm, top_k
out). Otherwise fusing two top_k lists would hand up to ``2 * top_k`` candidates
downstream, and hybrid's recall would rise simply because it draws from a pool twice
dense's size — a bigger pool, not better retrieval. The fusion function returns the
full fused union; ``HybridRetriever`` applies the cap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from ledgion.config import Settings
from ledgion.interfaces import RetrievedChunk, Retriever


def reciprocal_rank_fusion(
    dense_ranked: Sequence[RetrievedChunk],
    sparse_ranked: Sequence[RetrievedChunk],
    k: int,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[RetrievedChunk]:
    """Fuse two rankings into one by weighted Reciprocal Rank Fusion.

    This is the Phase 6 fusion logic the user authors (Claude scaffolds around it).

    Contract to implement against:

    * ``dense_ranked`` and ``sparse_ranked`` are each a ranked list of
      ``RetrievedChunk``, **best-first** — list position ``i`` means rank ``i + 1``.
    * A chunk's identity across the two lists is ``rc.chunk.chunk_id`` (the same
      chunk may appear in both, at different ranks).
    * ``k`` is the RRF smoothing constant (``fusion.rrf_k``, default 60).
    * ``dense_weight`` and ``sparse_weight`` control how strongly each retrieval
      arm contributes. With both set to ``1.0`` this is standard RRF.

    Weighted RRF assigns each chunk the score
    ``sum(weight / (k + rank))`` over the lists it appears in (rank 1-based).
    Return the **full** fused ranking — every distinct chunk from either list,
    **best-first** (highest fused score), each appearing once, with ``score`` set
    to its fused RRF score. Do **not** truncate here: ``HybridRetriever`` caps the
    result at the ``top_k`` candidate budget, so this function returns the whole
    union. Break ties deterministically by ``chunk_id`` so the ranking — and
    therefore ``results/<hash>.json`` — is reproducible.

    With both weights at their defaults of ``1.0`` this is standard equal-weight
    RRF. Lowering ``sparse_weight`` allows sparse retrieval to influence ordering
    without necessarily allowing sparse-only candidates to displace dense
    candidates from the final candidate budget.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}

    # Each arm contributes at most once per chunk. If an upstream retriever ever
    # returns a duplicate chunk, keep its first occurrence because that is its
    # best rank in the already-best-first ranking.
    seen_dense: set[str] = set()

    for rank, rc in enumerate(dense_ranked, start=1):
        chunk_id = rc.chunk.chunk_id

        if chunk_id in seen_dense:
            continue

        seen_dense.add(chunk_id)
        chunks.setdefault(chunk_id, rc)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (
            dense_weight / (k + rank)
        )

    seen_sparse: set[str] = set()

    for rank, rc in enumerate(sparse_ranked, start=1):
        chunk_id = rc.chunk.chunk_id

        if chunk_id in seen_sparse:
            continue

        seen_sparse.add(chunk_id)
        chunks.setdefault(chunk_id, rc)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (
            sparse_weight / (k + rank)
        )

    # Highest RRF score first. chunk_id ascending gives deterministic ordering
    # when two chunks receive exactly the same fused score.
    ranked_chunk_ids = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], chunk_id),
    )

    return [
        replace(chunks[chunk_id], score=scores[chunk_id])
        for chunk_id in ranked_chunk_ids
    ]


# Type of a fusion function:
# two ranked lists + smoothing constant + arm weights -> one ranking.
FuseFn = Callable[
    [
        Sequence[RetrievedChunk],
        Sequence[RetrievedChunk],
        int,
        float,
        float,
    ],
    list[RetrievedChunk],
]


class HybridRetriever:
    """A ``Retriever`` (see interfaces.py) that fuses a dense and a sparse arm."""

    def __init__(
        self,
        *,
        dense: Retriever,
        sparse: Retriever,
        rrf_k: int,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        fuse: FuseFn = reciprocal_rank_fusion,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Injected so candidate gathering remains independently testable with a
        # fake fuse; defaults to the user's reciprocal_rank_fusion function.
        self.fuse = fuse

    @classmethod
    def from_config(cls, cfg: Settings) -> HybridRetriever:
        """Build the live hybrid retriever: real dense + real sparse arms, fused by
        ``reciprocal_rank_fusion`` with the configured RRF parameters."""
        from ledgion.retrieve.dense import DenseRetriever
        from ledgion.retrieve.sparse import SparseRetriever

        return cls(
            dense=DenseRetriever.from_config(cfg),
            sparse=SparseRetriever.from_config(cfg),
            rrf_k=cfg.fusion.rrf_k,
            dense_weight=cfg.fusion.dense_weight,
            sparse_weight=cfg.fusion.sparse_weight,
        )

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """Gather top_k from each arm, fuse, and return the top_k fused chunks.

        The fused list is truncated back to ``top_k`` so hybrid is scored over the
        **same candidate budget as dense** — top_k in from each arm, top_k out.
        Without this, fusing two top_k lists would feed up to ``2 * top_k`` candidates
        downstream, and any recall "gain" would just be a bigger pool, not better
        retrieval. (Truncation is the retriever's job, not the fusion maths': the
        fuse function returns the full fused union, best-first, and this caps it.)
        """
        dense_ranked = self.dense.retrieve(query, top_k=top_k)
        sparse_ranked = self.sparse.retrieve(query, top_k=top_k)

        fused = self.fuse(
            dense_ranked,
            sparse_ranked,
            self.rrf_k,
            self.dense_weight,
            self.sparse_weight,
        )

        return fused[:top_k]