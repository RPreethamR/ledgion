"""Hybrid retrieval — combine the dense and sparse arms into one ranking.

The ``HybridRetriever`` pulls ``retrieval.top_k`` candidates from each of the dense
(embedding) and sparse (BM25) retrievers and hands both ranked lists to a fusion
function, which merges them into a single ranking. The default fusion is Reciprocal
Rank Fusion (RRF), smoothed by ``fusion.rrf_k``.

Division of labour (see CLAUDE.md working agreement): **candidate gathering, config
wiring, and the Retriever interface here are Claude's; the fusion maths in
``reciprocal_rank_fusion`` is the user's** — it is the number that has to be
explained in an interview, so it is left as a documented stub. To keep the
surrounding scaffolding testable before that stub is filled in, ``HybridRetriever``
takes the fusion as an injectable ``fuse`` callable (defaulting to
``reciprocal_rank_fusion``): tests pass a fake fuse to verify gathering, and a
separate test pins that the real stub raises ``NotImplementedError``.

The fused ranking is truncated back to ``retrieval.top_k`` chunks, so hybrid is
scored over the **same candidate budget as dense** (top_k in from each arm, top_k
out). Otherwise fusing two top_k lists would hand up to ``2 * top_k`` candidates
downstream, and hybrid's recall would rise simply because it draws from a pool twice
dense's size — a bigger pool, not better retrieval. The fusion function returns the
full fused union; ``HybridRetriever`` applies the cap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ledgion.config import Settings
from ledgion.interfaces import RetrievedChunk, Retriever


def reciprocal_rank_fusion(
    dense_ranked: Sequence[RetrievedChunk],
    sparse_ranked: Sequence[RetrievedChunk],
    k: int,
) -> list[RetrievedChunk]:
    """Fuse two rankings into one by Reciprocal Rank Fusion.  **[USER WRITES THIS]**

    This is the Phase 6 fusion logic the user authors (Claude scaffolds around it).
    It is intentionally unimplemented; ``HybridRetriever`` accepts an injectable
    ``fuse`` so the rest of the pipeline is exercisable before this lands.

    Contract to implement against:

    * ``dense_ranked`` and ``sparse_ranked`` are each a ranked list of
      ``RetrievedChunk``, **best-first** — list position ``i`` means rank ``i + 1``.
    * A chunk's identity across the two lists is ``rc.chunk.chunk_id`` (the same
      chunk may appear in both, at different ranks).
    * ``k`` is the RRF smoothing constant (``fusion.rrf_k``, default 60).

    RRF assigns each chunk the score ``sum(1 / (k + rank))`` over the lists it
    appears in (rank 1-based). Return the **full** fused ranking — every distinct
    chunk from either list, **best-first** (highest fused score), each appearing once,
    with ``score`` set to its fused RRF score. Do **not** truncate here:
    ``HybridRetriever`` caps the result at the ``top_k`` candidate budget, so this
    function returns the whole union. Break ties deterministically (e.g. by chunk_id)
    so the ranking — and therefore ``results/<hash>.json`` — is reproducible.

    (``fusion.dense_weight`` / ``fusion.sparse_weight`` exist in config for a later
    weighted-RRF ablation; plain RRF ignores them.)
    """
    raise NotImplementedError(
        "reciprocal_rank_fusion is the user's to implement (Phase 6 fusion logic). "
        "See this function's docstring for the RRF contract."
    )


# Type of a fusion function: two ranked lists + smoothing constant -> one ranking.
FuseFn = Callable[[Sequence[RetrievedChunk], Sequence[RetrievedChunk], int], list[RetrievedChunk]]


class HybridRetriever:
    """A ``Retriever`` (see interfaces.py) that fuses a dense and a sparse arm."""

    def __init__(
        self,
        *,
        dense: Retriever,
        sparse: Retriever,
        rrf_k: int,
        fuse: FuseFn = reciprocal_rank_fusion,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        # Injected so candidate gathering is testable with a fake fuse before the
        # real reciprocal_rank_fusion is written; defaults to the user's function.
        self.fuse = fuse

    @classmethod
    def from_config(cls, cfg: Settings) -> HybridRetriever:
        """Build the live hybrid retriever: real dense + real sparse arms, fused by
        ``reciprocal_rank_fusion`` with ``fusion.rrf_k``."""
        from ledgion.retrieve.dense import DenseRetriever
        from ledgion.retrieve.sparse import SparseRetriever

        return cls(
            dense=DenseRetriever.from_config(cfg),
            sparse=SparseRetriever.from_config(cfg),
            rrf_k=cfg.fusion.rrf_k,
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
        fused = self.fuse(dense_ranked, sparse_ranked, self.rrf_k)
        return fused[:top_k]
