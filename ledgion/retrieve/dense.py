"""Dense (embedding-only) retrieval — the Phase 3 baseline.

No BM25, no fusion, no reranking: embed the query with the *same* pinned bge
model used at ingest, search the ``dense`` named vector in Qdrant by cosine, and
return the top_k points as ``RetrievedChunk``s. This is the floor every later
ablation (sparse, fusion, rerank, query rewriting) is measured against, so it
stays deliberately minimal.

**bge query prefix.** bge-base-en-v1.5's model card asks that *queries* (not
documents) be prefixed with an instruction — "Represent this sentence for
searching relevant passages:" — for retrieval; passages are embedded bare. That
asymmetry already lives in ``BGEEmbedder.embed_query`` (the prefix is a config
knob, ``embedding.query_instruction``, so "instruction vs none" stays a
measurable ablation). This retriever therefore just calls ``embed_query`` and
never re-applies the prefix itself — the query is encoded exactly the way the
corpus was, which is the whole point of using one embedder for both sides.
"""

from __future__ import annotations

from ledgion.config import Settings
from ledgion.ingest.embed import BGEEmbedder
from ledgion.ingest.index import DENSE_VECTOR
from ledgion.interfaces import Chunk, RetrievedChunk


class DenseRetriever:
    """A ``Retriever`` (see interfaces.py) over the dense Qdrant vector."""

    def __init__(
        self,
        *,
        embedder: BGEEmbedder,
        client,
        collection_name: str,
        vector_name: str = DENSE_VECTOR,
    ) -> None:
        self.embedder = embedder
        self.client = client
        self.collection_name = collection_name
        self.vector_name = vector_name

    @classmethod
    def from_config(cls, cfg: Settings) -> DenseRetriever:
        """Build from resolved ``Settings``, opening a Qdrant client.

        Mirrors ``QdrantIndexer``'s client construction. ``mode="memory"`` builds
        a fresh in-process store that is *empty* unless the ingest client is
        reused — the offline-eval harness injects that shared client through the
        constructor; a standalone ``ledgion ask`` runs against the docker server,
        where the collection already holds the corpus.
        """
        from qdrant_client import QdrantClient

        if cfg.qdrant.mode == "memory":
            client = QdrantClient(location=":memory:")
        elif cfg.qdrant.mode == "docker":
            client = QdrantClient(url=cfg.qdrant.url, prefer_grpc=cfg.qdrant.prefer_grpc)
        else:
            raise ValueError(f"unknown qdrant mode: {cfg.qdrant.mode!r}")

        return cls(
            embedder=BGEEmbedder.from_config(cfg),
            client=client,
            collection_name=cfg.qdrant.collection_name,
        )

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """Return the top_k chunks for ``query``, best match first.

        The returned list order *is* the rank: position 0 is the nearest
        neighbour. (Retrieval is scored downstream at page level, so what matters
        is the order and each chunk's ``page_num``, not the chunk's absolute
        score — see interfaces.py.)
        """
        vector = self.embedder.embed_query(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            using=self.vector_name,
            limit=top_k,
            with_payload=True,
        )
        return [self._to_retrieved(point) for point in response.points]

    @staticmethod
    def _to_retrieved(point) -> RetrievedChunk:
        """Rebuild a ``Chunk`` from the Qdrant payload (the full metadata that
        ``QdrantIndexer.upsert`` stored) and pair it with the point's score."""
        payload = point.payload or {}
        chunk = Chunk(
            chunk_id=payload["chunk_id"],
            doc_id=payload["doc_id"],
            page_num=payload["page_num"],
            text=payload["text"],
            company=payload["company"],
            ticker=payload["ticker"],
            fiscal_year=payload["fiscal_year"],
            form_type=payload["form_type"],
        )
        return RetrievedChunk(chunk=chunk, score=point.score)
