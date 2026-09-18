"""Qdrant indexing.

Upserts ``(Chunk, vector)`` pairs into a Qdrant collection. Two forward-looking
choices matter here:

* **Named vectors.** The collection stores a vector under the name ``"dense"``
  rather than as the anonymous default. A collection created with a single
  unnamed vector cannot gain a second vector later without being recreated;
  naming it now means Phase 6 can add a ``"sparse"`` BM25 vector to the same
  points with no migration and no re-embed.
* **Filterable payload fields.** ``doc_id``, ``page_num`` and ``ticker`` are
  given payload indexes, so metadata-filtered retrieval (e.g. "only this
  filing", a Phase 6 ablation) is a query-time filter, not a re-index.

The full chunk metadata is stored in the payload — including the chunk text,
which the generator needs to build grounded context. Point ids are derived
deterministically from ``chunk_id`` so re-running ingest overwrites points in
place (idempotent) instead of duplicating them.
"""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from ledgion.interfaces import Chunk

DENSE_VECTOR = "dense"


def _point_id(chunk_id: str) -> int:
    """Map a 16-hex-char ``chunk_id`` to a stable uint64 Qdrant point id.

    16 hex digits are exactly 64 bits, so this is a lossless, deterministic id:
    the same chunk always lands on the same point, making upserts idempotent.
    """
    return int(chunk_id, 16)


class QdrantIndexer:
    """Creates the collection on first use and upserts dense-vector points."""

    def __init__(
        self,
        *,
        mode: str,
        url: str,
        path: str,
        collection_name: str,
        prefer_grpc: bool = False,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        self.collection_name = collection_name
        self.distance = distance
        if mode == "memory":
            # In-process client for CI / offline eval — no server, no network.
            self.client = QdrantClient(location=":memory:")
        elif mode == "docker":
            self.client = QdrantClient(url=url, prefer_grpc=prefer_grpc)
        else:
            raise ValueError(f"unknown qdrant mode: {mode!r}")

    @classmethod
    def from_config(cls, cfg) -> QdrantIndexer:
        return cls(
            mode=cfg.qdrant.mode,
            url=cfg.qdrant.url,
            path=str(cfg.qdrant.path),
            collection_name=cfg.qdrant.collection_name,
            prefer_grpc=cfg.qdrant.prefer_grpc,
            # Normalised embeddings -> cosine is equivalent to dot but keeps the
            # collection self-describing if a non-normalised embedder is swapped in.
            distance=models.Distance.COSINE,
        )

    def ensure_collection(self, dim: int) -> None:
        """Create the collection (named ``dense`` vector) and its payload indexes
        if they don't already exist. Idempotent."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(size=dim, distance=self.distance)
                },
            )
            # Filterable fields for Phase-6 metadata filtering (no re-index later).
            self.client.create_payload_index(
                self.collection_name, "doc_id", models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                self.collection_name, "page_num", models.PayloadSchemaType.INTEGER
            )
            self.client.create_payload_index(
                self.collection_name, "ticker", models.PayloadSchemaType.KEYWORD
            )

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, batch_size: int = 256
    ) -> int:
        """Upsert chunks with their dense vectors. Returns the number upserted."""
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks/vectors length mismatch: {len(chunks)} != {len(vectors)}")
        if not chunks:
            return 0
        self.ensure_collection(len(vectors[0]))

        points = [
            models.PointStruct(
                id=_point_id(c.chunk_id),
                vector={DENSE_VECTOR: _to_list(v)},
                payload={
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "page_num": c.page_num,
                    "text": c.text,
                    "company": c.company,
                    "ticker": c.ticker,
                    "fiscal_year": c.fiscal_year,
                    "form_type": c.form_type,
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(self.collection_name, points=points[start : start + batch_size])
        return len(points)

    def count(self) -> int:
        """Number of points currently in the collection."""
        return self.client.count(self.collection_name).count


def _to_list(vec: Sequence[float]) -> list[float]:
    """Qdrant wants a plain list; accept numpy arrays or sequences."""
    tolist = getattr(vec, "tolist", None)
    return tolist() if callable(tolist) else list(vec)
