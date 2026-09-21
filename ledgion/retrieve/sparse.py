"""Sparse (BM25) retrieval — the Phase 6 lexical arm of hybrid search.

Where ``DenseRetriever`` embeds the query and searches cosine neighbours, this
scores the query against the corpus with Okapi BM25 over the chunk *text*. It is
the complement dense retrieval needs: exact tokens (tickers, GAAP line items,
fiscal years) that an embedder smooths over match here verbatim.

Two properties this file guarantees:

* **Built once, never per query.** ``rank_bm25`` precomputes IDF and document
  statistics at construction; a query is then a single ``get_scores`` pass. The
  index is persisted (pickled under ``<cache_dir>/sparse/``) so a second process
  reloads it instead of re-scrolling Qdrant and re-tokenising the whole corpus.
* **Tokenisation is explicit and recordable.** Lowercasing and the token regex
  are config knobs (``sparse.lowercase`` / ``sparse.token_pattern``), not baked-in
  constants, so "which tokeniser" is a measurable ablation and shows up in
  ``results/<hash>.json`` and the fixture manifest. The BM25 hyperparameters
  (``k1``/``b``/``epsilon``) are config too, for the same reason.

Retrieval is scored downstream at **page level** (see interfaces.py), so — exactly
like dense — what matters is the returned order and each chunk's ``page_num``, not
the absolute BM25 score. The returned list order *is* the rank (position 0 = best).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections.abc import Sequence
from pathlib import Path

from ledgion.config import Settings
from ledgion.interfaces import Chunk, RetrievedChunk

# Persisted-index format version. Bumped if the pickle layout changes so a stale
# cache from an older Ledgion is treated as a miss rather than mis-loaded.
_PICKLE_VERSION = 1


class BM25Tokenizer:
    """Turns text into BM25 tokens under an explicit, recordable policy.

    ``lowercase`` is applied first, then ``token_pattern`` (a regex) is matched
    globally — every match is one token. Kept as a small object (not a bare
    function) so its policy can be serialised into the manifest and compared by the
    fixture guard.
    """

    def __init__(self, *, lowercase: bool, token_pattern: str) -> None:
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self._re = re.compile(token_pattern)

    @classmethod
    def from_config(cls, cfg: Settings) -> BM25Tokenizer:
        return cls(lowercase=cfg.sparse.lowercase, token_pattern=cfg.sparse.token_pattern)

    def tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return self._re.findall(text)

    def config_dict(self) -> dict:
        """The policy, as a plain dict — for the manifest and the staleness guard."""
        return {"lowercase": self.lowercase, "token_pattern": self.token_pattern}


def tokenizer_config(cfg: Settings) -> dict:
    """The tokeniser policy the resolved config asks for (manifest + guard)."""
    return {"lowercase": cfg.sparse.lowercase, "token_pattern": cfg.sparse.token_pattern}


def bm25_params(cfg: Settings) -> dict:
    """The BM25 hyperparameters the resolved config asks for (manifest + guard)."""
    return {"k1": cfg.sparse.k1, "b": cfg.sparse.b, "epsilon": cfg.sparse.epsilon}


def corpus_fingerprint(chunk_ids: Sequence[str]) -> str:
    """A stable digest of the corpus identity: sha256 over the *sorted* chunk_ids.

    Page rankings are only comparable if dense and sparse were built over the same
    corpus. This fingerprint is written into the manifest at ``make fixture`` time
    and re-checked when the sparse/hybrid fixture path loads, so a corpus that
    changed under the fixtures fails loudly instead of scoring against stale ids.
    Sorting makes it independent of scroll/enumeration order.
    """
    h = hashlib.sha256()
    for cid in sorted(chunk_ids):
        h.update(cid.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class SparseRetriever:
    """A ``Retriever`` (see interfaces.py) backed by an Okapi BM25 index."""

    def __init__(
        self,
        *,
        chunks: Sequence[Chunk],
        tokenizer: BM25Tokenizer,
        k1: float,
        b: float,
        epsilon: float,
        corpus_fp: str | None = None,
    ) -> None:
        from rank_bm25 import BM25Okapi  # lazy: keep module import torch/-dep free

        self.chunks = list(chunks)
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        # Fingerprint of the corpus this index was built over; used to invalidate a
        # persisted index when the corpus changes underneath it.
        if corpus_fp is None:
            corpus_fp = corpus_fingerprint([c.chunk_id for c in self.chunks])
        self.corpus_fp = corpus_fp
        # Build once: IDF + doc stats are precomputed here, so retrieve() is a single
        # scoring pass and never rebuilds.
        tokenised = [self.tokenizer.tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenised, k1=k1, b=b, epsilon=epsilon)

    # -- Retriever contract --------------------------------------------------

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """Return the top_k chunks for ``query``, best (highest BM25) first.

        Ties are broken by chunk_id so the ranking is fully deterministic (rank_bm25
        preserves corpus order, but an explicit key keeps results stable regardless
        of the underlying sort). List position is the rank.
        """
        scores = self._bm25.get_scores(self.tokenizer.tokenize(query))
        order = sorted(
            range(len(self.chunks)),
            key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id),
        )
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i])) for i in order[:top_k]
        ]

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Pickle the whole index (BM25 stats + chunks + policy) to ``path``.

        Pickling the built ``BM25Okapi`` (not just the tokenised corpus) means load
        does zero recomputation, so a reloaded index yields byte-identical rankings.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": _PICKLE_VERSION,
            "bm25": self._bm25,
            "chunks": self.chunks,
            "tokenizer": self.tokenizer.config_dict(),
            "params": {"k1": self.k1, "b": self.b, "epsilon": self.epsilon},
            "corpus_fp": self.corpus_fp,
        }
        with path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Path) -> SparseRetriever:
        """Reload a persisted index without touching Qdrant or rebuilding BM25."""
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        if state.get("version") != _PICKLE_VERSION:
            raise ValueError(f"unsupported sparse index version: {state.get('version')!r}")
        obj = cls.__new__(cls)
        obj._bm25 = state["bm25"]
        obj.chunks = state["chunks"]
        tok = state["tokenizer"]
        obj.tokenizer = BM25Tokenizer(
            lowercase=tok["lowercase"], token_pattern=tok["token_pattern"]
        )
        params = state["params"]
        obj.k1, obj.b, obj.epsilon = params["k1"], params["b"], params["epsilon"]
        obj.corpus_fp = state["corpus_fp"]
        return obj

    # -- construction from the live corpus -----------------------------------

    @classmethod
    def from_config(cls, cfg: Settings, *, client=None) -> SparseRetriever:
        """Build (or reload) the index from the live Qdrant collection.

        A persisted index at ``<cache_dir>/sparse/<policy-hash>.pkl`` is reused only
        when both the tokeniser/BM25 policy *and* the corpus fingerprint still match
        the live setup; otherwise it is rebuilt by scrolling chunk text and saved.
        Rebuilding is fast (regex tokenisation + IDF over a few thousand chunks); the
        cache exists so it happens once per corpus, not once per process.
        """
        if client is None:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=cfg.qdrant.url, prefer_grpc=cfg.qdrant.prefer_grpc)

        collection = cfg.qdrant.collection_name
        tokenizer = BM25Tokenizer.from_config(cfg)
        live_fp = corpus_fingerprint(_scroll_chunk_ids(client, collection))
        cache_path = _cache_path(cfg)

        if cache_path.exists():
            cached = cls.load(cache_path)
            if cached.corpus_fp == live_fp and _policy_matches(cached, cfg):
                return cached

        chunks = _scroll_chunks(client, collection)
        retriever = cls(
            chunks=chunks,
            tokenizer=tokenizer,
            k1=cfg.sparse.k1,
            b=cfg.sparse.b,
            epsilon=cfg.sparse.epsilon,
            corpus_fp=live_fp,
        )
        retriever.save(cache_path)
        return retriever


# -- helpers -----------------------------------------------------------------


def _policy_matches(retriever: SparseRetriever, cfg: Settings) -> bool:
    return (
        retriever.tokenizer.config_dict() == tokenizer_config(cfg)
        and {"k1": retriever.k1, "b": retriever.b, "epsilon": retriever.epsilon}
        == bm25_params(cfg)
    )


def _cache_path(cfg: Settings) -> Path:
    """Where the persisted index lives — keyed by the tokeniser/BM25 policy so two
    policies never overwrite each other's cache."""
    policy = json.dumps(
        {
            "backend": cfg.sparse.backend,
            "tokenizer": tokenizer_config(cfg),
            "params": bm25_params(cfg),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(policy.encode("utf-8")).hexdigest()[:16]
    return Path(cfg.paths.cache_dir) / "sparse" / f"bm25_{digest}.pkl"


def _scroll_chunk_ids(client, collection: str) -> list[str]:
    """Every chunk_id in the collection — cheap (no text/vectors), for the fingerprint."""
    ids: list[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=1024,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        ids.extend((p.payload or {})["chunk_id"] for p in points)
        if offset is None:
            return ids


def _scroll_chunks(client, collection: str) -> list[Chunk]:
    """Every chunk with the full payload BM25 needs (text + metadata), sorted by
    chunk_id so the built index is deterministic across runs."""
    chunks: list[Chunk] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            chunks.append(
                Chunk(
                    chunk_id=payload["chunk_id"],
                    doc_id=payload["doc_id"],
                    page_num=int(payload["page_num"]),
                    text=payload["text"],
                    company=payload["company"],
                    ticker=payload["ticker"],
                    fiscal_year=payload["fiscal_year"],
                    form_type=payload["form_type"],
                )
            )
        if offset is None:
            break
    chunks.sort(key=lambda c: c.chunk_id)
    return chunks
