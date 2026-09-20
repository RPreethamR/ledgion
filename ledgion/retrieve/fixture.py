"""Fixture-backed retrieval — the offline Tier-1 path (no model, no live Qdrant).

This is the same ``Retriever`` contract as ``DenseRetriever``, wired so a CI
runner with no torch, no network, and no populated Qdrant can still reproduce the
exact page rankings the real retriever produces. It does that by *replaying*
committed fixtures instead of computing anything:

* ``fixtures/index.npz`` — every corpus chunk's dense vector plus parallel
  ``chunk_id``/``doc_id``/``page_num`` arrays — is loaded into an **in-process**
  Qdrant collection built by the very same ``QdrantIndexer`` used at ingest, so
  the collection is byte-for-byte the shape production queries hit.
* ``fixtures/query_vectors.npz`` — one pre-computed query embedding per golden
  qid — supplies the query vector by lookup, so no query is ever embedded.

Identical rankings are guaranteed *by construction*, not by coincidence: the
actual similarity search is delegated to a real ``DenseRetriever`` whose only
change is a ``_PrecomputedEmbedder`` in place of the bge model. Same collection,
same ``query_points`` call, same page-collapse downstream — the only thing swapped
out is the multi-GB model.

Two guards fire loudly at construction (both saying ``run make fixture``), because
a stale fixture would silently score against the wrong vectors:

1. the fixture's embedding revision must equal the configured revision;
2. every golden qid must have a pre-computed query vector.

The fixture stores pages, not text (Tier 1 scores pages only), so the chunk
payloads carry placeholder metadata — retrieval only ever reads ``page_num``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ledgion.config import REPO_ROOT, Settings
from ledgion.ingest.index import DENSE_VECTOR, QdrantIndexer
from ledgion.interfaces import Chunk, RetrievedChunk
from ledgion.retrieve.dense import DenseRetriever

FIXTURES_DIR = REPO_ROOT / "fixtures"
INDEX_FILE = "index.npz"
QUERY_VECTORS_FILE = "query_vectors.npz"
MANIFEST_FILE = "manifest.json"


def _fixture_error(fixtures_dir: Path, detail: str) -> SystemExit:
    """A loud, actionable failure. Every fixture problem points at the one fix."""
    return SystemExit(
        f"{detail}\n(fixtures in {fixtures_dir} are stale or missing — run make fixture)"
    )


class _PrecomputedEmbedder:
    """An ``Embedder`` that returns a *pre-computed* query vector by question text.

    The retriever contract passes the question string, not a qid, so this maps
    question -> vector (built from golden qid -> vector at load time). It embeds
    nothing: ``embed_documents`` is unsupported because a fixture run must never
    silently fall back to computing vectors — that would defeat the whole point.
    """

    def __init__(self, vector_by_question: dict[str, np.ndarray]) -> None:
        self._by_question = vector_by_question

    def embed_query(self, text: str) -> np.ndarray:
        try:
            return self._by_question[text]
        except KeyError as exc:
            raise KeyError(
                f"no pre-computed query vector for question {text!r}; "
                f"it is not a golden question this fixture set was built for (run make fixture)"
            ) from exc

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        raise NotImplementedError(
            "FixtureRetriever embeds nothing; regenerate vectors with `make fixture`"
        )


class FixtureRetriever:
    """A ``Retriever`` (see interfaces.py) backed by committed fixtures."""

    def __init__(self, *, dense: DenseRetriever) -> None:
        # Composition, not inheritance: the real query path is reused verbatim so
        # rankings match the live retriever exactly.
        self._dense = dense

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        return self._dense.retrieve(query, top_k=top_k)

    @classmethod
    def from_config(
        cls,
        cfg: Settings,
        *,
        fixtures_dir: Path = FIXTURES_DIR,
        golden: Sequence[dict] | None = None,
    ) -> FixtureRetriever:
        """Build the retriever from ``fixtures/`` (+ the golden set for the qid guard).

        ``golden`` defaults to the project golden set; it is a parameter so the
        guards are testable on a fabricated fixture set with no real data.
        """
        fixtures_dir = Path(fixtures_dir)
        manifest = _load_manifest(fixtures_dir)

        # Guard 1: the fixture was built with the model revision we're evaluating.
        fixture_rev = manifest.get("embedding", {}).get("revision")
        if fixture_rev != cfg.embedding.revision:
            raise _fixture_error(
                fixtures_dir,
                f"fixture embedding revision {fixture_rev!r} != config revision "
                f"{cfg.embedding.revision!r}",
            )

        if golden is None:
            from ledgion.eval.runner import load_golden

            golden = load_golden()

        # Guard 2: every golden question has a pre-computed query vector.
        query_vectors = _load_query_vectors(fixtures_dir)
        missing = [row["qid"] for row in golden if row["qid"] not in query_vectors]
        if missing:
            raise _fixture_error(
                fixtures_dir,
                f"{len(missing)} golden qid(s) have no pre-computed query vector: "
                f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}",
            )

        client = _load_index_into_memory(cfg, fixtures_dir)
        vector_by_question = {row["question"]: query_vectors[row["qid"]] for row in golden}
        dense = DenseRetriever(
            embedder=_PrecomputedEmbedder(vector_by_question),
            client=client,
            collection_name=cfg.qdrant.collection_name,
            vector_name=DENSE_VECTOR,
        )
        return cls(dense=dense)


# -- fixture loading ---------------------------------------------------------


def _load_manifest(fixtures_dir: Path) -> dict:
    path = fixtures_dir / MANIFEST_FILE
    if not path.exists():
        raise _fixture_error(fixtures_dir, f"{path} not found")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_query_vectors(fixtures_dir: Path) -> dict[str, np.ndarray]:
    path = fixtures_dir / QUERY_VECTORS_FILE
    if not path.exists():
        raise _fixture_error(fixtures_dir, f"{path} not found")
    with np.load(path) as data:
        return {qid: data[qid].astype(np.float32) for qid in data.files}


def _load_index_into_memory(cfg: Settings, fixtures_dir: Path):
    """Load ``index.npz`` into a fresh in-process Qdrant collection and return the
    client. Built with the ingest-time ``QdrantIndexer`` so the collection — named
    vector, cosine distance, point ids derived from chunk_id — is identical to
    production, which is what makes rankings comparable."""
    path = fixtures_dir / INDEX_FILE
    if not path.exists():
        raise _fixture_error(fixtures_dir, f"{path} not found")

    with np.load(path) as data:
        vectors = data["vectors"].astype(np.float32)
        chunk_ids = data["chunk_id"]
        doc_ids = data["doc_id"]
        page_nums = data["page_num"]

    # The fixture holds pages, not text (Tier 1 scores pages only), so metadata
    # the ranking never reads is filled with placeholders.
    chunks = [
        Chunk(
            chunk_id=str(cid),
            doc_id=str(did),
            page_num=int(pn),
            text="",
            company="",
            ticker="",
            fiscal_year=0,
            form_type="",
        )
        for cid, did, pn in zip(chunk_ids, doc_ids, page_nums, strict=True)
    ]

    indexer = QdrantIndexer(
        mode="memory",
        url=cfg.qdrant.url,
        path=str(cfg.qdrant.path),
        collection_name=cfg.qdrant.collection_name,
    )
    indexer.upsert(chunks, list(vectors))
    return indexer.client
