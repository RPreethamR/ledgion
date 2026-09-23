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
from ledgion.retrieve.rerank import rerank_by_scores
from ledgion.retrieve.sparse import bm25_params, corpus_fingerprint, tokenizer_config

FIXTURES_DIR = REPO_ROOT / "fixtures"
INDEX_FILE = "index.npz"
QUERY_VECTORS_FILE = "query_vectors.npz"
SPARSE_RANKINGS_FILE = "sparse_rankings.npz"
RERANK_SCORES_FILE = "rerank_scores.npz"
MANIFEST_FILE = "manifest.json"

# Depth at which sparse rankings are stored, independent of retrieval.top_k. The
# fixture holds the top-STORED_DEPTH sparse chunk_ids per qid; a run truncates to
# top_k and hard-fails if top_k exceeds this (there is nothing deeper to serve).
STORED_DEPTH = 100


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


# -- fixture-backed sparse retrieval -----------------------------------------


class FixtureSparseRetriever:
    """The offline sparse arm: replays frozen BM25 rankings instead of a live index.

    ``fixtures/sparse_rankings.npz`` holds, per golden qid, the top-STORED_DEPTH
    chunk_ids and their BM25 scores — computed once, locally, over the full corpus by
    ``make fixture``. This retriever looks them up by question (mapped to qid via the
    golden set), truncates to ``top_k``, and joins each chunk_id to its ``page_num``
    via ``index.npz`` (the fixture stores pages, not text). The hybrid path can then
    run in CI as: live dense search over fixture vectors → these frozen sparse
    rankings → the user's fusion → page-level scoring, all with no model or network.

    Construction runs the same class of guard as the dense fixture, so a stale sparse
    fixture fails loudly rather than scoring against the wrong rankings:

    1. the manifest must carry a ``sparse`` block (fixtures were built with sparse);
    2. its backend, tokeniser settings and BM25 parameters must equal the resolved
       config (the sparse ablation knobs actually used);
    3. its corpus fingerprint must equal the live one recomputed from ``index.npz``;
    4. every golden qid must have stored rankings.

    A fifth guard is deferred to query time: ``top_k`` must not exceed the stored
    depth (there is nothing deeper to serve).
    """

    def __init__(
        self,
        *,
        rankings_by_qid: dict[str, tuple[list[str], list[float]]],
        qid_by_question: dict[str, str],
        page_by_chunk_id: dict[str, int],
        doc_by_chunk_id: dict[str, str],
        stored_depth: int,
    ) -> None:
        self._rankings = rankings_by_qid
        self._qid_by_question = qid_by_question
        self._page_by_chunk_id = page_by_chunk_id
        self._doc_by_chunk_id = doc_by_chunk_id
        self._stored_depth = stored_depth

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        # Depth guard: the fixture only stores STORED_DEPTH candidates per qid.
        if top_k > self._stored_depth:
            raise SystemExit(
                f"retrieval.top_k={top_k} exceeds the stored sparse depth "
                f"{self._stored_depth}: the fixture has nothing deeper to serve. "
                f"Lower retrieval.top_k, or raise STORED_DEPTH and run make fixture."
            )
        try:
            qid = self._qid_by_question[query]
        except KeyError as exc:
            raise KeyError(
                f"no frozen sparse ranking for question {query!r}; it is not a golden "
                f"question this fixture set was built for (run make fixture)"
            ) from exc

        chunk_ids, scores = self._rankings[qid]
        results: list[RetrievedChunk] = []
        for cid, score in zip(chunk_ids[:top_k], scores[:top_k], strict=True):
            # Pages, not text: metadata the page-level ranking never reads is placeholder.
            chunk = Chunk(
                chunk_id=cid,
                doc_id=self._doc_by_chunk_id.get(cid, ""),
                page_num=self._page_by_chunk_id[cid],
                text="",
                company="",
                ticker="",
                fiscal_year=0,
                form_type="",
            )
            results.append(RetrievedChunk(chunk=chunk, score=float(score)))
        return results

    @classmethod
    def from_config(
        cls,
        cfg: Settings,
        *,
        fixtures_dir: Path = FIXTURES_DIR,
        golden: Sequence[dict] | None = None,
    ) -> FixtureSparseRetriever:
        fixtures_dir = Path(fixtures_dir)
        manifest = _load_manifest(fixtures_dir)

        index = _load_index_arrays(fixtures_dir)
        live_fp = corpus_fingerprint([str(cid) for cid in index["chunk_id"]])
        _check_sparse_manifest(cfg, manifest, fixtures_dir, live_fp)

        if golden is None:
            from ledgion.eval.runner import load_golden

            golden = load_golden()

        rankings = _load_sparse_rankings(fixtures_dir)
        missing = [row["qid"] for row in golden if row["qid"] not in rankings]
        if missing:
            raise _fixture_error(
                fixtures_dir,
                f"{len(missing)} golden qid(s) have no frozen sparse ranking: "
                f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}",
            )

        page_by_chunk_id = {
            str(cid): int(pn) for cid, pn in zip(index["chunk_id"], index["page_num"], strict=True)
        }
        doc_by_chunk_id = {
            str(cid): str(did) for cid, did in zip(index["chunk_id"], index["doc_id"], strict=True)
        }
        qid_by_question = {row["question"]: row["qid"] for row in golden}
        stored_depth = int(manifest["sparse"]["stored_depth"])
        return cls(
            rankings_by_qid=rankings,
            qid_by_question=qid_by_question,
            page_by_chunk_id=page_by_chunk_id,
            doc_by_chunk_id=doc_by_chunk_id,
            stored_depth=stored_depth,
        )


def _check_sparse_manifest(
    cfg: Settings, manifest: dict, fixtures_dir: Path, live_fingerprint: str
) -> None:
    """Fail loudly (run make fixture) if the fixture's sparse provenance has drifted
    from the resolved config or the committed corpus. Mirrors the dense revision guard."""
    sparse = manifest.get("sparse")
    if not sparse:
        raise _fixture_error(
            fixtures_dir,
            "manifest has no 'sparse' block — the sparse fixtures were never generated",
        )
    checks = [
        ("sparse backend", sparse.get("backend"), cfg.sparse.backend),
        ("sparse tokenizer", sparse.get("tokenizer"), tokenizer_config(cfg)),
        ("sparse bm25 parameters", sparse.get("bm25"), bm25_params(cfg)),
        ("corpus fingerprint", sparse.get("corpus_fingerprint"), live_fingerprint),
    ]
    for label, fixture_val, config_val in checks:
        if fixture_val != config_val:
            raise _fixture_error(
                fixtures_dir,
                f"fixture {label} {fixture_val!r} != {config_val!r}",
            )


def _load_index_arrays(fixtures_dir: Path) -> dict:
    """The chunk_id/doc_id/page_num arrays from index.npz (no vectors loaded)."""
    path = fixtures_dir / INDEX_FILE
    if not path.exists():
        raise _fixture_error(fixtures_dir, f"{path} not found")
    with np.load(path) as data:
        return {
            "chunk_id": data["chunk_id"],
            "doc_id": data["doc_id"],
            "page_num": data["page_num"],
        }


def _load_sparse_rankings(fixtures_dir: Path) -> dict[str, tuple[list[str], list[float]]]:
    """Load sparse_rankings.npz into {qid: (chunk_ids, scores)}, best-first per qid."""
    path = fixtures_dir / SPARSE_RANKINGS_FILE
    if not path.exists():
        raise _fixture_error(fixtures_dir, f"{path} not found")
    with np.load(path) as data:
        qids = [str(q) for q in data["qids"]]
        chunk_ids = data["chunk_ids"]
        scores = data["scores"]
    return {
        qid: ([str(c) for c in chunk_ids[i]], [float(s) for s in scores[i]])
        for i, qid in enumerate(qids)
    }


# -- fixture-backed reranking ------------------------------------------------


class FixtureReranker:
    """The offline reranking stage: replays frozen cross-encoder scores instead of a
    live model, so a reranked run scores in CI exactly like the live path.

    ``fixtures/rerank_scores.npz`` holds, per golden qid, the cross-encoder score for
    every candidate chunk_id in that question's pool — frozen once, locally, by
    ``make fixture`` at ``candidate_depth`` (the dense top-``depth`` pool, which
    contains every reranked config's pool). This stage maps the question to its qid,
    looks the frozen scores up, and hands them to ``rerank_by_scores`` — the *same*
    reordering primitive the live ``CrossEncoderReranker`` uses, so the two produce
    identical rankings by construction.

    ⚠️ If a candidate has no frozen score it **raises** (run make fixture). It never
    skips the candidate and never defaults the score to zero or the minimum: a PR that
    changes retrieval can pull a chunk into the pool that was never scored, and
    silently reranking that partial pool would be a wrong result nobody noticed. Loud
    failure is the contract — the same role the missing-query-vector guard plays for
    the dense fixture.

    Construction runs the same class of guard as the sparse fixture (run make fixture
    on any):

    1. the manifest carries a ``reranker`` block (fixtures were built with reranking);
    2. its model_id and revision equal the resolved config's — the lookup is keyed on
       the revision, so changing the model can't reuse stale frozen scores;
    3. its corpus fingerprint equals the live one recomputed from ``index.npz``;
    4. every golden qid has frozen scores.
    """

    def __init__(
        self,
        *,
        scores_by_qid: dict[str, dict[str, float]],
        qid_by_question: dict[str, str],
        top_n: int,
    ) -> None:
        self._scores = scores_by_qid
        self._qid_by_question = qid_by_question
        self._top_n = top_n

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        try:
            qid = self._qid_by_question[query]
        except KeyError as exc:
            raise KeyError(
                f"no frozen rerank scores for question {query!r}; it is not a golden "
                f"question this fixture set was built for (run make fixture)"
            ) from exc

        frozen = self._scores[qid]
        scores: list[float] = []
        for rc in candidates:
            cid = rc.chunk.chunk_id
            if cid not in frozen:
                # A candidate the frozen pool never scored — never skip it, never
                # default the score. Fail loudly so a retrieval change that widened
                # the pool can't silently rerank a partial one.
                raise SystemExit(
                    f"candidate chunk_id {cid!r} (qid {qid}) has no precomputed "
                    f"cross-encoder score: retrieval pulled a chunk the frozen rerank "
                    f"pool never scored. Never rerank a partial pool — run make fixture."
                )
            scores.append(frozen[cid])
        return rerank_by_scores(candidates, scores, top_n=top_n)

    @classmethod
    def from_config(
        cls,
        cfg: Settings,
        *,
        fixtures_dir: Path = FIXTURES_DIR,
        golden: Sequence[dict] | None = None,
    ) -> FixtureReranker:
        fixtures_dir = Path(fixtures_dir)
        manifest = _load_manifest(fixtures_dir)

        index = _load_index_arrays(fixtures_dir)
        live_fp = corpus_fingerprint([str(cid) for cid in index["chunk_id"]])
        _check_rerank_manifest(cfg, manifest, fixtures_dir, live_fp)

        if golden is None:
            from ledgion.eval.runner import load_golden

            golden = load_golden()

        scores = _load_rerank_scores(fixtures_dir)
        missing = [row["qid"] for row in golden if row["qid"] not in scores]
        if missing:
            raise _fixture_error(
                fixtures_dir,
                f"{len(missing)} golden qid(s) have no frozen rerank scores: "
                f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}",
            )

        qid_by_question = {row["question"]: row["qid"] for row in golden}
        return cls(
            scores_by_qid=scores,
            qid_by_question=qid_by_question,
            top_n=cfg.reranker.top_n,
        )


def _check_rerank_manifest(
    cfg: Settings, manifest: dict, fixtures_dir: Path, live_fingerprint: str
) -> None:
    """Fail loudly (run make fixture) if the fixture's reranker provenance has drifted
    from the resolved config or the committed corpus. Mirrors the sparse guard; the
    revision check is what keys the frozen scores to a model version."""
    reranker = manifest.get("reranker")
    if not reranker:
        raise _fixture_error(
            fixtures_dir,
            "manifest has no 'reranker' block — the rerank fixtures were never generated",
        )
    checks = [
        ("reranker model_id", reranker.get("model_id"), cfg.reranker.model_id),
        ("reranker revision", reranker.get("revision"), cfg.reranker.revision),
        ("corpus fingerprint", reranker.get("corpus_fingerprint"), live_fingerprint),
    ]
    for label, fixture_val, config_val in checks:
        if fixture_val != config_val:
            raise _fixture_error(
                fixtures_dir,
                f"fixture {label} {fixture_val!r} != {config_val!r}",
            )


def _load_rerank_scores(fixtures_dir: Path) -> dict[str, dict[str, float]]:
    """Load rerank_scores.npz into {qid: {chunk_id: score}} — the frozen cross-encoder
    score for every candidate in each golden qid's pool."""
    path = fixtures_dir / RERANK_SCORES_FILE
    if not path.exists():
        raise _fixture_error(fixtures_dir, f"{path} not found")
    with np.load(path) as data:
        qids = [str(q) for q in data["qids"]]
        chunk_ids = data["chunk_ids"]
        scores = data["scores"]
    return {
        qid: {
            str(c): float(s)
            for c, s in zip(chunk_ids[i], scores[i], strict=True)
        }
        for i, qid in enumerate(qids)
    }
