"""Swappable-component contracts for the Ledgion pipeline.

Every component that a future ablation might replace is defined here as a
``typing.Protocol`` rather than a concrete base class. The project runs many
ablations (chunkers, retrievers, rerankers, fusion strategies); structural
typing lets us swap an implementation without touching call sites or forcing
an inheritance hierarchy.

The three dataclasses below are the *only* objects these components exchange.

Note on identifiers: retrieval is scored at **page level, never chunk level**
(see CLAUDE.md). ``chunk_id`` changes whenever the chunker changes, so it is
useless for cross-strategy comparison; ``page_num`` is stable across chunkers
and is what evaluation keys on. Every ``Chunk`` therefore carries its
``page_num``, and citations are (doc_id, page_num) pairs.

This module is contracts only — no implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# A dense embedding and a matrix of them. Typed as plain float sequences so the
# contract stays free of a numpy dependency; concrete embedders may return
# ndarrays, which satisfy this structurally at the call sites that consume them.
Vector = Sequence[float]
Matrix = Sequence[Vector]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A unit of retrievable text that never spans a page boundary.

    ``slots=True`` keeps per-chunk memory down — a full 10-K produces thousands
    of these and the machine has ~5.5GB usable RAM. ``frozen=True`` makes chunks
    hashable and safe to share across retrievers.
    """

    chunk_id: str
    doc_id: str
    page_num: int
    text: str
    company: str
    ticker: str
    fiscal_year: int
    form_type: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A ``Chunk`` paired with the score that surfaced it.

    ``score`` is whatever the producing stage assigns (dense similarity, BM25,
    a fused rank score, or a reranker logit). Its scale is only meaningful
    within one stage; downstream stages re-order, they do not compare across
    scales.
    """

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Answer:
    """A generated answer with the page-level provenance it was grounded on.

    ``citations`` holds only *validated* provenance: (doc_id, page_num) pairs for
    chunk_ids the model cited that were actually in the context sent to it (see
    generate/citations.py). ``insufficient_evidence`` is the model's own signal
    that the context did not contain the answer. ``citation_validity`` and
    ``dropped_citations`` are the deterministic, judge-free faithfulness signal:
    the fraction of cited chunk_ids that were real, and the fabricated ids that
    were discarded. The last three are defaulted so a caller that skips citation
    validation still builds a valid ``Answer``.
    """

    question: str
    text: str
    citations: list[tuple[str, int]]  # validated (doc_id, page_num) provenance
    contexts: list[RetrievedChunk]
    insufficient_evidence: bool = False
    citation_validity: float = 1.0
    dropped_citations: list[str] = field(default_factory=list)


@runtime_checkable
class Chunker(Protocol):
    """Splits a parsed document into page-bounded chunks."""

    def chunk(
        self,
        doc_id: str,
        pages: Sequence[tuple[int, str]],
        *,
        company: str,
        ticker: str,
        fiscal_year: int,
        form_type: str,
    ) -> list[Chunk]:
        """Chunk ``pages`` (each a ``(page_num, page_text)`` pair) for one doc.

        Implementations must not merge text across pages: every returned
        ``Chunk`` carries the ``page_num`` it came from.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Maps text to dense vectors. Documents and queries may be encoded
    differently (e.g. an instruction prefix on the query side)."""

    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Vector: ...


@runtime_checkable
class Retriever(Protocol):
    """Returns candidate chunks for a query, best first."""

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """Re-orders retrieved candidates and keeps the top ``top_n``."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_n: int,
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class Generator(Protocol):
    """Produces a grounded ``Answer`` from a question and retrieved context."""

    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> Answer: ...
