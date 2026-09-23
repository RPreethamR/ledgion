"""Cross-encoder reranking — a stage applied *after* retrieval, not a Retriever.

A bi-encoder (the dense arm) embeds the query and every chunk independently and
compares vectors; a cross-encoder instead reads ``(query, chunk_text)`` *together*
and scores their relevance directly. That joint attention is far more accurate at
ordering a small candidate set — and far too slow to run over the whole corpus —
so it is used exactly as a second pass: retrieve a wide ``top_k`` pool cheaply,
then let the cross-encoder re-order it.

``ms-marco-MiniLM-L-6-v2`` via sentence-transformers, CPU-only, threads pinned to
the machine's physical cores (see CLAUDE.md) — the same discipline as
``BGEEmbedder``. Two properties carried over from the embedder:

* **Never load unpinned.** ``rerank(...)`` refuses to run if ``reranker.revision``
  is ``None``: an unpinned reranker makes every reordering — and every frozen
  fixture score keyed on that revision — unreproducible. A pin nothing reads is
  worse than no pin.
* **Deterministic ordering.** Ties in the cross-encoder score are broken by
  ``chunk_id`` so the reordering — and therefore ``results/<hash>.json`` — is
  byte-reproducible, exactly like the sparse arm's tie-break.

Division of labour (CLAUDE.md working agreement): this is a model *client* +
reordering plumbing (Claude's), not fusion maths or a metric (the user's). The
reordering is a single pure function, ``rerank_by_scores``, so it is unit-testable
with no model — and it is the *same* function the offline ``FixtureReranker`` uses,
so live and frozen-score reranking reorder identically by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

from ledgion.config import Settings
from ledgion.interfaces import RetrievedChunk


def rerank_by_scores(
    candidates: Sequence[RetrievedChunk], scores: Sequence[float], *, top_n: int
) -> list[RetrievedChunk]:
    """Reorder ``candidates`` by ``scores`` (highest first) and keep the top ``top_n``.

    ``scores[i]`` is the reranker score for ``candidates[i]`` — the two are parallel
    and must be the same length (a mismatch means a candidate was scored against the
    wrong pool, so it raises rather than silently zip-truncating). The returned
    ``RetrievedChunk``s carry the *reranker* score in ``.score``, replacing the
    upstream retrieval score; downstream stages re-order, they never compare scales
    across stages (see interfaces.py).

    Ties are broken by ``chunk_id`` ascending, so the ordering is fully deterministic
    regardless of the sort's stability or the cross-encoder's floating-point noise.
    This is the one reordering primitive both the live and the fixture rerankers use.
    """
    if len(candidates) != len(scores):
        raise ValueError(
            f"candidates/scores length mismatch: {len(candidates)} != {len(scores)}"
        )
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-float(scores[i]), candidates[i].chunk.chunk_id),
    )
    return [
        RetrievedChunk(chunk=candidates[i].chunk, score=float(scores[i]))
        for i in order[:top_n]
    ]


class CrossEncoderReranker:
    """A ``Reranker`` (see interfaces.py) over a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        device: str = "cpu",
        batch_size: int = 32,
        num_threads: int = 4,
        top_n: int = 5,
    ) -> None:
        if revision is None:
            # Fail fast, exactly like BGEEmbedder: an unpinned reranker makes every
            # reordering and every frozen fixture score unreproducible.
            raise ValueError(
                "reranker.revision is None — pin a commit SHA before reranking "
                "(an unpinned model breaks reproducibility and the fixture score guard)"
            )
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.top_n = top_n
        self._model = None

    @classmethod
    def from_config(cls, cfg: Settings) -> CrossEncoderReranker:
        return cls(
            model_id=cfg.reranker.model_id,
            revision=cfg.reranker.revision,
            device=cfg.reranker.device,
            batch_size=cfg.reranker.batch_size,
            num_threads=cfg.torch.num_threads,
            top_n=cfg.reranker.top_n,
        )

    # -- model loading -------------------------------------------------------

    def _ensure_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            # Physical cores only — hyperthreading hurts this CPU workload (CLAUDE.md).
            torch.set_num_threads(self.num_threads)
            self._model = CrossEncoder(self.model_id, revision=self.revision, device=self.device)
        return self._model

    # -- scoring + Reranker contract -----------------------------------------

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Raw cross-encoder relevance scores for ``(query, text)`` pairs, in input
        order (no reordering, no truncation).

        Kept separate from ``rerank`` so fixture generation can freeze the score of
        *every* candidate in a pool — reordering happens later, in ``rerank_by_scores``.
        Returns the model's raw logits (``predict`` with no softmax): the activation
        is monotonic, so it never changes the ranking, and skipping it keeps the frozen
        scores identical to what a bare ``predict`` produces.
        """
        if not texts:
            return []
        model = self._ensure_model()
        scores = model.predict(
            [[query, text] for text in texts],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_n: int,
    ) -> list[RetrievedChunk]:
        """Score every candidate against ``query`` and return the top ``top_n`` reordered.

        The eval pipeline passes ``top_n = retrieval.top_k`` so the *whole* pool is
        reordered and scored (keeping recall@k comparable to dense — see DECISIONS.md
        Phase 7); the serving path passes ``reranker.top_n`` to trim the context handed
        to the generator. Either way the reordering is ``rerank_by_scores``, the same
        primitive the offline ``FixtureReranker`` uses.
        """
        if not candidates:
            return []
        scores = self.score(query, [rc.chunk.text for rc in candidates])
        return rerank_by_scores(candidates, scores, top_n=top_n)
