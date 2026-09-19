"""Tier 1: the deterministic, fully-offline retrieval eval.

Tier 1 scores **retrieval only** — which pages came back, never what a model
generated — so it needs no API key, no judge, and no network. It runs the
injected ``Retriever`` over every golden question, collapses the retrieved chunks
to a deduplicated page ranking, and scores that ranking against the golden
``evidence_pages`` with the user's metric functions in ``metrics.py``.

Two CLAUDE.md rules shape this file:

* **Scored at page level, never chunk level.** chunk_ids move with the chunker;
  page numbers are the stable cross-strategy key. ``collapse_to_pages`` is the
  one bridge from a chunk ranking to a page ranking.
* **Deterministic.** No wall-clock, no RNG, no dict-ordering dependence — the
  same retriever output always yields the same numbers, so a re-run of a config
  reproduces its ``results/<hash>.json`` byte-for-byte.

The retriever is a *parameter*, not a hard-wired ``DenseRetriever``: the runner
injects the real one (which needs the bge model + Qdrant), while the tests inject
a canned fake. That is what keeps Tier 1 exercisable with no network.

Metric cutoffs. A spec like ``recall@k`` / ``ndcg@10`` carries its cutoff after
``@``; a *literal* ``k`` resolves to ``default_k`` — the retrieval candidate depth
(``retrieval.top_k``). After page-collapse the candidate list is shorter than
``top_k`` chunks, so ``recall@k`` reads as "did the evidence page appear anywhere
in the returned candidates". ``mrr`` and ``hit@1`` take no cutoff.
"""

from __future__ import annotations

from collections.abc import Sequence

from ledgion.eval import metrics
from ledgion.interfaces import RetrievedChunk, Retriever

# Answer-type groups reported alongside "overall". Fixed (not derived from the
# data) so the results schema is stable across runs even when a group is empty.
_GROUPS = ("numeric", "prose")


def collapse_to_pages(retrieved: Sequence[RetrievedChunk]) -> list[int]:
    """Collapse a chunk ranking to a deduplicated page ranking.

    The retriever returns chunks best-first, so list position *is* the rank. Each
    page enters the result once, at its best (lowest) rank; later chunks from an
    already-seen page contribute nothing. The result is therefore ordered by best
    rank ascending.

    Example: chunks from page 52 at ranks 1, 3 and 7 → page 52 appears once, at
    position 0; ranks 3 and 7 are dropped.
    """
    seen: set[int] = set()
    pages: list[int] = []
    for rc in retrieved:
        page = rc.chunk.page_num
        if page not in seen:
            seen.add(page)
            pages.append(page)
    return pages


def _apply_metric(
    spec: str, ranked: list[int], relevant: set[int], *, default_k: int
) -> float:
    """Dispatch one ``name@k`` metric spec to the matching function in metrics.py."""
    name, _, cutoff = spec.partition("@")
    if name == "mrr":
        return metrics.mrr(ranked, relevant)
    if name == "hit":
        # hit@1 is fixed at rank 1; any cutoff in the spec is cosmetic.
        return metrics.hit_at_1(ranked, relevant)
    k = default_k if cutoff in ("", "k") else int(cutoff)
    if name == "recall":
        return metrics.recall_at_k(ranked, relevant, k)
    if name == "ndcg":
        return metrics.ndcg_at_k(ranked, relevant, k)
    raise ValueError(f"unknown metric spec: {spec!r}")


def score_row(
    ranked: Sequence[int],
    relevant: set[int],
    metric_specs: Sequence[str],
    *,
    default_k: int,
) -> dict[str, float]:
    """Score one page ranking against its relevant pages for every metric spec.

    Values are rounded to 6 dp so the emitted numbers are stable and readable;
    the rounding is deterministic, so it never breaks the byte-identical guarantee.
    """
    ranked_list = list(ranked)
    return {
        spec: round(_apply_metric(spec, ranked_list, relevant, default_k=default_k), 6)
        for spec in metric_specs
    }


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6)


def _aggregate(per_question: Sequence[dict], metric_specs: Sequence[str]) -> dict:
    """Mean of each metric overall, and separately for numeric and prose rows."""
    groups: dict[str, list[dict]] = {"overall": list(per_question)}
    for group in _GROUPS:
        groups[group] = [r for r in per_question if r["answer_type"] == group]

    out: dict = {}
    for name, rows in groups.items():
        if rows:  # an empty group is omitted from the metric table…
            out[name] = {
                spec: _mean([r["metrics"][spec] for r in rows]) for spec in metric_specs
            }
    # …but its count is always reported, so the split is legible even at zero.
    out["counts"] = {name: len(rows) for name, rows in groups.items()}
    return out


def run_tier1(
    golden_rows: Sequence[dict],
    retriever: Retriever,
    *,
    top_k: int,
    metric_specs: Sequence[str],
    default_k: int | None = None,
) -> dict:
    """Run retrieval + page-level scoring over every golden row.

    Returns ``{"results": [per-question…], "metrics": {overall/numeric/prose/counts}}``.
    ``default_k`` (for a literal ``recall@k``) defaults to ``top_k``.
    """
    default_k = top_k if default_k is None else default_k
    per_question: list[dict] = []
    for row in golden_rows:
        retrieved = retriever.retrieve(row["question"], top_k=top_k)
        ranked = collapse_to_pages(retrieved)
        relevant = set(row["evidence_pages"])
        per_question.append(
            {
                "qid": row["qid"],
                "doc_id": row["doc_id"],
                "answer_type": row["answer_type"],
                "evidence_pages": list(row["evidence_pages"]),
                "retrieved_pages": ranked,
                "metrics": score_row(ranked, relevant, metric_specs, default_k=default_k),
            }
        )
    return {"results": per_question, "metrics": _aggregate(per_question, metric_specs)}
