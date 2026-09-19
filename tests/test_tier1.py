"""Tier 1: the offline retrieval eval — chunk→page collapse, scoring, determinism.

Everything here runs with **no network, no API key, no model download and no
Qdrant**: retrieval is a hand-built fake returning canned ``RetrievedChunk``s, so
the scoring harness is exercised in isolation. (The real ``DenseRetriever`` — which
needs the bge model + a populated Qdrant — is wired only in ``runner.run``, never
here.) That is exactly how the project rule "Tier 1 runs fully offline" is enforced
at the test level, so it stays green on a fork's CI runner.
"""

from __future__ import annotations

from ledgion.eval.tier1 import collapse_to_pages, run_tier1
from ledgion.interfaces import Chunk, RetrievedChunk


def _chunk(page_num: int, idx: int) -> Chunk:
    return Chunk(
        chunk_id=f"{idx:016x}",
        doc_id="AMCOR_2023_10K",
        page_num=page_num,
        text=f"chunk on page {page_num}",
        company="Amcor",
        ticker="AMCR",
        fiscal_year=2023,
        form_type="10-K",
    )


def _retrieved(pages: list[int]) -> list[RetrievedChunk]:
    # list position IS the rank; score strictly decreasing so nothing re-orders it.
    return [RetrievedChunk(chunk=_chunk(p, i), score=1.0 - 0.01 * i) for i, p in enumerate(pages)]


class _FakeRetriever:
    """Returns a fixed page ranking per question — no model, no network."""

    def __init__(self, pages_by_question: dict[str, list[int]]) -> None:
        self._pages = pages_by_question

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        return _retrieved(self._pages[query])[:top_k]


# --- the exact case from the phase spec -------------------------------------


def test_collapse_keeps_page_at_best_rank():
    # Chunks from page 52 come back at ranks 1, 3 and 7 (positions 0, 2, 6). Page
    # 52 must enter the page list exactly ONCE, at position 0 (its best rank); the
    # ranks-3-and-7 duplicates contribute nothing. The result is best-rank ascending.
    retrieved = _retrieved([52, 10, 52, 20, 30, 40, 52])
    assert collapse_to_pages(retrieved) == [52, 10, 20, 30, 40]


def test_collapse_empty_ranking():
    assert collapse_to_pages([]) == []


# --- scoring, reported overall and per answer type --------------------------

_GOLDEN = [
    {
        "qid": "q_num",
        "doc_id": "AMCOR_2023_10K",
        "question": "num?",
        "answer_type": "numeric",
        "evidence_pages": [52],
    },
    {
        "qid": "q_prose",
        "doc_id": "AMCOR_2023_10K",
        "question": "prose?",
        "answer_type": "prose",
        "evidence_pages": [5],
    },
]


def _run() -> dict:
    retriever = _FakeRetriever(
        {
            "num?": [52, 10, 20],  # relevant page 52 at rank 1
            "prose?": [10, 20, 5],  # relevant page 5 at rank 3
        }
    )
    return run_tier1(
        _GOLDEN,
        retriever,
        top_k=20,
        metric_specs=["recall@k", "ndcg@10", "mrr", "hit@1"],
        default_k=20,
    )


def test_tier1_runs_offline_and_reports_by_answer_type():
    result = _run()
    metrics = result["metrics"]

    # overall + both answer-type groups + counts are present.
    assert set(metrics) >= {"overall", "numeric", "prose", "counts"}
    assert metrics["counts"] == {"overall": 2, "numeric": 1, "prose": 1}

    # numeric question: relevant page at rank 1 -> hit@1 and mrr are perfect.
    assert metrics["numeric"]["hit@1"] == 1.0
    assert metrics["numeric"]["mrr"] == 1.0
    assert metrics["numeric"]["recall@k"] == 1.0

    # prose question: first relevant page at rank 3 -> miss on hit@1, mrr = 1/3.
    assert metrics["prose"]["hit@1"] == 0.0
    assert metrics["prose"]["mrr"] == round(1 / 3, 6)

    # per-question records carry the collapsed page ranking (so a moved number is
    # debuggable without re-running retrieval).
    by_qid = {r["qid"]: r for r in result["results"]}
    assert by_qid["q_num"]["retrieved_pages"] == [52, 10, 20]
    assert by_qid["q_prose"]["evidence_pages"] == [5]


def test_tier1_two_runs_are_identical():
    # Same config (same golden, same fake retriever) -> identical numbers, every
    # metric, every per-question record. This is the determinism guarantee that
    # makes results/<hash>.json reproducible.
    assert _run() == _run()
