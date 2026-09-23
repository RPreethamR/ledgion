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
    doc = "AMCOR_2023_10K"  # _chunk's doc_id — keys are now (doc_id, page_num)
    assert collapse_to_pages(retrieved) == [(doc, 52), (doc, 10), (doc, 20), (doc, 30), (doc, 40)]


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
    doc = "AMCOR_2023_10K"
    assert by_qid["q_num"]["retrieved_pages"] == [(doc, 52), (doc, 10), (doc, 20)]
    assert by_qid["q_prose"]["evidence_pages"] == [5]  # golden provenance stays bare


def test_recall_at_10_and_recall_at_k_are_distinct_columns():
    # Relevant page 52 is retrieved at rank 11 (positions 0..10). recall@k spans
    # the full candidate depth (top_k=20) and finds it; recall@10's fixed cutoff
    # does not — proving the two recall columns are genuinely distinct, both from
    # the single recall_at_k function at different k.
    golden = [
        {
            "qid": "q",
            "doc_id": "AMCOR_2023_10K",
            "question": "q?",
            "answer_type": "numeric",
            "evidence_pages": [52],
        },
    ]
    pages = list(range(1, 11)) + [52]  # 10 non-relevant pages, then page 52 at rank 11
    retriever = _FakeRetriever({"q?": pages})

    result = run_tier1(
        golden,
        retriever,
        top_k=20,
        metric_specs=["recall@k", "recall@10"],
        default_k=20,
    )

    per_q = result["results"][0]["metrics"]
    assert per_q["recall@k"] == 1.0  # found within the full top_k (=20)
    assert per_q["recall@10"] == 0.0  # but not within the fixed top 10
    # both are reported as their own aggregate columns.
    assert set(result["metrics"]["overall"]) == {"recall@k", "recall@10"}


def test_wrong_filing_page_collision_is_not_a_hit():
    # Golden evidence is AMD page 56. A chunk from a DIFFERENT filing that happens to
    # share page 56 must NOT count — scoring is on (doc_id, page), not bare page.
    golden = [
        {
            "qid": "q",
            "doc_id": "AMD_2022_10K",
            "question": "q?",
            "answer_type": "numeric",
            "evidence_pages": [56],
        }
    ]

    def _mixed(*pairs: tuple[str, int]) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id=f"{i:016x}",
                    doc_id=doc,
                    page_num=page,
                    text="",
                    company="",
                    ticker="",
                    fiscal_year=2022,
                    form_type="10-K",
                ),
                score=1.0 - 0.01 * i,
            )
            for i, (doc, page) in enumerate(pairs)
        ]

    class _Canned:
        def __init__(self, ranking: list[RetrievedChunk]) -> None:
            self._ranking = ranking

        def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
            return self._ranking[:top_k]

    specs = ["recall@k", "recall@10", "ndcg@10", "mrr", "hit@1"]

    # rank 1 = wrong filing at the evidence page number; rank 2 = right filing, wrong page.
    collision = _Canned(_mixed(("AMERICANEXPRESS_2022_10K", 56), ("AMD_2022_10K", 99)))
    rec = run_tier1(golden, collision, top_k=20, metric_specs=specs, default_k=20)["results"][0]
    assert rec["metrics"]["hit@1"] == 0.0
    assert rec["metrics"]["recall@10"] == 0.0
    assert rec["metrics"]["mrr"] == 0.0
    assert rec["retrieved_pages"] == [("AMERICANEXPRESS_2022_10K", 56), ("AMD_2022_10K", 99)]

    # Positive control: the SAME page number from the RIGHT filing does hit.
    right = _Canned(_mixed(("AMD_2022_10K", 56)))
    rec2 = run_tier1(golden, right, top_k=20, metric_specs=specs, default_k=20)["results"][0]
    assert rec2["metrics"]["hit@1"] == 1.0
    assert rec2["metrics"]["recall@10"] == 1.0


def test_tier1_two_runs_are_identical():
    # Same config (same golden, same fake retriever) -> identical numbers, every
    # metric, every per-question record. This is the determinism guarantee that
    # makes results/<hash>.json reproducible.
    assert _run() == _run()


# --- reranking as a post-retrieval stage (Phase 7) --------------------------


class _ReversingReranker:
    """Reverses the candidate pool — a pure permutation, no model. Reversing is the
    starkest way to show the pool-integrity invariant: it changes every rank yet adds
    and removes nothing, so recall@k (full depth) must not move."""

    def rerank(self, query, candidates, *, top_n):
        return list(reversed(candidates))[:top_n]


def test_reranker_preserves_recall_at_k_but_moves_recall_at_10():
    # The pool-integrity property check, in miniature (DECISIONS.md Phase 7): reranking
    # reorders the top_k pool, it never adds to it — so recall@k (scored over the whole
    # reordered pool) is invariant, while recall@10 is exactly what reranking can move.
    golden = [
        {
            "qid": "q",
            "doc_id": "AMCOR_2023_10K",
            "question": "q?",
            "answer_type": "numeric",
            "evidence_pages": [52],
        }
    ]
    # 20 distinct pages; the evidence page is buried at rank 20 (past the top 10).
    pages = list(range(1, 20)) + [52]
    retriever = _FakeRetriever({"q?": pages})
    specs = ["recall@k", "recall@10"]

    base = run_tier1(golden, retriever, top_k=20, metric_specs=specs, default_k=20)
    reranked = run_tier1(
        golden,
        retriever,
        top_k=20,
        metric_specs=specs,
        default_k=20,
        reranker=_ReversingReranker(),
    )
    base_m = base["results"][0]["metrics"]
    rr_m = reranked["results"][0]["metrics"]

    # recall@k (full pool) is unchanged — the invariant.
    assert base_m["recall@k"] == 1.0
    assert rr_m["recall@k"] == 1.0
    # recall@10 is what reranking promoted: buried past rank 10, now surfaced.
    assert base_m["recall@10"] == 0.0
    assert rr_m["recall@10"] == 1.0
