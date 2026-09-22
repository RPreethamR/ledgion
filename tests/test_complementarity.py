"""Complementarity analysis — pure, offline, over fabricated results dicts.

No files, no retrieval: build two tiny in-memory "results" reports with a known
overlap and check the 2x2 classification, that union recall can never dip below
either system, and that the consistency guard raises on a tampered file.
"""

from __future__ import annotations

import pytest

from ledgion.eval.complementarity import ConsistencyError, analyze

# Four one-page questions, two numeric and two prose.
_GOLDEN = [
    {"qid": "q1", "doc_id": "D1", "answer_type": "numeric", "evidence_pages": [10]},
    {"qid": "q2", "doc_id": "D2", "answer_type": "prose", "evidence_pages": [20]},
    {"qid": "q3", "doc_id": "D3", "answer_type": "numeric", "evidence_pages": [30]},
    {"qid": "q4", "doc_id": "D4", "answer_type": "prose", "evidence_pages": [40]},
]


def _row(qid, doc, atype, evidence, retrieved):
    return {
        "qid": qid,
        "doc_id": doc,
        "answer_type": atype,
        "evidence_pages": evidence,
        "retrieved_pages": [list(x) for x in retrieved],  # JSON-array form, as written
        "metrics": {},
    }


def _report(backend, top_k, results, recall_k):
    return {
        "config_hash": backend + "0000",
        "config": {"retrieval": {"backend": backend, "top_k": top_k}},
        "metrics": {"overall": {"recall@k": recall_k}},
        "results": results,
    }


def _runs(a_recall=0.5, b_recall=0.5):
    # A finds q1, q3 ; B finds q1, q2  =>  q1 both, q2 B-only, q3 A-only, q4 neither.
    a = _report(
        "dense",
        50,
        [
            _row("q1", "D1", "numeric", [10], [("D1", 10), ("D1", 11)]),
            _row("q2", "D2", "prose", [20], [("D2", 21)]),
            _row("q3", "D3", "numeric", [30], [("D3", 30)]),
            _row("q4", "D4", "prose", [40], [("D4", 41)]),
        ],
        a_recall,
    )
    b = _report(
        "sparse",
        50,
        [
            _row("q1", "D1", "numeric", [10], [("D1", 10)]),
            _row("q2", "D2", "prose", [20], [("D2", 20)]),
            _row("q3", "D3", "numeric", [30], [("D3", 31)]),
            _row("q4", "D4", "prose", [40], [("D4", 42)]),
        ],
        b_recall,
    )
    return a, b


def test_known_overlap_produces_expected_2x2():
    a, b = _runs()
    result = analyze(_GOLDEN, a, b)

    assert result.counts["overall"] == {"both": 1, "A only": 1, "B only": 1, "neither": 1}
    assert result.counts["numeric"] == {"both": 1, "A only": 1, "B only": 0, "neither": 0}
    assert result.counts["prose"] == {"both": 0, "A only": 0, "B only": 1, "neither": 1}

    by_qid = {p.qid: p for p in result.pairs}
    assert by_qid["q1"].cell == "both" and by_qid["q1"].rank_a == 1 and by_qid["q1"].rank_b == 1
    assert by_qid["q3"].cell == "A only" and by_qid["q3"].rank_b is None
    assert by_qid["q2"].cell == "B only" and by_qid["q2"].rank_a is None
    assert by_qid["q4"].cell == "neither"


def test_union_recall_never_below_either_system():
    a, b = _runs()
    result = analyze(_GOLDEN, a, b)
    # A finds 2/4, B finds 2/4, union finds 3/4 (only q4 is missed by both).
    assert result.recall_a == 0.5
    assert result.recall_b == 0.5
    assert result.recall_union == 0.75
    assert result.recall_union >= result.recall_a
    assert result.recall_union >= result.recall_b


def test_consistency_check_fails_loudly_on_tampered_file():
    # Tamper A's recorded recall@k so it no longer matches the classification.
    a, b = _runs(a_recall=0.9)
    with pytest.raises(ConsistencyError, match="MISMATCH"):
        analyze(_GOLDEN, a, b)


def test_multi_page_evidence_counts_each_pair():
    # A question with two evidence pages is two units; recall is per-question (macro).
    golden = [{"qid": "m", "doc_id": "D", "answer_type": "numeric", "evidence_pages": [1, 2]}]
    # A finds page 1 only -> question recall 0.5; B finds neither.
    a = _report("dense", 50, [_row("m", "D", "numeric", [1, 2], [("D", 1)])], 0.5)
    b = _report("sparse", 50, [_row("m", "D", "numeric", [1, 2], [("D", 9)])], 0.0)
    result = analyze(golden, a, b)
    assert len(result.pairs) == 2
    assert result.counts["overall"] == {"both": 0, "A only": 1, "B only": 0, "neither": 1}
    assert result.recall_a == 0.5
    assert result.recall_union == 0.5  # B added nothing
