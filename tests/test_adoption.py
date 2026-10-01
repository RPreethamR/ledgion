"""The adoption rule (DECISIONS.md) as code — pure, offline, boundary-exact.

Deltas are computed on 6-decimal values and rounded to 6 decimals before comparison,
so a drop of exactly the tolerance passes condition 1 and a gain of exactly the
tolerance fails condition 2. recall@50 is guarded (condition 1) but cannot qualify a
config on its own (condition 2 is recall@10/ndcg@10 only).
"""

from __future__ import annotations

from ledgion.eval.adoption import evaluate, resolve_recall50, select_adopted

BASE = {"recall@10": 0.5, "ndcg@10": 0.3, "recall@50": 0.816667}


def test_recall50_regression_fails_condition1():
    # recall@50 -0.03 (past tolerance) even though ndcg@10 gains: condition 1 fails.
    v = evaluate({"recall@10": 0.5, "ndcg@10": 0.35, "recall@50": 0.786667}, BASE, top_k=50)
    assert not v.condition1
    assert not v.qualifies


def test_recall50_only_gain_fails_condition2():
    # Only recall@50 improves: no regression (condition 1 ok) but nothing recall@10/
    # ndcg@10 rose, so condition 2 fails — recall@50 can't qualify a config.
    v = evaluate({"recall@10": 0.5, "ndcg@10": 0.3, "recall@50": 0.9}, BASE, top_k=50)
    assert v.condition1
    assert not v.condition2
    assert not v.qualifies


def test_exactly_minus_002_passes_condition1():
    # A drop of exactly the tolerance is allowed; only *more than* tolerance regresses.
    v = evaluate({"recall@10": 0.48, "ndcg@10": 0.3, "recall@50": 0.816667}, BASE, top_k=50)
    assert v.condition1


def test_exactly_plus_002_fails_condition2():
    # A gain of exactly the tolerance is not enough; condition 2 needs *more than* it.
    v = evaluate({"recall@10": 0.52, "ndcg@10": 0.3, "recall@50": 0.816667}, BASE, top_k=50)
    assert not v.condition2


def test_two_qualifiers_larger_ndcg_selected():
    a = evaluate({"recall@10": 0.55, "ndcg@10": 0.33, "recall@50": 0.816667}, BASE, top_k=50)
    b = evaluate({"recall@10": 0.60, "ndcg@10": 0.35, "recall@50": 0.816667}, BASE, top_k=50)
    assert a.qualifies and b.qualifies
    assert select_adopted([("A", a), ("B", b)]) == "B"  # larger ndcg@10 gain wins


def test_ties_go_to_recall10():
    # Equal ndcg@10 gain -> the larger recall@10 gain breaks the tie.
    a = evaluate({"recall@10": 0.55, "ndcg@10": 0.35, "recall@50": 0.816667}, BASE, top_k=50)
    b = evaluate({"recall@10": 0.60, "ndcg@10": 0.35, "recall@50": 0.816667}, BASE, top_k=50)
    assert select_adopted([("A", a), ("B", b)]) == "B"


def test_recall_k_substitutes_recall50_only_at_top_k_50():
    # At top_k=50, recall@k IS recall@50.
    assert resolve_recall50({"recall@k": 0.8}, 50) == 0.8
    # At top_k=100, recall@k is recall@100 — not a substitute; recall@50 is missing.
    assert resolve_recall50({"recall@k": 0.88}, 100) is None
    v = evaluate({"recall@10": 0.5, "ndcg@10": 0.3, "recall@k": 0.88}, BASE, top_k=100)
    assert "recall@50" in v.missing
    assert not v.condition1  # a missing guarded metric can't be shown non-regressing
