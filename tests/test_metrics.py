import pytest

from ledgion.eval.metrics import (
    hit_at_1,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def test_nothing_relevant_retrieved():
    ranked = [1, 2, 3, 4, 5]
    relevant = {10, 20}

    assert recall_at_k(ranked, relevant, 5) == 0.0
    assert hit_at_1(ranked, relevant) == 0.0
    assert mrr(ranked, relevant) == 0.0
    assert ndcg_at_k(ranked, relevant, 5) == 0.0

def test_perfect_retrieval():
    ranked = [10, 20, 30]
    relevant = {10, 20, 30}

    assert recall_at_k(ranked, relevant, 3) == 1.0
    assert hit_at_1(ranked, relevant) == 1.0
    assert mrr(ranked, relevant) == 1.0
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(1.0)

def test_relevant_page_at_rank_one():
    ranked = [10, 1, 2, 3]
    relevant = {10}

    assert hit_at_1(ranked, relevant) == 1.0
    assert mrr(ranked, relevant) == 1.0

def test_relevant_page_at_rank_ten():
    ranked = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    relevant = {10}

    assert hit_at_1(ranked, relevant) == 0.0
    assert mrr(ranked, relevant) == pytest.approx(0.1)

def test_k_larger_than_ranked_list():
    ranked = [10, 20]
    relevant = {10, 20, 30}

    assert recall_at_k(ranked, relevant, 100) == pytest.approx(2 / 3)

def test_empty_ranked_list():
    ranked = []
    relevant = {10}

    assert recall_at_k(ranked, relevant, 5) == 0.0
    assert hit_at_1(ranked, relevant) == 0.0
    assert mrr(ranked, relevant) == 0.0
    assert ndcg_at_k(ranked, relevant, 5) == 0.0

@pytest.mark.parametrize(
    "metric_call",
    [
        lambda: recall_at_k([1, 2], set(), 5),
        lambda: hit_at_1([1, 2], set()),
        lambda: mrr([1, 2], set()),
        lambda: ndcg_at_k([1, 2], set(), 5),
    ],
)
def test_empty_relevant_raises(metric_call):
    with pytest.raises(ValueError):
        metric_call()

def test_partial_recall():
    ranked = [10, 99, 20, 98]
    relevant = {10, 20, 30, 40}

    assert recall_at_k(ranked, relevant, 4) == 0.5
