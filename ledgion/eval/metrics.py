import math


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant pages appearing in the top k."""
    if not relevant:
        raise ValueError("relevant must not be empty")
    if k <= 0:
        raise ValueError("k must be greater than 0")
    
    retrieved_relevant = sum(
        1 for page in ranked[:k] if page in relevant
    )

    return retrieved_relevant / len(relevant)


def hit_at_1(ranked: list[int], relevant: set[int]) -> float:
    """1.0 if the top-ranked page is relevant, else 0.0."""
    if not relevant:
        raise ValueError("relevant must not be empty")

    if not ranked:
        return 0.0

    return 1.0 if ranked[0] in relevant else 0.0


def mrr(ranked: list[int], relevant: set[int]) -> float:
    """Reciprocal rank of the first relevant page. 0.0 if none."""
    if not relevant:
        raise ValueError("relevant must not be empty")

    for rank, page in enumerate(ranked, start=1):
        if page in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    """Binary-gain nDCG. IDCG is the ideal ordering given |relevant|."""
    if not relevant:
        raise ValueError("relevant must not be empty")
    if k <= 0:
        raise ValueError("k must be greater than 0")

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, page in enumerate(ranked[:k])
        if page in relevant
    )

    ideal_relevant_count = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(i + 2)
        for i in range(ideal_relevant_count)
    )

    return dcg / idcg