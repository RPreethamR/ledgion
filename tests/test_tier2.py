"""Tier 2 caches every judge result by (qid, config_hash) so a re-run never
re-hits the quota-limited judge.

Fakes throughout — a fake retriever, a fake generator, a fake judge — so this
needs neither ragas nor a network. It proves the one thing that protects the
~1,500/day free-tier budget: the second run of a config is served entirely from
the disk cache.
"""

from __future__ import annotations

from ledgion.eval.tier2 import run_tier2
from ledgion.interfaces import Answer, Chunk, RetrievedChunk


def _ctx(page: int) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{page:016x}",
        doc_id="AMCOR_2023_10K",
        page_num=page,
        text=f"context text for page {page}",
        company="Amcor",
        ticker="AMCR",
        fiscal_year=2023,
        form_type="10-K",
    )
    return RetrievedChunk(chunk=chunk, score=1.0)


class _FakeRetriever:
    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        return [_ctx(50)]


class _FakeGenerator:
    def generate(self, question: str, contexts) -> Answer:
        contexts = list(contexts)
        # citation_validity 0.5 is the judge-free signal that must ride along.
        return Answer(
            question=question,
            text="Net sales were $14,694M.",
            citations=[("AMCOR_2023_10K", 50)],
            contexts=contexts,
            citation_validity=0.5,
        )


def test_tier2_caches_judge_by_qid_and_config_hash(tmp_path):
    golden = [{"qid": "financebench_id_00684", "question": "net sales?", "answer": "14,694"}]
    calls = {"n": 0}

    def judge(*, question, answer, contexts, reference):
        calls["n"] += 1
        return {"faithfulness": 0.8, "answer_correctness": 0.6}

    kwargs = dict(top_k=5, cache_dir=tmp_path, config_hash="deadbeef")
    first = run_tier2(golden, _FakeRetriever(), _FakeGenerator(), judge, **kwargs)
    second = run_tier2(golden, _FakeRetriever(), _FakeGenerator(), judge, **kwargs)

    # The judge ran exactly once; the second run was served from the cache.
    assert calls["n"] == 1
    assert first == second

    record = first["results"][0]
    assert record["faithfulness"] == 0.8
    assert record["answer_correctness"] == 0.6
    assert record["citation_validity"] == 0.5  # from the Answer, no judge call

    # aggregate means over the (single) question.
    assert first["metrics"]["faithfulness"] == 0.8
    assert first["metrics"]["answer_correctness"] == 0.6
    assert first["metrics"]["citation_validity"] == 0.5


def test_tier2_cache_key_separates_configs(tmp_path):
    golden = [{"qid": "q1", "question": "q?", "answer": "a"}]
    calls = {"n": 0}

    def judge(*, question, answer, contexts, reference):
        calls["n"] += 1
        return {"faithfulness": 1.0, "answer_correctness": 1.0}

    args = (golden, _FakeRetriever(), _FakeGenerator(), judge)
    run_tier2(*args, top_k=5, cache_dir=tmp_path, config_hash="hashA")
    run_tier2(*args, top_k=5, cache_dir=tmp_path, config_hash="hashB")

    # A different config_hash is a different cache key -> the judge runs again.
    assert calls["n"] == 2
