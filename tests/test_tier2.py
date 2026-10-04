"""Tier 2 orchestration + the content-addressed judgment cache.

Fakes throughout — a fake retriever, a fake generator, a fake judge — so this needs
neither ragas nor openai nor a network. It proves the guarantees that protect the
quota-limited judge and keep the numbers honest:

* a judge failure raises and nothing is written as NaN;
* an interrupted run resumes from the cache and judges only the remaining questions;
* refusals never reach the judge;
* an uncited answer scores 0 and is counted;
* the judgment cache keys on the judge signature (rubric/judge/ragas), not config;
* a truncated/failed judge response raises and isn't cached.
"""

from __future__ import annotations

import json

import pytest

from ledgion.config import load_config
from ledgion.eval.judge import (
    CorrectnessVerdict,
    FaithfulnessScore,
    judge_signature,
    parse_correctness_response,
)
from ledgion.eval.tier2 import run_tier2
from ledgion.interfaces import Answer, Chunk, RetrievedChunk

# -- fakes -------------------------------------------------------------------


def _ctx(page: int, *, text: str | None = None) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{page:016x}",
        doc_id="AMCOR_2023_10K",
        page_num=page,
        text=text or f"context text for page {page}",
        company="Amcor",
        ticker="AMCR",
        fiscal_year=2023,
        form_type="10-K",
    )
    return RetrievedChunk(chunk=chunk, score=1.0)


class _FakeRetriever:
    def __init__(self, contexts: list[RetrievedChunk]) -> None:
        self._contexts = contexts

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        return list(self._contexts[:top_k])


class _FakeGenerator:
    """Returns a scripted Answer per question. ``specs`` maps question -> dict with
    ``text``, ``cite_pages`` (pages to cite from the context), ``insufficient``."""

    def __init__(self, specs: dict[str, dict]) -> None:
        self.specs = specs

    def generate(self, question: str, contexts) -> Answer:
        spec = self.specs[question]
        contexts = list(contexts)
        cite_pages = set(spec.get("cite_pages", []))
        citations = [
            (c.chunk.doc_id, c.chunk.page_num) for c in contexts if c.chunk.page_num in cite_pages
        ]
        return Answer(
            question=question,
            text=spec["text"],
            citations=citations,
            contexts=contexts,
            insufficient_evidence=spec.get("insufficient", False),
            citation_validity=spec.get("validity", 1.0),
        )


class _FakeJudge:
    def __init__(self, verdicts: dict[str, str], *, faith: float = 0.9, fail_on: str | None = None):
        self.verdicts = verdicts
        self.faith = faith
        self.fail_on = fail_on
        self.correctness_calls: list[str] = []
        self.faithfulness_calls: list[str] = []

    def judge_correctness(self, *, question, gold_answer, answer) -> CorrectnessVerdict:
        self.correctness_calls.append(question)
        if self.fail_on == question:
            raise RuntimeError("simulated judge failure (e.g. truncated or daily cap)")
        return CorrectnessVerdict(self.verdicts[question], "because", 10, 5)

    def judge_faithfulness(self, *, question, answer, contexts) -> FaithfulnessScore:
        self.faithfulness_calls.append(question)
        return FaithfulnessScore(self.faith, 8, 4)


def _no_nan_in(cache_dir) -> None:
    # Scores are only ever a real number or null — never NaN/Infinity (which json.loads
    # would happily round-trip), so assert on the raw text.
    for path in cache_dir.glob("*.json"):
        raw = path.read_text(encoding="utf-8")
        assert "NaN" not in raw and "Infinity" not in raw, f"NaN/Infinity in cache: {path}"
        json.loads(raw)


# -- tests -------------------------------------------------------------------


def test_judge_failure_raises_and_writes_nothing_nan(tmp_path):
    golden = [{"qid": "q1", "question": "net sales?", "answer": "14,694", "answer_type": "numeric"}]
    gen = _FakeGenerator({"net sales?": {"text": "Net sales were $14,694M.", "cite_pages": [50]}})
    judge = _FakeJudge({"net sales?": "correct"}, fail_on="net sales?")

    with pytest.raises(SystemExit) as exc:
        run_tier2(
            golden,
            _FakeRetriever([_ctx(50)]),
            gen,
            judge,
            context_size=5,
            cache_dir=tmp_path,
            judge_signature="sigA",
        )
    assert "Scored 0/1" in str(exc.value)
    # Nothing cached for the failed question, and nothing NaN anywhere.
    assert not list(tmp_path.glob("*.json"))
    _no_nan_in(tmp_path)


def test_interrupted_run_resumes_from_cache(tmp_path):
    golden = [
        {"qid": "q1", "question": "qa", "answer": "a", "answer_type": "numeric"},
        {"qid": "q2", "question": "qb", "answer": "b", "answer_type": "prose"},
    ]
    gen = _FakeGenerator(
        {
            "qa": {"text": "answer a", "cite_pages": [50]},
            "qb": {"text": "answer b", "cite_pages": [51]},
        }
    )
    retriever = _FakeRetriever([_ctx(50), _ctx(51)])
    kwargs = dict(context_size=5, cache_dir=tmp_path, judge_signature="sigA")

    # First run fails on q2 -> q1 is judged and cached, then SystemExit.
    judge1 = _FakeJudge({"qa": "correct", "qb": "incorrect"}, fail_on="qb")
    with pytest.raises(SystemExit) as exc:
        run_tier2(golden, retriever, gen, judge1, **kwargs)
    assert "Scored 1/2" in str(exc.value)
    assert judge1.correctness_calls == ["qa", "qb"]  # qa succeeded, qb raised

    # Second run (judge no longer fails): q1 is served from cache, only q2 is judged.
    judge2 = _FakeJudge({"qa": "correct", "qb": "incorrect"})
    out = run_tier2(golden, retriever, gen, judge2, **kwargs)
    assert judge2.correctness_calls == ["qb"]  # q1 NOT re-judged
    assert out["completed"] == 2
    verdicts = {r["qid"]: r["correctness"] for r in out["results"]}
    assert verdicts == {"q1": "correct", "q2": "incorrect"}


def test_refusals_never_reach_the_judge(tmp_path):
    golden = [{"qid": "q1", "question": "q?", "answer": "a", "answer_type": "prose"}]
    gen = _FakeGenerator({"q?": {"text": "The filings do not answer this.", "insufficient": True}})
    judge = _FakeJudge({})  # empty: a judge call would KeyError, proving none happens

    out = run_tier2(
        golden, _FakeRetriever([_ctx(50)]), gen, judge,
        context_size=5, cache_dir=tmp_path, judge_signature="sigA",
    )
    assert judge.correctness_calls == [] and judge.faithfulness_calls == []
    record = out["results"][0]
    assert record["refused"] is True
    assert record["correctness"] is None and record["faithfulness"] is None
    # Refused counts as not correct in the report.
    corr = out["report"]["correctness"]["overall"]
    assert corr["refused"] == 1 and corr["correct"] == 0
    assert corr["pct_refused"] == 1.0 and corr["pct_correct"] == 0.0


def test_uncited_answer_scores_zero_and_is_counted(tmp_path):
    golden = [{"qid": "q1", "question": "q?", "answer": "a", "answer_type": "numeric"}]
    # Non-refusal, but cites nothing valid -> uncited.
    gen = _FakeGenerator({"q?": {"text": "Some answer with no citation.", "cite_pages": []}})
    judge = _FakeJudge({"q?": "correct"})

    out = run_tier2(
        golden, _FakeRetriever([_ctx(50)]), gen, judge,
        context_size=5, cache_dir=tmp_path, judge_signature="sigA",
    )
    # Correctness is still judged; faithfulness is 0 WITHOUT a ragas call.
    assert judge.correctness_calls == ["q?"]
    assert judge.faithfulness_calls == []
    record = out["results"][0]
    assert record["uncited"] is True and record["faithfulness"] == 0.0
    faith = out["report"]["faithfulness"]["overall"]
    assert faith["uncited"] == 1 and faith["n"] == 1 and faith["mean"] == 0.0


def test_judgment_cache_keys_on_signature_not_config(tmp_path):
    golden = [{"qid": "q1", "question": "q?", "answer": "a", "answer_type": "numeric"}]
    gen = _FakeGenerator({"q?": {"text": "answer", "cite_pages": [50]}})
    retriever = _FakeRetriever([_ctx(50)])

    judge = _FakeJudge({"q?": "correct"})
    base = dict(context_size=5, cache_dir=tmp_path)

    run_tier2(golden, retriever, gen, judge, judge_signature="sigA", **base)
    run_tier2(golden, retriever, gen, judge, judge_signature="sigA", **base)
    # Same signature (an unrelated config change doesn't touch it) -> served from cache.
    assert judge.correctness_calls == ["q?"]

    # A new signature (rubric/judge/ragas change) -> re-judged.
    run_tier2(golden, retriever, gen, judge, judge_signature="sigB", **base)
    assert judge.correctness_calls == ["q?", "q?"]


def test_judge_signature_tracks_rubric_and_judge_not_unrelated_config():
    base = load_config()
    sig = judge_signature(base, rubric_text="RUBRIC", ragas_ver="0.4.3")

    # An unrelated knob (fusion smoothing) -> SAME judge signature (no re-judge).
    unrelated = base.model_copy(update={"fusion": base.fusion.model_copy(update={"rrf_k": 999})})
    assert judge_signature(unrelated, rubric_text="RUBRIC", ragas_ver="0.4.3") == sig

    # Rubric text, judge model, and ragas version each change it.
    assert judge_signature(base, rubric_text="RUBRIC v2", ragas_ver="0.4.3") != sig
    jchg = base.model_copy(update={"judge": base.judge.model_copy(update={"model": "x"})})
    assert judge_signature(jchg, rubric_text="RUBRIC", ragas_ver="0.4.3") != sig
    assert judge_signature(base, rubric_text="RUBRIC", ragas_ver="0.4.9") != sig


def test_truncated_or_failed_judge_is_not_cached(tmp_path):
    # A truncated response surfaces as a RuntimeError before caching (same path as the
    # generator's MAX_TOKENS guard): the run stops and nothing is written for it.
    golden = [{"qid": "q1", "question": "q?", "answer": "a", "answer_type": "numeric"}]
    gen = _FakeGenerator({"q?": {"text": "answer", "cite_pages": [50]}})
    judge = _FakeJudge({"q?": "correct"}, fail_on="q?")

    with pytest.raises(SystemExit):
        run_tier2(
            golden, _FakeRetriever([_ctx(50)]), gen, judge,
            context_size=5, cache_dir=tmp_path, judge_signature="sigA",
        )
    assert not list(tmp_path.glob("*.json"))  # not cached


def test_parse_correctness_rejects_malformed():
    assert parse_correctness_response('{"verdict": "correct", "reason": "matches"}') == (
        "correct",
        "matches",
    )
    for bad in (
        None,
        "",
        "not json",
        '{"verdict": "maybe", "reason": "x"}',  # verdict out of range
        '{"verdict": "correct"}',  # missing reason
        '{"verdict": "correct", "reason": "   "}',  # empty reason
        '["correct"]',  # not an object
    ):
        with pytest.raises(RuntimeError):
            parse_correctness_response(bad)
