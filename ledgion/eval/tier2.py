"""Tier 2: the judged tier. Never gates a build; off by default.

Tier 2 adds three per-question signals on top of Tier 1's retrieval metrics:

* **faithfulness** and **answer_correctness** — Ragas judges (an LLM scores
  whether the answer is grounded in the retrieved context, and how well it
  matches the reference answer). These cost judge calls.
* **citation_validity** — the deterministic, judge-free rate from Phase 3
  (the fraction of the model's cited chunk_ids that were actually in its
  context). It rides along on the ``Answer`` for free.

**Quota discipline.** The Gemini free tier is ~1,500 requests/day and a sweep
multiplies (questions × judge metrics) fast, so *every* judged record is cached
on disk keyed by ``(qid, config_hash)``: a re-run of the same config over the
same question reads the cached scores and makes no judge call. Generation is
already cached by ``GeminiGenerator``; this caches the judging on top of it, so a
warm sweep spends zero quota.

Ragas is imported lazily inside ``ragas_judge`` only — importing this module, and
the cache-path tests, need neither ragas nor a network. The judge is passed in as
a callable, so the orchestration and cache are exercised with a fake.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from ledgion.interfaces import Generator, Retriever

# A judge maps one (question, answer, contexts, reference) to a scores dict with
# at least the keys in ``_JUDGE_METRICS``.
Judge = Callable[..., dict]

_JUDGE_METRICS = ("faithfulness", "answer_correctness")
# citation_validity is judge-free (from the Answer) but reported in the same tier.
_TIER2_METRICS = (*_JUDGE_METRICS, "citation_validity")


# -- (qid, config_hash) disk cache -------------------------------------------


def _cache_key(qid: str, config_hash: str) -> str:
    # NUL separator so (qid, hash) can't be forged by concatenation — same
    # rationale as the generation/embedding cache keys elsewhere.
    return hashlib.sha256(f"{qid}\x00{config_hash}".encode()).hexdigest()


def _load_cached(cache_dir: Path, qid: str, config_hash: str) -> dict | None:
    path = cache_dir / f"{_cache_key(qid, config_hash)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupt/unreadable entry is a miss, not a crash — re-judge it.
        return None


def _store_cached(cache_dir: Path, qid: str, config_hash: str, record: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_cache_key(qid, config_hash)}.json"
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")


# -- orchestration -----------------------------------------------------------


def run_tier2(
    golden_rows: Sequence[dict],
    retriever: Retriever,
    generator: Generator,
    judge: Judge,
    *,
    top_k: int,
    cache_dir: Path,
    config_hash: str,
) -> dict:
    """Generate + judge every golden row, caching each record by (qid, config_hash).

    Returns ``{"results": [per-question…], "metrics": {mean per Tier 2 metric}}``.
    On a cache hit the retriever, generator, and judge are all skipped for that row.
    """
    cache_dir = Path(cache_dir)
    results: list[dict] = []
    for row in golden_rows:
        qid = row["qid"]
        record = _load_cached(cache_dir, qid, config_hash)
        if record is None:
            contexts = retriever.retrieve(row["question"], top_k=top_k)
            answer = generator.generate(row["question"], contexts)
            scores = judge(
                question=row["question"],
                answer=answer.text,
                contexts=[rc.chunk.text for rc in contexts],
                reference=row["answer"],
            )
            record = {
                "qid": qid,
                "faithfulness": scores["faithfulness"],
                "answer_correctness": scores["answer_correctness"],
                # judge-free faithfulness signal from Phase 3.
                "citation_validity": answer.citation_validity,
            }
            _store_cached(cache_dir, qid, config_hash, record)
        results.append(record)

    metrics = (
        {name: round(sum(r[name] for r in results) / len(results), 6) for name in _TIER2_METRICS}
        if results
        else {}
    )
    return {"results": results, "metrics": metrics}


# -- the real Ragas judge (heavy, approval-gated dependency) ------------------


def ragas_judge(cfg) -> Judge:
    """Build the real Ragas judge (faithfulness + answer_correctness).

    Deliberately unwired: Ragas pulls in LangChain + datasets (a >500MB install),
    which CLAUDE.md requires be approved before adding. This is the single seam to
    complete once that dependency lands — the closure it returns must accept
    ``(question, answer, contexts, reference)`` keyword args and return a dict with
    ``faithfulness`` and ``answer_correctness`` floats, exactly like the fakes in
    ``tests/test_tier2.py``. The judge model is ``cfg.eval.judge_model`` (Gemini).

    Until then the orchestration, the (qid, config_hash) cache, and the
    citation_validity aggregation above are all live and tested with a fake judge.
    """
    raise NotImplementedError(
        "ragas_judge is not wired: installing ragas is a >500MB, approval-gated "
        "dependency (see CLAUDE.md). Enable eval.ragas_enabled only after it is added "
        "and this factory is completed."
    )
