"""Tier 2: the judged tier. Never gates a build; off by default; never runs in CI.

Tier 2 generates an answer for every golden question and scores it on four signals:

* **refusal** — deterministic, from the generator's ``insufficient_evidence`` flag. A
  refusal is never sent to the judge (it would waste tokens on a non-answer) and is
  counted as *not correct*, matching how FinanceBench reports accuracy.
* **citation_validity** — the deterministic, judge-free Phase-3 rate (fraction of the
  model's cited chunk_ids that were real), carried on the ``Answer``.
* **faithfulness** — ragas's ``Faithfulness``, scored against the text of the chunks
  the answer *validly cited* (not everything retrieved). A non-refusal answer with no
  valid citations scores 0 and is counted separately as *uncited* (no judge call).
* **correctness** — a binary FinanceBench-style verdict from the judge, graded against
  the versioned rubric, for every non-refusal.

**Quota discipline (the binding constraint).** The Groq free tier caps tokens hard
(8k/min, 200k/day), so nothing is ever wasted or silently lost:

* Questions are scored **one at a time** and each judgment is cached **the moment it
  lands**. If a judge call fails after retries — including hitting the daily cap — the
  run stops with a message saying how many are scored; a re-run resumes from the cache.
* The judgment cache is **content-addressed**: keyed on *what was judged* (question,
  gold answer, generated answer text, cited chunk texts) plus a ``judge_signature``
  that folds the judge model/settings, the rubric hash, and the ragas version — **not**
  ``config_hash``. So editing the rubric re-judges, a ragas upgrade re-judges, but an
  unrelated config change never burns a token re-judging an identical answer.
* **No NaN, ever.** A judge error raises (it is never written as a score), means are
  never taken over a shrunken denominator, and every mean is reported beside its ``n``.

The judge is injected (the ``Judge`` protocol in ``judge.py``), so the orchestration
and cache are exercised with a fake judge and no ragas, no openai, and no network.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from ledgion.eval.judge import Judge
from ledgion.interfaces import Answer, Generator, Retriever

_GROUPS = ("numeric", "prose")


# -- hashing: answer identity + the content-addressed judgment cache key ------


def answer_hash(text: str) -> str:
    """Stable hash of an answer's text — the join key for blind hand-grading."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _judgment_key(
    *, question: str, gold: str, answer_text: str, cited_texts: Sequence[str], judge_signature: str
) -> str:
    # NUL-separated so field boundaries can't be forged by concatenation. The cited
    # texts (and their count) are part of the key because faithfulness is scored against
    # exactly them: a different valid-citation set is a different judgment.
    parts = [question, gold, answer_text, judge_signature, str(len(cited_texts)), *cited_texts]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def _load_cached(cache_dir: Path, key: str) -> dict | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupt entry is a miss, not a crash — re-judge it.
        return None


def _store_cached(cache_dir: Path, key: str, record: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(
        json.dumps(record, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )


# -- per-answer helpers ------------------------------------------------------


def cited_chunk_texts(answer: Answer) -> list[str]:
    """The text of the chunks the answer *validly* cited.

    Citations are page-level validated provenance ((doc_id, page_num) pairs), so a cited
    chunk is any context chunk on a validly-cited page. This is the faithfulness context
    — the evidence the answer claims to rest on, not the whole retrieved pool. Empty iff
    the answer cited nothing valid (an *uncited* answer).
    """
    cited_pages = set(answer.citations)
    return [
        rc.chunk.text
        for rc in answer.contexts
        if (rc.chunk.doc_id, rc.chunk.page_num) in cited_pages
    ]


def _judge_one(row: dict, answer: Answer, cited_texts: list[str], judge: Judge) -> dict:
    """Produce one complete per-question judgment record. May raise (judge failure)."""
    record = {
        "qid": row["qid"],
        "answer_type": row["answer_type"],
        "question": row["question"],
        "gold_answer": row["answer"],
        "answer_text": answer.text,
        "answer_hash": answer_hash(answer.text),
        "cited_pages": [[doc_id, page] for doc_id, page in answer.citations],
        "citation_validity": answer.citation_validity,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    if answer.insufficient_evidence:
        # Refusal: deterministic, never judged, counted as not correct.
        record.update(
            refused=True,
            uncited=False,
            faithfulness=None,
            correctness=None,
            correctness_reason=None,
        )
        return record

    # Non-refusal: correctness for every one; faithfulness only when validly cited.
    verdict = judge.judge_correctness(
        question=row["question"], gold_answer=row["answer"], answer=answer.text
    )
    prompt_tokens = verdict.prompt_tokens
    completion_tokens = verdict.completion_tokens

    uncited = not cited_texts
    if uncited:
        # No valid citations -> faithfulness 0 by definition, no ragas call.
        faithfulness = 0.0
    else:
        fs = judge.judge_faithfulness(
            question=row["question"], answer=answer.text, contexts=cited_texts
        )
        faithfulness = fs.score
        prompt_tokens += fs.prompt_tokens
        completion_tokens += fs.completion_tokens

    record.update(
        refused=False,
        uncited=uncited,
        faithfulness=faithfulness,
        correctness=verdict.verdict,
        correctness_reason=verdict.reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return record


# -- orchestration -----------------------------------------------------------


def run_tier2(
    golden_rows: Sequence[dict],
    retriever: Retriever,
    generator: Generator,
    judge: Judge,
    *,
    context_size: int,
    cache_dir: Path,
    judge_signature: str,
) -> dict:
    """Generate + judge every golden row one at a time, caching each judgment as it lands.

    Returns ``{"results": [...], "report": {...}, "completed": n, "total": N}``. On a judge
    failure after retries (including the daily token cap) it stops with ``SystemExit``
    stating how many were scored; already-cached judgments persist, so a re-run resumes.
    """
    cache_dir = Path(cache_dir)
    records: list[dict] = []
    completed = 0
    total = len(golden_rows)

    for row in golden_rows:
        try:
            # Retrieve exactly the chunks the generator should see, then generate.
            # Generation is cached on its own content-addressed key, so a re-run spends
            # no generation quota here even before we reach the judge.
            contexts = retriever.retrieve(row["question"], top_k=context_size)
            answer = generator.generate(row["question"], contexts)
            cited_texts = cited_chunk_texts(answer)

            key = _judgment_key(
                question=row["question"],
                gold=row["answer"],
                answer_text=answer.text,
                cited_texts=cited_texts,
                judge_signature=judge_signature,
            )
            record = _load_cached(cache_dir, key)
            if record is None:
                record = _judge_one(row, answer, cited_texts, judge)
                _store_cached(cache_dir, key, record)
        except RuntimeError as exc:
            # A quota/transient failure after retries — from the GENERATOR (e.g. the
            # Gemini daily cap) or the JUDGE (the Groq cap, a truncated response, or a
            # malformed verdict). Stop cleanly: nothing partial or NaN is written for this
            # question, and everything before it is cached, so a re-run resumes.
            raise SystemExit(
                f"Tier 2 stopped on {row['qid']}: {exc}\n"
                f"Scored {completed}/{total} questions; cached progress persists — "
                f"rerun to resume from the cache."
            ) from exc
        records.append(record)
        completed += 1

    return {
        "results": records,
        "report": aggregate_tier2(records),
        "completed": completed,
        "total": total,
    }


# -- aggregation / reporting -------------------------------------------------


def _round(value: float, n: int) -> float:
    return round(value / n, 6)


def _group_correctness(rows: Sequence[dict]) -> dict:
    """Percent correct / incorrect / refused over ALL rows in the group (refusals count
    as not correct, per FinanceBench). ``n`` is always the full group size."""
    n = len(rows)
    correct = sum(1 for r in rows if r["correctness"] == "correct")
    incorrect = sum(1 for r in rows if r["correctness"] == "incorrect")
    refused = sum(1 for r in rows if r["refused"])
    return {
        "n": n,
        "correct": correct,
        "incorrect": incorrect,
        "refused": refused,
        "pct_correct": _round(correct, n) if n else None,
        "pct_incorrect": _round(incorrect, n) if n else None,
        "pct_refused": _round(refused, n) if n else None,
    }


def _group_faithfulness(rows: Sequence[dict]) -> dict:
    """Mean faithfulness over non-refusals (uncited answers contribute 0), with ``n`` and
    the separate uncited count. Never averaged over refusals or a shrunken denominator."""
    scored = [r for r in rows if not r["refused"]]
    uncited = sum(1 for r in scored if r["uncited"])
    n = len(scored)
    mean = _round(sum(r["faithfulness"] for r in scored), n) if n else None
    return {"mean": mean, "n": n, "uncited": uncited}


def _group_citation_validity(rows: Sequence[dict]) -> dict:
    """Mean citation validity over non-refusals (the answers that actually attempted
    citations), with ``n``. Refusals cite nothing, so including them would mislead."""
    scored = [r for r in rows if not r["refused"]]
    n = len(scored)
    mean = _round(sum(r["citation_validity"] for r in scored), n) if n else None
    return {"mean": mean, "n": n}


def aggregate_tier2(records: Sequence[dict]) -> dict:
    """Assemble the Tier 2 report: correctness / faithfulness / citation validity, overall
    and split by numeric/prose, each with its ``n``, plus the run's total judge tokens."""
    groups: dict[str, list[dict]] = {"overall": list(records)}
    for group in _GROUPS:
        groups[group] = [r for r in records if r["answer_type"] == group]

    report: dict = {
        "counts": {name: len(rows) for name, rows in groups.items()},
        "correctness": {},
        "faithfulness": {},
        "citation_validity": {},
    }
    for name, rows in groups.items():
        if not rows:
            continue  # empty group omitted from the metric tables; its count still shows
        report["correctness"][name] = _group_correctness(rows)
        report["faithfulness"][name] = _group_faithfulness(rows)
        report["citation_validity"][name] = _group_citation_validity(rows)

    prompt = sum(r["prompt_tokens"] for r in records)
    completion = sum(r["completion_tokens"] for r in records)
    report["tokens"] = {
        "prompt": prompt,
        "completion": completion,
        "total": prompt + completion,
    }
    return report
