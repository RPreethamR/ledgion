"""The Tier 2 LLM judge: Groq's gpt-oss over an OpenAI-compatible endpoint.

Two signals, two mechanisms, one model:

* **correctness** — a binary, FinanceBench-style verdict (``correct``/``incorrect``
  + a one-sentence reason) produced by a *direct* structured-output call, graded
  against the versioned rubric in ``config/judge/correctness_rubric.md``.
* **faithfulness** — ragas's ``Faithfulness`` metric, scored against the text of the
  chunks the answer *validly cited* (not everything retrieved). Ragas is a heavy,
  approval-gated dependency (the ``tier2`` group), imported lazily here.

gpt-oss is a reasoning model, so hidden reasoning counts against both the Groq
token caps and ``max_tokens``. A response the limit cut off is **raised, not
cached** — the same guard as the Phase-3 generator — so a truncated verdict can
never be mistaken for a real one. Every judge call logs its prompt and completion
tokens and returns them, so the orchestration can total the run's token spend and
stop cleanly when the daily cap is hit.

This module is import-safe without ragas/openai: the dataclasses, the ``Judge``
protocol, the rubric helpers, and ``parse_correctness_response`` are pure, so the
Tier 2 tests exercise the orchestration with a fake judge and no network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ledgion.config import REPO_ROOT, Settings

logger = logging.getLogger(__name__)

_VALID_VERDICTS = ("correct", "incorrect")
# HTTP status codes worth retrying (rate limit + server overload); mirrors the Gemini
# generator. A daily-cap 429 simply exhausts the retry budget and then stops the run.
_TRANSIENT_CODES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class CorrectnessVerdict:
    """One binary correctness judgment plus the tokens it cost."""

    verdict: str  # "correct" | "incorrect"
    reason: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class FaithfulnessScore:
    """One ragas faithfulness score (0..1) plus the tokens it cost."""

    score: float
    prompt_tokens: int
    completion_tokens: int


@runtime_checkable
class Judge(Protocol):
    """The Tier 2 judge seam. The orchestration (``run_tier2``) depends only on this,
    so tests inject a fake and make no Groq call."""

    def judge_correctness(
        self, *, question: str, gold_answer: str, answer: str
    ) -> CorrectnessVerdict: ...

    def judge_faithfulness(
        self, *, question: str, answer: str, contexts: Sequence[str]
    ) -> FaithfulnessScore: ...


# -- rubric + provenance helpers (pure) --------------------------------------


def rubric_hash(text: str) -> str:
    """Stable hash of the rubric text — part of the judgment cache key and provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_rubric(cfg: Settings) -> str:
    """Read the correctness rubric. Resolved against the repo root (like docling_dir)
    so it is found regardless of the working directory."""
    path = Path(cfg.judge.rubric_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise SystemExit(f"correctness rubric not found: {path} (set judge.rubric_path)")
    return path.read_text(encoding="utf-8")


def ragas_version() -> str:
    """The installed ragas version string (part of the judgment cache key).

    Lazy: importing this module must not require ragas. A change here invalidates
    cached faithfulness scores, because a ragas upgrade can change how faithfulness
    is computed.
    """
    from importlib.metadata import version

    return version("ragas")


def judge_signature(cfg: Settings, *, rubric_text: str, ragas_ver: str) -> str:
    """Fold the judge's identity into one hash for the judgment cache key.

    Includes the judge model and every setting that changes a verdict, plus the rubric
    hash (so editing the rubric re-judges) and the ragas version (so a ragas upgrade
    re-judges faithfulness). Deliberately excludes ``config_hash``: an unrelated knob
    change must not burn tokens re-judging identical answers.
    """
    j = cfg.judge
    payload = json.dumps(
        {
            "model": j.model,
            "base_url": j.base_url,
            "temperature": j.temperature,
            "reasoning_effort": j.reasoning_effort,
            "max_tokens": j.max_tokens,
            "rubric_hash": rubric_hash(rubric_text),
            "ragas_version": ragas_ver,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_correctness_response(content: str | None) -> tuple[str, str]:
    """Parse the judge's JSON into ``(verdict, reason)``, raising on anything malformed.

    A malformed verdict (not JSON, missing keys, a verdict outside the allowed set, or
    an empty reason) raises ``RuntimeError`` so the caller stops and nothing is cached —
    a guessed verdict is worse than a halted run. Pure, so it is unit-tested with no
    network.
    """
    if not content:
        raise RuntimeError("judge returned empty correctness content")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"judge correctness output was not JSON: {content!r}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"judge correctness output was not an object: {content!r}")
    verdict = data.get("verdict")
    reason = data.get("reason")
    if verdict not in _VALID_VERDICTS:
        raise RuntimeError(
            f"judge verdict must be one of {_VALID_VERDICTS}, got {verdict!r}: {content!r}"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError(f"judge correctness reason missing/empty: {content!r}")
    return verdict, reason.strip()


# -- the correctness prompt --------------------------------------------------


def build_correctness_prompt(
    rubric_text: str, *, question: str, gold_answer: str, answer: str
) -> list[dict]:
    """Assemble the chat messages for one binary correctness judgment.

    The gold answer and the answer-under-test are clearly delimited so the model grades
    the latter against the former under the rubric. The rubric already specifies the
    required JSON output shape.
    """
    system = (
        "You are a careful grader for a financial-filings QA system. "
        "Apply the rubric exactly and output only the requested JSON.\n\n" + rubric_text
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Gold answer (ground truth):\n{gold_answer}\n\n"
        f"Answer to grade:\n{answer}\n\n"
        'Return only: {"verdict": "correct"|"incorrect", "reason": "<one sentence>"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# -- the real Groq judge (heavy, approval-gated: ragas + openai) --------------


class GroqJudge:
    """A ``Judge`` over Groq's gpt-oss. Correctness is a direct structured call;
    faithfulness is ragas. Only built for a real Tier 2 run — not in the tests."""

    def __init__(self, cfg: Settings, rubric_text: str) -> None:
        self.cfg = cfg.judge
        self.rubric_text = rubric_text
        self._client = None  # openai client, built lazily
        self._faith_metric = None  # ragas Faithfulness metric, built lazily
        self._faith_llm = None  # ragas-wrapped judge LLM, built lazily

    # -- correctness: a direct OpenAI-compatible structured-output call -------

    def judge_correctness(
        self, *, question: str, gold_answer: str, answer: str
    ) -> CorrectnessVerdict:
        messages = build_correctness_prompt(
            self.rubric_text, question=question, gold_answer=gold_answer, answer=answer
        )
        content, prompt_tokens, completion_tokens = self._with_retries(
            lambda: self._correctness_once(messages)
        )
        verdict, reason = parse_correctness_response(content)
        logger.info(
            "judge correctness: verdict=%s prompt_tokens=%d completion_tokens=%d",
            verdict,
            prompt_tokens,
            completion_tokens,
        )
        return CorrectnessVerdict(verdict, reason, prompt_tokens, completion_tokens)

    def _correctness_once(self, messages: list[dict]) -> tuple[str | None, int, int]:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            response_format={"type": "json_object"},
            # reasoning_effort is a Groq extension, sent through the OpenAI SDK's escape
            # hatch so the typed client doesn't reject it.
            extra_body={"reasoning_effort": self.cfg.reasoning_effort},
        )
        choice = resp.choices[0]
        # Same guard as the generator: a limit cutoff is truncated output — raise so it
        # is neither parsed nor cached. gpt-oss reasoning can eat the whole budget.
        if choice.finish_reason == "length":
            raise RuntimeError(
                "judge response truncated (finish_reason=length). Raise judge.max_tokens "
                "or lower judge.reasoning_effort."
            )
        usage = resp.usage
        return (
            choice.message.content,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    # -- faithfulness: ragas, against the validly-cited chunk texts -----------

    def judge_faithfulness(
        self, *, question: str, answer: str, contexts: Sequence[str]
    ) -> FaithfulnessScore:
        # NOTE (verify against installed ragas==0.4.3 before the first Groq run): this is
        # the stable ragas metrics path — a one-row EvaluationDataset scored by evaluate()
        # with a RunConfig and raise_exceptions=True (so an error raises, never NaN), and
        # a token-usage parser so faithfulness's internal calls are counted too. If a
        # faithfulness sub-call truncates, ragas's structured parse fails and (with
        # raise_exceptions) propagates — so a cut-off judgment is not cached.
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.cost import get_token_usage_for_openai
        from ragas.run_config import RunConfig

        metric = self._ensure_faithfulness()
        sample = SingleTurnSample(
            user_input=question, response=answer, retrieved_contexts=list(contexts)
        )
        try:
            result = evaluate(
                dataset=EvaluationDataset(samples=[sample]),
                metrics=[metric],
                llm=self._faith_llm,
                run_config=RunConfig(
                    max_workers=self.cfg.max_workers,
                    max_retries=self.cfg.max_retries,
                    max_wait=self.cfg.max_wait,
                ),
                raise_exceptions=True,
                token_usage_parser=get_token_usage_for_openai,
                show_progress=False,
            )
            score = float(result["faithfulness"][0])
        except Exception as exc:  # ragas/openai failure (incl. a truncated sub-call)
            # Normalise to RuntimeError so the orchestration stops cleanly and caches
            # nothing — a re-run resumes from the content-addressed cache.
            raise RuntimeError(f"ragas faithfulness failed: {exc}") from exc
        if math.isnan(score):
            # raise_exceptions=True should prevent this, but never let a NaN be cached.
            raise RuntimeError("ragas faithfulness returned NaN")
        usage = result.total_tokens()
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        logger.info(
            "judge faithfulness: score=%.4f prompt_tokens=%d completion_tokens=%d",
            score,
            prompt_tokens,
            completion_tokens,
        )
        return FaithfulnessScore(score, prompt_tokens, completion_tokens)

    # -- lazy clients --------------------------------------------------------

    def _api_key(self) -> str:
        key = os.environ.get(self.cfg.api_key_env)
        if not key:
            raise SystemExit(
                f"no judge API key: set the env var named by judge.api_key_env "
                f"({self.cfg.api_key_env})"
            )
        return key

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self._api_key())
        return self._client

    def _ensure_faithfulness(self):
        if self._faith_metric is None:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import Faithfulness

            chat = ChatOpenAI(
                model=self.cfg.model,
                base_url=self.cfg.base_url,
                api_key=self._api_key(),
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                # reasoning_effort is a Groq extension; ChatOpenAI forwards extra_body to
                # the API call (passed explicitly, not via model_kwargs, which it warns on).
                extra_body={"reasoning_effort": self.cfg.reasoning_effort},
            )
            self._faith_llm = LangchainLLMWrapper(chat)
            self._faith_metric = Faithfulness(llm=self._faith_llm)
        return self._faith_metric

    def _with_retries(self, call):
        """Bounded exponential backoff on transient errors for the direct correctness
        call (ragas handles its own retries for faithfulness). A daily-cap 429 exhausts
        the budget and raises, which stops the run; a re-run resumes from the cache."""
        from openai import APIError, APIStatusError

        for attempt in range(self.cfg.max_retries + 1):
            try:
                return call()
            except APIStatusError as exc:
                code = getattr(exc, "status_code", None)
                if code not in _TRANSIENT_CODES or attempt == self.cfg.max_retries:
                    raise RuntimeError(f"judge API error ({code}): {exc}") from exc
                delay = min(2.0 * (2**attempt), float(self.cfg.max_wait))
                logger.warning(
                    "judge transient error %s; retrying in %.1fs (attempt %d/%d)",
                    code,
                    delay,
                    attempt + 1,
                    self.cfg.max_retries,
                )
                time.sleep(delay)
            except APIError as exc:
                # Connection/timeout with no HTTP status — retry a bounded number of times.
                if attempt == self.cfg.max_retries:
                    raise RuntimeError(f"judge API error: {exc}") from exc
                time.sleep(min(2.0 * (2**attempt), float(self.cfg.max_wait)))
        raise RuntimeError("judge retry loop exited without returning")


def build_judge(cfg: Settings, rubric_text: str) -> GroqJudge:
    """Build the real Groq judge for a Tier 2 run."""
    return GroqJudge(cfg, rubric_text)
