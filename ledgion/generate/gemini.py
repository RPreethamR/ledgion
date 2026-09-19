"""Gemini Flash generator with a disk response cache.

Implements the ``Generator`` contract (interfaces.py): question + retrieved
context -> grounded ``Answer``. Three things matter here.

* **Structured output.** The model is constrained to JSON
  ``{"answer": str, "citations": [chunk_id], "insufficient_evidence": bool}`` via
  a response schema, so decoding is total and the citation list is
  machine-checkable rather than scraped from prose.
* **Determinism.** ``temperature=0`` (from config), and never a local model in
  the serving path (CLAUDE.md) — generation is a Gemini API call.
* **Cache before network.** Responses are cached on disk keyed by
  ``sha256(question + config_hash + retrieved_chunk_ids)``. A re-run with the
  same question, config, and retrieved context reads the cached JSON and makes
  **no** API call — so eval sweeps and repeated ``ask``s don't burn quota. The
  key includes ``config_hash`` because a different model / temperature / prompt is
  a different answer, and the chunk_ids because an answer is only valid for the
  exact context it saw.

After decoding, citations are validated against the context actually sent (see
citations.py): fabricated ids are dropped, kept ids are resolved to their
(doc_id, page_num), and the validity rate rides along on the ``Answer``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel

from ledgion.config import Settings
from ledgion.config import config_hash as compute_config_hash
from ledgion.generate.citations import validate_citations
from ledgion.generate.prompt import SYSTEM_INSTRUCTION, build_user_prompt, context_chunk_ids
from ledgion.interfaces import Answer, RetrievedChunk

logger = logging.getLogger(__name__)

# HTTP status codes worth retrying: rate limit + server-side/overload. Everything
# else (bad key 401/403, unknown model 404, malformed request 400) is a permanent
# failure that retrying only delays.
_TRANSIENT_CODES = frozenset({429, 500, 502, 503, 504})


class _AnswerSchema(BaseModel):
    """The structured-output contract. Passed to the API as ``response_schema``
    and mirrored by the JSON we cache/parse. Defined with pydantic (already a
    dependency) so importing this module needs no genai import."""

    answer: str
    citations: list[str]
    insufficient_evidence: bool


class GeminiGenerator:
    """A ``Generator`` over Gemini Flash + a disk response cache."""

    def __init__(
        self,
        *,
        model: str,
        config_hash: str,
        cache_dir: Path,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        thinking_budget: int | None = 0,
        max_retries: int = 5,
        retry_base_delay_s: float = 2.0,
    ) -> None:
        self.model = model
        self.config_hash = config_hash
        self.cache_dir = Path(cache_dir)
        self.api_key = api_key
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s
        self._client = None
        # Incremented only on a real request; tests assert generation stays at 0
        # extra calls once a response is cached.
        self.api_calls = 0

    @classmethod
    def from_config(cls, cfg: Settings) -> GeminiGenerator:
        # config_hash folds every knob (model, temperature, chunking, etc.) into
        # the cache key: change any of them and cached answers are bypassed.
        return cls(
            model=cfg.generation.model,
            config_hash=compute_config_hash(cfg),
            cache_dir=Path(cfg.paths.cache_dir) / "generation",
            api_key=os.environ.get(cfg.generation.api_key_env),
            temperature=cfg.generation.temperature,
            max_output_tokens=cfg.generation.max_output_tokens,
            thinking_budget=cfg.generation.thinking_budget,
            max_retries=cfg.generation.max_retries,
            retry_base_delay_s=cfg.generation.retry_base_delay_s,
        )

    # -- Generator contract --------------------------------------------------

    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> Answer:
        contexts = list(contexts)
        chunk_ids = context_chunk_ids(contexts)

        raw = self._cached_or_call(question, contexts, chunk_ids)
        data = json.loads(raw)

        cited = data.get("citations") or []
        check = validate_citations(cited, chunk_ids)
        by_id = {rc.chunk.chunk_id: rc.chunk for rc in contexts}
        citations = [(by_id[cid].doc_id, by_id[cid].page_num) for cid in check.kept]

        return Answer(
            question=question,
            text=data.get("answer", ""),
            citations=citations,
            contexts=contexts,
            insufficient_evidence=bool(data.get("insufficient_evidence", False)),
            citation_validity=check.validity,
            dropped_citations=check.dropped,
        )

    def _cached_or_call(
        self, question: str, contexts: Sequence[RetrievedChunk], chunk_ids: Sequence[str]
    ) -> str:
        """Return the raw JSON response string, from disk if present else the API."""
        key = self._cache_key(question, chunk_ids)
        raw = self._cache_load(key)
        if raw is None:
            raw = self._call_model(build_user_prompt(question, contexts))
            self._cache_store(key, raw)
        return raw

    # -- cache key + I/O -----------------------------------------------------

    def _cache_key(self, question: str, chunk_ids: Sequence[str]) -> str:
        # NUL separators so field boundaries can't be forged by concatenation
        # (same rationale as the embedding cache key in ingest/embed.py).
        payload = "\x00".join([question, self.config_hash, *chunk_ids])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_load(self, key: str) -> str | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            # A corrupt/unreadable cache file is a miss, not a crash — regenerate.
            return None
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            # Self-heal a poisoned entry (e.g. a truncated response cached before
            # the MAX_TOKENS guard existed): treat unparseable JSON as a miss.
            return None
        return raw

    def _cache_store(self, key: str, raw: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(raw, encoding="utf-8")

    # -- the one place that hits the network ---------------------------------

    def _call_model(self, user_prompt: str) -> str:
        """Call Gemini with structured output and return the raw JSON string.

        The only method that makes a network request; it is wrapped by the disk
        cache in ``_cached_or_call`` and overridden in tests, so the cache path
        can be exercised fully offline. Transient failures are retried (see
        ``_with_retries``).
        """
        self.api_calls += 1
        client = self._ensure_client()
        return self._with_retries(lambda: self._generate_once(client, user_prompt))

    def _generate_once(self, client, user_prompt: str) -> str:
        """One structured-output call; returns the raw JSON text."""
        from google.genai import types

        # Thinking tokens count against max_output_tokens; budget=0 disables them
        # so the whole budget goes to the answer. None omits the field.
        thinking = (
            types.ThinkingConfig(thinking_budget=self.thinking_budget)
            if self.thinking_budget is not None
            else None
        )
        response = client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
                response_schema=_AnswerSchema,
                thinking_config=thinking,
            ),
        )
        self._raise_if_truncated(response)
        raw = response.text
        if raw is None:
            raise RuntimeError(f"Gemini returned no text (blocked/finish?): {response!r}")
        return raw

    @staticmethod
    def _raise_if_truncated(response) -> None:
        """Reject a ``MAX_TOKENS`` cutoff before it can be parsed or cached.

        A truncated response is invalid JSON (an unterminated string). Raised as a
        plain ``RuntimeError`` — not an ``APIError`` — so ``_with_retries`` won't
        retry it (the result would be identical) and, because it fires inside
        ``_call_model``, the caller never reaches ``_cache_store``: a truncated
        answer is never cached. The message is actionable because the usual cause
        is thinking tokens starving the answer (see generation.thinking_budget).
        """
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return
        reason = getattr(candidates[0], "finish_reason", None)
        reason_name = getattr(reason, "name", None) or str(reason or "")
        if reason_name == "MAX_TOKENS":
            usage = getattr(response, "usage_metadata", None)
            thoughts = getattr(usage, "thoughts_token_count", None)
            raise RuntimeError(
                "Gemini response truncated (finish_reason=MAX_TOKENS"
                + (f", thoughts_tokens={thoughts}" if thoughts else "")
                + "). Raise generation.max_output_tokens or lower "
                "generation.thinking_budget (0 disables thinking)."
            )

    def _with_retries(self, call: Callable[[], str]) -> str:
        """Run ``call`` with bounded exponential backoff on transient API errors.

        Retries only 429/5xx (overload, rate limit) — the failures that clear on
        their own. A non-transient error (bad key, unknown model, malformed
        request) or an exhausted retry budget is translated to a ``RuntimeError``,
        so the CLI prints one clean line instead of an SDK stack trace.
        """
        from google.genai import errors

        for attempt in range(self.max_retries + 1):
            try:
                return call()
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                if code not in _TRANSIENT_CODES or attempt == self.max_retries:
                    status = getattr(exc, "status", "") or ""
                    raise RuntimeError(f"Gemini API error ({code} {status}): {exc}") from exc
                delay = self.retry_base_delay_s * (2**attempt)
                logger.warning(
                    "Gemini transient error %s; retrying in %.1fs (attempt %d/%d)",
                    code,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(delay)
        # The loop always returns or raises; this satisfies the type checker.
        raise RuntimeError("Gemini retry loop exited without returning")

    def _ensure_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "no Gemini API key: set the env var named by "
                    "generation.api_key_env (default GEMINI_API_KEY)"
                )
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client
