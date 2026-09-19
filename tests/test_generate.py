"""Tests for Phase 3 generation: prompt assembly, citation validation, cache.

All three run fully offline — no API key, no network, no model download. The
prompt and validator tests are pure; the cache test stubs the one method that
would hit the network (``_call_model``) so it can prove a second call is served
from disk without a request.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from google.genai import errors

from ledgion.generate.citations import validate_citations
from ledgion.generate.gemini import GeminiGenerator
from ledgion.generate.prompt import build_user_prompt
from ledgion.interfaces import Chunk, RetrievedChunk


def _chunk(chunk_id: str, *, page_num: int = 50, text: str = "Net sales $14,694") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="AMCOR_2023_10K",
        page_num=page_num,
        text=text,
        company="Amcor",
        ticker="AMCR",
        fiscal_year=2023,
        form_type="10-K",
    )


def _contexts(ids: list[str]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=_chunk(cid, page_num=50 + i), score=1.0 - 0.1 * i)
        for i, cid in enumerate(ids)
    ]


def test_prompt_includes_every_retrieved_chunk_id():
    ids = ["a1b2c3d4e5f60718", "deadbeefdeadbeef", "0011ffaa0011ffaa"]
    prompt = build_user_prompt("What were net sales?", _contexts(ids))
    for cid in ids:
        assert cid in prompt, f"prompt is missing chunk_id {cid}"
    # the question must survive into the prompt too
    assert "What were net sales?" in prompt


def test_validate_citations_drops_fabricated_keeps_real():
    check = validate_citations(
        ["a1b2c3d4e5f60718", "fabricated0000000"],
        ["a1b2c3d4e5f60718", "0011ffaa0011ffaa"],
    )
    assert check.kept == ["a1b2c3d4e5f60718"]
    assert check.dropped == ["fabricated0000000"]
    assert check.validity == 0.5


def test_generation_cache_skips_second_api_call(tmp_path):
    ids = ["a1b2c3d4e5f60718", "0011ffaa0011ffaa"]
    contexts = _contexts(ids)
    gen = GeminiGenerator(
        model="gemini-test",
        config_hash="cfghash",
        cache_dir=tmp_path,
        api_key=None,
    )

    calls = {"n": 0}

    def fake_call(user_prompt: str) -> str:
        calls["n"] += 1
        return json.dumps(
            {
                "answer": "Net sales were $14,694M.",
                # one real id, one fabricated id the validator must drop
                "citations": ["a1b2c3d4e5f60718", "nope999nope99999"],
                "insufficient_evidence": False,
            }
        )

    gen._call_model = fake_call  # the only network method — stub it

    first = gen.generate("What were net sales?", contexts)
    second = gen.generate("What were net sales?", contexts)

    # Second call is served from disk: no additional request.
    assert calls["n"] == 1
    # ... and it reproduces the first answer exactly.
    assert first.text == second.text
    assert first.citations == second.citations
    assert first.citation_validity == second.citation_validity

    # The real citation is kept and resolved to (doc_id, page_num); the
    # fabricated one is dropped and the validity rate reflects it.
    assert first.citations == [("AMCOR_2023_10K", 50)]
    assert first.dropped_citations == ["nope999nope99999"]
    assert first.citation_validity == 0.5


def _generator(tmp_path):
    # retry_base_delay_s=0 so the retry tests don't actually sleep.
    return GeminiGenerator(
        model="gemini-test",
        config_hash="cfghash",
        cache_dir=tmp_path,
        api_key=None,
        max_retries=3,
        retry_base_delay_s=0.0,
    )


def test_retries_transient_error_then_succeeds(tmp_path):
    gen = _generator(tmp_path)
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise errors.ServerError(503, {"error": {"message": "overloaded"}})
        return '{"answer": "ok", "citations": [], "insufficient_evidence": true}'

    out = gen._with_retries(flaky)
    assert attempts["n"] == 3  # failed twice, succeeded on the third
    assert json.loads(out)["answer"] == "ok"


def test_does_not_retry_non_transient_error(tmp_path):
    gen = _generator(tmp_path)
    attempts = {"n": 0}

    def bad_request() -> str:
        attempts["n"] += 1
        raise errors.ClientError(400, {"error": {"message": "bad request"}})

    # A 400 is permanent: surfaced as a clean RuntimeError, not retried.
    with pytest.raises(RuntimeError):
        gen._with_retries(bad_request)
    assert attempts["n"] == 1


def _fake_response(finish_reason: str, *, thoughts: int = 0):
    return SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason))],
        usage_metadata=SimpleNamespace(thoughts_token_count=thoughts),
    )


def test_raise_if_truncated_rejects_max_tokens():
    # A MAX_TOKENS cutoff is invalid (unterminated) JSON — must raise, not return.
    with pytest.raises(RuntimeError, match="MAX_TOKENS"):
        GeminiGenerator._raise_if_truncated(_fake_response("MAX_TOKENS", thoughts=979))
    # A normal STOP finish is fine.
    GeminiGenerator._raise_if_truncated(_fake_response("STOP"))


def test_truncated_response_is_not_cached(tmp_path):
    gen = _generator(tmp_path)

    def truncating_call(user_prompt: str) -> str:
        # mimic _generate_once hitting the guard before it can return/cache
        raise RuntimeError("Gemini response truncated (finish_reason=MAX_TOKENS)")

    gen._call_model = truncating_call
    with pytest.raises(RuntimeError):
        gen.generate("What were net sales?", _contexts(["a1b2c3d4e5f60718"]))
    # Nothing was written: the guard fires before _cache_store.
    assert not list(tmp_path.glob("*.json"))


def test_cache_load_treats_poisoned_entry_as_miss(tmp_path):
    gen = _generator(tmp_path)
    # A truncated entry (as poisoned the cache before the guard existed).
    (tmp_path / "poison.json").write_text('{\n  "answer": "unterminated', encoding="utf-8")
    assert gen._cache_load("poison") is None
    # A valid entry loads back verbatim.
    (tmp_path / "good.json").write_text('{"answer": "ok"}', encoding="utf-8")
    assert gen._cache_load("good") == '{"answer": "ok"}'
