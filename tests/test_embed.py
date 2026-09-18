"""Tests for the dense embedder and its revision-aware cache.

Split by what they need:

* **Offline, always run** — the cache key depends on the model revision, the
  embedder refuses an unpinned revision, and the shipped config *is* pinned.
  None of these load a model.
* **Model-dependent, auto-skipped** — "the cache returns identical vectors on a
  second call" needs the real embedder. It runs only when bge-base is already in
  the local HuggingFace cache, so it never triggers a download (Tier 1 stays
  offline on CI and on fork PRs).
"""

from __future__ import annotations

import numpy as np
import pytest
from huggingface_hub import try_to_load_from_cache

from ledgion.config import load_config
from ledgion.ingest.embed import BGEEmbedder, cache_key


def _model_cached(model_id: str, revision: str | None) -> bool:
    """True iff the model weights are already in the local HF cache (no network)."""
    path = try_to_load_from_cache(model_id, "model.safetensors", revision=revision)
    return isinstance(path, str)


# --- offline: cache key + revision guards -----------------------------------


def test_cache_key_depends_on_revision():
    # Same text under different revisions must not collide (else a model change
    # silently reuses stale vectors).
    assert cache_key("rev-a", "hello") != cache_key("rev-b", "hello")
    # And it's deterministic for a fixed (revision, text).
    assert cache_key("rev-a", "hello") == cache_key("rev-a", "hello")


def test_cache_key_no_separator_collision():
    # The NUL separator prevents (rev="ab", text="c") == (rev="a", text="bc").
    assert cache_key("ab", "c") != cache_key("a", "bc")


def test_embedder_requires_pinned_revision(tmp_path):
    # A pin nothing reads is worse than no pin: refuse to construct without one.
    with pytest.raises(ValueError):
        BGEEmbedder(model_id="BAAI/bge-base-en-v1.5", revision=None, cache_dir=tmp_path)


def test_shipped_config_pins_revisions():
    cfg = load_config()
    assert cfg.embedding.revision is not None
    assert cfg.reranker.revision is not None


# --- model-dependent: cache identity ----------------------------------------


def test_cache_returns_identical_vectors_on_second_call(tmp_path):
    cfg = load_config()
    if not _model_cached(cfg.embedding.model_id, cfg.embedding.revision):
        pytest.skip("bge-base not in local HF cache; skipping to stay offline")

    embedder = BGEEmbedder.from_config(cfg)
    embedder.cache_dir = tmp_path  # isolate from the real cache

    first = embedder.embed_documents(["net sales increased year over year"])
    assert embedder.cache_misses == 1 and embedder.cache_hits == 0

    second = embedder.embed_documents(["net sales increased year over year"])
    # Second call is served from disk -> byte-identical, and counted as a hit.
    assert embedder.cache_hits == 1
    assert np.array_equal(first[0], second[0])
