"""Config loading + hashing. The hash keys every results/<hash>.json file,
so it must be deterministic and independent of dict/field ordering."""

from ledgion.config import Settings, config_hash, load_config


def test_config_loads_from_default_yaml():
    cfg = load_config()

    assert isinstance(cfg, Settings)
    # Fixed technical choices from CLAUDE.md must survive the round trip.
    assert cfg.embedding.model_id == "BAAI/bge-base-en-v1.5"
    assert cfg.embedding.device == "cpu"
    assert cfg.reranker.model_id == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert cfg.generation.provider == "gemini"
    assert cfg.generation.temperature == 0.0
    assert cfg.chunk.respect_page_boundary is True
    assert cfg.torch.num_threads == 4
    assert cfg.qdrant.collection_name == "ledgion_filings"


def test_config_hash_is_stable():
    fixed = {"b": 2, "a": 1, "nested": {"y": [3, 2, 1], "x": "v"}}
    reordered = {"nested": {"x": "v", "y": [3, 2, 1]}, "a": 1, "b": 2}

    h = config_hash(fixed)

    assert h == config_hash(fixed)  # deterministic
    assert h == config_hash(reordered)  # order-independent
    assert len(h) == 64  # sha-256 hex digest
    # Pinned so a change to the canonicalisation (which would silently rename
    # every results file) breaks this test loudly.
    assert h == "fdd0976613714aa44286d75569778b224607dc49e783a20d36d499701b024c01"
