"""Config loading + hashing. The hash keys every results/<hash>.json file, so it
must be deterministic, independent of dict/field ordering, AND independent of the
host path separator (so a Windows dev run and a Linux CI run agree)."""

from pathlib import PurePosixPath, PureWindowsPath

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
    # A Path is included so the pin guards the POSIX normalisation too, not just
    # the key-sorted JSON canonicalisation.
    win_path = PureWindowsPath("data\\pdfs")
    fixed = {"b": 2, "a": 1, "nested": {"y": [3, 2, 1], "x": "v"}, "root": win_path}
    reordered = {"root": win_path, "nested": {"x": "v", "y": [3, 2, 1]}, "a": 1, "b": 2}

    h = config_hash(fixed)

    assert h == config_hash(fixed)  # deterministic
    assert h == config_hash(reordered)  # order-independent
    assert len(h) == 64  # sha-256 hex digest
    # Pinned so a change to the canonicalisation (which would silently rename
    # every results file) breaks this test loudly.
    assert h == "2c669b44d465d79579ae0e9969c0f150f03d0aece1e8abee87c9e8875c864b78"


def test_config_hash_is_path_separator_independent():
    # The same config expressed with Windows vs POSIX separators must hash the
    # same (the normaliser recurses into nested dicts), so results/<hash>.json
    # lines up across a Windows dev run and a Linux CI run.
    win = {"paths": {"pdf_dir": PureWindowsPath("data\\pdfs")}}
    posix = {"paths": {"pdf_dir": PurePosixPath("data/pdfs")}}
    assert config_hash(win) == config_hash(posix)
    # Normalisation only rewrites separators — genuinely different paths still differ.
    assert config_hash(win) != config_hash({"paths": {"pdf_dir": PurePosixPath("data/other")}})
