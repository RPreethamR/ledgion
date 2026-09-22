.PHONY: ingest ask eval fixture test lint ci

ingest:
	uv run ledgion ingest

ask:
	uv run ledgion ask

eval:
	uv run ledgion eval

# Regenerate the offline Tier-1 fixtures from the live setup (docker Qdrant + bge).
# One atomic pass writes the dense index, query vectors, AND the frozen sparse BM25
# rankings + manifest — computed from a single corpus scroll so dense and sparse can
# never drift apart. Run after the corpus, the golden set, the embedding revision, or
# any sparse knob changes, then commit fixtures/*.npz + manifest.json.
fixture:
	uv run ledgion fixture

test:
	uv run pytest

lint:
	uv run ruff check .

# What CI runs: a torch-free env (no `models` group) scoring retrieval against the
# committed fixtures, then the gate. `--no-sync` reuses the env `uv sync` built so
# `uv run` never re-adds torch via the default groups.
ci:
	uv sync --no-group models
	uv run --no-sync ruff check .
	uv run --no-sync pytest -q
	LEDGION_RETRIEVAL__BACKEND=fixture LEDGION_QDRANT__MODE=memory \
		uv run --no-sync ledgion eval --tier 1 --gate
