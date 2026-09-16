.PHONY: ingest ask eval test lint

ingest:
	uv run ledgion ingest

ask:
	uv run ledgion ask

eval:
	uv run ledgion eval

test:
	uv run pytest

lint:
	uv run ruff check .
