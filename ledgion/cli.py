import sys

if sys.platform == "win32":
    # 10-K text contains ☑ and non-breaking spaces that Windows' cp1252 console
    # cannot encode; force UTF-8 so printing filing text (or config) never crashes.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import json
import logging
from pathlib import Path

import typer

from ledgion.config import config_hash, load_config

app = typer.Typer(add_completion=False, help="Ledgion — RAG QA over SEC filings.")

_ConfigOption = typer.Option(None, "--config", "-c", help="Path to a config YAML.")


def _show_config(config_path: Path | None) -> None:
    """Resolve config and print it plus its hash (the results/<hash>.json key)."""
    cfg = load_config(config_path)
    typer.echo(json.dumps(cfg.model_dump(mode="json"), indent=2, default=str))
    typer.echo(f"config_hash: {config_hash(cfg)}")


@app.command()
def ingest(
    config: Path | None = _ConfigOption,
    force: bool = typer.Option(
        False, "--force", help="Bypass the embedding cache and recompute every vector."
    ),
) -> None:
    """Parse, chunk, and embed filings into the vector store."""
    # INFO-level so the embedder's truncation tripwire (a warning) is visible.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(config)
    # Imported lazily: keeps the torch/qdrant stack out of `ask`/`eval` startup.
    from ledgion.ingest.pipeline import run

    run(cfg, force=force)


@app.command()
def ask(
    question: str | None = typer.Argument(None, help="Question to answer."),
    config: Path | None = _ConfigOption,
) -> None:
    """Answer a question with page-level citations. [stub]"""
    _show_config(config)
    typer.echo("ask: not implemented in Phase 0.")


@app.command("eval")
def run_eval(config: Path | None = _ConfigOption) -> None:
    """Run the offline eval and write results/<hash>.json. [stub]"""
    _show_config(config)
    typer.echo("eval: not implemented in Phase 0.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
