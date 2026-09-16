import sys

if sys.platform == "win32":
    # 10-K text contains ☑ and non-breaking spaces that Windows' cp1252 console
    # cannot encode; force UTF-8 so printing filing text (or config) never crashes.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import json
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
def ingest(config: Path | None = _ConfigOption) -> None:
    """Parse, chunk, and embed filings into the vector store. [stub]"""
    _show_config(config)
    typer.echo("ingest: not implemented in Phase 0.")


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
