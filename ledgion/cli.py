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
    question: str = typer.Argument(..., help="Question to answer."),
    config: Path | None = _ConfigOption,
) -> None:
    """Answer a question from the filings with validated page-level citations."""
    # WARNING keeps the embedder's truncation tripwire visible without the INFO
    # chatter of model loading drowning the answer.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(config)
    # Lazily imported so `--help` and other commands don't pull in torch/qdrant.
    from ledgion.generate.gemini import GeminiGenerator
    from ledgion.retrieve.dense import DenseRetriever

    retriever = DenseRetriever.from_config(cfg)
    contexts = retriever.retrieve(question, top_k=cfg.retrieval.top_k)
    generator = GeminiGenerator.from_config(cfg)
    try:
        answer = generator.generate(question, contexts)
    except RuntimeError as exc:
        # e.g. a missing API key — a one-line message, not a stack trace. (A cached
        # answer needs no key, so this only fires on a genuine cache-miss call.)
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(answer.text)
    typer.echo("")
    if answer.insufficient_evidence:
        typer.echo("[insufficient evidence: the model reported the filings don't answer this]")

    if answer.citations:
        typer.echo(f"Citations ({len(answer.citations)}, validity {answer.citation_validity:.2f}):")
        for doc_id, page_num in answer.citations:
            typer.echo(f"  - {doc_id}  p.{page_num}")
    else:
        typer.echo("Citations: none")

    if answer.dropped_citations:
        typer.echo(
            f"Dropped {len(answer.dropped_citations)} fabricated citation(s) "
            f"not in context: {', '.join(answer.dropped_citations)}"
        )


@app.command("eval")
def run_eval(config: Path | None = _ConfigOption) -> None:
    """Run the offline eval and write results/<hash>.json. [stub]"""
    _show_config(config)
    typer.echo("eval: not implemented in Phase 0.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
