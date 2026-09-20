import sys

if sys.platform == "win32":
    # 10-K text contains ☑ and non-breaking spaces that Windows' cp1252 console
    # cannot encode; force UTF-8 so printing filing text (or config) never crashes.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import logging
from pathlib import Path

import typer

from ledgion.config import load_config

app = typer.Typer(add_completion=False, help="Ledgion — RAG QA over SEC filings.")

_ConfigOption = typer.Option(None, "--config", "-c", help="Path to a config YAML.")
# Module-level singletons (per the B008 convention) for the eval-specific options.
_TierOption = typer.Option(
    1, "--tier", "-t", min=1, max=2, help="1 = offline retrieval, 2 = judged."
)
_CompareOption = typer.Option(
    None, "--compare", help="A previous results/<hash>.json to print a delta table against."
)
_GateOption = typer.Option(
    False,
    "--gate",
    help="Fail (exit non-zero) if a gated metric falls more than eval.gate_tolerance "
    "below fixtures/baseline_metrics.json. Improvements always pass.",
)


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


def _print_eval_summary(report: dict) -> None:
    """Print the metric table (overall + per answer type) for one results file."""
    metrics = report["metrics"]
    counts = metrics.get("counts", {})
    dirty = "  (dirty)" if report.get("git_dirty") else ""
    typer.echo(
        f"Tier {report['tier']}  ·  config {report['config_hash'][:12]}  ·  "
        f"git {report['git_sha'][:12]}{dirty}"
    )
    specs = list(metrics["overall"].keys())
    typer.echo(f"{'group':<8} " + " ".join(f"{s:>10}" for s in specs) + f"  {'n':>4}")
    for group in ("overall", "numeric", "prose"):
        if group not in metrics:
            continue
        cells = " ".join(f"{metrics[group][s]:>10.4f}" for s in specs)
        typer.echo(f"{group:<8} {cells}  {counts.get(group, 0):>4}")

    tier2 = report.get("tier2")
    if tier2 and tier2.get("metrics"):
        judged = " ".join(f"{k}={v:.4f}" for k, v in sorted(tier2["metrics"].items()))
        typer.echo(f"tier2    {judged}")


@app.command("eval")
def run_eval(
    config: Path | None = _ConfigOption,
    tier: int = _TierOption,
    compare: Path | None = _CompareOption,
    gate: bool = _GateOption,
) -> None:
    """Run the offline eval, write results/<hash>.json, and print a summary."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(config)
    # Lazily imported so `--help` doesn't pull in torch/qdrant.
    from ledgion.eval import runner

    path = runner.run(cfg, tier=tier)
    report = runner.load_report(path)
    _print_eval_summary(report)
    typer.echo(f"\nwrote {path}")

    if compare is not None:
        old = runner.load_report(compare)
        typer.echo("")
        typer.echo(runner.format_compare(old, report))

    if gate:
        from ledgion.eval.gate import check_gate, load_baseline

        try:
            result = check_gate(
                report["metrics"]["overall"],
                load_baseline(),
                gated_metrics=cfg.eval.gate_metrics,
                tolerance=cfg.eval.gate_tolerance,
            )
        except KeyError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        typer.echo("")
        typer.echo(result.message())
        if not result.ok:
            raise typer.Exit(code=1)


@app.command()
def fixture(config: Path | None = _ConfigOption) -> None:
    """Regenerate the offline Tier-1 fixtures from the live local setup.

    Reads the populated Qdrant + the bge model and writes fixtures/index.npz,
    query_vectors.npz, and manifest.json — the artifacts the CI gate replays with
    no model or network. Run this whenever the corpus, the golden set, or the
    embedding revision changes.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(config)
    # Lazily imported so `--help` and other commands don't pull in torch/qdrant.
    from ledgion.eval.fixtures import generate_fixtures

    summary = generate_fixtures(cfg)
    typer.echo(
        f"wrote {summary['chunk_count']} chunk vectors, "
        f"{summary['query_count']} query vectors"
    )
    for key in ("index", "query_vectors", "manifest"):
        typer.echo(f"  {summary[key]}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
