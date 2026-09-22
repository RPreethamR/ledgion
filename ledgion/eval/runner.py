"""Eval orchestration + the ``results/<config_hash>.json`` writer.

The runner is the seam between the offline scoring core (``tier1``/``tier2``) and
the outside world: it loads the golden set, builds the *real* retriever/generator
from config, runs the requested tier, and writes a fully-provenanced results file.

Everything that makes a run reproducible lives here:

* the file is named ``results/<config_hash>.json`` — the hash keys the run;
* it carries the git SHA, the pinned model revisions, and the *entire* resolved
  config, so any number can be traced back to exactly what produced it;
* it is serialised canonically — sorted keys, LF newline, no wall-clock — so two
  runs of the same config produce a byte-identical file (stronger than merely
  "identical numbers").

``--compare`` reads a previous file and prints a delta table, which is how an
ablation's effect on the headline metrics is read off.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from ledgion.config import REPO_ROOT, Settings, config_hash, load_config
from ledgion.eval.tier1 import run_tier1

GOLDEN_PATH = REPO_ROOT / "golden" / "dev.jsonl"

# Groups printed by the compare table, in a fixed order.
_COMPARE_GROUPS = ("overall", "numeric", "prose")


# -- golden set --------------------------------------------------------------


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    """Read the golden set as a list of rows (one JSON object per line)."""
    if not path.exists():
        raise SystemExit(
            f"{path} not found; run `uv run python -m ledgion.eval.build_golden` first."
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# -- provenance --------------------------------------------------------------


def _git(args: list[str], repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_sha(repo_root: Path = REPO_ROOT) -> str:
    """HEAD commit SHA, or ``"unknown"`` if git isn't available."""
    return _git(["rev-parse", "HEAD"], repo_root) or "unknown"


def git_dirty(repo_root: Path = REPO_ROOT) -> bool:
    """True if the working tree has uncommitted changes.

    Recorded for honesty: a results file produced from a dirty tree can't be tied
    to the committed SHA alone, and the reader should know.
    """
    return bool(_git(["status", "--porcelain"], repo_root))


def model_revisions(cfg: Settings) -> dict:
    """The pinned (model_id, revision) pairs the retrieval path depends on."""
    return {
        "embedding": {"model_id": cfg.embedding.model_id, "revision": cfg.embedding.revision},
        "reranker": {"model_id": cfg.reranker.model_id, "revision": cfg.reranker.revision},
    }


# -- report assembly + I/O ---------------------------------------------------


def build_report(
    cfg: Settings,
    *,
    tier: int,
    tier1_result: dict,
    tier2_result: dict | None = None,
) -> dict:
    """Assemble the full results object: metrics + every provenance field.

    Pure assembly (no retrieval), so it is unit-testable from a fabricated
    ``tier1_result``.
    """
    report = {
        "tier": tier,
        "config_hash": config_hash(cfg),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "model_revisions": model_revisions(cfg),
        "metrics": tier1_result["metrics"],
        "results": tier1_result["results"],
        "config": cfg.model_dump(mode="json"),
    }
    if tier2_result is not None:
        report["tier2"] = tier2_result
    return report


def write_report(report: dict, *, out_dir: Path | None = None) -> Path:
    """Write ``<out_dir>/<config_hash>.json`` canonically; return the path.

    Sorted keys + LF newline + no timestamp → the same report serialises to the
    same bytes on any machine, so a re-run overwrites its file with identical
    content. ``out_dir`` defaults to ``results/`` at the repo root.
    """
    out_dir = Path(out_dir) if out_dir is not None else REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report['config_hash']}.json"
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def load_report(path: Path) -> dict:
    """Read a previously-written results file."""
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


# -- compare -----------------------------------------------------------------


def format_compare(old: dict, new: dict) -> str:
    """A delta table (new − old) for each metric shared by both runs, per group."""
    old_h = old.get("config_hash", "?")[:12]
    new_h = new.get("config_hash", "?")[:12]
    lines = [
        f"compare   old {old_h}   ->   new {new_h}",
        f"{'group':<8} {'metric':<12} {'old':>9} {'new':>9} {'delta':>10}",
        "-" * 52,
    ]
    old_m, new_m = old["metrics"], new["metrics"]
    for group in _COMPARE_GROUPS:
        if group not in new_m or group not in old_m:
            continue
        for metric, new_val in new_m[group].items():
            if metric not in old_m[group]:
                continue
            old_val = old_m[group][metric]
            delta = round(new_val - old_val, 6)
            lines.append(
                f"{group:<8} {metric:<12} {old_val:>9.4f} {new_val:>9.4f} {delta:>+10.4f}"
            )
    return "\n".join(lines)


# -- ablation table (many runs at once) --------------------------------------

# Canonical column order for the ablation table; any extra metric a run carries is
# appended after these. recall@k is the full-depth recall (retrieval.top_k).
_METRIC_ORDER = ("hit@1", "mrr", "ndcg@10", "recall@10", "recall@k")
_LABEL_WIDTH = 26
_COL_WIDTH = 11


def _run_backend(report: dict) -> str:
    return report.get("config", {}).get("retrieval", {}).get("backend", "?")


def _run_label(report: dict) -> str:
    """A row label that surfaces the ablation knobs, not just an opaque hash — so a
    weight sweep reads as ``hybrid sw0.25 stop <hash>`` rather than six ``hybrid`` rows."""
    cfg = report.get("config", {})
    parts = [_run_backend(report)]
    if _run_backend(report) == "hybrid":
        sw = cfg.get("fusion", {}).get("sparse_weight")
        if sw is not None:
            parts.append(f"sw{sw}")
    if cfg.get("sparse", {}).get("remove_stopwords"):
        parts.append("stop")
    parts.append(report.get("config_hash", "?")[:6])
    return " ".join(parts)


def _delta_cell(current: float, baseline: float) -> str:
    return f"{round(current - baseline, 4):>+{_COL_WIDTH}.4f}"


def _ablation_specs(report: dict) -> list[str]:
    overall = report["metrics"].get("overall", {})
    ordered = [m for m in _METRIC_ORDER if m in overall]
    ordered += [m for m in overall if m not in ordered]
    return ordered


def format_ablation(reports: Sequence[dict]) -> str:
    """A multi-run ablation table: one row per config, every metric, split by answer
    type (overall / numeric / prose), with deltas against the dense baseline.

    The baseline is the dense run among ``reports`` (else the first run). Each
    non-baseline row is followed by a ``Δ vs dense`` line so a reader sees both the
    absolute number and the movement the ablation bought.
    """
    if not reports:
        return "no runs to compare"

    baseline = next((r for r in reports if _run_backend(r) == "dense"), reports[0])
    specs = _ablation_specs(baseline)
    lines = [
        f"ablation   baseline = {_run_label(baseline)}   ({len(reports)} runs)",
    ]

    for group in _COMPARE_GROUPS:
        base_group = baseline["metrics"].get(group)
        if base_group is None:
            continue
        lines.append("")
        lines.append(f"{group:<{_LABEL_WIDTH}}" + "".join(f"{s:>{_COL_WIDTH}}" for s in specs))
        lines.append("-" * (_LABEL_WIDTH + _COL_WIDTH * len(specs)))
        for report in reports:
            group_metrics = report["metrics"].get(group)
            if group_metrics is None:
                continue
            values = "".join(
                f"{group_metrics.get(s, float('nan')):>{_COL_WIDTH}.4f}" for s in specs
            )
            lines.append(f"{_run_label(report):<{_LABEL_WIDTH}}{values}")
            if report is not baseline:
                deltas = "".join(
                    _delta_cell(group_metrics.get(s, 0.0), base_group.get(s, 0.0)) for s in specs
                )
                lines.append(f"{'  delta':<{_LABEL_WIDTH}}{deltas}")
    return "\n".join(lines)


# -- end-to-end run (the real, model-backed path) ----------------------------


def _build_retriever(cfg: Settings):
    """Build the retriever ``retrieval.backend`` selects.

    ``"dense"``/``"sparse"``/``"hybrid"`` are the real, model-backed strategies (bge
    and/or BM25 over a populated Qdrant); ``"fixture"`` is the offline path that
    replays the committed dense fixtures with no model (this is what the CI gate
    runs). Everything is imported lazily so tier1's offline tests — which inject a
    fake retriever — never pull in torch/qdrant, and ``ledgion eval --help`` stays
    light.
    """
    backend = cfg.retrieval.backend
    if backend == "fixture":
        from ledgion.retrieve.fixture import FixtureRetriever

        return FixtureRetriever.from_config(cfg)
    if backend == "sparse":
        from ledgion.retrieve.sparse import SparseRetriever

        return SparseRetriever.from_config(cfg)
    if backend == "hybrid":
        from ledgion.retrieve.fusion import HybridRetriever

        return HybridRetriever.from_config(cfg)

    from ledgion.retrieve.dense import DenseRetriever

    return DenseRetriever.from_config(cfg)


def run(cfg: Settings, *, tier: int, out_dir: Path | None = None) -> Path:
    """Run the eval end-to-end and write the results file. Returns its path."""
    golden = load_golden()
    retriever = _build_retriever(cfg)

    tier1_result = run_tier1(
        golden,
        retriever,
        top_k=cfg.retrieval.top_k,
        metric_specs=cfg.eval.metrics,
    )

    tier2_result = None
    if tier == 2:
        if not cfg.eval.ragas_enabled:
            raise SystemExit(
                "Tier 2 is judged and off by default; set eval.ragas_enabled: true to run it."
            )
        # Lazy: keeps ragas + the generator out of the Tier 1 path entirely.
        from ledgion.eval.tier2 import ragas_judge, run_tier2
        from ledgion.generate.gemini import GeminiGenerator

        tier2_result = run_tier2(
            golden,
            retriever,
            GeminiGenerator.from_config(cfg),
            ragas_judge(cfg),
            top_k=cfg.retrieval.top_k,
            cache_dir=Path(cfg.paths.cache_dir) / "tier2",
            config_hash=config_hash(cfg),
        )

    report = build_report(cfg, tier=tier, tier1_result=tier1_result, tier2_result=tier2_result)
    return write_report(report, out_dir=out_dir)


def main() -> None:
    """``python -m ledgion.eval.runner`` — run Tier 1 with the default config."""
    path = run(load_config(), tier=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
