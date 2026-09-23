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
import time
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
    latency: dict | None = None,
) -> dict:
    """Assemble the full results object: metrics + every provenance field.

    Pure assembly (no retrieval), so it is unit-testable from a fabricated
    ``tier1_result``.

    ``latency`` (retrieval/rerank p50/p95) is included only when it was measured —
    i.e. on a live, model-backed run. The offline "fixture" path measures nothing
    (its timings would be meaningless dict lookups) so its report carries no
    ``latency`` block and stays byte-identical across re-runs, exactly as before;
    the byte-identical guarantee is scoped to everything *except* this block, since
    wall-clock can't be deterministic.
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
    if latency is not None:
        report["latency"] = latency
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
# appended after these. recall@k is the full-depth recall (retrieval.top_k); recall@50
# is a fixed-cutoff column comparable across configs (a pool-integrity invariant only
# for the top_k=50 rows — see the Phase 7 note in DECISIONS.md).
_METRIC_ORDER = ("hit@1", "mrr", "ndcg@10", "recall@10", "recall@50", "recall@k")
_LABEL_WIDTH = 26
_COL_WIDTH = 11


def _run_backend(report: dict) -> str:
    return report.get("config", {}).get("retrieval", {}).get("backend", "?")


def _run_label(report: dict) -> str:
    """A row label that surfaces the ablation knobs, not just an opaque hash — so a
    weight sweep reads as ``hybrid sw0.25 stop <hash>`` rather than six ``hybrid`` rows,
    and a reranked deeper-pool run reads as ``dense rerank k100 <hash>``."""
    cfg = report.get("config", {})
    parts = [_run_backend(report)]
    if _run_backend(report) == "hybrid":
        sw = cfg.get("fusion", {}).get("sparse_weight")
        if sw is not None:
            parts.append(f"sw{sw}")
    if cfg.get("sparse", {}).get("remove_stopwords"):
        parts.append("stop")
    if cfg.get("reranker", {}).get("enabled"):
        parts.append("rerank")
    # top_k=50 is the baseline pool depth; only surface it when a run deepens it,
    # so Phase-6 labels stay unchanged and dense_rerank_k100 is legible.
    top_k = cfg.get("retrieval", {}).get("top_k")
    if top_k is not None and top_k != 50:
        parts.append(f"k{top_k}")
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

    lines += _format_latency_section(reports)
    return "\n".join(lines)


# -- latency (the other side of the rerank trade) ----------------------------

_LATENCY_COLS = ("retr p50", "retr p95", "rrank p50", "rrank p95")


def _latency_cell(stage: dict | None, key: str) -> str:
    """One p50/p95 cell; a dash when the run didn't measure that stage (e.g. no
    reranker, or the offline fixture path which measures nothing)."""
    value = (stage or {}).get(key)
    if value is None:
        return f"{'-':>{_COL_WIDTH}}"
    return f"{value:>{_COL_WIDTH}.2f}"


def _format_latency_section(reports: Sequence[dict]) -> list[str]:
    """A per-run latency table (ms): retrieval and rerank p50/p95 side by side, so the
    ablation shows both halves of a reranker's quality-for-latency trade. Rendered only
    if at least one run carries a measured ``latency`` block; runs without one are
    skipped (rather than shown as zero)."""
    if not any(r.get("latency") for r in reports):
        return []
    lines = [
        "",
        f"{'latency (ms)':<{_LABEL_WIDTH}}" + "".join(f"{c:>{_COL_WIDTH}}" for c in _LATENCY_COLS),
        "-" * (_LABEL_WIDTH + _COL_WIDTH * len(_LATENCY_COLS)),
    ]
    for report in reports:
        latency = report.get("latency")
        if not latency:
            continue
        retrieval, rerank = latency.get("retrieval"), latency.get("rerank")
        cells = (
            _latency_cell(retrieval, "p50")
            + _latency_cell(retrieval, "p95")
            + _latency_cell(rerank, "p50")
            + _latency_cell(rerank, "p95")
        )
        lines.append(f"{_run_label(report):<{_LABEL_WIDTH}}{cells}")
    return lines


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


def _build_reranker(cfg: Settings):
    """Build the reranking stage when ``reranker.enabled``, else ``None``.

    Reranking is a stage applied *after* retrieval, not a Retriever, so the runner
    composes it around ``_build_retriever`` rather than selecting it as a backend.
    The offline ``"fixture"`` backend replays frozen cross-encoder scores (no model,
    no torch — the CI path); every model-backed backend uses the live CrossEncoder.
    """
    if not cfg.reranker.enabled:
        return None
    if cfg.retrieval.backend == "fixture":
        from ledgion.retrieve.fixture import FixtureReranker

        return FixtureReranker.from_config(cfg)

    from ledgion.retrieve.rerank import CrossEncoderReranker

    return CrossEncoderReranker.from_config(cfg)


# -- latency measurement (live path only) ------------------------------------


class _TimedRetriever:
    """Wraps a retriever to record each ``retrieve`` call's wall-clock (ms), in order."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.durations_ms: list[float] = []

    def retrieve(self, query: str, *, top_k: int):
        start = time.perf_counter()
        out = self._inner.retrieve(query, top_k=top_k)
        self.durations_ms.append((time.perf_counter() - start) * 1000.0)
        return out


class _TimedReranker:
    """Wraps a reranker to record each ``rerank`` call's wall-clock (ms), in order."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.durations_ms: list[float] = []

    def rerank(self, query: str, candidates, *, top_n: int):
        start = time.perf_counter()
        out = self._inner.rerank(query, candidates, top_n=top_n)
        self.durations_ms.append((time.perf_counter() - start) * 1000.0)
        return out


def _percentiles(durations_ms: Sequence[float]) -> dict | None:
    """p50/p95 (ms, 2dp) + sample count for one stage; ``None`` if nothing was timed."""
    if not durations_ms:
        return None
    import numpy as np

    p50, p95 = (float(x) for x in np.percentile(np.asarray(durations_ms, dtype=float), [50, 95]))
    return {"p50": round(p50, 2), "p95": round(p95, 2), "n": len(durations_ms)}


def _latency_block(retrieval_ms: Sequence[float], rerank_ms: Sequence[float]) -> dict:
    """Assemble the report's ``latency`` block from per-question stage timings."""
    block: dict = {"unit": "ms", "retrieval": _percentiles(retrieval_ms)}
    rerank = _percentiles(rerank_ms)
    if rerank is not None:
        block["rerank"] = rerank
    return block


def run(cfg: Settings, *, tier: int, out_dir: Path | None = None) -> Path:
    """Run the eval end-to-end and write the results file. Returns its path."""
    golden = load_golden()
    retriever = _build_retriever(cfg)
    reranker = _build_reranker(cfg)

    # Latency is meaningful only on the live, model-backed path — the fixture backend
    # replays precomputed vectors/scores, so its timings would measure dict lookups.
    timed_retriever = timed_reranker = None
    if cfg.retrieval.backend != "fixture" and golden:
        # Warm up the lazy model loads once (untimed) so p50/p95 reflect steady-state
        # latency rather than a one-off cold start on the first golden question.
        warm_q = golden[0]["question"]
        warm = retriever.retrieve(warm_q, top_k=cfg.retrieval.top_k)
        if reranker is not None:
            reranker.rerank(warm_q, warm, top_n=cfg.retrieval.top_k)
        timed_retriever = _TimedRetriever(retriever)
        retriever = timed_retriever
        if reranker is not None:
            timed_reranker = _TimedReranker(reranker)
            reranker = timed_reranker

    tier1_result = run_tier1(
        golden,
        retriever,
        top_k=cfg.retrieval.top_k,
        metric_specs=cfg.eval.metrics,
        reranker=reranker,
    )

    # Snapshot latency from the Tier-1 loop before any Tier-2 retrieval adds calls.
    latency = None
    if timed_retriever is not None:
        rerank_ms = timed_reranker.durations_ms if timed_reranker is not None else []
        latency = _latency_block(timed_retriever.durations_ms, rerank_ms)

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

    report = build_report(
        cfg,
        tier=tier,
        tier1_result=tier1_result,
        tier2_result=tier2_result,
        latency=latency,
    )
    return write_report(report, out_dir=out_dir)


def main() -> None:
    """``python -m ledgion.eval.runner`` — run Tier 1 with the default config."""
    path = run(load_config(), tier=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
