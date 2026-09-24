"""The adoption rule as code (see DECISIONS.md, "Adoption rule").

A configuration replaces the current default only if, on the Tier-1 overall metrics
(30 golden questions, clean tree, against ``fixtures/baseline_metrics.json``):

1. **No regression** — ``recall@10``, ``ndcg@10`` and ``recall@50`` each fall no more
   than 0.02 below the default.
2. **Real improvement** — ``recall@10`` or ``ndcg@10`` rises by *more than* 0.02.

Among qualifiers, the largest ``ndcg@10`` gain wins; ties go to ``recall@10``.
``recall@50`` is guarded but cannot qualify a config on its own.

This module is **read-only and pure** — it computes verdicts from metric dicts and
never changes a default or writes a file. Deltas are computed on the 6-decimal values
the results files already carry and rounded to 6 decimals before comparison, so the
boundary cases (a drop of exactly 0.02, a gain of exactly 0.02) are exact. For a
top_k=50 run with no ``recall@50`` column, ``recall@k`` *is* ``recall@50``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TOLERANCE = 0.02
# Condition 1 guards all three; only these two can satisfy condition 2 (recall@50 is
# a pool metric — gaining it is a budget change, not a quality gain).
NO_REGRESSION = ("recall@10", "ndcg@10", "recall@50")
CAN_QUALIFY = ("recall@10", "ndcg@10")


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    baseline: float
    current: float
    delta: float  # round(current - baseline, 6); negative is a regression


@dataclass(frozen=True)
class Verdict:
    condition1: bool  # no regression on any guarded metric
    condition2: bool  # a real improvement on recall@10 or ndcg@10
    deltas: dict[str, MetricDelta]
    missing: tuple[str, ...]  # guarded metrics absent from the run (or baseline)

    @property
    def qualifies(self) -> bool:
        return self.condition1 and self.condition2


def resolve_recall50(metrics: dict, top_k: int | None) -> float | None:
    """``recall@50`` if present; else ``recall@k`` **only when top_k is 50** (there
    ``recall@k`` == ``recall@50``). At top_k=100 ``recall@k`` is ``recall@100`` and is
    *not* a substitute, so this returns None and the metric is reported missing."""
    if metrics.get("recall@50") is not None:
        return metrics["recall@50"]
    if top_k == 50 and metrics.get("recall@k") is not None:
        return metrics["recall@k"]
    return None


def _baseline_recall50(baseline: dict) -> float | None:
    value = baseline.get("recall@50")
    return value if value is not None else baseline.get("recall@k")


def evaluate(
    metrics: dict, baseline: dict, *, top_k: int | None, tolerance: float = TOLERANCE
) -> Verdict:
    """Apply the rule to one run's overall ``metrics`` against ``baseline``."""
    current = {
        "recall@10": metrics.get("recall@10"),
        "ndcg@10": metrics.get("ndcg@10"),
        "recall@50": resolve_recall50(metrics, top_k),
    }
    base = {
        "recall@10": baseline.get("recall@10"),
        "ndcg@10": baseline.get("ndcg@10"),
        "recall@50": _baseline_recall50(baseline),
    }
    missing = tuple(m for m in NO_REGRESSION if current[m] is None or base[m] is None)
    deltas = {
        m: MetricDelta(m, base[m], current[m], round(current[m] - base[m], 6))
        for m in NO_REGRESSION
        if current[m] is not None and base[m] is not None
    }
    # Condition 1: every guarded metric falls no more than tolerance (a missing metric
    # can't be shown non-regressing, so it fails condition 1).
    condition1 = all(
        m in deltas and round(base[m] - current[m], 6) <= tolerance for m in NO_REGRESSION
    )
    # Condition 2: recall@10 or ndcg@10 rises by strictly more than tolerance.
    condition2 = any(
        m in deltas and round(current[m] - base[m], 6) > tolerance for m in CAN_QUALIFY
    )
    return Verdict(condition1, condition2, deltas, missing)


def select_adopted(candidates: list[tuple[str, Verdict]]) -> str | None:
    """The adopted label among qualifiers: largest ndcg@10 gain, ties → recall@10.
    ``None`` if nothing qualifies (the default stays)."""
    qualifiers = [(label, v) for label, v in candidates if v.qualifies]
    if not qualifiers:
        return None
    return max(
        qualifiers,
        key=lambda item: (item[1].deltas["ndcg@10"].delta, item[1].deltas["recall@10"].delta),
    )[0]


# -- file-backed analysis (read-only) ----------------------------------------


@dataclass(frozen=True)
class Candidate:
    label: str
    top_k: int | None
    dirty: bool
    metrics: dict
    verdict: Verdict


def load_candidates(paths: list[Path], baseline: dict) -> list[Candidate]:
    """Evaluate each results file. Reads the file's own ``metrics.overall`` and
    ``config.retrieval.top_k``; keeps the ``git_dirty`` flag for exclusion."""
    from ledgion.eval.runner import _run_label

    out: list[Candidate] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as fh:
            report = json.load(fh)
        overall = report.get("metrics", {}).get("overall", {})
        top_k = report.get("config", {}).get("retrieval", {}).get("top_k")
        verdict = evaluate(overall, baseline, top_k=top_k)
        out.append(
            Candidate(
                label=_run_label(report),
                top_k=top_k,
                dirty=bool(report.get("git_dirty")),
                metrics=overall,
                verdict=verdict,
            )
        )
    return out


def _fail_reason(verdict: Verdict) -> str:
    """Which condition failed, and why — for the per-row report."""
    reasons = []
    if not verdict.condition1:
        drops = [
            f"{m} {d.delta:+.4f}"
            for m, d in verdict.deltas.items()
            if round(d.baseline - d.current, 6) > TOLERANCE
        ]
        if verdict.missing:
            drops.append("missing " + ",".join(verdict.missing))
        reasons.append("cond1(" + "; ".join(drops) + ")")
    if not verdict.condition2:
        reasons.append("cond2(no recall@10/ndcg@10 gain > 0.02)")
    return " ".join(reasons) if reasons else "qualifies"


def format_report(candidates: list[Candidate]) -> str:
    """Per-candidate verdict table + the adoption decision (clean qualifiers only)."""
    order = ("recall@10", "ndcg@10", "recall@50")
    lines = [
        f"adoption rule (tolerance {TOLERANCE}); baseline = fixtures/baseline_metrics.json",
        f"{'candidate':32} {'r@10':>8} {'ndcg@10':>8} {'r@50':>8}  c1  c2  verdict",
        "-" * 92,
    ]
    clean: list[tuple[str, Verdict]] = []
    dirty: list[Candidate] = []
    for c in sorted(candidates, key=lambda x: x.label):
        cells = " ".join(
            f"{c.verdict.deltas[m].delta:>+8.4f}" if m in c.verdict.deltas else f"{'-':>8}"
            for m in order
        )
        c1 = "ok" if c.verdict.condition1 else "NO"
        c2 = "ok" if c.verdict.condition2 else "NO"
        flag = " [DIRTY]" if c.dirty else ""
        lines.append(
            f"{c.label:32} {cells}  {c1:>3} {c2:>3}  {_fail_reason(c.verdict)}{flag}"
        )
        if c.dirty:
            dirty.append(c)
        else:
            clean.append((c.label, c.verdict))

    lines.append("-" * 92)
    if dirty:
        lines.append(f"excluded from adoption (dirty tree): {', '.join(c.label for c in dirty)}")
    adopted = select_adopted(clean)
    if adopted:
        lines.append(f"ADOPT: {adopted}  (largest ndcg@10 gain among clean qualifiers)")
    else:
        lines.append("ADOPT: none — no clean candidate qualifies; the default stays.")
    return "\n".join(lines)


def analyze(paths: list[Path], baseline: dict) -> str:
    return format_report(load_candidates(paths, baseline))
