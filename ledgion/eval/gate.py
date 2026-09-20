"""The CI gate — the one place a build is failed on a retrieval regression.

The gate does **no scoring of its own**: it reads the metrics an eval run already
produced (with the user's functions in ``metrics.py``) and compares the *overall*
group against a committed baseline. A gated metric that falls more than
``tolerance`` below its baseline fails the build; an improvement, or a dip within
tolerance, passes. This is the sibling of ``runner.format_compare`` — the same
new-minus-old delta, but with a pass/fail verdict attached.

The baseline is a small, hand-maintained file (``fixtures/baseline_metrics.json``)
rather than a ``results/<hash>.json`` so it survives config changes and is updated
deliberately, in the PR that improves it — never silently overwritten by a run.

Everything here is pure and offline (two dicts in, a verdict out), so it is
exercised with no retriever, model, or network.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ledgion.config import REPO_ROOT

# Committed baseline the gate compares against. A fixed repo path (like
# runner.GOLDEN_PATH), not a config knob, because it is an artifact of the repo,
# not a tunable of a run.
BASELINE_PATH = REPO_ROOT / "fixtures" / "baseline_metrics.json"


@dataclass(frozen=True)
class MetricCheck:
    """One gated metric's verdict.

    ``delta`` is ``current - baseline`` (so a regression reads negative, matching
    the compare table). ``regressed`` is the failing condition: the metric fell
    *more than* ``tolerance`` below baseline — an exactly-tolerance dip does not
    regress, and any improvement never does.
    """

    metric: str
    baseline: float
    current: float
    delta: float
    regressed: bool


@dataclass(frozen=True)
class GateResult:
    ok: bool
    tolerance: float
    checks: list[MetricCheck]

    def message(self) -> str:
        """A one-line-per-metric report; the failing lines name metric, baseline,
        current, and delta so a red build says exactly what moved and by how much."""
        verdict = "PASS" if self.ok else "FAIL"
        lines = [f"gate {verdict}  (tolerance {self.tolerance:+.4f})"]
        lines.append(f"{'metric':<12} {'baseline':>10} {'current':>10} {'delta':>10}  status")
        for c in self.checks:
            status = "REGRESSED" if c.regressed else "ok"
            lines.append(
                f"{c.metric:<12} {c.baseline:>10.4f} {c.current:>10.4f} "
                f"{c.delta:>+10.4f}  {status}"
            )
        return "\n".join(lines)


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, float]:
    """Read the baseline's ``metrics`` block (the overall-group metric values)."""
    if not Path(path).exists():
        raise SystemExit(
            f"{path} not found; commit a baseline (run `ledgion eval` on main and record "
            f"its overall metrics) before gating."
        )
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)["metrics"]


def check_gate(
    current: dict[str, float],
    baseline: dict[str, float],
    *,
    gated_metrics: Sequence[str],
    tolerance: float,
) -> GateResult:
    """Compare ``current`` metrics to ``baseline`` for each of ``gated_metrics``.

    ``current`` and ``baseline`` are flat ``{metric: value}`` dicts (the eval
    report's overall group, and the baseline file's ``metrics`` block). A gated
    metric missing from either is a loud ``KeyError``: it means the run did not
    compute a metric the gate needs (fix ``eval.metrics``), not a silent pass.
    """
    checks: list[MetricCheck] = []
    for metric in gated_metrics:
        if metric not in current:
            raise KeyError(
                f"gated metric {metric!r} not in the run's metrics "
                f"(add it to eval.metrics so the gate can enforce it)"
            )
        if metric not in baseline:
            raise KeyError(f"gated metric {metric!r} not in the baseline file")
        base = baseline[metric]
        cur = current[metric]
        delta = round(cur - base, 6)
        # Regress only on a drop strictly greater than tolerance; improvements pass.
        # Compare the *rounded* drop (metrics are already 6-dp) so a dip of exactly
        # the tolerance passes rather than tripping on float noise (0.5 - 0.48 is
        # 0.020000000000000018, not 0.02).
        regressed = round(base - cur, 6) > tolerance
        checks.append(
            MetricCheck(
                metric=metric, baseline=base, current=cur, delta=delta, regressed=regressed
            )
        )
    return GateResult(ok=not any(c.regressed for c in checks), tolerance=tolerance, checks=checks)
