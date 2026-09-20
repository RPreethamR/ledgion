"""The CI gate: compare a run's metrics to the committed baseline, per tolerance.

Fully offline — the gate is pure arithmetic over two metric dicts, so these tests
need no retriever, no model, and no fixtures. They pin the three behaviours the
build depends on: an improvement passes, a dip within tolerance passes, and a
drop past tolerance fails with a message that names the metric, the baseline, the
current value, and the delta.
"""

from __future__ import annotations

import json

import pytest

from ledgion.eval.gate import check_gate, load_baseline

_GATED = ["recall@10", "ndcg@10"]
_TOL = 0.02
_BASELINE = {"recall@10": 0.5, "ndcg@10": 0.3}


def test_gate_passes_on_improvement():
    current = {"recall@10": 0.6, "ndcg@10": 0.35}
    result = check_gate(current, _BASELINE, gated_metrics=_GATED, tolerance=_TOL)
    assert result.ok
    # deltas are current - baseline, so improvements are positive.
    by_metric = {c.metric: c for c in result.checks}
    assert by_metric["recall@10"].delta == pytest.approx(0.1)
    assert not by_metric["recall@10"].regressed


def test_gate_passes_within_tolerance():
    # A 0.01 dip on each metric is inside the 0.02 tolerance -> still green.
    current = {"recall@10": 0.49, "ndcg@10": 0.29}
    result = check_gate(current, _BASELINE, gated_metrics=_GATED, tolerance=_TOL)
    assert result.ok
    assert all(c.delta < 0 for c in result.checks)  # dipped, but not enough to fail


def test_gate_fails_on_regression_past_tolerance():
    # recall@10 drops 0.05 (> 0.02) -> the build must fail.
    current = {"recall@10": 0.45, "ndcg@10": 0.30}
    result = check_gate(current, _BASELINE, gated_metrics=_GATED, tolerance=_TOL)
    assert not result.ok
    regressed = [c.metric for c in result.checks if c.regressed]
    assert regressed == ["recall@10"]


def test_failure_message_names_metric_baseline_current_and_delta():
    current = {"recall@10": 0.45, "ndcg@10": 0.30}
    result = check_gate(current, _BASELINE, gated_metrics=_GATED, tolerance=_TOL)
    msg = result.message()
    assert "recall@10" in msg
    assert "0.5000" in msg  # baseline
    assert "0.4500" in msg  # current
    assert "-0.0500" in msg  # delta (current - baseline)


def test_exactly_at_tolerance_passes():
    # A dip of exactly the tolerance is allowed; only *more than* tolerance fails.
    current = {"recall@10": 0.48, "ndcg@10": 0.30}  # 0.02 dip == tolerance
    result = check_gate(current, _BASELINE, gated_metrics=_GATED, tolerance=_TOL)
    assert result.ok


def test_missing_gated_metric_is_a_loud_error():
    # If the run never computed a gated metric, that's a config mistake, not a pass.
    current = {"recall@10": 0.6}  # ndcg@10 absent
    with pytest.raises(KeyError):
        check_gate(current, _BASELINE, gated_metrics=_GATED, tolerance=_TOL)


def test_load_baseline_reads_overall_metrics(tmp_path):
    path = tmp_path / "baseline_metrics.json"
    path.write_text(
        json.dumps({"metrics": {"recall@10": 0.5, "ndcg@10": 0.3}}), encoding="utf-8"
    )
    assert load_baseline(path) == {"recall@10": 0.5, "ndcg@10": 0.3}
