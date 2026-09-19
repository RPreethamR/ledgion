"""The results file must carry every provenance field and be reproducible.

These tests assemble a report from a *fake* Tier 1 result (no retrieval), so they
stay offline; they assert the provenance fields the spec requires and that writing
the same report twice yields byte-identical files.
"""

from __future__ import annotations

import json

from ledgion.config import config_hash, load_config
from ledgion.eval import runner


def _fake_tier1_result() -> dict:
    return {
        "results": [
            {
                "qid": "financebench_id_00799",
                "doc_id": "AMCOR_2023_10K",
                "answer_type": "numeric",
                "evidence_pages": [52],
                "retrieved_pages": [52, 10, 20],
                "metrics": {"recall@k": 1.0, "mrr": 1.0, "hit@1": 1.0},
            },
        ],
        "metrics": {
            "overall": {"recall@k": 1.0, "mrr": 1.0, "hit@1": 1.0},
            "numeric": {"recall@k": 1.0, "mrr": 1.0, "hit@1": 1.0},
            "counts": {"overall": 1, "numeric": 1, "prose": 0},
        },
    }


def test_report_has_every_provenance_field(tmp_path):
    cfg = load_config()
    report = runner.build_report(cfg, tier=1, tier1_result=_fake_tier1_result())

    # config hash (the results-file key) + git provenance.
    assert report["config_hash"] == config_hash(cfg)
    assert report["tier"] == 1
    assert isinstance(report["git_sha"], str) and report["git_sha"]
    assert "git_dirty" in report

    # pinned model revisions, surfaced explicitly (not just buried in the config).
    revs = report["model_revisions"]
    assert revs["embedding"]["model_id"] == cfg.embedding.model_id
    assert revs["embedding"]["revision"] == cfg.embedding.revision
    assert revs["reranker"]["revision"] == cfg.reranker.revision

    # the FULL resolved config, and the metrics + per-question results.
    assert report["config"] == cfg.model_dump(mode="json")
    assert report["results"]
    assert report["metrics"]["overall"]

    # written to <out_dir>/<config_hash>.json and round-trips unchanged.
    path = runner.write_report(report, out_dir=tmp_path)
    assert path.name == f"{config_hash(cfg)}.json"
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_report_write_is_byte_identical(tmp_path):
    cfg = load_config()
    report = runner.build_report(cfg, tier=1, tier1_result=_fake_tier1_result())

    p1 = runner.write_report(report, out_dir=tmp_path / "a")
    p2 = runner.write_report(report, out_dir=tmp_path / "b")

    assert p1.read_bytes() == p2.read_bytes()


def test_compare_prints_metric_deltas():
    old = runner.build_report(load_config(), tier=1, tier1_result=_fake_tier1_result())
    worse = _fake_tier1_result()
    worse["metrics"]["overall"]["hit@1"] = 0.5
    new = runner.build_report(load_config(), tier=1, tier1_result=worse)

    table = runner.format_compare(old, new)
    assert "hit@1" in table
    # new - old = 0.5 - 1.0 = -0.5 for the changed metric.
    assert "-0.5000" in table
