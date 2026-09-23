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


# --- latency (Phase 7): measured on the live path, rendered in the ablation --


def _rerank_report(*, enabled=False, top_k=50, latency=None, hash_="abcdef012345") -> dict:
    """A minimal report shaped like a real one, for label/table rendering tests."""
    report = {
        "config_hash": hash_,
        "config": {
            "retrieval": {"backend": "dense", "top_k": top_k},
            "reranker": {"enabled": enabled},
            "sparse": {"remove_stopwords": False},
            "fusion": {},
        },
        "metrics": {
            "overall": {"recall@10": 0.5, "recall@50": 0.8167, "recall@k": 0.8167}
        },
    }
    if latency is not None:
        report["latency"] = latency
    return report


def test_percentiles_and_latency_block():
    block = runner._latency_block([10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 300.0])
    assert block["unit"] == "ms"
    assert block["retrieval"]["n"] == 4
    assert isinstance(block["retrieval"]["p50"], float)
    assert "rerank" in block and block["rerank"]["n"] == 3

    # No rerank timings -> no rerank key (a dense-only run measures retrieval only).
    retrieval_only = runner._latency_block([10.0, 20.0], [])
    assert "rerank" not in retrieval_only
    assert retrieval_only["retrieval"]["p50"] is not None


def test_latency_cell_dashes_when_unmeasured():
    assert runner._latency_cell(None, "p50").strip() == "-"
    assert runner._latency_cell({"p50": 12.0}, "p50").strip() == "12.00"


def test_run_label_reflects_rerank_and_deep_pool():
    assert runner._run_label(_rerank_report(enabled=True)).startswith("dense rerank")
    assert "k100" in runner._run_label(_rerank_report(enabled=True, top_k=100))
    # top_k=50 is the baseline depth, so it isn't surfaced (Phase-6 labels unchanged).
    assert "k100" not in runner._run_label(_rerank_report(enabled=False, top_k=50))


def test_ablation_table_renders_latency_section():
    base = _rerank_report(enabled=False, hash_="aaaaaaaaaaaa")
    reranked = _rerank_report(
        enabled=True,
        hash_="bbbbbbbbbbbb",
        latency={
            "unit": "ms",
            "retrieval": {"p50": 12.0, "p95": 18.0, "n": 30},
            "rerank": {"p50": 140.0, "p95": 210.0, "n": 30},
        },
    )
    table = runner.format_ablation([base, reranked])

    assert "latency (ms)" in table
    assert "12.00" in table and "140.00" in table  # retrieval + rerank p50
    assert "dense rerank" in table  # the reranked row is labelled distinctly


def test_ablation_latency_section_absent_without_measurements():
    # No run carries a latency block -> the metric table renders, the latency table
    # is omitted rather than shown empty.
    table = runner.format_ablation([_rerank_report(), _rerank_report(enabled=True)])
    assert "latency (ms)" not in table


def test_build_report_includes_latency_only_when_measured(tmp_path):
    cfg = load_config()
    latency = {"unit": "ms", "retrieval": {"p50": 5.0, "p95": 9.0, "n": 2}}
    with_lat = runner.build_report(
        cfg, tier=1, tier1_result=_fake_tier1_result(), latency=latency
    )
    without_lat = runner.build_report(cfg, tier=1, tier1_result=_fake_tier1_result())

    assert with_lat["latency"] == latency
    assert "latency" not in without_lat  # fixture/CI path stays byte-identical
