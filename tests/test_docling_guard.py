"""The Docling golden-alignment guard, as an auto-skipped test.

This is the Phase-1 evidence verification, automated: every golden evidence string
must best-match its golden page in Docling's page-labeled text (after first-page
assignment). The ingestion pipeline runs the same check and aborts on any
disagreement; here it also guards on every PR *when the artifact is present*.

It is **skipped when the artifact is absent** — exactly like the PDF/model-dependent
tests — so CI on a bare runner (and fork PRs) stay green without the 8.7 MB artifact.
"""

from __future__ import annotations

import json

import pytest

from ledgion.config import REPO_ROOT, load_config
from ledgion.ingest.docling_artifact import golden_page_disagreements, has_artifact

GOLDEN_PATH = REPO_ROOT / "golden" / "dev.jsonl"


def _docling_dir():
    docling_dir = load_config().parser.docling_dir
    return docling_dir if docling_dir.is_absolute() else REPO_ROOT / docling_dir


def test_every_golden_evidence_matches_its_docling_page():
    docling_dir = _docling_dir()
    if not has_artifact(docling_dir):
        pytest.skip("no Docling artifact present; skipping to stay offline (like PDF tests)")

    lines = GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
    golden = [json.loads(line) for line in lines if line.strip()]
    disagreements = golden_page_disagreements(docling_dir, golden)
    assert disagreements == [], (
        "golden evidence best-matches the wrong page in Docling's text:\n"
        + "\n".join(
            f"  {d['qid']} {d['doc_id']}: best p{d['best_page']} not in {d['evidence_pages']}"
            for d in disagreements
        )
    )
