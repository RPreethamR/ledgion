"""Tests for the golden evaluation set.

Per the working agreement, ``eval/`` code is test-first. Two kinds of test live
here:

* **Pure-rule tests** — the page-number conversion (``to_one_based``) and the
  answer typing. These need no data and always run.
* **Golden-set integrity tests** — every ``doc_id`` in ``golden/dev.jsonl``
  must resolve to a PDF, and every evidence page must be inside that PDF. These
  need the filings, which are gitignored (``data/`` is never committed), so they
  *skip* when the PDFs or the built golden file are absent — e.g. on a fork's CI
  runner, where the Tier 1 eval must still pass offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ledgion.config import REPO_ROOT, load_config
from ledgion.eval.golden import classify_answer_type, to_one_based

GOLDEN_PATH = REPO_ROOT / "golden" / "dev.jsonl"


def _pdf_dir() -> Path:
    pdf_dir = load_config().paths.pdf_dir
    return pdf_dir if pdf_dir.is_absolute() else REPO_ROOT / pdf_dir


def _load_golden() -> list[dict]:
    if not GOLDEN_PATH.exists():
        pytest.skip(
            f"{GOLDEN_PATH} not built; run `uv run python -m ledgion.eval.build_golden`"
        )
    with GOLDEN_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- pure rules: page-number conversion -------------------------------------


def test_to_one_based_zero():
    # FinanceBench's first page is 0; ours is 1.
    assert to_one_based(0) == 1


def test_to_one_based_one():
    assert to_one_based(1) == 2


def test_to_one_based_typical():
    # The Amcor balance sheet: FinanceBench evidence_page_num 51 -> our page 52.
    assert to_one_based(51) == 52


def test_to_one_based_rejects_negative():
    # A negative source page means a bug upstream, not a valid 0-based index.
    with pytest.raises(ValueError):
        to_one_based(-1)


# --- pure rules: answer typing ----------------------------------------------


def test_classify_answer_type_by_question_intent():
    # Questions whose answer requires pulling/comparing specific figures.
    assert (
        classify_answer_type("How much did the effective tax rate change from FY21 to FY22?")
        == "numeric"
    )
    assert classify_answer_type("What is the FY2022 unadjusted EBITDA margin?") == "numeric"
    assert (
        classify_answer_type("Has the quick ratio improved between FY2022 and FY2023?")
        == "numeric"
    )
    # Questions answered with narrative / qualitative content.
    assert classify_answer_type("What industry does the company operate in?") == "prose"
    # "what drove ..." asks for a cause -> prose, even though it names a margin.
    assert classify_answer_type("What drove the change in operating margin?") == "prose"
    assert classify_answer_type("Who are the company's primary customers?") == "prose"


# --- golden-set integrity: need the filings on disk -------------------------


def test_every_doc_id_resolves_to_a_pdf():
    rows = _load_golden()
    pdf_dir = _pdf_dir()
    if not pdf_dir.exists():
        pytest.skip(f"{pdf_dir} not present; filings are gitignored")
    missing = sorted(
        {r["doc_id"] for r in rows if not (pdf_dir / f"{r['doc_id']}.pdf").exists()}
    )
    assert not missing, f"golden doc_ids without a matching PDF: {missing}"


def test_evidence_pages_within_pdf_page_count():
    rows = _load_golden()
    pdf_dir = _pdf_dir()
    if not pdf_dir.exists():
        pytest.skip(f"{pdf_dir} not present; filings are gitignored")

    import pymupdf

    page_counts: dict[str, int] = {}
    for row in rows:
        doc_id = row["doc_id"]
        if doc_id not in page_counts:
            with pymupdf.open(pdf_dir / f"{doc_id}.pdf") as doc:
                page_counts[doc_id] = doc.page_count
        n = page_counts[doc_id]
        assert row["evidence_pages"], f"{row['qid']} has no evidence_pages"
        for page in row["evidence_pages"]:
            assert 1 <= page <= n, (
                f"{doc_id}: evidence page {page} outside 1..{n} for {row['qid']}"
            )
