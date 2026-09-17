"""Verify ``golden/dev.jsonl`` page labels against a fresh PyMuPDF extraction.

For every golden row, re-extract the filing's pages and use
``rapidfuzz.fuzz.partial_ratio`` to find the page whose text best matches the
labelled ``evidence_text``. A row **AGREE**s when that best-match page is one of
its labelled ``evidence_pages``; otherwise **DISAGREE**.

This is the one-time check that our page convention (see
``ledgion.eval.golden.to_one_based``) lines up with FinanceBench's own
``evidence_page_num``. Once it passes, we trust the labels and never fuzzy-match
at eval time — retrieval is scored directly against ``evidence_pages``.

    uv run python -m ledgion.eval.verify_golden
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf
from rapidfuzz import fuzz

from ledgion.config import REPO_ROOT, load_config

GOLDEN_PATH = REPO_ROOT / "golden" / "dev.jsonl"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_golden() -> list[dict]:
    if not GOLDEN_PATH.exists():
        raise SystemExit(
            f"{GOLDEN_PATH} not found; run `uv run python -m ledgion.eval.build_golden` first."
        )
    with GOLDEN_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _extract_pages(pdf_path: Path) -> list[str]:
    """Return page text in reading order; index i is our 1-based page i+1."""
    with pymupdf.open(pdf_path) as doc:
        return [doc[i].get_text() for i in range(doc.page_count)]


def _best_match_page(evidence_text: str, pages: list[str]) -> tuple[int, float]:
    """Return the (1-based page, score) whose text best matches ``evidence_text``."""
    best_page, best_score = 0, -1.0
    for index, page_text in enumerate(pages):
        score = fuzz.partial_ratio(evidence_text, page_text)
        if score > best_score:
            best_page, best_score = index + 1, score
    return best_page, best_score


def verify() -> int:
    """Print the per-row report and summary. Return the number of DISAGREEs.

    Rows are printed as they are computed (flushed) so progress is visible: a
    full-length ``partial_ratio`` over PepsiCo's ~500 pages takes seconds per
    row, and this is a one-time check we favour correctness over speed on.
    """
    rows = _load_golden()
    pdf_dir = _resolve(load_config().paths.pdf_dir)

    header = f"{'qid':<22} {'labelled':<14} {'best':>5} {'score':>7}  verdict"
    print(header)
    print("-" * len(header))

    pages_by_doc: dict[str, list[str]] = {}
    results: list[tuple[str, list[int], int, float, bool]] = []
    for row in rows:
        doc_id = row["doc_id"]
        if doc_id not in pages_by_doc:
            pages_by_doc[doc_id] = _extract_pages(pdf_dir / f"{doc_id}.pdf")
        labelled = row["evidence_pages"]
        best_page, score = _best_match_page(row["evidence_text"], pages_by_doc[doc_id])
        agree = best_page in labelled
        results.append((row["qid"], labelled, best_page, score, agree))
        verdict = "AGREE" if agree else "DISAGREE"
        print(
            f"{row['qid']:<22} {str(labelled):<14} {best_page:>5} {score:>7.1f}  {verdict}",
            flush=True,
        )

    disagreements = [r for r in results if not r[4]]
    scores = [r[3] for r in results]
    print("-" * len(header))
    print(
        f"\n{len(results)} rows: {len(results) - len(disagreements)} AGREE, "
        f"{len(disagreements)} DISAGREE  |  score min={min(scores):.1f} "
        f"max={max(scores):.1f} mean={sum(scores) / len(scores):.1f}"
    )
    if disagreements:
        print("\nDISAGREE rows (labelled page not the best fuzzy match):")
        for qid, labelled, best_page, score, _ in disagreements:
            print(f"  {qid}: labelled {labelled}, best-match page {best_page} @ {score:.1f}")
    return len(disagreements)


def main() -> None:
    if sys.platform == "win32":
        # Filing text carries ☑ / non-breaking spaces; force UTF-8 before print.
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    verify()


if __name__ == "__main__":
    main()
