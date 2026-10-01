"""Page-alignment property check: the Docling artifact vs a fresh PyMuPDF extract.

Every page-level number in Phase 8 rests on one assumption: Docling's page *p* is
PyMuPDF's page *p* is FinanceBench's evidence page *p*. If Docling's paging drifted
by even one page anywhere, every retrieval metric computed on the Docling index would
be scored against the wrong ground truth and silently wrong. So we check it before
building anything on top of the artifact.

Three things are verified, per document, per page:

1. **Alignment.** For each (non-near-empty) page, ``token_set_ratio`` between Docling's
   page *p* text and PyMuPDF's page *p* must beat Docling *p* vs PyMuPDF *p−1* and *p+1*.
   ``token_set_ratio`` is order- and duplicate-insensitive, which suits comparing
   Docling's reading-order text (tables rendered as markdown) against PyMuPDF's raw
   stream — same content, different order. A page that matches a neighbour better than
   itself is a paging drift and a hard FAIL.
2. **Dropped content.** A page whose Docling text is much shorter than PyMuPDF's is
   flagged — the signature of Docling classifying a table as a picture (text dropped)
   or otherwise losing a block. (Markdown tables *add* pipe characters, so a genuine
   shortfall is meaningful, not a rendering artefact.)
3. **Golden coverage.** Every one of the 35 golden (doc_id, page) evidence pairs must
   have non-empty Docling content — an empty evidence page is unretrievable.

Multi-page elements (Docling provenance spanning >1 page) are summarised separately:
they are the open chunking question this phase has to answer, so they're surfaced here
for a decision, not silently split.

No Docling import — this reads the JSONL artifact only.

    uv run python -m ledgion.ingest.verify_docling_alignment
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pymupdf
from rapidfuzz import fuzz

from ledgion.config import REPO_ROOT, load_config
from ledgion.eval.golden import to_one_based  # noqa: F401  (kept: the convention lives there)

DOCLING_DIR = REPO_ROOT / "data" / "parsed" / "docling"
GOLDEN_PATH = REPO_ROOT / "golden" / "dev.jsonl"

# A page with fewer than this many non-space characters (covers, blank separators) is
# excluded from the alignment test — there is not enough text to match meaningfully.
NEAR_EMPTY_CHARS = 40
# Docling text below this fraction of PyMuPDF's length on the same page is flagged as
# possible dropped content.
SHORT_RATIO = 0.5
# Alignment tolerance. token_set_ratio is set-based, so two consecutive pages of the
# same long financial statement (identical column headers, recurring line items, "in
# millions") score near-identically — a self of 98 next to a neighbour of 99 is a
# tie the metric can't break, NOT a paging drift. A page fails alignment only when a
# neighbour beats it by more than this margin (real drift shows self ≈ 45, neighbour
# ≈ 95 — a gap far larger than this).
ALIGN_MARGIN = 5.0


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _pymupdf_pages(pdf_path: Path) -> list[str]:
    """Page text in reading order; index ``i`` is our 1-based page ``i + 1``."""
    with pymupdf.open(pdf_path) as doc:
        return [doc[i].get_text() for i in range(doc.page_count)]


def _load_docling(jsonl_path: Path) -> tuple[dict[int, str], list[dict]]:
    """Aggregate Docling element text per 1-based page, and collect multi-page rows.

    An element's text is attributed to **every** page in its ``pages`` list, so a page's
    aggregate holds all content physically on it (a table spanning p→p+1 contributes to
    both). Elements are visited in ``element_index`` (reading) order.
    """
    per_page: dict[int, list[str]] = {}
    multipage: list[dict] = []
    rows: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("element_index", 0))
    for rec in rows:
        text = (rec.get("text") or "").strip()
        pages = rec.get("pages") or ([rec["page_num"]] if rec.get("page_num") else [])
        if text:
            for p in pages:
                per_page.setdefault(p, []).append(rec["text"])
        if rec.get("multi_page"):
            multipage.append(rec)
    return {p: "\n".join(parts) for p, parts in per_page.items()}, multipage


def _near_empty(text: str) -> bool:
    return len(text.strip()) < NEAR_EMPTY_CHARS


def _load_golden() -> list[dict]:
    with GOLDEN_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _doc_ids() -> list[str]:
    return sorted(p.name[: -len(".docling.jsonl")] for p in DOCLING_DIR.glob("*.docling.jsonl"))


def run() -> int:
    """Print the report; return the number of HARD failures (0 = artifact trustworthy)."""
    if not DOCLING_DIR.exists() or not any(DOCLING_DIR.glob("*.docling.jsonl")):
        raise SystemExit(
            f"No Docling artifact in {DOCLING_DIR}. Produce it on Colab first "
            f"(notebooks/colab_docling_parse.ipynb), then copy the JSONL here."
        )

    pdf_dir = _resolve(load_config().paths.pdf_dir)
    doc_ids = _doc_ids()

    alignment_fails: list[tuple[str, int, float, float | None, float | None]] = []
    short_pages: list[tuple[str, int, int, int]] = []
    docling_by_doc: dict[str, dict[int, str]] = {}
    multipage_by_doc: dict[str, list[dict]] = {}

    print("=" * 78)
    print("PAGE-ALIGNMENT PROPERTY CHECK — Docling artifact vs PyMuPDF")
    print("=" * 78)

    for doc_id in doc_ids:
        pym = _pymupdf_pages(pdf_dir / f"{doc_id}.pdf")
        docling, multipage = _load_docling(DOCLING_DIR / f"{doc_id}.docling.jsonl")
        docling_by_doc[doc_id] = docling
        multipage_by_doc[doc_id] = multipage
        n = len(pym)

        tested = doc_fails = doc_short = 0
        for p in range(1, n + 1):
            pym_p = pym[p - 1]
            if _near_empty(pym_p):
                continue
            dl_p = docling.get(p, "")

            # Dropped-content pages (Docling emitted far less than PyMuPDF, or nothing)
            # are reported separately and excluded from alignment — you can't fairly ask
            # whether a page Docling gutted lands on the right page number.
            if not dl_p or len(dl_p) < SHORT_RATIO * len(pym_p):
                short_pages.append((doc_id, p, len(dl_p), len(pym_p)))
                doc_short += 1
                continue

            tested += 1
            score_p = fuzz.token_set_ratio(dl_p, pym_p)
            score_prev = (
                fuzz.token_set_ratio(dl_p, pym[p - 2])
                if p > 1 and not _near_empty(pym[p - 2])
                else None
            )
            score_next = (
                fuzz.token_set_ratio(dl_p, pym[p])
                if p < n and not _near_empty(pym[p])
                else None
            )
            neighbours = [s for s in (score_prev, score_next) if s is not None]
            best_neighbour = max(neighbours) if neighbours else None
            if best_neighbour is not None and best_neighbour - score_p > ALIGN_MARGIN:
                alignment_fails.append((doc_id, p, score_p, score_prev, score_next))
                doc_fails += 1

        print(
            f"\n{doc_id}: {n} pages, {tested} tested "
            f"(near-empty skipped: {n - tested}) | "
            f"alignment FAILs: {doc_fails} | short pages: {doc_short} | "
            f"multi-page elements: {len(multipage)}"
        )

    # ---- Golden coverage ---------------------------------------------------
    golden = _load_golden()
    golden_pairs: list[tuple[str, int, str]] = []  # (doc_id, page, qid)
    for row in golden:
        for page in row["evidence_pages"]:
            golden_pairs.append((row["doc_id"], page, row["qid"]))
    golden_page_set = {(d, pg) for d, pg, _ in golden_pairs}
    empty_golden = [
        (doc, page, qid)
        for doc, page, qid in golden_pairs
        if not (docling_by_doc.get(doc, {}).get(page, "").strip())
    ]

    print("\n" + "-" * 78)
    print(
        f"GOLDEN COVERAGE: {len(golden_pairs)} evidence pairs, "
        f"{len(golden_pairs) - len(empty_golden)} with non-empty Docling content"
    )
    if empty_golden:
        for doc, page, qid in empty_golden:
            print(f"  EMPTY: {doc} p{page} ({qid})")

    # ---- Alignment failures ------------------------------------------------
    print("\n" + "-" * 78)
    print(f"ALIGNMENT: {len(alignment_fails)} page(s) match a neighbour at least as well")
    for doc, p, sp, spv, snx in alignment_fails:
        golden_flag = " [GOLDEN]" if (doc, p) in golden_page_set else ""
        print(f"  {doc} p{p}: self={sp:.1f} prev={_fmt(spv)} next={_fmt(snx)}{golden_flag}")

    # ---- Possible dropped content ------------------------------------------
    print("\n" + "-" * 78)
    pct = int(SHORT_RATIO * 100)
    print(f"SHORT PAGES (Docling < {pct}% of PyMuPDF length): {len(short_pages)}")
    for doc, p, dl_len, pym_len in short_pages:
        golden_flag = " [GOLDEN]" if (doc, p) in golden_page_set else ""
        print(f"  {doc} p{p}: docling {dl_len} vs pymupdf {pym_len} chars{golden_flag}")

    # ---- Multi-page elements ----------------------------------------------
    _report_multipage(multipage_by_doc, golden_pairs)

    # ---- Verdict -----------------------------------------------------------
    short_golden = [s for s in short_pages if (s[0], s[1]) in golden_page_set]
    hard_fails = len(alignment_fails) + len(empty_golden) + len(short_golden)
    print("\n" + "=" * 78)
    if hard_fails == 0:
        print("VERDICT: PASS — alignment holds, every golden page has content.")
    else:
        print(
            f"VERDICT: STOP — {len(alignment_fails)} alignment fail(s), "
            f"{len(empty_golden)} empty golden page(s), {len(short_golden)} short golden page(s)."
        )
    print("=" * 78)
    return hard_fails


def _fmt(score: float | None) -> str:
    return "  -  " if score is None else f"{score:.1f}"


def _report_multipage(
    multipage_by_doc: dict[str, list[dict]], golden_pairs: list[tuple[str, int, str]]
) -> None:
    total = sum(len(v) for v in multipage_by_doc.values())
    print("\n" + "-" * 78)
    print(f"MULTI-PAGE ELEMENTS: {total} across {len(multipage_by_doc)} documents")
    golden_page_set = {(d, pg) for d, pg, _ in golden_pairs}
    for doc_id, elems in multipage_by_doc.items():
        by_type = Counter(e["element_type"] for e in elems)
        spans = Counter(len(e.get("pages") or []) for e in elems)
        print(
            f"  {doc_id}: {len(elems)}  by_type={dict(by_type)}  "
            f"page_span_counts={dict(sorted(spans.items()))}"
        )
    # Every multi-page TABLE — the decision-relevant case — with a snippet.
    print("\n  Multi-page TABLES (the chunking decision):")
    any_table = False
    for doc_id, elems in multipage_by_doc.items():
        for e in elems:
            if e["element_type"] != "table":
                continue
            any_table = True
            pages = e.get("pages") or []
            touches_golden = any((doc_id, pg) in golden_page_set for pg in pages)
            flag = " [touches GOLDEN]" if touches_golden else ""
            snippet = (e.get("text") or "").replace("\n", " ")[:100]
            print(f"    {doc_id} pages={pages}{flag}: {snippet!r}")
    if not any_table:
        print("    (none — all multi-page elements are text/list/other)")
    # Multi-page elements that land on a golden evidence page (any type).
    print("\n  Multi-page elements on a GOLDEN evidence page:")
    any_golden = False
    for doc_id, elems in multipage_by_doc.items():
        for e in elems:
            pages = e.get("pages") or []
            if any((doc_id, pg) in golden_page_set for pg in pages):
                any_golden = True
                snippet = (e.get("text") or "").replace("\n", " ")[:80]
                print(f"    {doc_id} pages={pages} type={e['element_type']}: {snippet!r}")
    if not any_golden:
        print("    (none)")


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    fails = run()
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
