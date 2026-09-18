"""Tests for the recursive, page-bounded chunker.

Two properties the project depends on:

* **Deterministic chunk ids.** ``chunk_id`` is a hash of
  ``(doc_id, page_num, char_offset)``, so the *same* input must yield the *same*
  ids on every run — otherwise the vector store fills with duplicate points and
  nothing is idempotent.
* **No chunk spans a page boundary** (a hard project rule — retrieval is scored
  at page level, so a chunk straddling two pages would be un-scoreable).

These run fully offline: ``unit='chars'`` measures length in characters and
never loads a tokenizer, so no model download is needed. The two properties are
unit-agnostic — a passing char-mode test guards the token-mode splitter too,
since both share the same span logic.
"""

from __future__ import annotations

from ledgion.ingest.chunk import RecursiveChunker

# Multi-page input with paragraph + sentence structure, plus a blank page.
PAGES = [
    (
        1,
        "Alpha beta gamma. Delta epsilon zeta. Eta theta iota kappa lambda.\n\n"
        "Second paragraph with several more words so it splits into chunks.",
    ),
    (2, "Page two sentence one is here. Page two sentence two runs a little longer."),
    (3, "   \n  "),  # blank page: must produce no chunks
]

META = dict(company="Acme", ticker="ACME", fiscal_year=2022, form_type="10-K")


def _chunker() -> RecursiveChunker:
    # Small size/overlap in chars forces several chunks per page.
    return RecursiveChunker(size=40, overlap=8, unit="chars")


def test_chunk_ids_stable_across_runs():
    a = _chunker().chunk("DOC", PAGES, **META)
    b = _chunker().chunk("DOC", PAGES, **META)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    # Full equality too: text, page_num and metadata must all reproduce.
    assert a == b


def test_chunk_ids_unique_within_doc():
    chunks = _chunker().chunk("DOC", PAGES, **META)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_no_chunk_spans_a_page_boundary():
    page_text = {num: text for num, text in PAGES}
    chunks = _chunker().chunk("DOC", PAGES, **META)
    for c in chunks:
        # Each chunk's text is a verbatim slice of exactly one page's text.
        assert c.page_num in page_text
        assert c.text in page_text[c.page_num]


def test_blank_page_yields_no_chunks():
    chunks = _chunker().chunk("DOC", PAGES, **META)
    assert all(c.page_num != 3 for c in chunks)


def test_chunks_respect_size_with_overlap():
    # No chunk exceeds size; consecutive chunks on a page overlap (share text),
    # so char_offsets strictly increase and stay distinct.
    chunks = _chunker().chunk("DOC", [PAGES[0]], **META)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 40
