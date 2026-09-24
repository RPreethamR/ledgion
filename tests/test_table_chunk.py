"""Tests for the table-aware chunker over the Docling artifact.

All offline: ``unit='chars'`` measures length in characters, so nothing loads a
tokenizer or a model, and every element is a synthetic dict (no file I/O — the pure
``chunk_elements`` entry point is exercised directly). The properties checked mirror
the project's hard rules: one page label per chunk, deterministic ids, and — in
markdown mode — a header row in every table chunk including each part of a split.
"""

from __future__ import annotations

import sys

from ledgion.ingest.chunk import TableAwareChunker

META = dict(company="Acme", ticker="ACME", fiscal_year=2022, form_type="10-K")


def _text(idx: int, page: int, text: str, *, etype: str = "text") -> dict:
    return {
        "doc_id": "DOC",
        "element_index": idx,
        "page_num": page,
        "pages": [page],
        "multi_page": False,
        "element_type": etype,
        "docling_label": etype,
        "text": text,
    }


def _table(
    idx: int, page: int, headers: list[str], rows: list[list[str]], caption: str = ""
) -> dict:
    cells = [
        {"text": h, "row": 0, "col": c, "row_span": 1, "col_span": 1,
         "column_header": True, "row_header": False, "row_section": False}
        for c, h in enumerate(headers)
    ]
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cells.append(
                {"text": val, "row": r, "col": c, "row_span": 1, "col_span": 1,
                 "column_header": False, "row_header": c == 0, "row_section": False}
            )
    return {
        "doc_id": "DOC",
        "element_index": idx,
        "page_num": page,
        "pages": [page],
        "multi_page": False,
        "element_type": "table",
        "docling_label": "table",
        "text": "FALLBACK-PLAINTEXT",
        "table": {
            "num_rows": len(rows) + 1,
            "num_cols": len(headers),
            "caption": caption,
            "cells": cells,
        },
    }


def _chunker(table_mode: str, size: int = 400) -> TableAwareChunker:
    return TableAwareChunker(size=size, overlap=8, unit="chars", table_mode=table_mode)


# --- page-boundary invariant ------------------------------------------------


def _two_page_elements() -> list[dict]:
    return [
        _text(0, 1, "Consolidated Statements of Income", etype="heading"),
        _text(1, 1, "(in millions, except per share data)"),
        _table(2, 1, ["Metric", "2022", "2021"], [["Net sales", "100", "90"], ["Cost", "60", "55"]],
               caption="Consolidated Statements of Income"),
        _text(3, 2, "PAGETWO marker paragraph with several words so it forms a chunk."),
    ]


def test_no_chunk_spans_a_page_flat():
    chunks = _chunker("flat").chunk_elements("DOC", _two_page_elements(), **META)
    assert {c.page_num for c in chunks} <= {1, 2}
    # Page-2 content never appears on a page-1 chunk and vice versa.
    assert all("PAGETWO" not in c.text for c in chunks if c.page_num == 1)
    assert all("Net sales" not in c.text for c in chunks if c.page_num == 2)


def test_no_chunk_spans_a_page_markdown():
    chunks = _chunker("markdown").chunk_elements("DOC", _two_page_elements(), **META)
    assert {c.page_num for c in chunks} <= {1, 2}
    assert all("PAGETWO" not in c.text for c in chunks if c.page_num == 1)


# --- markdown table handling ------------------------------------------------


def test_markdown_table_chunk_contains_header_row():
    chunker = _chunker("markdown")
    chunks = chunker.chunk_elements("DOC", _two_page_elements(), **META)
    table_chunks = [c for c in chunks if c.chunk_id in chunker.last_table_chunk_ids]
    assert table_chunks
    for c in table_chunks:
        # The markdown header row (with every column name) is present.
        assert "| Metric | 2022 | 2021 |" in c.text


def test_markdown_table_prefixed_with_caption_and_units_line():
    chunker = _chunker("markdown")
    chunks = chunker.chunk_elements("DOC", _two_page_elements(), **META)
    table_chunks = [c for c in chunks if c.chunk_id in chunker.last_table_chunk_ids]
    text = table_chunks[0].text
    # Caption (statement title) and the nearest short line (units) both prefix the table.
    assert "Consolidated Statements of Income" in text
    assert "(in millions, except per share data)" in text


def test_oversized_table_splits_by_rows_with_header_repeated():
    # Tiny size forces the body to split across several parts.
    chunker = _chunker("markdown", size=90)
    rows = [[f"Row{i}", str(i), str(i * 2)] for i in range(1, 21)]
    elements = [_table(0, 1, ["Metric", "A", "B"], rows)]
    chunks = chunker.chunk_elements("DOC", elements, **META)
    table_chunks = [c for c in chunks if c.chunk_id in chunker.last_table_chunk_ids]
    assert len(table_chunks) >= 2  # the table was split
    for c in table_chunks:
        assert "| Metric | A | B |" in c.text  # header repeated in every part
    # No content lost: every body row lands in exactly one part.
    joined = "\n".join(c.text for c in table_chunks)
    for i in range(1, 21):
        assert f"| Row{i} |" in joined


# --- flat mode --------------------------------------------------------------


def test_giant_cell_table_never_exceeds_size():
    # An exhibit-index row can be one enormous cell that a row-group split can't
    # shrink; it must still be capped to the budget (else the embedder crashes on a
    # >512-token input, as PepsiCo p126 did).
    chunker = _chunker("markdown", size=90)
    giant = "senior notes due 2044 " * 40  # ~880 chars in one cell
    elements = [_table(0, 1, ["Ref", "Description"], [["4.60", giant]])]
    chunks = chunker.chunk_elements("DOC", elements, **META)
    table_chunks = [c for c in chunks if c.chunk_id in chunker.last_table_chunk_ids]
    assert len(table_chunks) >= 2  # the giant cell was split
    assert all(len(c.text) <= 90 for c in table_chunks)  # size measured in chars here
    # Content preserved across the split.
    assert "senior notes due 2044" in " ".join(c.text for c in table_chunks)


def test_flat_mode_renders_table_as_plaintext_no_pipes():
    chunks = _chunker("flat").chunk_elements("DOC", _two_page_elements(), **META)
    # Flat mode never emits markdown pipes; numbers sit next to their labels.
    assert all("|" not in c.text for c in chunks)
    page1 = " ".join(c.text for c in chunks if c.page_num == 1)
    assert "Net sales 100 90" in page1


def test_flat_mode_has_no_table_chunks():
    chunker = _chunker("flat")
    chunker.chunk_elements("DOC", _two_page_elements(), **META)
    assert chunker.last_table_chunk_ids == set()  # flat folds tables into prose


# --- determinism ------------------------------------------------------------


def test_chunk_ids_stable_across_runs_markdown():
    a = _chunker("markdown").chunk_elements("DOC", _two_page_elements(), **META)
    b = _chunker("markdown").chunk_elements("DOC", _two_page_elements(), **META)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert a == b


def test_chunk_ids_stable_across_runs_flat():
    a = _chunker("flat").chunk_elements("DOC", _two_page_elements(), **META)
    b = _chunker("flat").chunk_elements("DOC", _two_page_elements(), **META)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_chunk_ids_unique_within_doc():
    chunker = _chunker("markdown")
    chunks = chunker.chunk_elements("DOC", _two_page_elements(), **META)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# --- no Docling import ------------------------------------------------------


def test_ingestion_never_imports_docling():
    # Importing the whole ingestion stack must not pull the Docling package (it's a
    # Colab-only, multi-GB dependency). Our own modules are named *docling_artifact*
    # etc. under `ledgion.`, so match only the real top-level `docling` package.
    import ledgion.ingest.chunk  # noqa: F401
    import ledgion.ingest.docling_artifact  # noqa: F401
    import ledgion.ingest.pipeline  # noqa: F401

    loaded = [m for m in sys.modules if m == "docling" or m.startswith("docling.")]
    assert loaded == []
