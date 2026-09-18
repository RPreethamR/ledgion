"""PDF -> per-page text with PyMuPDF.

Deliberately self-contained: this module imports **nothing from the rest of the
package**. It is pure file-in / data-out, so the exact same function can be
lifted onto Colab (where a GPU or more RAM is available) and run unchanged to
pre-extract a corpus. Everything downstream (chunking, metadata) is layered on
top by the pipeline, never in here.

Pages are **1-based** to match the project-wide convention (see
``ledgion/eval/golden.py``: FinanceBench's 0-based ``evidence_page_num`` is
converted once, and PyMuPDF's own 0-based index maps to our page ``N+1``). The
first page a human sees is page 1, and that is the stable key retrieval is
scored on.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf


def parse_pdf(path: str | Path) -> list[dict]:
    """Extract text from every page of ``path``.

    Returns one dict per page, in reading order::

        [{"page_num": 1, "text": "..."}, {"page_num": 2, "text": "..."}, ...]

    ``page_num`` is 1-based. ``text`` is PyMuPDF's plain-text extraction
    (``page.get_text()``) — layout is not preserved, but the spike (DECISIONS.md)
    confirmed numbers stay adjacent to their row labels, which is what matters
    for financial tables. Empty/near-empty pages (covers, blank separators) are
    kept, not dropped, so ``page_num`` always equals the physical PDF page and
    never drifts from FinanceBench's evidence pages.
    """
    path = Path(path)
    pages: list[dict] = []
    # `with` closes the file handle deterministically — important on Windows,
    # where a lingering handle blocks re-opening the same PDF in a later step.
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc):  # index is 0-based (PyMuPDF)
            pages.append({"page_num": index + 1, "text": page.get_text()})
    return pages
