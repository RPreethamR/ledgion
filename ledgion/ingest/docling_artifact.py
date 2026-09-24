"""Read the Docling JSONL artifact and render its elements — **no Docling import**.

Phase 8 runs Docling once on Colab (see ``parse_docling.py``) and everything
downstream reads the resulting ``<doc_id>.docling.jsonl`` here, on the laptop, with
only ``rapidfuzz`` (a core dep) touched. This module is that reader plus the two
table renderers the chunker needs:

* **flat** — a table as plain text (numbers next to their row labels, like PyMuPDF),
  for the parser-only ablation.
* **markdown** — a table as a markdown grid with an explicit header block, so the
  chunker can split an oversized table by row groups and repeat the header in each
  part.

It also hosts the ingestion **guard** (``golden_page_disagreements``): the Phase-1
evidence verification, re-run against Docling's page-labeled text as the chunker
labels it (multi-page elements → their first page). Every golden evidence string
must still best-match its golden page; the pipeline fails loudly otherwise.

Page convention (load-bearing): an element's ``page_num`` is its **first** physical
page — the same page FinanceBench labels evidence on — so a multi-page element is
labeled with its first page and every chunk carries exactly one page label.
"""

from __future__ import annotations

import json
from pathlib import Path

from rapidfuzz import fuzz

ARTIFACT_SUFFIX = ".docling.jsonl"
MANIFEST_NAME = "manifest.json"

# The guard matches an evidence string to the page that contains it with
# rapidfuzz.partial_ratio, which always makes the *shorter* string the pattern. Two
# consequences shape how we call it:
#  * A near-empty page (a stray page-number fragment like "10") is a substring magnet:
#    it scores 100 against any long evidence. Pages below MIN_MATCH_CHARS are excluded
#    — every golden evidence page holds a full statement/paragraph (>1100 chars here),
#    the fragments are <120, so this removes magnets without touching a real page.
#  * The full evidence string can be huge (a whole statement, ~4k chars); partial_ratio
#    over ~500 such pairs is minutes of CPU. We match on the evidence's leading SNIPPET
#    instead — the distinctive statement title / first sentence that identifies the page
#    — which is far shorter than a page, so the page is the searched text (correct
#    containment) and the match is ~100x cheaper.
MIN_MATCH_CHARS = 300
EVIDENCE_SNIPPET_CHARS = 500

# In markdown mode, a table cell text is escaped for markdown and newlines are
# flattened so one cell stays on one line of the grid.
_MD_ESCAPE = str.maketrans({"|": r"\|", "\n": " ", "\r": " "})


# -- locating & loading ------------------------------------------------------


def artifact_path(docling_dir: Path, doc_id: str) -> Path:
    return Path(docling_dir) / f"{doc_id}{ARTIFACT_SUFFIX}"


def doc_ids_in(docling_dir: Path) -> list[str]:
    """The doc_ids the artifact directory holds (one JSONL per document)."""
    return sorted(
        p.name[: -len(ARTIFACT_SUFFIX)] for p in Path(docling_dir).glob(f"*{ARTIFACT_SUFFIX}")
    )


def has_artifact(docling_dir: Path) -> bool:
    return bool(Path(docling_dir).exists() and any(Path(docling_dir).glob(f"*{ARTIFACT_SUFFIX}")))


def load_manifest(docling_dir: Path) -> dict | None:
    """The run manifest (Docling version, parse options, per-doc stats), or None."""
    path = Path(docling_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_elements(jsonl_path: Path) -> list[dict]:
    """Every element for one document, in reading (``element_index``) order."""
    rows: list[dict] = []
    with Path(jsonl_path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("element_index", 0))
    return rows


def group_by_page(elements: list[dict]) -> dict[int, list[dict]]:
    """Group elements by their (first-page) ``page_num``, preserving reading order.

    Elements without a page (rare — no provenance) are dropped: they can't be page-
    scored. A multi-page element lands wholly on its first page — the labeling
    convention this whole phase rests on.
    """
    by_page: dict[int, list[dict]] = {}
    for el in elements:
        page = el.get("page_num")
        if page is None:
            continue
        by_page.setdefault(page, []).append(el)
    return by_page


# -- table rendering ---------------------------------------------------------


def _grid_dims(table: dict) -> tuple[int, int]:
    cells = table.get("cells") or []
    num_rows = table.get("num_rows") or (max((c.get("row", -1) for c in cells), default=-1) + 1)
    num_cols = table.get("num_cols") or (max((c.get("col", -1) for c in cells), default=-1) + 1)
    return int(num_rows or 0), int(num_cols or 0)


def _build_grid(table: dict, *, escape: bool) -> tuple[list[list[str]], list[bool]]:
    """Place each cell's text at its top-left (row, col); return the grid and, per
    row, whether it holds a column-header cell. Spans aren't expanded — markdown has
    no spans — so a spanned cell's extra positions stay blank."""
    num_rows, num_cols = _grid_dims(table)
    grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    header_row = [False] * num_rows
    for c in table.get("cells") or []:
        r, col = c.get("row"), c.get("col")
        if r is None or col is None or not (0 <= r < num_rows and 0 <= col < num_cols):
            continue
        text = (c.get("text") or "").strip()
        if escape:
            text = text.translate(_MD_ESCAPE)
        if not grid[r][col]:
            grid[r][col] = text
        if c.get("column_header"):
            header_row[r] = True
    return grid, header_row


def render_table_flat(table: dict) -> str:
    """A table as plain text: caption, then each row's non-empty cells joined by
    spaces — numbers stay adjacent to their labels, as PyMuPDF's extraction gives."""
    grid, _ = _build_grid(table, escape=False)
    lines: list[str] = []
    caption = (table.get("caption") or "").strip()
    if caption:
        lines.append(caption)
    for row in grid:
        cells = [x for x in row if x]
        if cells:
            lines.append(" ".join(cells))
    return "\n".join(lines)


def table_markdown_parts(table: dict) -> tuple[list[str], list[str]]:
    """Return ``(header_lines, body_lines)`` as markdown.

    ``header_lines`` are the leading header rows plus the ``| --- |`` separator;
    ``body_lines`` is one markdown string per remaining row. If Docling flagged no
    column-header cell, the first row is treated as the header — so there is always a
    header to repeat when an oversized table is split by row groups.
    """
    num_rows, num_cols = _grid_dims(table)
    if num_rows == 0 or num_cols == 0:
        return [], []
    grid, header_row = _build_grid(table, escape=True)

    n_head = 0
    while n_head < num_rows and header_row[n_head]:
        n_head += 1
    if n_head == 0:
        n_head = 1  # fallback: first row is the header, so every split part carries one

    def row_md(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join(["---"] * num_cols) + " |"
    header_lines = [row_md(r) for r in grid[:n_head]] + [separator]
    body_lines = [row_md(r) for r in grid[n_head:]]
    return header_lines, body_lines


# -- page-labeled text (for the guard) ---------------------------------------


def _element_flat_text(element: dict) -> str:
    if element.get("element_type") == "table":
        return render_table_flat(element.get("table") or {})
    return element.get("text") or ""


def page_flat_texts(docling_dir: Path, doc_id: str) -> dict[int, str]:
    """Per-page plain text as the chunker labels it (first-page assignment, tables
    flattened). This is what the guard matches golden evidence against — the page
    *assignment* is what it verifies, and that is identical in flat and markdown
    modes, so flat rendering is used for both."""
    by_page = group_by_page(load_elements(artifact_path(docling_dir, doc_id)))
    out: dict[int, str] = {}
    for page, elements in by_page.items():
        parts = [t for el in elements if (t := _element_flat_text(el).strip())]
        out[page] = "\n".join(parts)
    return out


# -- the ingestion guard -----------------------------------------------------


def _best_match_page(evidence_text: str, pages: dict[int, str]) -> tuple[int, float]:
    snippet = evidence_text[:EVIDENCE_SNIPPET_CHARS]
    best_page, best_score = 0, -1.0
    for page, text in pages.items():
        if len(text) < MIN_MATCH_CHARS:  # skip near-empty pages (substring magnets)
            continue
        score = fuzz.partial_ratio(snippet, text)
        if score > best_score:
            best_page, best_score = page, score
    return best_page, best_score


def golden_page_disagreements(docling_dir: Path, golden_rows: list[dict]) -> list[dict]:
    """Phase-1 evidence verification against Docling's page-labeled text.

    For every golden row, the labelled ``evidence_text`` must best-match (rapidfuzz
    ``partial_ratio``) one of its ``evidence_pages`` among that document's Docling
    page texts. Returns one dict per DISAGREEMENT (empty list = artifact trustworthy);
    the pipeline raises on any, exactly as the manual Phase-1 check did.
    """
    texts_by_doc: dict[str, dict[int, str]] = {}
    disagreements: list[dict] = []
    for row in golden_rows:
        doc_id = row["doc_id"]
        if doc_id not in texts_by_doc:
            texts_by_doc[doc_id] = page_flat_texts(docling_dir, doc_id)
        best_page, score = _best_match_page(row["evidence_text"], texts_by_doc[doc_id])
        if best_page not in row["evidence_pages"]:
            disagreements.append(
                {
                    "qid": row["qid"],
                    "doc_id": doc_id,
                    "evidence_pages": list(row["evidence_pages"]),
                    "best_page": best_page,
                    "score": round(score, 1),
                }
            )
    return disagreements
