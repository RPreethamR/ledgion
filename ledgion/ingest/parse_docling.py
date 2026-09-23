"""PDF -> per-element JSONL with Docling. **Colab-only, run once, artifact out.**

Docling is a layout-model + TableFormer parser: multiple GB of torch and model
weights that do not fit this project's 4GB-VRAM / 8GB-RAM laptop (see CLAUDE.md).
So it never runs in the local pipeline. Instead it runs **once on Colab's GPU**
(``notebooks/colab_docling_parse.ipynb``) and emits a portable artifact — one
JSONL per document — that every downstream step (chunking, table serialization,
indexing) consumes locally, with no Docling dependency in sight.

Design constraints this file must honour, all load-bearing:

* **Zero package imports.** Like ``parse.py``, this module imports *nothing* from
  the rest of ``ledgion``. Docling itself is imported **lazily inside the
  functions**, not at module top, so the file imports with only the standard
  library present — a fresh Colab env needs just ``pip install docling`` for the
  parse calls to work, and importing the module never pulls torch. Keep the
  package ``__init__.py`` files free of heavy imports for the same reason.
* **Store the grid, not only a string.** Table serialization (markdown vs. a
  flattened "label: value" form vs. HTML) is a *local* ablation knob for Phase 8.
  So every table's raw cell grid — cells with row/col indices, spans, and header
  flags — is written to the artifact verbatim. A different serialization is then
  a local re-read of this JSONL, never a re-run of Colab.
* **Page numbers are 1-based physical PDF pages**, identical to ``parse.py``'s
  PyMuPDF convention (Docling's ``ProvenanceItem.page_no`` is already 1-based).
  This is the stable key retrieval is scored on, so the two parsers must agree on
  it to the page or the parser ablation is meaningless.
* **Multi-page elements are recorded, never split.** If Docling's provenance for
  one element spans more than one page, every page is listed in ``pages``,
  ``multi_page`` is set, and ``parse_document`` counts them so the run manifest
  can report how many exist. Splitting is a downstream decision, made locally.

Public surface:

* :class:`ParseConfig`        — the parse knobs (OCR off, ACCURATE tables, GPU).
* :func:`build_converter`     — a configured Docling ``DocumentConverter`` (loads
                                models once; reuse it across documents).
* :func:`environment_manifest`— Docling/model versions, resolved options, GPU —
                                everything the run manifest needs *except* the
                                per-document rows.
* :func:`parse_document`      — parse one PDF with a prebuilt converter, write its
                                JSONL, return per-document stats for the manifest.
* :func:`parse_pdf`           — the pure one-shot: PDF path in, JSONL path out.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Latest Docling at authoring time (2026-09). Informational: the *binding* pin is
# the ``docling`` dependency group in pyproject.toml, and the notebook installs the
# same ``==`` so the Colab run and the recorded manifest version agree.
DOCLING_PIN = "2.130.0"

# Docling's OCR was verified off in the spike (filings are text-native, no scanned
# pages — DECISIONS.md). Every field here is echoed into the manifest so a reader
# knows exactly how the artifact was produced.
@dataclass(frozen=True, slots=True)
class ParseConfig:
    use_gpu: bool = True          # use CUDA when torch reports a device; else CPU
    do_ocr: bool = False          # filings are text-native; OCR off (verified)
    table_mode: str = "accurate"  # TableFormer: "accurate" | "fast"
    do_cell_matching: bool = True  # map predicted cells back onto extracted PDF cells


# --- Element category -------------------------------------------------------
#
# Docling emits a fine-grained DocItemLabel; we collapse it to the five coarse
# buckets the artifact promises (heading | text | table | list | other). The raw
# label is *also* written to every record (``docling_label``) so a local ablation
# can remap categories — e.g. treat captions as text — without re-parsing.
_ELEMENT_TYPE: dict[str, str] = {
    "title": "heading",
    "section_header": "heading",
    "list_item": "list",
    "table": "table",
    "text": "text",
    "paragraph": "text",
    "caption": "text",
    "footnote": "text",
    "code": "text",
    "formula": "text",
}


def _element_type(label_value: str) -> str:
    """Coarse bucket for a Docling label; anything unmapped is ``"other"``."""
    return _ELEMENT_TYPE.get(label_value, "other")


def _cuda_available(use_gpu: bool) -> bool:
    """True iff we should place Docling's models on CUDA.

    Docling depends on torch, so torch imports in any env that has Docling. Kept
    tolerant: a torch without CUDA, or any import hiccup, falls back to CPU rather
    than crashing the run.
    """
    if not use_gpu:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def build_converter(config: ParseConfig | None = None) -> Any:
    """Build a Docling ``DocumentConverter`` for the given knobs.

    Loading a converter loads the layout + TableFormer models, so build it **once**
    and pass it to :func:`parse_document` for every PDF — the notebook does exactly
    this so a 5-document run pays the model-load cost a single time.
    """
    config = config or ParseConfig()

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # AcceleratorOptions moved to its own module in modern Docling but is still
    # re-exported from pipeline_options in older builds — try the canonical path
    # first so the pin can move without editing this import.
    try:
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )
    except ImportError:  # pragma: no cover - depends on installed docling version
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )

    device = AcceleratorDevice.CUDA if _cuda_available(config.use_gpu) else AcceleratorDevice.CPU

    pipeline_options = PdfPipelineOptions(
        do_ocr=config.do_ocr,
        do_table_structure=True,
        accelerator_options=AcceleratorOptions(device=device),
    )
    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE if config.table_mode == "accurate" else TableFormerMode.FAST
    )
    pipeline_options.table_structure_options.do_cell_matching = config.do_cell_matching

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _package_version(name: str) -> str | None:
    """Installed version of a distribution, or ``None`` if it isn't present."""
    import importlib.metadata as md

    try:
        return md.version(name)
    except Exception:
        return None


def environment_manifest(config: ParseConfig | None = None) -> dict:
    """The run-level manifest block: versions, resolved options, and the GPU.

    Everything the run manifest needs *except* the per-document rows (those come
    from :func:`parse_document`). "Model versions where exposed" resolves to the
    installed ``docling-ibm-models`` / ``docling-parse`` distribution versions —
    the layout and TableFormer weights are versioned with those packages — plus
    Docling itself. The device recorded here is the one that will actually be used
    (same GPU check :func:`build_converter` applies), not merely the one requested.
    """
    config = config or ParseConfig()

    gpu: dict[str, Any] = {"cuda_available": False, "name": None, "torch_version": None}
    try:
        import torch

        gpu["torch_version"] = torch.__version__
        gpu["cuda_available"] = bool(torch.cuda.is_available())
        if gpu["cuda_available"]:
            gpu["name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    device = "cuda" if (config.use_gpu and gpu["cuda_available"]) else "cpu"

    return {
        "docling_version": _package_version("docling"),
        "docling_core_version": _package_version("docling-core"),
        "docling_ibm_models_version": _package_version("docling-ibm-models"),
        "docling_parse_version": _package_version("docling-parse"),
        "parse_options": {
            "do_ocr": config.do_ocr,
            "do_table_structure": True,
            "table_mode": config.table_mode,
            "do_cell_matching": config.do_cell_matching,
            "requested_gpu": config.use_gpu,
            "device": device,
        },
        "gpu": gpu,
    }


def _element_pages(item: Any) -> list[int]:
    """Sorted, de-duplicated 1-based page numbers an element's provenance covers."""
    prov = getattr(item, "prov", None) or []
    return sorted({p.page_no for p in prov})


def _table_markdown(item: Any, doc: Any) -> str:
    """Plain-text (markdown) rendering of a table, for the record's ``text`` field.

    Modern docling-core's ``export_to_markdown`` takes the owning document; older
    builds took no argument — accept both so the pin can move.
    """
    try:
        return item.export_to_markdown(doc)
    except TypeError:
        return item.export_to_markdown()
    except Exception:
        return ""


def _table_grid(item: Any, doc: Any) -> dict:
    """The raw cell grid for a table — the whole point of the Colab artifact.

    One entry per cell with its position (``row``/``col`` = the cell's top-left
    offset), its ``row_span``/``col_span``, and Docling's header flags. Spans are
    read from Docling's own fields when present and otherwise derived from the
    start/end offsets, so the grid survives minor schema drift across versions.
    """
    data = item.data
    cells: list[dict] = []
    for cell in getattr(data, "table_cells", []) or []:
        start_row = getattr(cell, "start_row_offset_idx", None)
        end_row = getattr(cell, "end_row_offset_idx", None)
        start_col = getattr(cell, "start_col_offset_idx", None)
        end_col = getattr(cell, "end_col_offset_idx", None)
        row_span = getattr(cell, "row_span", None)
        col_span = getattr(cell, "col_span", None)
        if row_span is None and start_row is not None and end_row is not None:
            row_span = end_row - start_row
        if col_span is None and start_col is not None and end_col is not None:
            col_span = end_col - start_col
        cells.append(
            {
                "text": getattr(cell, "text", "") or "",
                "row": start_row,
                "col": start_col,
                "row_span": row_span,
                "col_span": col_span,
                "column_header": bool(getattr(cell, "column_header", False)),
                "row_header": bool(getattr(cell, "row_header", False)),
                "row_section": bool(getattr(cell, "row_section", False)),
            }
        )

    caption = ""
    try:
        caption = item.caption_text(doc) or ""
    except Exception:
        caption = ""

    return {
        "num_rows": getattr(data, "num_rows", None),
        "num_cols": getattr(data, "num_cols", None),
        "caption": caption,
        "cells": cells,
    }


def _element_record(item: Any, doc: Any, doc_id: str, index: int) -> dict:
    """Build the JSON record for one Docling item, in the artifact's schema."""
    label = getattr(item, "label", None)
    label_value = getattr(label, "value", None) or (str(label) if label is not None else "")

    pages = _element_pages(item)
    element_type = _element_type(label_value)

    if element_type == "table":
        text = _table_markdown(item, doc)
    else:
        # TextItem / SectionHeaderItem / ListItem expose .text; pictures and other
        # non-textual items have none, and get an empty string (still page-located).
        text = getattr(item, "text", "") or ""

    record = {
        "doc_id": doc_id,
        "element_index": index,
        "page_num": pages[0] if pages else None,  # first physical page, 1-based
        "pages": pages,
        "multi_page": len(pages) > 1,
        "element_type": element_type,
        "docling_label": label_value,
        "text": text,
    }
    if element_type == "table":
        record["table"] = _table_grid(item, doc)
    return record


def parse_document(
    converter: Any,
    pdf_path: str | Path,
    out_path: str | Path,
    *,
    doc_id: str | None = None,
) -> dict:
    """Parse one PDF with a prebuilt ``converter`` and write its JSONL to ``out_path``.

    Returns a per-document manifest row: ``doc_id``, ``page_count``, element and
    multi-page counts, conversion status, and parse time. The JSONL is written
    **atomically** — to ``<out_path>.tmp`` then ``os.replace`` — so a file that
    exists at ``out_path`` is always complete. That is what makes the notebook's
    "skip documents already parsed" resume safe against a mid-write disconnect.
    """
    pdf_path = Path(pdf_path)
    out_path = Path(out_path)
    doc_id = doc_id or pdf_path.stem

    t0 = time.perf_counter()
    result = converter.convert(pdf_path)
    doc = result.document

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.parent / (out_path.name + ".tmp")

    n_elements = 0
    n_multi_page = 0
    with tmp_path.open("w", encoding="utf-8") as fh:
        # iterate_items() walks the body tree in reading order and omits furniture
        # (page headers/footers) and groups by default — exactly the body content
        # we want, in order. It yields (item, level); the depth is not needed here.
        for item, _level in doc.iterate_items():
            record = _element_record(item, doc, doc_id, n_elements)
            if record["multi_page"]:
                n_multi_page += 1
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_elements += 1
    os.replace(tmp_path, out_path)

    elapsed = time.perf_counter() - t0
    return {
        "doc_id": doc_id,
        "output": out_path.name,
        "page_count": len(getattr(doc, "pages", {}) or {}),
        "n_elements": n_elements,
        "n_multi_page_elements": n_multi_page,
        "status": str(getattr(result, "status", "")),
        "parse_time_s": round(elapsed, 2),
    }


def parse_pdf(
    pdf_path: str | Path,
    out_path: str | Path,
    *,
    config: ParseConfig | None = None,
    doc_id: str | None = None,
) -> dict:
    """One-shot pure parse: PDF path in, JSONL path out, per-document stats returned.

    Builds a fresh converter for this single document — convenient for a one-off or
    a test, but it reloads the models every call. For a multi-document run reuse a
    single :func:`build_converter` across :func:`parse_document` calls instead.
    """
    converter = build_converter(config)
    return parse_document(converter, pdf_path, out_path, doc_id=doc_id)


# Re-exported so ``from parse_docling import *`` in a notebook pulls the surface.
__all__ = [
    "DOCLING_PIN",
    "ParseConfig",
    "build_converter",
    "environment_manifest",
    "parse_document",
    "parse_pdf",
]
