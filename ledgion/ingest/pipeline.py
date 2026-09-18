"""Ingestion pipeline: parse -> chunk -> embed -> index, over ``data/pdfs/``.

This is the one stage that does I/O and wiring; ``parse``/``chunk``/``embed``/
``index`` stay independently usable. It also owns **document metadata
resolution**: ``company``, ``fiscal_year`` and ``form_type`` come from
FinanceBench's ``financebench_document_information.jsonl`` (never parsed from
filenames), and ``ticker`` from the hand-maintained ``tickers`` map in config
(FinanceBench has no ticker field). An unmapped company is a hard error, so the
map can never silently drift behind the corpus.

Per-document progress (pages, chunks, cache hits, elapsed) is printed as each
filing lands, and totals at the end.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ledgion.config import REPO_ROOT, Settings, load_config
from ledgion.ingest.chunk import RecursiveChunker
from ledgion.ingest.embed import BGEEmbedder
from ledgion.ingest.index import QdrantIndexer
from ledgion.ingest.parse import parse_pdf

logger = logging.getLogger(__name__)

DOC_INFO_FILE = "financebench_document_information.jsonl"

# FinanceBench records doc_type lowercase and compact ("10k"); we store the
# conventional SEC form name. Anything unmapped falls back to upper-case as-is.
_FORM_TYPE = {"10k": "10-K", "10q": "10-Q"}


@dataclass(frozen=True, slots=True)
class DocMeta:
    """The four metadata fields every ``Chunk`` from a document carries."""

    company: str
    ticker: str
    fiscal_year: int
    form_type: str


def _resolve(path: Path) -> Path:
    """Resolve a (possibly relative) config path against the repo root."""
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_metadata(cfg: Settings, doc_ids: set[str]) -> dict[str, DocMeta]:
    """Map each ``doc_id`` (a PDF stem) to its ``DocMeta`` from FinanceBench.

    Raises if a filing is missing from FinanceBench's document information, or if
    its company has no ticker in the config map — both are corpus/config drift we
    want to catch loudly at ingest, not paper over with blank fields.
    """
    info_path = _resolve(cfg.paths.financebench_dir) / DOC_INFO_FILE
    records: dict[str, dict] = {}
    with info_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["doc_name"] in doc_ids:
                records[rec["doc_name"]] = rec

    missing = sorted(doc_ids - records.keys())
    if missing:
        raise KeyError(f"no FinanceBench document information for: {missing}")

    meta: dict[str, DocMeta] = {}
    for doc_id, rec in records.items():
        company = rec["company"]
        ticker = cfg.tickers.get(company)
        if ticker is None:
            raise KeyError(
                f"company {company!r} (from {doc_id}) has no ticker in config.tickers; "
                f"add it to config/default.yaml"
            )
        doc_type = str(rec["doc_type"]).lower()
        meta[doc_id] = DocMeta(
            company=company,
            ticker=ticker,
            fiscal_year=int(rec["doc_period"]),
            form_type=_FORM_TYPE.get(doc_type, doc_type.upper()),
        )
    return meta


def run(cfg: Settings | None = None, *, force: bool = False) -> dict:
    """Run the full pipeline over every PDF in ``paths.pdf_dir``.

    ``force`` bypasses the embedding cache (recompute every vector). Returns a
    summary dict (per-document counts, totals, collection size).
    """
    cfg = cfg or load_config()

    # Pin OpenMP threads before torch is imported (inside the embedder) so the
    # cap actually takes effect; mirrors torch.set_num_threads in the embedder.
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.torch.num_threads))

    pdf_dir = _resolve(cfg.paths.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}; nothing to ingest.")

    metadata = resolve_metadata(cfg, {p.stem for p in pdfs})
    chunker = RecursiveChunker.from_config(cfg)
    embedder = BGEEmbedder.from_config(cfg, force=force)
    indexer = QdrantIndexer.from_config(cfg)

    print(
        f"Ingesting {len(pdfs)} filing(s)  "
        f"[chunk unit={cfg.chunk.unit} size={cfg.chunk.size} overlap={cfg.chunk.overlap}, "
        f"force={force}]"
    )
    print(f"{'doc_id':<26} {'pages':>6} {'chunks':>7} {'cache_hits':>11} {'elapsed':>9}")

    per_doc: list[dict] = []
    total_pages = total_chunks = 0
    t_start = time.perf_counter()

    for pdf in pdfs:
        doc_id = pdf.stem
        meta = metadata[doc_id]
        t0 = time.perf_counter()

        pages = parse_pdf(pdf)
        chunks = chunker.chunk(
            doc_id,
            [(p["page_num"], p["text"]) for p in pages],
            company=meta.company,
            ticker=meta.ticker,
            fiscal_year=meta.fiscal_year,
            form_type=meta.form_type,
        )

        hits_before = embedder.cache_hits
        vectors = embedder.embed_documents([c.text for c in chunks])
        hits = embedder.cache_hits - hits_before

        indexer.upsert(chunks, vectors)
        elapsed = time.perf_counter() - t0

        print(f"{doc_id:<26} {len(pages):>6} {len(chunks):>7} {hits:>11} {elapsed:>8.1f}s")
        per_doc.append(
            {
                "doc_id": doc_id,
                "pages": len(pages),
                "chunks": len(chunks),
                "cache_hits": hits,
                "elapsed_s": round(elapsed, 2),
            }
        )
        total_pages += len(pages)
        total_chunks += len(chunks)

    wall = time.perf_counter() - t_start
    collection_count = indexer.count()
    print(
        f"{'TOTAL':<26} {total_pages:>6} {total_chunks:>7} "
        f"{embedder.cache_hits:>11} {wall:>8.1f}s"
    )
    print(
        f"collection {cfg.qdrant.collection_name!r}: {collection_count} points  "
        f"(cache misses this run: {embedder.cache_misses})"
    )

    return {
        "documents": per_doc,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "wall_s": round(wall, 2),
        "collection_count": collection_count,
        "cache_hits": embedder.cache_hits,
        "cache_misses": embedder.cache_misses,
    }
