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
from ledgion.ingest.chunk import RecursiveChunker, TableAwareChunker
from ledgion.ingest.docling_artifact import (
    doc_ids_in,
    golden_page_disagreements,
    has_artifact,
    load_manifest,
)
from ledgion.ingest.embed import BGEEmbedder
from ledgion.ingest.index import QdrantIndexer
from ledgion.ingest.parse import parse_pdf

logger = logging.getLogger(__name__)

DOC_INFO_FILE = "financebench_document_information.jsonl"
GOLDEN_PATH = REPO_ROOT / "golden" / "dev.jsonl"

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


def _load_golden() -> list[dict]:
    if not GOLDEN_PATH.exists():
        raise SystemExit(f"{GOLDEN_PATH} not found; run `python -m ledgion.eval.build_golden`.")
    with GOLDEN_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _build_chunker(cfg: Settings):
    """The chunker ``parser.backend`` selects: PyMuPDF pages → RecursiveChunker, or the
    Docling artifact → TableAwareChunker (table_mode flat|markdown)."""
    if cfg.parser.backend == "docling":
        return TableAwareChunker.from_config(cfg)
    return RecursiveChunker.from_config(cfg)


def _docling_preflight(cfg: Settings, docling_dir: Path) -> dict:
    """Fail fast if the Docling artifact is missing, then run the golden guard.

    The guard is the Phase-1 evidence verification, automated: every golden evidence
    string must best-match its golden page in Docling's page-labeled text (after the
    first-page assignment the chunker uses). A disagreement aborts ingestion loudly —
    a page-level index built on a mislabeled corpus would score against wrong truth.
    """
    if not has_artifact(docling_dir):
        raise SystemExit(
            f"No Docling artifact in {docling_dir}. Produce it once on Colab "
            f"(notebooks/colab_docling_parse.ipynb) and copy the *.docling.jsonl + "
            f"manifest.json into {docling_dir}."
        )
    golden = _load_golden()
    disagreements = golden_page_disagreements(docling_dir, golden)
    if disagreements:
        detail = "\n".join(
            f"  {d['qid']} {d['doc_id']}: best-match p{d['best_page']} "
            f"(score {d['score']}) not in golden {d['evidence_pages']}"
            for d in disagreements
        )
        raise SystemExit(
            "Docling golden-alignment guard FAILED — the artifact and the golden set "
            f"disagree on {len(disagreements)} evidence page(s):\n{detail}\n"
            "Re-check the Colab parse before indexing; page-level metrics would be wrong."
        )
    print(f"golden guard OK: all {len(golden)} evidence strings best-match their golden page")
    manifest = load_manifest(docling_dir) or {}
    return {d["doc_id"]: d for d in manifest.get("documents", [])}


def run(cfg: Settings | None = None, *, force: bool = False) -> dict:
    """Run the full ingestion pipeline over the corpus. Returns a summary dict.

    ``parser.backend`` chooses the source: ``pymupdf`` parses PDFs from
    ``paths.pdf_dir`` live; ``docling`` reads the pre-parsed artifact in
    ``parser.docling_dir`` (no Docling, no PDF parse) and applies the golden guard
    first. ``force`` bypasses the embedding cache.
    """
    cfg = cfg or load_config()

    # Pin OpenMP threads before torch is imported (inside the embedder) so the
    # cap actually takes effect; mirrors torch.set_num_threads in the embedder.
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.torch.num_threads))

    parser = cfg.parser.backend
    docling_stats: dict[str, dict] = {}
    if parser == "docling":
        docling_dir = _resolve(cfg.parser.docling_dir)
        docling_stats = _docling_preflight(cfg, docling_dir)
        doc_ids = doc_ids_in(docling_dir)
        pdf_by_doc: dict[str, Path] = {}
    else:
        pdf_dir = _resolve(cfg.paths.pdf_dir)
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            raise SystemExit(f"No PDFs found in {pdf_dir}; nothing to ingest.")
        pdf_by_doc = {p.stem: p for p in pdfs}
        doc_ids = sorted(pdf_by_doc)

    if not doc_ids:
        raise SystemExit("No documents to ingest.")

    metadata = resolve_metadata(cfg, set(doc_ids))
    chunker = _build_chunker(cfg)
    embedder = BGEEmbedder.from_config(cfg, force=force)
    indexer = QdrantIndexer.from_config(cfg)

    table_mode = f" table_mode={cfg.chunk.table_mode}" if parser == "docling" else ""
    print(
        f"Ingesting {len(doc_ids)} filing(s)  [parser={parser}{table_mode} "
        f"chunk unit={cfg.chunk.unit} size={cfg.chunk.size} overlap={cfg.chunk.overlap}, "
        f"force={force}]  ->  collection {cfg.qdrant.collection_name!r}"
    )
    print(
        f"{'doc_id':<26} {'pages':>6} {'chunks':>7} {'tbl_chk':>8} "
        f"{'cache_hits':>11} {'elapsed':>9}"
    )

    per_doc: list[dict] = []
    total_pages = total_chunks = total_table_chunks = 0
    t_start = time.perf_counter()

    for doc_id in doc_ids:
        meta = metadata[doc_id]
        t0 = time.perf_counter()

        if parser == "docling":
            chunks = chunker.chunk(doc_id, [], **_meta_kwargs(meta))
            table_chunks = len(chunker.last_table_chunk_ids)
            n_pages = docling_stats.get(doc_id, {}).get("page_count")
            if n_pages is None:  # manifest lacked it — derive from the chunks
                n_pages = len({c.page_num for c in chunks})
        else:
            pages = parse_pdf(pdf_by_doc[doc_id])
            chunks = chunker.chunk(
                doc_id, [(p["page_num"], p["text"]) for p in pages], **_meta_kwargs(meta)
            )
            table_chunks = 0
            n_pages = len(pages)

        hits_before = embedder.cache_hits
        vectors = embedder.embed_documents(
            [c.text for c in chunks], ids=[c.chunk_id for c in chunks]
        )
        hits = embedder.cache_hits - hits_before

        indexer.upsert(chunks, vectors)
        elapsed = time.perf_counter() - t0

        print(
            f"{doc_id:<26} {n_pages:>6} {len(chunks):>7} {table_chunks:>8} "
            f"{hits:>11} {elapsed:>8.1f}s"
        )
        per_doc.append(
            {
                "doc_id": doc_id,
                "pages": n_pages,
                "chunks": len(chunks),
                "table_chunks": table_chunks,
                "cache_hits": hits,
                "elapsed_s": round(elapsed, 2),
            }
        )
        total_pages += n_pages
        total_chunks += len(chunks)
        total_table_chunks += table_chunks

    wall = time.perf_counter() - t_start
    collection_count = indexer.count()
    print(
        f"{'TOTAL':<26} {total_pages:>6} {total_chunks:>7} {total_table_chunks:>8} "
        f"{embedder.cache_hits:>11} {wall:>8.1f}s"
    )
    chunks_per_page = round(total_chunks / total_pages, 3) if total_pages else 0.0
    table_frac = round(total_table_chunks / total_chunks, 3) if total_chunks else 0.0
    print(
        f"collection {cfg.qdrant.collection_name!r}: {collection_count} points  "
        f"({chunks_per_page} chunks/page, table-chunk fraction {table_frac}, "
        f"cache misses {embedder.cache_misses})"
    )
    if parser == "docling":
        docling_parse_s = sum((d.get("parse_time_s") or 0) for d in docling_stats.values())
        print(
            f"Docling parse (Colab GPU, from manifest): {docling_parse_s:.1f}s total  "
            f"| local ingest (chunk+embed+index) this run: {wall:.1f}s"
        )

    return {
        "parser": parser,
        "table_mode": cfg.chunk.table_mode if parser == "docling" else None,
        "collection_name": cfg.qdrant.collection_name,
        "documents": per_doc,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "total_table_chunks": total_table_chunks,
        "chunks_per_page": chunks_per_page,
        "table_chunk_fraction": table_frac,
        "wall_s": round(wall, 2),
        "collection_count": collection_count,
        "cache_hits": embedder.cache_hits,
        "cache_misses": embedder.cache_misses,
    }


def _meta_kwargs(meta: DocMeta) -> dict:
    return {
        "company": meta.company,
        "ticker": meta.ticker,
        "fiscal_year": meta.fiscal_year,
        "form_type": meta.form_type,
    }
