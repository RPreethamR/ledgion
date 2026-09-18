## Spike — corpus verification
FinanceBench public subset: 150 questions, 84 documents, all
downloadable. PyMuPDF extracts clean text; no scanned pages.
Evidence strings match FinanceBench's own evidence_page_num at
96-99 partial_ratio, with a consistent 0-based → 1-based offset.

Decisions:
- Use FinanceBench's evidence_page_num directly as ground truth;
  fuzzy matching becomes a one-time verification pass only.
- PyMuPDF stays through Phase 7. Tables extract with numbers
  adjacent to row labels (checked on Amex). Docling remains a
  Phase 8 ablation, not a dependency.
- Working set: AMD, Amex, Boeing, PepsiCo, Amcor 10-Ks.
  PepsiCo at 503pp is the stress case.

Open: table quality unverified on AMD/PepsiCo. If numeric
questions underperform prose ones, suspect extraction first.

## Phase 1 — golden set
30 questions across 5 filings. Ground truth from FinanceBench's
evidence_page_num via to_one_based(); verification 30/30 AGREE,
mean partial_ratio 96.7. Boeing id_01290 scores 61.8 — evidence
joins 3 pages, best match still on the labelled page.

answer_type classified on what the question requires, not the
answer's surface form. Initial heuristic gave 2/28 , numeric/prose; corrected
to 16/14 by reclassification and by hand.

Selected set derived from PDFs on disk, not a manifest file.

## Phase 2 — ingestion
2181 chunks / 1230 pages across 5 filings (1.77 chunks/page).
Cold run 16.7 min, warm 41s with 2181/2181 cache hits.
Qdrant volume 22 MiB, embedding cache 6.6 MiB.

Token-length recursive splitter, size=448 (headroom below bge's
512 max; truncation tripwire never fired). Named dense vector
(768-d, cosine) so Phase 6 sparse needs no migration.
Ticker from hand-maintained config map; unmapped company is a
hard error. Indexed filterable alongside doc_id/page_num.
Cache key = sha256(revision + text) — 2178 files for 2181 chunks,
3 boilerplate duplicates dedup correctly.
Qdrant pinned v1.19.1 to match client 1.19.0.