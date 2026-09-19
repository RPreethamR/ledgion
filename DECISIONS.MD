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

## Phase 3 — baseline pipeline
Dense-only retrieval + gemini-3.8-flash generation, temp=0.
Citations validated server-side against the context sent.
Refusal path verified — out-of-corpus question returns
insufficient_evidence with no citations.

gemini-3.8-flash is a thinking model; reasoning tokens count
against max_output_tokens. For example, 979 of 1024 went to thinking, leaving 31 for the answer → truncated JSON → poisoned cache.
Fixed: thinking_budget knob (default 0), max_output_tokens 2048,
MAX_TOKENS guard raises instead of caching a truncated response,
cache treats unparseable JSON as a miss.

## Phase 4 — eval harness + baseline
Tier 1 offline, retriever injected. 48 tests green.
Config hash normalises Paths via as_posix() — was platform-
dependent, would have broken the CI delta gate.

BASELINE (dense-only, 30 questions, config 4af90eda0d31):
  overall  hit@1 .133  mrr .263  ndcg@10 .300  r@10 .500  r@50 .817
  numeric  hit@1 .188  mrr .305  ndcg@10 .330  r@10 .531  r@50 .938
  prose    hit@1 .071  mrr .216  ndcg@10 .265  r@10 .464  r@50 .679

Read: large r@10/r@50 gap (.50 vs .82) — evidence is retrieved
but buried. Ranking problem, not a finding problem. Cross-encoder
has ~32pp of headroom. Prose r@50 .679 is the hard floor —
~4-5 questions never surface evidence at all; reranking can't
reach those, Phase 6 might.

Ragas deferred (500MB LangChain+datasets). Faithfulness via
deterministic citation_validity instead.