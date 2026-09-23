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

## Phase 5 — CI gate (offline Tier 1)
Tier 1 now runs on a bare runner: no GPU, network, API keys, or
model download — fork PRs included. `make fixture` dumps the live
setup to three committed artifacts: index.npz (all 2181 dense
vectors + parallel chunk_id/doc_id/page_num, no text, ~7MB),
query_vectors.npz (one embedding per golden qid, same bge revision
and query prefix the retriever applies), manifest.json (model/
revision, chunk count, qids, git SHA, timestamp). Both .npz written
sorted → byte-deterministic; only the manifest carries a timestamp.

FixtureRetriever replays index.npz into an in-process Qdrant and
looks up the query vector by qid. Parity by construction: it composes
the real DenseRetriever with a precomputed-lookup embedder over the
same QdrantIndexer, so query path, point ids, and page-collapse are
identical. Offline metrics matched the live baseline to the digit
(r@10 .500, ndcg@10 .300); parity holds across all 30 questions.
Two startup guards fail loudly with "run make fixture": manifest
revision != config revision, and any golden qid missing a vector.

Gate (eval/gate.py) compares overall recall@10/ndcg@10 to
fixtures/baseline_metrics.json; fails the build only on a drop >
eval.gate_tolerance (.02) — improvements always pass. Compares the
6dp-rounded drop so an exactly-tolerance dip passes (else .5-.48 =
.02000…018 trips). baseline_metrics.json is hand-updated in the PR
that improves it, never auto-written by a run.

sentence-transformers (→ torch, multi-GB) moved out of core into a
PEP 735 `models` default group; CI runs `uv sync --no-group models`
for a torch-free env (verified: torch unimportable, 0 mentions in
`uv tree --no-group models`). numpy promoted to explicit core dep.
Retriever picked by config (retrieval.backend dense|fixture) so CI
uses the literal `ledgion eval --tier 1 --gate` with
LEDGION_RETRIEVAL__BACKEND=fixture — no CLI flag needed.

CI (.github/workflows/ci.yml, on PR + push main): ruff → pytest →
gate. Model/Qdrant/PDF tests skip cleanly (importorskip / pytest.skip),
never error. Verified in a throwaway torch-free env: ruff+pytest pass
with 2 skips, gate green (exit 0); synthetic regression exits 1.
60 tests green locally.

Read: the gate guards the ranking metrics (r@10/ndcg@10), not the
finding metric (r@50) — a strategy that finds more evidence but ranks
it worse still fails, which is the right bar. Cheap + hermetic is the
whole point: a gate that needed the 4GB model would get skipped on
forks and guard nothing.

## Phase 6 — hypothesis (recorded before running)
Baseline: r@10 .500, r@50 .817, ndcg@10 .300.
Prose r@50 .679 is the weak point — ~4-5 questions never surface
evidence at all.

Predict BM25 helps two ways:
1. Recall floor — surfaces some of the ~18% dense misses, most
   likely prose questions where the evidence page names a
   specific entity or section the embedder smooths over.
2. Rank position — exact tokens (tickers, GAAP line items,
   fiscal years) appear verbatim, so sparse should push buried
   pages up. Expect the r@10 gain to exceed the r@50 gain.

Expect numeric to benefit more than prose on ndcg@10, since
numeric questions name line items that match literally.

Risk: at 2181 chunks over 5 filings, BM25 may just retrieve the
same pages dense already finds, and RRF adds nothing.

## Metric correction — (doc_id, page) scoring
Tier 1 scored bare page numbers, so a wrong filing's page N
counted as a hit for the right filing's page N. Now keyed on
(doc_id, page_num). Metric logic unchanged — membership tests
over hashable keys; the keys were fixed upstream in
collapse_to_pages. Regression test added.

Dense baseline, old → new:
  hit@1 .133 → .133    r@10 .500 → .500    r@50 .817 → .817
  mrr   .263 → .252    ndcg@10 .300 → .291
Only rank-sensitive metrics moved: phantom hits ranked above
the real evidence page in a few questions, mostly prose.

The bug was biased toward sparse: under correct scoring sparse
r@10 fell .267 → .233 while dense r@10 was unchanged. Fixed
before running hybrid for that reason.

Baseline lowered deliberately — the old metric was wrong.
Supersedes the Phase 4 baseline numbers above.

## Phase 6 — revised prediction (after 2×2, before hybrid)
Complementarity at K=50, 35 evidence pairs:
  dense × sparse:     both 11, dense-only 16, sparse-only 1, neither 7
  dense × sparse_sw:  both 13, dense-only 14, sparse-only 1, neither 7
Fusion recall ceiling .828 / .833 vs dense .817. Original
prediction 1 (BM25 raises the recall floor) is effectively
falsified in advance.

Pre-registered grid, sparse_weight ∈ {1.0, 0.5, 0.25}, k=60:
- 1.0: recall@50 falls — sparse noise displaces deep dense-only
  evidence; agreement on wrong pages risks hit@1.
- 0.5, 0.25: below the 61/110 threshold, fused top-50 = dense's
  top-50 reordered. recall@50 fixed at .817; any gain is ranking.
All rows reported regardless of outcome.

## Hybrid budget fix
HybridRetriever returned the full fused union (up to 2×top_k
chunks) rather than top_k, so hybrid would have been scored on a
candidate pool twice dense's size. Now capped at top_k in the
retriever; reciprocal_rank_fusion still returns the full fused list.
Second evaluation bug, after bare-page scoring, that would have
flattered hybrid. Both caught before the first hybrid run.

## Phase 6 — result
7 rows, pre-registered grid, all reported:
                     hit@1   mrr    ndcg@10  r@10    r@50
  dense              .133    .252   .291     .500    .817
  hybrid sw1.0       +.033   −.024  −.055    −.117   −.167
  hybrid sw0.5       ±0      −.021  −.033    −.033   ±0
  hybrid sw0.25      +.033   +.013  −.011    −.033   ±0
  hybrid sw1.0 stop  +.033   −.011  −.029    −.083   −.150
  hybrid sw0.5 stop  +.033   +.003  −.018    −.067   ±0
  hybrid sw0.25 stop +.067   +.036  +.014    −.050   ±0

Original hypothesis:
  1. BM25 raises the recall floor — WRONG. One evidence page
     dense misses.
  2. r@10 gain exceeds r@50 gain — WRONG. r@10 falls in every row.
  3. Numeric benefits more than prose — WRONG. Where hybrid helps
     at all, it's prose.
  4. Risk: RRF adds nothing — effectively CORRECT.
Revised (after 2×2):
  sw1.0 recall@50 falls — CORRECT, by five questions' worth.
  sw1.0 hit@1 at risk — WRONG. Rose by one question.
  sw ≤ 0.5 recall@50 exactly .817 — CORRECT, 4/4 to the digit.

Decision: dense stays default. No hybrid row passes the recall@10
gate. Below sw 0.55 the candidate set equals dense's, so under a
reranker hybrid is identical to dense by construction.
Hybrid code and sparse fixture kept; re-run at corpus scale.

## Phase 7 — depth diagnostic (before reranking)
Dense at top_k=200, 35 evidence pairs:
  ranks 1-10: 17 | 11-50: 11 | 51-100: 3 | 101-200: 1 | absent: 3
Depth 50 → 200 rescues 4 pairs. Keeping top_k=50; deeper pool
isn't worth the reranker latency. Will report a k=100 row.

Reranker ceiling is exact: 28/35 pairs are in the top 50.
  recall@50 MUST stay .8167 — reranking reorders, never adds.
  recall@10 can rise from .500 to at most .800.
Pre-registered: large recall@10 gain, recall@50 unchanged.
Any change to recall@50 means the pipeline is broken.

Amex p96 is the evidence page for two questions: id_00720
(prose) at rank 156, id_00723 (numeric) absent at 200. Same page,
same index, same embedder — the difference is query phrasing
alone. Strongest motivating case for Phase 9 query rewriting.

Three pairs absent at depth 200 are a floor for bge-base with the
question as written, not an absolute floor.

## Phase 7 — result (two rerankers, measured)
Metric semantics decided before running: score the FULL reordered
top_k pool, not the served top_n=5 (DECISIONS above / interview note).
recall@top_k is the pool-integrity invariant — reranking permutes,
never adds. Held to the digit on every row:
  top_k=50  recall@50 = .8167  (dense, +MiniLM, +bge, hybrid+MiniLM)
  top_k=100 recall@k  = .8889  (dense_k100 and +MiniLM companion)
No deviation → pipeline correct; the quality moves below are real.

                       hit@1  mrr    ndcg@10 r@10   r@50   retr   rerank
                                                          p50ms  p50ms
  dense (baseline)     .1333  .2521  .2906   .5000  .8167    83     —
  dense +MiniLM        .0333  .2033  .2624   .5167  .8167    88   4826
  dense +bge           .1000  .2036  .2644   .5333  .8167    97  27375
  dense_k100           .1333  .2533  .2906   .5000  .8500    83     —
  dense_k100 +MiniLM   .0333  .1913  .2617   .5333  .8444   110  10860
  hybrid +MiniLM       .0333  .2033  .2624   .5167  .8167   120   5668
(30 questions, CPU 4-thread. rerank p95: MiniLM 5963, bge 30598.)

Read:
- No reranker beats dense on hit@1, mrr, or ndcg@10. Both DROP hit@1
  (.133 → .033 MiniLM, → .100 bge). Reranking is net-negative here.
- +bge is the ONLY row to beat baseline recall@10 (.5333 vs .5000) —
  it rescues buried evidence into the top-10 — but costs ~27s/question
  (~285x retrieval, ~5.7x MiniLM). Not worth adopting.
- hybrid+MiniLM == dense+MiniLM on every metric. Confirms sw0.25 fused
  pool = dense's pool; reranking the same pool gives the same answer,
  for ~0.8s more (the BM25 arm).
- k100 deepens recall (r@k .8889, r@50 rises) but hit@1 unchanged at
  .033 and rerank cost ~2.25x (linear in pool size).

Why hit@1 collapses (diagnostic, not truncation):
- Combined (query+chunk) tokens rarely exceed the 512 cap — MiniLM
  tokenizer: 34/1500 pairs over, by ≤25 tokens. 3 of the 4 dense-rank-1
  evidence pages pushed out fit; the 1 truncated lost 13 tail tokens
  with its statement header intact. Truncation is NOT the cause.
- Domain mismatch. On all 4 hit@1 regressions the CE promoted a prose
  MD&A page or a different summary statement over the numeric
  statement/notes holding the answer. MS-MARCO-trained cross-encoders
  reward passages that lexically restate the question; 10-K evidence is
  a dense table whose header does not.

Score distributions (1500 pairs, top-50 pool; NOTE different scales —
MiniLM raw logits, bge sigmoid probabilities, its native activation):
  MiniLM  median -6.96, 3.6% > 0, evidence-vs-non mean sep +1.93 logit
  bge     median 0.004, max 0.994, evidence mean 2.7x non (0.178/0.067)
MiniLM sits in a flat low-confidence negative band → ordering among
near-ties is noise. bge has real dynamic range and separates evidence
better — yet its sharper confidence still doesn't put the right page
at rank 1 (hit@1 .100 < dense .133).

bge-reranker-base is XLM-RoBERTa, 512-token cap (NOT 8192; that's
bge-reranker-v2-m3) — no max-length handling differs from MiniLM.
Frozen scores keyed by reranker revision so the two models can't share
a cache; fixture holds both (+12.3 KiB).

Decision: dense stays default, no reranker adopted. Reranking code +
both frozen score sets kept for re-measurement at corpus scale, where
the cross-encoder has more to reorder and the domain-mismatch may ease.