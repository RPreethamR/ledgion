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