# Ledgion

Retrieval-augmented QA over SEC filings (10-K/10-Q), benchmarked against
FinanceBench. Every answer cites the page it came from.

Portfolio project for AI engineer roles. The deliverable is a public repo
with an **ablation table**, a **headline metric** stated against the
published FinanceBench baseline, and a **CI pipeline** that runs an offline
eval on every PR and fails the build on regression. Evaluation quality
matters more than feature count — every component must justify itself with a
measured delta.

## Hardware constraints (binding, not advisory)

Lenovo Legion Y540, Windows 10.
- CPU: Intel i5-9300H, 4 physical cores / 8 threads, AVX2, no AVX-512.
- RAM: 8GB total, ~5.5GB realistically usable.
- GPU: GTX 1650, **4GB VRAM** (Windows reports 8GB; half is shared system
  RAM). Treat budget as 4GB.
- Disk: ~17GB free — the tightest constraint.
- Mobile chassis: sustained CPU load thermally throttles.

Rules:
- **CPU-only torch. Never CUDA:**
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  (CUDA wheels are 5-7GB.)
- `torch.set_num_threads(4)` and `OMP_NUM_THREADS=4` — physical cores, not
  logical. Hyperthreading hurts this workload.
- **Never run an LLM locally in the serving path.** An 8B model at Q4 is
  4.7GB and does not fit in 4GB VRAM.
- Before installing anything over ~500MB: state the size and ask first.
- Avoid duplicate HuggingFace downloads (.bin + .safetensors + ONNX of the
  same model). Load via library APIs, not `git clone`.

## Fixed technical choices (don't substitute without asking)

- Python 3.11, **uv** for dependency management.
- Embeddings: `BAAI/bge-base-en-v1.5`, pinned by revision SHA, CPU.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU.
- Sparse/BM25: `rank_bm25` first, then FastEmbed BM25 into Qdrant.
- Vector DB: **Qdrant** — local Docker for dev, in-process client for CI.
- Generation: **Gemini Flash** via API, `temperature=0`. Never a local model.
- Tracing: Langfuse Cloud.
- Eval: custom deterministic metrics + Ragas for the judged tier.
- PDF parsing: **PyMuPDF** to start. Docling comes later as a *measured
  ablation*, not an assumption.
- FastEmbed is already a dep for sparse vectors and ships ONNX int8 BGE
  builds. TODO: benchmark it against sentence-transformers for the dense
  embedder too — may be 2-3x faster on this CPU for free.

## Architecture rules

- Every swappable component implements a `typing.Protocol` in
  `interfaces.py`. The project runs 8-15 ablations; concrete wired classes
  turn each ablation into a refactor.
- **No hard-coded constants.** Model ids, top_k, chunk size, fusion params
  live in `config/`, resolved via `pydantic-settings`.
- Chunks never span a page boundary.
- Retrieval is scored at **PAGE level, never chunk level**. Chunk ids change
  with the chunker; page numbers are stable and allow cross-strategy
  comparison.
- The **Tier 1 eval runs fully offline**: no network, no GPU, no API keys.
  Must work on a GitHub Actions runner and on PRs from forks.
- Every eval run writes `results/<config_hash>.json` containing metrics, the
  full resolved config, the git SHA, and pinned model revisions.

## Working agreement

- One phase at a time. Do not scaffold ahead of the current task.
- Prefer editing one file over generating five.
- For anything in `eval/`, write tests before implementation.
- **User authors** the fusion logic, metric functions, and golden-set
  mapping — review these, don't write them (he must explain why a number
  moved in an interview).
- **Claude authors** parsers, clients, CLI plumbing, CI YAML, caching.
