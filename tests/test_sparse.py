"""BM25 sparse retrieval, exercised on a synthetic in-file corpus.

Fully offline: ~20 short documents defined inline, built into an index under
pytest's tmp_path, persisted, reloaded, and queried — no committed fixture file, no
Qdrant, no model. These pin the four things the sparse arm must guarantee: it
returns valid RetrievedChunks, an obviously-matching document ranks first,
tokenisation is applied as configured, and a persist/load roundtrip reproduces the
rankings exactly.
"""

from __future__ import annotations

from ledgion.interfaces import Chunk, RetrievedChunk
from ledgion.retrieve.sparse import BM25Tokenizer, SparseRetriever

# ~20 tiny documents. Doc 0 is the only one that mentions "xilinx", so a query for
# it must rank doc 0 first; the rest are financial-flavoured filler to give BM25 a
# realistic IDF spread.
_DOCS = [
    "AMD completed the Xilinx acquisition and recorded amortization of intangible assets",
    "Net revenue increased driven by higher sales of EPYC server processors",
    "The Data Center segment revenue grew year over year",
    "Gaming segment revenue rose on semi custom product sales",
    "Cost of sales and gross margin were roughly flat",
    "Operating income declined compared to the prior year",
    "Cash flows from operating activities were strong",
    "Short term investments matured during the period",
    "Accounts receivable increased with revenue",
    "Inventories grew to support demand",
    "The company repurchased common stock",
    "Long term debt was reduced during the year",
    "Research and development expense rose",
    "Selling general and administrative expenses increased",
    "Effective tax rate changed versus the prior year",
    "Goodwill was recognized on the acquisition",
    "Deferred tax assets and liabilities were remeasured",
    "One customer accounted for a large share of revenue",
    "The board declared a quarterly dividend",
    "Foreign exchange had an immaterial effect on results",
]


def _chunk(idx: int, text: str) -> Chunk:
    # 16-hex chunk_id keeps parity with the real corpus; page = idx + 1.
    return Chunk(
        chunk_id=f"{idx:016x}",
        doc_id="SYNTH_10K",
        page_num=idx + 1,
        text=text,
        company="Synthetic",
        ticker="SYN",
        fiscal_year=2022,
        form_type="10-K",
    )


def _corpus() -> list[Chunk]:
    return [_chunk(i, text) for i, text in enumerate(_DOCS)]


def _retriever() -> SparseRetriever:
    return SparseRetriever(
        chunks=_corpus(),
        tokenizer=BM25Tokenizer(lowercase=True, token_pattern=r"[A-Za-z0-9]+"),
        k1=1.5,
        b=0.75,
        epsilon=0.25,
    )


def test_returns_valid_retrieved_chunks():
    results = _retriever().retrieve("revenue growth", top_k=5)

    assert 0 < len(results) <= 5
    assert all(isinstance(rc, RetrievedChunk) for rc in results)
    assert all(isinstance(rc.chunk, Chunk) for rc in results)
    # Scores are non-increasing: list position IS the rank.
    scores = [rc.score for rc in results]
    assert scores == sorted(scores, reverse=True)
    # Every returned chunk is a real corpus chunk with its page populated.
    corpus_ids = {c.chunk_id for c in _corpus()}
    assert all(rc.chunk.chunk_id in corpus_ids for rc in results)
    assert all(rc.chunk.page_num >= 1 for rc in results)


def test_obvious_match_ranks_first():
    # Only doc 0 mentions "xilinx"; it must come back at rank 1.
    results = _retriever().retrieve("xilinx acquisition", top_k=5)
    assert results[0].chunk.page_num == 1  # doc 0 -> page 1
    assert results[0].score > 0.0


def test_lowercasing_is_applied():
    # Query in upper case still matches the lower-case corpus term because the
    # tokeniser lowercases both sides — the configured policy, not a coincidence.
    results = _retriever().retrieve("XILINX", top_k=1)
    assert results[0].chunk.page_num == 1


def test_persist_load_roundtrip_is_identical(tmp_path):
    retriever = _retriever()
    path = retriever.save(tmp_path / "bm25.pkl")
    reloaded = SparseRetriever.load(path)

    for query in ("xilinx acquisition", "revenue growth", "dividend", "tax rate"):
        before = [(rc.chunk.chunk_id, rc.score) for rc in retriever.retrieve(query, top_k=10)]
        after = [(rc.chunk.chunk_id, rc.score) for rc in reloaded.retrieve(query, top_k=10)]
        assert before == after
    # The persisted policy survived the roundtrip.
    assert reloaded.tokenizer.config_dict() == {"lowercase": True, "token_pattern": r"[A-Za-z0-9]+"}
    assert (reloaded.k1, reloaded.b, reloaded.epsilon) == (1.5, 0.75, 0.25)
