"""Prompt assembly for grounded generation.

Turns a question plus its retrieved chunks into (1) a fixed system instruction
and (2) a user prompt whose context block labels every chunk with its
``chunk_id``. The labelling is load-bearing: the model is told to cite the
chunk_ids it used, and generate/citations.py later validates those ids against
exactly the ids we put in this block — so a citation is verifiable provenance,
not an unchecked claim.

The instruction pins three behaviours the eval depends on: answer *only* from the
provided context (no parametric knowledge), cite the chunk_ids used, and set
``insufficient_evidence`` when the context does not contain the answer (so an
unanswerable question yields an honest abstention, not a fabrication).
"""

from __future__ import annotations

from collections.abc import Sequence

from ledgion.interfaces import RetrievedChunk

SYSTEM_INSTRUCTION = """\
You are a financial-filings analyst. Answer the question using ONLY the context \
passages provided in the user message. Do not use any outside or prior knowledge.

Each passage is labelled with a [chunk_id]. Rules:
- Base every part of your answer strictly on the passages. Never invent figures, \
names, or dates that are not present in them.
- In "citations", list the exact [chunk_id] of every passage you actually used, \
and nothing you did not use.
- If the passages do not contain enough information to answer, set \
"insufficient_evidence" to true, leave "citations" empty, and make "answer" a \
brief statement that the provided filings do not contain the answer.
- Keep the answer concise and factual."""


def render_context(contexts: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks into a labelled context block, best match first.

    Each block is headed by ``[chunk_id]`` (the citation key) plus its filing and
    page, which give the model something to ground on and let a human trace a
    cited id back to a page; the body is the chunk text verbatim.
    """
    blocks: list[str] = []
    for rc in contexts:
        c = rc.chunk
        header = f"[{c.chunk_id}] ({c.doc_id}, page {c.page_num})"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, contexts: Sequence[RetrievedChunk]) -> str:
    """Assemble the user prompt: the labelled context block, then the question."""
    return f"Context passages:\n\n{render_context(contexts)}\n\n----\nQuestion: {question}"


def context_chunk_ids(contexts: Sequence[RetrievedChunk]) -> list[str]:
    """The chunk_ids present in the context, in render order.

    One definition, reused twice so the two can never drift: it is both the
    allow-list generate/citations.py validates cited ids against, and the id list
    folded into the generation cache key.
    """
    return [rc.chunk.chunk_id for rc in contexts]
