"""Recursive, page-bounded chunking.

Splits each page's text on a descending hierarchy of separators
(``"\\n\\n"`` -> ``"\\n"`` -> ``". "`` -> ``" "`` -> ``""``): keep whole
paragraphs together when they fit, fall back to sentences, then words, then
characters only if a "word" is itself larger than a chunk. This is the classic
"recursive character" strategy; the *separators* are characters, but chunk
*length* is measured in whichever ``unit`` the config picks (tokens by default).

Two hard requirements shape the implementation:

* **Chunks never span a page boundary** (CLAUDE.md). Enforced structurally:
  each page is chunked independently and a ``Chunk`` only ever carries the
  ``page_num`` it came from. There is no code path that joins two pages.
* **Chunk ids are stable across runs.** ``chunk_id`` is
  ``sha256(f"{doc_id}:{page_num}:{char_offset}")[:16]`` where ``char_offset`` is
  the chunk's exact starting character offset within the page. To make that
  offset exact and unique we never rejoin split text — the splitter works
  entirely in ``(start, end)`` index spans over the original page string, so a
  chunk's text is always a verbatim ``page_text[start:end]`` slice and its
  offset is unambiguous. (Chunk ids move when the chunker changes; that is why
  retrieval is scored on ``page_num``, not ``chunk_id`` — see interfaces.py.)

Token length is measured with the embedder's own fast tokenizer via its
offset mapping: the page is tokenised **once**, and the token count of any
character range is a pair of ``bisect`` lookups. So sizing chunks to the
embedder's window costs one tokenisation per page, not one per candidate chunk.
The tokenizer is loaded lazily (``unit='chars'`` never touches it), keeping the
chunker importable and testable with no model download.
"""

from __future__ import annotations

import bisect
import hashlib
from collections.abc import Callable, Sequence

from ledgion.interfaces import Chunk

# Descending granularity. The empty string is the terminal fallback: split into
# individual characters, reached only if a single run has none of the higher
# separators — vanishingly rare in extracted filing text, but it guarantees the
# recursion always terminates with spans no larger than one character.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

# A length function scores a character range [a, b) of the page in the configured
# unit (tokens or chars). Threaded through the recursion so the splitter is
# unit-agnostic.
LengthFn = Callable[[int, int], int]


def _chunk_id(doc_id: str, page_num: int, char_offset: int) -> str:
    """Deterministic 16-hex-char id. Same (doc, page, offset) -> same id, always."""
    key = f"{doc_id}:{page_num}:{char_offset}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class RecursiveChunker:
    """A ``Chunker`` (see interfaces.py) that splits recursively within a page."""

    def __init__(
        self,
        *,
        size: int,
        overlap: int,
        unit: str = "tokens",
        tokenizer_model_id: str | None = None,
        tokenizer_revision: str | None = None,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if overlap >= size:
            # Overlap must be a strict tail of a chunk; >= size makes the merge
            # window never advance.
            raise ValueError(f"chunk overlap ({overlap}) must be < size ({size})")
        if unit not in ("tokens", "chars"):
            raise ValueError(f"unknown chunk unit: {unit!r}")
        self.size = size
        self.overlap = overlap
        self.unit = unit
        self.tokenizer_model_id = tokenizer_model_id
        self.tokenizer_revision = tokenizer_revision
        self.separators = tuple(separators)
        self._tok = None  # lazily loaded fast tokenizer, only when unit == "tokens"

    @classmethod
    def from_config(cls, cfg) -> RecursiveChunker:
        """Build from resolved ``Settings``. Token length uses the *embedder's*
        tokenizer so chunk sizes are measured in the same tokens the model sees."""
        if not cfg.chunk.respect_page_boundary:
            # We only implement page-bounded chunking (a hard project rule); refuse
            # rather than silently ignore a config that asks to cross boundaries.
            raise NotImplementedError("respect_page_boundary=False is not supported")
        return cls(
            size=cfg.chunk.size,
            overlap=cfg.chunk.overlap,
            unit=cfg.chunk.unit,
            tokenizer_model_id=cfg.embedding.model_id,
            tokenizer_revision=cfg.embedding.revision,
        )

    # -- public contract -----------------------------------------------------

    def chunk(
        self,
        doc_id: str,
        pages: Sequence[tuple[int, str]],
        *,
        company: str,
        ticker: str,
        fiscal_year: int,
        form_type: str,
    ) -> list[Chunk]:
        """Chunk each ``(page_num, page_text)`` pair independently."""
        chunks: list[Chunk] = []
        for page_num, text in pages:
            for start, end in self._chunk_page(text):
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(doc_id, page_num, start),
                        doc_id=doc_id,
                        page_num=page_num,
                        text=text[start:end],
                        company=company,
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        form_type=form_type,
                    )
                )
        return chunks

    # -- per-page splitting --------------------------------------------------

    def _chunk_page(self, text: str) -> list[tuple[int, int]]:
        """Return chunk spans ``(start, end)`` for one page, in order."""
        if not text.strip():
            return []  # blank cover/separator pages produce nothing to embed
        length = self._length_fn(text)
        spans = self._recursive_spans(text, 0, len(text), self.separators, length)
        return self._merge_spans(text, spans, length)

    def _recursive_spans(
        self,
        text: str,
        start: int,
        end: int,
        separators: Sequence[str],
        length: LengthFn,
    ) -> list[tuple[int, int]]:
        """Split ``text[start:end]`` into atomic spans each <= ``size`` (unless a
        single character already exceeds it — impossible for tokens, so ignored)."""
        segment = text[start:end]
        # Pick the finest separator that still occurs here; "" is the terminal.
        chosen = len(separators) - 1
        for i, sep in enumerate(separators):
            if sep == "" or sep in segment:
                chosen = i
                break
        sep = separators[chosen]
        remaining = separators[chosen + 1 :]

        result: list[tuple[int, int]] = []
        for s, e in self._split_span(text, start, end, sep):
            if s == e:
                continue
            if length(s, e) <= self.size:
                result.append((s, e))
            elif remaining:
                result.extend(self._recursive_spans(text, s, e, remaining, length))
            else:
                # No finer separator left (sep was ""): accept the oversized span.
                result.append((s, e))
        return result

    @staticmethod
    def _split_span(text: str, start: int, end: int, sep: str) -> list[tuple[int, int]]:
        """Split ``[start, end)`` on ``sep`` into contiguous spans that exactly
        tile the range. The separator is kept on the tail of each piece, so
        concatenating the slices reproduces the original text and offsets never
        drift."""
        if sep == "":
            return [(i, i + 1) for i in range(start, end)]
        spans: list[tuple[int, int]] = []
        pos = start
        idx = text.find(sep, start, end)
        while idx != -1:
            piece_end = idx + len(sep)
            spans.append((pos, piece_end))
            pos = piece_end
            idx = text.find(sep, pos, end)
        if pos < end:
            spans.append((pos, end))
        return spans

    def _merge_spans(
        self, text: str, spans: Sequence[tuple[int, int]], length: LengthFn
    ) -> list[tuple[int, int]]:
        """Greedily pack atomic spans into chunks up to ``size``, carrying an
        ``overlap``-sized tail from each chunk into the next. Because atomic spans
        are far smaller than ``size`` (tokens bottom out at words), chunk start
        offsets strictly increase — so every chunk on a page gets a distinct
        ``char_offset`` and therefore a distinct ``chunk_id``."""
        chunks: list[tuple[int, int]] = []
        window: list[tuple[int, int]] = []
        window_len = 0
        for span in spans:
            slen = length(span[0], span[1])
            if window and window_len + slen > self.size:
                chunks.append((window[0][0], window[-1][1]))
                # Drop from the front until the retained tail is <= overlap.
                while window and window_len > self.overlap:
                    removed = window.pop(0)
                    window_len -= length(removed[0], removed[1])
            window.append(span)
            window_len += slen
        if window:
            chunks.append((window[0][0], window[-1][1]))
        return chunks

    # -- length measurement --------------------------------------------------

    def _length_fn(self, text: str) -> LengthFn:
        """Build a ``(start, end) -> length`` function for this page."""
        if self.unit == "chars":
            return lambda a, b: b - a

        # tokens: tokenise the page once, then count tokens whose start offset
        # falls in [a, b) via bisect over the (sorted) token start offsets.
        tok = self._tokenizer()
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        starts = [off[0] for off in enc["offset_mapping"]]

        def token_length(a: int, b: int) -> int:
            return bisect.bisect_left(starts, b) - bisect.bisect_left(starts, a)

        return token_length

    def _tokenizer(self):
        if self._tok is None:
            if self.tokenizer_model_id is None:
                raise ValueError("unit='tokens' requires a tokenizer_model_id")
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                self.tokenizer_model_id, revision=self.tokenizer_revision
            )
            # We only ever ask for offsets, never feed the model, so lift the
            # length cap — otherwise transformers warns on every >512-token page.
            tok.model_max_length = int(1e9)
            self._tok = tok
        return self._tok
