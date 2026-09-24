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

from ledgion.config import REPO_ROOT
from ledgion.ingest.docling_artifact import (
    artifact_path,
    group_by_page,
    load_elements,
    render_table_flat,
    table_markdown_parts,
)
from ledgion.interfaces import Chunk

# A "context line" for a markdown table's prefix: a heading, or a short one-line text
# such as the units line "(in millions, except per share data)". Anything longer is a
# paragraph, not a label, and is left to chunk as prose.
SHORT_LINE_CHARS = 120

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

    def count_tokens(self, text: str) -> int:
        """Token count of a whole string in the configured unit (chars → ``len``).

        Used by the table-aware chunker to size a markdown table (prefix + header +
        rows) against ``size`` before deciding whether to split it by row groups.
        """
        if self.unit == "chars":
            return len(text)
        if not text:
            return 0
        tok = self._tokenizer()
        return len(tok(text, add_special_tokens=False)["input_ids"])

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


def _resolve(path) -> object:
    """Resolve a (possibly relative) config path against the repo root."""
    from pathlib import Path

    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _table_chunk_id(doc_id: str, page_num: int, element_index: int, part: int) -> str:
    """Deterministic id for a table chunk. Keyed on the stable ``element_index`` from
    the artifact and the split ``part`` — a different namespace from text chunks
    (which key on a char offset), so the two can never collide."""
    key = f"{doc_id}:{page_num}:t{element_index}:{part}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class TableAwareChunker:
    """A ``Chunker`` over the Docling artifact (see ``docling_artifact.py``).

    Text elements chunk exactly as ``RecursiveChunker`` does (composed below, same
    size/overlap/tokenizer). Tables follow ``chunk.table_mode``:

    * **flat** — every element (tables flattened to plain text) is concatenated per
      page and chunked recursively, so the only variable vs the PyMuPDF baseline is
      the parser.
    * **markdown** — tables are pulled out as their own chunk(s), each prefixed with
      its caption and the nearest preceding heading/short line; an oversized table
      splits by row groups with the header repeated in every part.

    Conventions (both hard rules):

    * **Every chunk carries exactly one page label.** A multi-page element is labeled
      with its **first** page (its ``page_num``) — the same page FinanceBench labels
      evidence on. So a paragraph flowing across a page break chunks entirely on the
      page it starts, and no chunk straddles a boundary.
    * **Tables never cross a page boundary** — a table is one element on one page, so
      its chunks inherit that single page.

    Imports nothing from Docling: the structured elements come from the JSONL artifact.
    ``last_table_chunk_ids`` holds the ids emitted for tables in the most recent
    ``chunk``/``chunk_elements`` call, so ingestion can report the table-chunk fraction.
    """

    def __init__(
        self,
        *,
        size: int,
        overlap: int,
        unit: str = "tokens",
        table_mode: str = "flat",
        tokenizer_model_id: str | None = None,
        tokenizer_revision: str | None = None,
        docling_dir=None,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if table_mode not in ("flat", "markdown"):
            raise ValueError(f"unknown table_mode: {table_mode!r}")
        self.size = size
        self.table_mode = table_mode
        self.docling_dir = docling_dir
        # Compose the recursive splitter for all prose (and, in flat mode, the whole
        # page). Sharing it means text chunks are byte-identical to the baseline's.
        self._text = RecursiveChunker(
            size=size,
            overlap=overlap,
            unit=unit,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_revision=tokenizer_revision,
            separators=separators,
        )
        self.last_table_chunk_ids: set[str] = set()

    @classmethod
    def from_config(cls, cfg) -> TableAwareChunker:
        if not cfg.chunk.respect_page_boundary:
            raise NotImplementedError("respect_page_boundary=False is not supported")
        return cls(
            size=cfg.chunk.size,
            overlap=cfg.chunk.overlap,
            unit=cfg.chunk.unit,
            table_mode=cfg.chunk.table_mode,
            tokenizer_model_id=cfg.embedding.model_id,
            tokenizer_revision=cfg.embedding.revision,
            docling_dir=_resolve(cfg.parser.docling_dir),
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
        """Chunk one document from its Docling artifact.

        Structured content comes from ``docling_dir/<doc_id>.docling.jsonl``, so the
        ``pages`` argument (plain ``(page_num, text)`` pairs, used by text-only
        chunkers) is accepted for ``Chunker``-protocol compatibility and ignored.
        """
        elements = load_elements(artifact_path(self.docling_dir, doc_id))
        return self.chunk_elements(
            doc_id,
            elements,
            company=company,
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
        )

    def chunk_elements(
        self,
        doc_id: str,
        elements: Sequence[dict],
        *,
        company: str,
        ticker: str,
        fiscal_year: int,
        form_type: str,
    ) -> list[Chunk]:
        """Chunk a list of Docling element dicts (grouped by page internally).

        The pure entry point — tests exercise it with synthetic elements, no file I/O.
        """
        meta = dict(company=company, ticker=ticker, fiscal_year=fiscal_year, form_type=form_type)
        self.last_table_chunk_ids = set()
        by_page = group_by_page(list(elements))
        chunks: list[Chunk] = []
        for page in sorted(by_page):
            page_elements = by_page[page]
            if self.table_mode == "flat":
                chunks.extend(self._chunk_page_flat(doc_id, page, page_elements, meta))
            else:
                chunks.extend(self._chunk_page_markdown(doc_id, page, page_elements, meta))
        return chunks

    # -- flat mode -----------------------------------------------------------

    def _chunk_page_flat(
        self, doc_id: str, page: int, elements: Sequence[dict], meta: dict
    ) -> list[Chunk]:
        parts = [t for el in elements if (t := self._element_flat(el).strip())]
        if not parts:
            return []
        page_text = "\n\n".join(parts)
        return self._text.chunk(doc_id, [(page, page_text)], **meta)

    @staticmethod
    def _element_flat(element: dict) -> str:
        if element.get("element_type") == "table":
            return render_table_flat(element.get("table") or {})
        return element.get("text") or ""

    # -- markdown mode -------------------------------------------------------

    def _chunk_page_markdown(
        self, doc_id: str, page: int, elements: Sequence[dict], meta: dict
    ) -> list[Chunk]:
        prose_parts: list[str] = []
        table_chunks: list[Chunk] = []
        context: str | None = None  # nearest preceding heading / short line
        for el in elements:
            if el.get("element_type") == "table":
                prefix = self._table_prefix(el, context)
                table_chunks.extend(self._table_chunks(doc_id, page, el, prefix, meta))
                continue
            text = (el.get("text") or "").strip()
            if text:
                prose_parts.append(text)
            if self._is_context_line(el, text):
                context = text

        prose_chunks: list[Chunk] = []
        if prose_parts:
            prose_chunks = self._text.chunk(doc_id, [(page, "\n\n".join(prose_parts))], **meta)
        return prose_chunks + table_chunks

    @staticmethod
    def _is_context_line(element: dict, text: str) -> bool:
        if not text:
            return False
        etype = element.get("element_type")
        if etype == "heading":
            return True
        # A short, single-line text is a label (a units line, a small caption).
        return etype == "text" and len(text) <= SHORT_LINE_CHARS and "\n" not in text

    @staticmethod
    def _table_prefix(element: dict, context: str | None) -> str:
        """Caption + nearest preceding heading/short line. Docling usually captures a
        financial statement's title as the table caption and the units line as the
        short text just above it, so together they restore what a table needs to be
        interpretable. De-duplicated in case the caption and context coincide."""
        caption = ((element.get("table") or {}).get("caption") or "").strip()
        lines: list[str] = []
        for line in (caption, context):
            if line and line not in lines:
                lines.append(line)
        return "\n".join(lines)

    def _table_chunks(
        self, doc_id: str, page: int, element: dict, prefix: str, meta: dict
    ) -> list[Chunk]:
        table = element.get("table") or {}
        header_lines, body_lines = table_markdown_parts(table)
        if not header_lines and not body_lines:
            # Degenerate/empty grid: fall back to the stored plain-text rendering so
            # the table is still represented rather than silently dropped.
            fallback = (element.get("text") or "").strip()
            texts = [f"{prefix}\n{fallback}".strip()] if fallback or prefix else []
        else:
            base = f"{prefix}\n" + "\n".join(header_lines) if prefix else "\n".join(header_lines)
            whole = base + ("\n" + "\n".join(body_lines) if body_lines else "")
            if not body_lines or self._text.count_tokens(whole) <= self.size:
                texts = [whole]
            else:
                texts = self._split_table_rows(base, body_lines)

        chunks: list[Chunk] = []
        for part, text in enumerate(texts):
            cid = _table_chunk_id(doc_id, page, element.get("element_index", 0), part)
            self.last_table_chunk_ids.add(cid)
            chunks.append(Chunk(chunk_id=cid, doc_id=doc_id, page_num=page, text=text, **meta))
        return chunks

    def _split_table_rows(self, base: str, body_lines: Sequence[str]) -> list[str]:
        """Greedily pack rows into parts so each ``base`` (prefix + header) + its rows
        stays within ``size``. ``base`` is repeated in every part, so the header row
        appears in each. A single row that alone overflows is emitted as its own part
        (a row can't be split)."""
        parts: list[str] = []
        current: list[str] = []
        for row in body_lines:
            trial = base + "\n" + "\n".join([*current, row])
            if current and self._text.count_tokens(trial) > self.size:
                parts.append(base + "\n" + "\n".join(current))
                current = [row]
            else:
                current.append(row)
        if current:
            parts.append(base + "\n" + "\n".join(current))
        return parts
