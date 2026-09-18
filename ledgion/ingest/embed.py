"""Dense embedding with a revision-aware disk cache.

``bge-base-en-v1.5`` via sentence-transformers, CPU-only, threads pinned to the
machine's physical cores. Like ``parse.py`` this is file-in / file-out: text in,
vectors out, **no Qdrant import** — indexing is a separate stage so the embedder
can be reused (and benchmarked) on its own.

Two properties the cache must hold:

* **Skip work already done.** Each chunk text is embedded once; a re-run reads
  the vector back from disk. The cache is one ``.npy`` per key, so a run that
  adds a few new chunks recomputes only those, not the whole corpus.
* **Never silently mix models.** The cache key is
  ``sha256(revision + "\\0" + text)`` — the *model revision*, not just the text.
  Change the pinned model and every key changes, so old vectors are never reused
  under a new model (which would produce a silently incomparable index). For the
  same reason the embedder refuses to load at all if ``revision`` is ``None``: a
  pin nothing reads is worse than no pin.

Truncation is made visible, not silent: before encoding, any input longer than
``max_seq_length`` tokens is logged with its length. With ``chunk.size`` held at
448 tokens this should never fire — the log is the tripwire that proves it.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def cache_key(revision: str, text: str) -> str:
    """Cache key for one chunk under one model revision.

    Pure and importable so the "key depends on revision" guarantee can be tested
    without loading a model. The NUL separator keeps ``(rev="a", text="bc")`` and
    ``(rev="ab", text="c")`` from colliding.
    """
    return hashlib.sha256(f"{revision}\x00{text}".encode()).hexdigest()


class BGEEmbedder:
    """An ``Embedder`` (see interfaces.py) over sentence-transformers + a cache."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        max_seq_length: int = 512,
        cache_dir: Path,
        num_threads: int = 4,
        query_instruction: str = "",
        force: bool = False,
    ) -> None:
        if revision is None:
            # Fail fast: an unpinned embedder makes every cached vector and every
            # eval result unreproducible. Better to refuse than to load "latest".
            raise ValueError(
                "embedding.revision is None — pin a commit SHA before embedding "
                "(an unpinned model breaks cache keys and reproducibility)"
            )
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_seq_length = max_seq_length
        self.cache_dir = Path(cache_dir)
        self.num_threads = num_threads
        self.query_instruction = query_instruction
        self.force = force
        self._model = None
        # Per-run counters the pipeline reads for its progress line.
        self.cache_hits = 0
        self.cache_misses = 0

    @classmethod
    def from_config(cls, cfg, *, force: bool = False) -> BGEEmbedder:
        return cls(
            model_id=cfg.embedding.model_id,
            revision=cfg.embedding.revision,
            device=cfg.embedding.device,
            batch_size=cfg.embedding.batch_size,
            normalize=cfg.embedding.normalize,
            max_seq_length=cfg.embedding.max_seq_length,
            cache_dir=Path(cfg.paths.cache_dir) / "embeddings",
            num_threads=cfg.torch.num_threads,
            query_instruction=cfg.embedding.query_instruction,
            force=force,
        )

    # -- model loading -------------------------------------------------------

    def _ensure_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            # Physical cores only — hyperthreading hurts this CPU workload.
            torch.set_num_threads(self.num_threads)
            model = SentenceTransformer(
                self.model_id, revision=self.revision, device=self.device
            )
            model.max_seq_length = self.max_seq_length
            # Raise the tokenizer's own cap so the truncation *check* below can
            # measure true length without transformers warning on long inputs;
            # real truncation still happens at model.max_seq_length during encode.
            model.tokenizer.model_max_length = int(1e9)
            self._model = model
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding width (768 for bge-base); loads the model if needed."""
        return self._ensure_model().get_sentence_embedding_dimension()

    # -- Embedder contract ---------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Embed document chunks, reading/writing the disk cache per text."""
        self._ensure_model()
        keys = [cache_key(self.revision, t) for t in texts]
        results: list[np.ndarray | None] = [None] * len(texts)

        to_encode: list[str] = []
        to_encode_idx: list[int] = []
        for i, key in enumerate(keys):
            cached = None if self.force else self._cache_load(key)
            if cached is not None:
                results[i] = cached
                self.cache_hits += 1
            else:
                to_encode.append(texts[i])
                to_encode_idx.append(i)
                self.cache_misses += 1

        if to_encode:
            self._warn_truncation(to_encode)
            vectors = self._model.encode(
                to_encode,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for j, idx in enumerate(to_encode_idx):
                vec = np.asarray(vectors[j], dtype=np.float32)
                results[idx] = vec
                self._cache_store(keys[idx], vec)

        # All slots are filled (cache hit or fresh encode).
        return [r for r in results if r is not None]

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. bge prepends a retrieval instruction to queries
        (documents get none); the instruction is a config knob so it stays a
        measurable ablation rather than a constant baked in here."""
        self._ensure_model()
        prompt = f"{self.query_instruction} {text}".strip() if self.query_instruction else text
        vec = self._model.encode(
            [prompt],
            batch_size=1,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vec, dtype=np.float32)

    # -- cache I/O -----------------------------------------------------------

    def _cache_load(self, key: str) -> np.ndarray | None:
        path = self.cache_dir / f"{key}.npy"
        if not path.exists():
            return None
        try:
            return np.load(path)
        except (OSError, ValueError):
            # A truncated/corrupt cache file (e.g. killed mid-write) is a miss,
            # not a crash — the vector is simply recomputed and overwritten.
            return None

    def _cache_store(self, key: str, vec: np.ndarray) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_dir / f"{key}.npy", vec)

    # -- truncation tripwire -------------------------------------------------

    def _warn_truncation(self, texts: Sequence[str]) -> None:
        """Log any input whose tokenised length exceeds ``max_seq_length`` — the
        point past which the model silently drops tokens."""
        encoded = self._model.tokenizer(list(texts), add_special_tokens=True)["input_ids"]
        for text, ids in zip(texts, encoded, strict=True):
            if len(ids) > self.max_seq_length:
                logger.warning(
                    "chunk exceeds max_seq_length and will be truncated: "
                    "%d > %d tokens: %.80r",
                    len(ids),
                    self.max_seq_length,
                    text,
                )
