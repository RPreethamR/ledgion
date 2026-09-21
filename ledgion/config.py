"""Configuration loading and deterministic hashing.

Every knob lives in ``config/default.yaml`` and is validated into typed
``pydantic`` models here. No constant is hard-coded in the pipeline — model
ids, revisions, top_k, chunk size, fusion params, etc. all resolve through
``load_config``.

Resolution order (highest precedence first):

    1. explicit keyword args passed to ``Settings(...)``
    2. process environment variables  (prefix ``LEDGION_``, nesting ``__``)
    3. a local ``.env`` file
    4. ``config/default.yaml``
    5. the field defaults declared below

So ``config/default.yaml`` is the source of truth for the pipeline, and the
environment / ``.env`` layer overrides it for a given machine or CI run
(e.g. ``LEDGION_QDRANT__MODE=memory`` to force the in-process client).

``config_hash`` produces the stable key that every eval run writes to
``results/<hash>.json``. It is order-independent and JSON-canonical, so the
same knobs always hash to the same value regardless of dict ordering.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePath
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Repo root is the parent of the ``ledgion`` package dir; resolve the default
# config relative to it so the loader works regardless of the current dir.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    pdf_dir: Path = Path("data/pdfs")
    financebench_dir: Path = Path("data/financebench")
    results_dir: Path = Path("results")
    # Embedding cache lives under here (<cache_dir>/embeddings); gitignored.
    cache_dir: Path = Path(".cache")


class TorchConfig(BaseModel):
    # Physical cores, not logical — hyperthreading hurts this workload.
    # Mirrors torch.set_num_threads(4) / OMP_NUM_THREADS=4 from CLAUDE.md.
    num_threads: int = 4


class EmbeddingConfig(BaseModel):
    model_id: str = "BAAI/bge-base-en-v1.5"
    # Pinned commit SHA (CLAUDE.md requires revision pinning). config/default.yaml
    # is the source of truth; this default mirrors it so a bare Settings() is valid.
    # The loader passes it to SentenceTransformer(revision=...) AND folds it into
    # the embedding cache key, so a model change can never silently reuse old vectors.
    revision: str | None = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True
    max_seq_length: int = 512
    # bge asks *queries* (not documents) to carry this instruction for retrieval.
    # Kept in config so "instruction vs none" stays a measurable ablation.
    query_instruction: str = "Represent this sentence for searching relevant passages:"


class RerankerConfig(BaseModel):
    model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Pinned commit SHA (see EmbeddingConfig). Wired into CrossEncoder(revision=...)
    # when the reranker lands in Phase 3; pinned now so config is stable meanwhile.
    revision: str | None = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    device: str = "cpu"
    top_n: int = 5  # rerank_top_n: how many survive reranking


class SparseConfig(BaseModel):
    # rank_bm25 first (simple, in-process), then FastEmbed BM25 into Qdrant.
    backend: Literal["rank_bm25", "fastembed"] = "rank_bm25"
    fastembed_model: str = "Qdrant/bm25"
    # BM25 tokenisation. An ablation knob, so it lives in config: it is folded into
    # config_hash (hence results/<hash>.json) and recorded in the fixture manifest.
    # lowercase is applied *before* token_pattern, whose every regex match is a token.
    lowercase: bool = True
    token_pattern: str = r"[A-Za-z0-9]+"
    # Okapi BM25 hyperparameters (the rank_bm25 BM25Okapi defaults). k1 tunes term-
    # frequency saturation, b the length normalisation, epsilon the IDF floor. Also
    # recorded in the manifest; the sparse fixture guard fails if they drift from config.
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25


class ChunkConfig(BaseModel):
    # 448 not 512: deliberate headroom below bge's 512-token window (2 special
    # tokens + margin) so a full chunk is never silently truncated at embed time.
    size: int = 448
    overlap: int = 64
    unit: Literal["tokens", "chars"] = "tokens"
    # Hard rule from CLAUDE.md: chunks never span a page boundary.
    respect_page_boundary: bool = True


class RetrievalConfig(BaseModel):
    top_k: int = 20  # candidates pulled before rerank
    # Which Retriever the eval harness builds. The real, model-backed strategies are
    # "dense" (bge + Qdrant), "sparse" (BM25 over chunk text), and "hybrid" (both,
    # fused). "fixture" is the offline CI path that reads committed fixtures/*.npz
    # into an in-process Qdrant and never loads a model. Config-driven (not a CLI
    # flag) so `ledgion eval --tier 1 --gate` runs the offline path in CI purely via
    # LEDGION_RETRIEVAL__BACKEND=fixture.
    backend: Literal["dense", "sparse", "hybrid", "fixture"] = "dense"


class FusionConfig(BaseModel):
    method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60  # reciprocal-rank-fusion smoothing constant
    dense_weight: float = 1.0
    sparse_weight: float = 1.0


class QdrantConfig(BaseModel):
    # "docker" -> local server for dev; "memory" -> in-process client for CI.
    mode: Literal["docker", "memory"] = "docker"
    url: str = "http://localhost:6333"
    path: Path = Path("qdrant_storage")
    collection_name: str = "ledgion_filings"
    prefer_grpc: bool = False


class GenerationConfig(BaseModel):
    provider: Literal["gemini"] = "gemini"
    model: str = "gemini-2.0-flash"
    temperature: float = 0.0  # deterministic; never a local model in the serving path
    max_output_tokens: int = 2048
    api_key_env: str = "GEMINI_API_KEY"  # name of the env var holding the secret
    # Gemini "thinking" budget in tokens. Thinking tokens COUNT AGAINST
    # max_output_tokens, so leaving it on can starve the actual answer and
    # truncate the JSON mid-string. 0 disables thinking (right for this
    # deterministic, temperature-0 extraction task); -1 lets the model choose;
    # a positive int caps it; None omits the field entirely (model default).
    thinking_budget: int | None = 0
    # Bounded retry with exponential backoff for *transient* API failures (HTTP
    # 429/5xx, e.g. "model overloaded"). Delay before attempt n is
    # retry_base_delay_s * 2**n. Non-transient errors (bad key, unknown model)
    # fail immediately — retrying them only delays the inevitable.
    max_retries: int = 5
    retry_base_delay_s: float = 2.0


class TracingConfig(BaseModel):
    enabled: bool = False
    provider: Literal["langfuse"] = "langfuse"
    host: str = "https://cloud.langfuse.com"
    public_key_env: str = "LANGFUSE_PUBLIC_KEY"
    secret_key_env: str = "LANGFUSE_SECRET_KEY"


class EvalConfig(BaseModel):
    # Tier 1 is deterministic and fully offline (no network/GPU/API keys).
    tier: Literal[1, 2] = 1
    ragas_enabled: bool = False  # Tier 2 (judged) only
    judge_model: str = "gemini-2.0-flash"
    metrics: list[str] = Field(default_factory=lambda: ["recall@k", "mrr", "hit@1"])
    # The CI gate (ledgion eval --gate). It fails the build only if one of
    # gate_metrics falls more than gate_tolerance below fixtures/baseline_metrics.json;
    # an improvement (or a dip within tolerance) always passes. Both are config, not
    # constants, so which metrics gate and how much slack they get stay tunable.
    gate_tolerance: float = 0.02
    gate_metrics: list[str] = Field(default_factory=lambda: ["recall@10", "ndcg@10"])


class Settings(BaseSettings):
    """Top-level resolved configuration."""

    # Path of the YAML file the customised source should read. Set by
    # ``load_config``; a ClassVar so it stays out of the model schema.
    yaml_path: ClassVar[Path] = DEFAULT_CONFIG_PATH

    model_config = SettingsConfigDict(
        env_prefix="LEDGION_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Hand-maintained company -> ticker map. FinanceBench ships no ticker field,
    # so ticker provenance lives in config (keyed by the FinanceBench `company`
    # name) rather than being parsed from filenames. config/default.yaml is the
    # source of truth; an unmapped company raises at ingest, never a blank ticker.
    tickers: dict[str, str] = Field(default_factory=dict)

    paths: PathsConfig = Field(default_factory=PathsConfig)
    torch: TorchConfig = Field(default_factory=TorchConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    sparse: SparseConfig = Field(default_factory=SparseConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Order = priority (first wins): init > env > .env > yaml > defaults.
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=settings_cls.yaml_path)
        return (init_settings, env_settings, dotenv_settings, yaml_source, file_secret_settings)


def load_config(path: str | Path | None = None) -> Settings:
    """Load and validate configuration.

    ``path`` overrides which YAML file is read (defaults to
    ``config/default.yaml``); environment and ``.env`` still layer on top.
    """
    # Load the repo-root ``.env`` into the process environment. pydantic reads
    # ``.env`` only to fill LEDGION_-prefixed *model* fields; secrets named
    # indirectly (``generation.api_key_env`` -> GEMINI_API_KEY, read via
    # ``os.environ``) would otherwise never see it. ``override=False`` keeps a real
    # environment variable ahead of ``.env``, matching the env > .env precedence
    # used for the model fields. Path is resolved from the repo root, not the CWD,
    # so it's found regardless of where the command is run.
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)

    Settings.yaml_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return Settings()


def _posix_paths(value: Any) -> Any:
    """Recursively rewrite every ``PurePath`` to its POSIX (forward-slash) string.

    Pydantic serialises a ``Path`` with the *host* separator — ``data\\pdfs`` on
    Windows, ``data/pdfs`` on Linux — so the identical config would otherwise hash
    differently on the dev machine and the CI runner, splitting
    ``results/<hash>.json`` and defeating ``--compare``. This normalises paths to
    forward slashes *for hashing only*: it runs on a throwaway copy built solely to
    compute the digest and does not change how paths are stored or used anywhere
    else.
    """
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: _posix_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_posix_paths(item) for item in value]
    return value


def config_hash(config: Settings | BaseModel | dict[str, Any]) -> str:
    """Return a stable SHA-256 hex digest for a config.

    Keys are sorted and the JSON is emitted canonically, so the digest depends
    only on the values, not on dict/field ordering. ``Path`` fields are normalised
    to POSIX separators first (see ``_posix_paths``), so the *same* config hashes
    identically on Windows and Linux. This is the key eval runs use for
    ``results/<hash>.json``.
    """
    if isinstance(config, BaseModel):
        # Python-mode dump (not mode="json") keeps Path fields as Path objects, so
        # _posix_paths can normalise their separators before they become strings.
        data: Any = config.model_dump()
    else:
        data = config
    data = _posix_paths(data)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
