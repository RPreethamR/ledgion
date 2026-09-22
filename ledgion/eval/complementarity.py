"""Complementarity analysis between two retrieval runs (analysis tooling only).

This changes nothing Tier 1 reports — it reads two ``results/<hash>.json`` files
post hoc and asks: for each piece of golden evidence, which of the two candidate
pools (A, B) contains it? That is the question that decides whether *fusing* the two
runs can help at all: fusion can only reorder the union of what its inputs already
retrieved, so the recall of that union is the hard ceiling any fusion of A and B
can reach.

The unit of analysis is each distinct ``(qid, evidence page)`` pair from the golden
set, keyed as ``(doc_id, page_num)`` — exactly the key Tier 1 scores on. A pair is
"found" by a system if that key appears within the top-K of the system's per-question
``retrieved_pages`` (the collapsed candidate ranking the run already wrote), where
K is the ``retrieval.top_k`` recorded in the results file — the candidate pool depth
fusion draws from.

The consistency check is the guard rail: it recomputes each system's recall@K purely
from this classification and requires it to reproduce, to the digit, the recall@K the
harness wrote. If it doesn't, the analysis is reading the files differently from the
eval harness and every number below is suspect — so it raises rather than mislead.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Cells of the 2x2, in a fixed print/count order.
CELLS = ("both", "A only", "B only", "neither")
# The metric key Tier 1 writes for recall at the full candidate depth (retrieval.top_k).
_RECALL_AT_K = "recall@k"


class ConsistencyError(RuntimeError):
    """Raised when recomputed recall@K disagrees with a results file's own recall@K."""


@dataclass(frozen=True)
class Pair:
    """One (qid, evidence page) unit and where each system ranked it (1-based)."""

    qid: str
    doc_id: str
    page: int
    answer_type: str
    rank_a: int | None  # None = not in A's top-K candidate pool
    rank_b: int | None

    @property
    def cell(self) -> str:
        in_a, in_b = self.rank_a is not None, self.rank_b is not None
        if in_a and in_b:
            return "both"
        if in_a:
            return "A only"
        if in_b:
            return "B only"
        return "neither"


@dataclass(frozen=True)
class SystemConsistency:
    system: str
    recomputed: float
    from_file: float

    @property
    def ok(self) -> bool:
        return self.recomputed == self.from_file


@dataclass(frozen=True)
class AnalysisResult:
    k: int
    pairs: list[Pair]
    counts: dict[str, dict[str, int]]  # group -> {cell -> count}
    recall_a: float
    recall_b: float
    recall_union: float
    consistency: list[SystemConsistency]


# -- reading results files ---------------------------------------------------


def resolve_k(a_report: dict, b_report: dict) -> int:
    """The common candidate depth K = retrieval.top_k. Both runs must agree on it,
    else a union at a shared depth is undefined."""
    ka = a_report["config"]["retrieval"]["top_k"]
    kb = b_report["config"]["retrieval"]["top_k"]
    if ka != kb:
        raise SystemExit(
            f"retrieval.top_k differs between the two runs (A={ka}, B={kb}); "
            f"they cannot be compared at a common candidate depth."
        )
    return ka


def ranked_pages(report: dict) -> dict[str, list[tuple[str, int]]]:
    """Each question's collapsed (doc_id, page_num) ranking, best-first, keyed by qid.

    Reads the ``retrieved_pages`` the runner already wrote (via ``collapse_to_pages``)
    — no retrieval is re-run. Fails loudly on a file that predates doc-qualified
    scoring (bare page ints), because comparing those keys to Tier 1's would silently
    mis-align."""
    out: dict[str, list[tuple[str, int]]] = {}
    for row in report["results"]:
        keys: list[tuple[str, int]] = []
        for item in row["retrieved_pages"]:
            is_key = isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)
            if not is_key:
                raise SystemExit(
                    "results file does not store (doc_id, page_num) rankings "
                    f"(saw {item!r}); it predates doc-qualified scoring — re-run `ledgion eval`."
                )
            keys.append((item[0], int(item[1])))
        out[row["qid"]] = keys
    return out


# -- classification + metrics ------------------------------------------------


def build_pairs(
    golden: Sequence[dict],
    a_ranked: dict[str, list[tuple[str, int]]],
    b_ranked: dict[str, list[tuple[str, int]]],
    k: int,
) -> list[Pair]:
    """One Pair per (qid, evidence page), with each system's 1-based rank within top-K."""
    pairs: list[Pair] = []
    for row in golden:
        qid = row["qid"]
        if qid not in a_ranked:
            raise SystemExit(f"golden qid {qid!r} is missing from run A's results.")
        if qid not in b_ranked:
            raise SystemExit(f"golden qid {qid!r} is missing from run B's results.")
        a_rank = {key: i + 1 for i, key in enumerate(a_ranked[qid][:k])}
        b_rank = {key: i + 1 for i, key in enumerate(b_ranked[qid][:k])}
        for page in row["evidence_pages"]:
            key = (row["doc_id"], page)
            pairs.append(
                Pair(
                    qid=qid,
                    doc_id=row["doc_id"],
                    page=page,
                    answer_type=row["answer_type"],
                    rank_a=a_rank.get(key),
                    rank_b=b_rank.get(key),
                )
            )
    return pairs


def cell_counts(pairs: Sequence[Pair]) -> dict[str, dict[str, int]]:
    """The 2x2 counts, overall and split by answer_type."""
    groups: dict[str, list[Pair]] = {
        "overall": list(pairs),
        "numeric": [p for p in pairs if p.answer_type == "numeric"],
        "prose": [p for p in pairs if p.answer_type == "prose"],
    }
    return {
        group: {cell: sum(1 for p in ps if p.cell == cell) for cell in CELLS}
        for group, ps in groups.items()
    }


def _recall(pairs: Sequence[Pair], found: Callable[[Pair], bool]) -> float:
    """Per-question mean recall, replicating Tier 1's rounding exactly: round each
    question's recall to 6dp, then round the mean to 6dp (see tier1.score_row/_mean).
    Questions are visited in first-seen (golden) order so the float sum order — and
    thus the rounded result — matches the harness."""
    by_q: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        by_q[p.qid].append(p)
    per_q = [round(sum(1 for p in ps if found(p)) / len(ps), 6) for ps in by_q.values()]
    return round(sum(per_q) / len(per_q), 6)


def check_consistency(
    pairs: Sequence[Pair], a_report: dict, b_report: dict
) -> list[SystemConsistency]:
    """Recompute A's and B's recall@K from the classification and compare to the files."""
    checks = []
    for system, report, found in (
        ("A", a_report, lambda p: p.rank_a is not None),
        ("B", b_report, lambda p: p.rank_b is not None),
    ):
        from_file = report["metrics"]["overall"].get(_RECALL_AT_K)
        if from_file is None:
            raise SystemExit(
                f"run {system} has no overall {_RECALL_AT_K!r} metric; the consistency "
                f"check needs it (add {_RECALL_AT_K} to eval.metrics and re-run)."
            )
        checks.append(SystemConsistency(system, _recall(pairs, found), from_file))
    return checks


def analyze(golden: Sequence[dict], a_report: dict, b_report: dict) -> AnalysisResult:
    """Full analysis. Raises ``ConsistencyError`` (stopping before any numbers are
    trusted) if the recomputed recall@K doesn't reproduce a file's own recall@K."""
    k = resolve_k(a_report, b_report)
    a_ranked, b_ranked = ranked_pages(a_report), ranked_pages(b_report)
    pairs = build_pairs(golden, a_ranked, b_ranked, k)

    consistency = check_consistency(pairs, a_report, b_report)
    if any(not c.ok for c in consistency):
        lines = ["consistency FAILED — the analysis reads the files differently from the harness:"]
        for c in consistency:
            verdict = "ok" if c.ok else "MISMATCH"
            lines.append(
                f"  system {c.system}: recomputed recall@{k}={c.recomputed:.6f}  "
                f"file={c.from_file:.6f}  [{verdict}]"
            )
        raise ConsistencyError("\n".join(lines))

    return AnalysisResult(
        k=k,
        pairs=pairs,
        counts=cell_counts(pairs),
        recall_a=_recall(pairs, lambda p: p.rank_a is not None),
        recall_b=_recall(pairs, lambda p: p.rank_b is not None),
        recall_union=_recall(pairs, lambda p: p.rank_a is not None or p.rank_b is not None),
        consistency=consistency,
    )


# -- rendering ---------------------------------------------------------------


def _label(report: dict) -> str:
    backend = report.get("config", {}).get("retrieval", {}).get("backend", "?")
    return f"{backend} {report.get('config_hash', '?')[:6]}"


def _rank(value: int | None) -> str:
    return str(value) if value is not None else "-"


def format_report(result: AnalysisResult, a_report: dict, b_report: dict) -> str:
    k = result.k
    a_label, b_label = _label(a_report), _label(b_report)
    lines = [
        f"complementarity   A = {a_label}   B = {b_label}   K={k}   pairs={len(result.pairs)}",
        "",
        "2x2 cell counts",
        f"{'cell':<10}{'overall':>9}{'numeric':>9}{'prose':>9}",
        "-" * 37,
    ]
    for cell in CELLS:
        c = result.counts
        lines.append(
            f"{cell:<10}{c['overall'][cell]:>9}{c['numeric'][cell]:>9}{c['prose'][cell]:>9}"
        )

    lines += [
        "",
        f"recall@{k} (per-question mean)",
        f"  A ({a_label}){'':<4}{result.recall_a:>10.4f}",
        f"  B ({b_label}){'':<4}{result.recall_b:>10.4f}",
        f"  union{'':<{len(a_label) + 6}}{result.recall_union:>10.4f}   <- fusion ceiling",
        "consistency: "
        + "   ".join(
            f"{c.system} recomputed {c.recomputed:.6f} == file {c.from_file:.6f} [ok]"
            for c in result.consistency
        ),
        "",
        "pairs by cell",
        f"{'qid':<24}{'doc_id':<24}{'page':>5}{'type':>9}{'rankA':>7}{'rankB':>7}",
        "-" * 76,
    ]
    for cell in CELLS:
        members = [p for p in result.pairs if p.cell == cell]
        lines.append(f"[{cell}]  (n={len(members)})")
        for p in members:
            lines.append(
                f"{p.qid:<24}{p.doc_id:<24}{p.page:>5}{p.answer_type:>9}"
                f"{_rank(p.rank_a):>7}{_rank(p.rank_b):>7}"
            )
    return "\n".join(lines)
