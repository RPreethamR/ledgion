"""Server-side citation validation — a judge-free faithfulness signal.

The generator asks the model to cite the chunk_ids it used. Nothing stops a model
from citing an id that was never in its context (a fabricated citation). This
module checks every cited id against the exact set of ids we put in the prompt,
keeps the real ones, drops the rest, and reports the fraction that were real as
``validity``. It is pure and deterministic — no model, no judge — so it doubles
as a cheap faithfulness metric the offline eval can aggregate later.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """The outcome of validating one answer's citations."""

    kept: list[str]  # cited ids that were in the context (valid provenance)
    dropped: list[str]  # cited ids that were NOT in the context (fabricated)
    validity: float  # kept / distinct-cited; 1.0 when the model cited nothing


def validate_citations(cited: Sequence[str], context_ids: Iterable[str]) -> CitationCheck:
    """Split ``cited`` ids into those present in ``context_ids`` and those not.

    Duplicates in ``cited`` are collapsed (a model may repeat an id) while
    first-seen order is preserved for stable output. ``validity`` is the share of
    *distinct* cited ids that were real; an answer that cites nothing is vacuously
    valid (1.0) — there is nothing fabricated to penalise, and that is the honest
    state for an ``insufficient_evidence`` abstention.
    """
    allowed = set(context_ids)
    seen: set[str] = set()
    kept: list[str] = []
    dropped: list[str] = []
    for cid in cited:
        if cid in seen:
            continue
        seen.add(cid)
        (kept if cid in allowed else dropped).append(cid)
    total = len(kept) + len(dropped)
    validity = 1.0 if total == 0 else len(kept) / total
    return CitationCheck(kept=kept, dropped=dropped, validity=validity)
