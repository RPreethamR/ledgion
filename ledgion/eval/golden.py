"""Golden-set rules for Ledgion's offline evaluation.

This module holds the *pure* decisions that define the golden set: how a
FinanceBench page number maps to our page convention, and how an answer is
typed. Keeping them here — importable and tested — means the build script, the
verifier, and the tests all agree by construction: there is exactly one
definition of each rule.

No I/O, no data loading. See ``build_golden.py`` for the loader that applies
these rules and ``verify_golden.py`` for the one-time page-alignment check.
"""

from __future__ import annotations

# Question-intent cues for answer typing (see ``classify_answer_type``). These
# lists were calibrated on the 30 dev questions and are the user's to own — the
# printed split is the review surface. Order matters: a "what drove ..." cause
# question is prose even when it names a margin, so the prose cues are matched
# first.
_PROSE_CUES = (
    "what drove",  # asks for a cause / driver -> narrative, not a figure
    "what industry",
    "geograph",  # geography / geographies
    "products and services",
    "major products",
    "acquisitions",
    "debt securities",
    "legal battles",
    "primary customers",
    "who are",
    "cyclicality",
    "retain",
)
_NUMERIC_CUES = (
    "how much",
    "how many",
    "quick ratio",
    "current ratio",
    "tax rate",
    "ebitda",
    "capex",
    "margin profile",  # "improving ... margin profile" -> compare margins over years
    "restructuring costs",
    "customer concentration",
    "production rate",  # a production rate is a quantity; "rate changes" -> figures
    "more than",  # "represent more than 20% of revenue"
    "proportionally",
    "the most",  # superlative ranking of reported figures
    "the least",
    "largest",
    "greatest",
    "highest",
    "lowest",
)


def to_one_based(page_num: int) -> int:
    """Convert a FinanceBench 0-based page number to our 1-based convention.

    FinanceBench records evidence with ``evidence_page_num`` counted from **0**
    (the first page of the PDF is page 0). Everything Ledgion stores, cites,
    prints, and *scores retrieval on* is **1-based** (the first page is page 1),
    because that is the page number a human sees when reading a filing and the
    stable key we compare retrieval strategies against (see CLAUDE.md: retrieval
    is scored at page level).

    The spike (see DECISIONS.md) confirmed FinanceBench's numbering aligns with
    our PyMuPDF extraction with exactly this +1 offset — PyMuPDF also indexes
    pages from 0, so ``evidence_page_num`` N is PyMuPDF index N is our page N+1.

    This ``+1`` must happen in **exactly one place** so the two conventions can
    never silently drift or be applied twice. That place is this function; no
    other code in the project may add or subtract a page offset.

    Args:
        page_num: A FinanceBench ``evidence_page_num`` (0-based, ``>= 0``).

    Returns:
        The same page in our 1-based convention (``page_num + 1``).

    Raises:
        ValueError: If ``page_num`` is negative — that signals a bug upstream,
            not a valid 0-based index, and must not silently yield page 0.
    """
    if page_num < 0:
        raise ValueError(f"page_num must be a 0-based page index (>= 0), got {page_num}")
    return page_num + 1


def classify_answer_type(question: str) -> str:
    """Classify a question by the *kind of content answering it requires*.

    (The user owns this golden-set decision and reviews the printed split; this
    is the first-pass mechanism.) The classification keys the metric used later:
    numeric answers admit exact / tolerance matching, prose answers need fuzzy
    or judged scoring — so what matters is what the question *demands*, not how
    the reference answer happens to be phrased.

    * ``"numeric"`` — answering requires pulling one or more specific figures
      from the filing (amounts, percentages, ratios, counts, line items) or
      comparing them, *even when the reference answer is a sentence*. E.g.
      "Has the quick ratio improved between FY2022 and FY2023?" or "Which was
      the largest liability?".
    * ``"prose"`` — answering requires narrative or qualitative content. E.g.
      "What industry does the company operate in?" or "What drove the change in
      operating margin?" (a cause, not a figure).

    Heuristic: match the lower-cased question against cue phrases. Prose cues
    are checked first, so a "what drove ..." cause question stays prose even
    though it names a numeric metric; then numeric cues; default prose (a
    question with no quantitative demand is treated as qualitative).
    """
    q = question.lower()
    if any(cue in q for cue in _PROSE_CUES):
        return "prose"
    if any(cue in q for cue in _NUMERIC_CUES):
        return "numeric"
    return "prose"
