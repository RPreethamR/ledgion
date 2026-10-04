"""Blind hand-grading support for Tier 2 correctness.

An LLM judge is only trustworthy if it agrees with a human. These two tools measure
that without letting the judge anchor the human:

* ``export_grading_sheet`` writes a CSV for **blind** grading — qid, question, gold
  answer, answer type, the generated answer, cited pages, an answer-text hash, and an
  empty ``human_verdict`` column. The judge's verdicts are deliberately **absent**:
  grading next to them would bias the grader. Refusals are pre-filled (they are
  deterministic — there is nothing for a human to grade).
* ``judge_agreement`` joins the completed sheet back to the judge's verdicts on
  ``(qid, answer_hash)`` and reports agreement, a confusion matrix, Cohen's kappa, and
  every disagreement with the judge's stated reason. It **fails** if any answer text
  differs from what was exported (grading a stale answer is meaningless).

Correctness verdicts are the human-authored half of this project's eval; the kappa /
agreement computation here is a candidate for the author to review.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ledgion.eval.tier2 import answer_hash

# The two human-gradable verdicts (refusals are pre-filled and excluded from kappa).
_VERDICTS = ("correct", "incorrect")
_REFUSED = "refused"

_SHEET_COLUMNS = (
    "qid",
    "answer_type",
    "question",
    "gold_answer",
    "generated_answer",
    "cited_pages",
    "answer_hash",
    "human_verdict",
)


def _format_cited_pages(cited_pages: Sequence[Sequence]) -> str:
    """Render ``[[doc_id, page], ...]`` as ``"doc p.N; doc p.M"`` for the sheet."""
    return "; ".join(f"{doc_id} p.{page}" for doc_id, page in cited_pages)


def export_grading_sheet(records: Sequence[dict], path: Path) -> int:
    """Write the blind grading CSV; return the number of rows a human must still grade.

    Refusals get ``human_verdict=refused`` pre-filled (deterministic); every other row's
    ``human_verdict`` is left empty for the grader. No judge verdict is written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_grade = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SHEET_COLUMNS)
        writer.writeheader()
        for r in records:
            refused = r.get("refused", False)
            if not refused:
                to_grade += 1
            writer.writerow(
                {
                    "qid": r["qid"],
                    "answer_type": r["answer_type"],
                    "question": r["question"],
                    "gold_answer": r["gold_answer"],
                    "generated_answer": r["answer_text"],
                    "cited_pages": _format_cited_pages(r.get("cited_pages", [])),
                    "answer_hash": r["answer_hash"],
                    "human_verdict": _REFUSED if refused else "",
                }
            )
    return to_grade


def read_grading_sheet(path: Path) -> list[dict]:
    """Read a completed grading sheet back as a list of row dicts."""
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# -- agreement ---------------------------------------------------------------


@dataclass(frozen=True)
class Disagreement:
    qid: str
    human: str
    judge: str
    judge_reason: str


@dataclass(frozen=True)
class AgreementReport:
    n: int  # non-refusal questions compared (both sides have a correct/incorrect verdict)
    agreement: float  # observed agreement (po)
    kappa: float  # Cohen's kappa
    confusion: dict  # {(human, judge): count} over _VERDICTS
    disagreements: list[Disagreement] = field(default_factory=list)
    refused: int = 0  # refusals (deterministic, excluded from kappa)
    ungraded: int = 0  # non-refusal sheet rows left blank by the human


class AgreementError(RuntimeError):
    """Raised when the sheet can't be trusted against the judge's records (answer drift,
    an unknown qid, or an invalid verdict)."""


def _cohen_kappa(confusion: dict, n: int) -> tuple[float, float]:
    """Return (observed agreement po, Cohen's kappa) for a confusion over _VERDICTS."""
    if n == 0:
        return 0.0, 0.0
    po = sum(confusion[(v, v)] for v in _VERDICTS) / n
    # Expected agreement by chance: sum_c P(human=c) * P(judge=c).
    pe = 0.0
    for c in _VERDICTS:
        human_c = sum(confusion[(c, j)] for j in _VERDICTS) / n
        judge_c = sum(confusion[(h, c)] for h in _VERDICTS) / n
        pe += human_c * judge_c
    if pe >= 1.0:
        # Degenerate (one category only): kappa is undefined; report 1.0 iff po is perfect.
        return po, 1.0 if po == 1.0 else 0.0
    return po, (po - pe) / (1.0 - pe)


def judge_agreement(sheet_rows: Sequence[dict], records: Sequence[dict]) -> AgreementReport:
    """Join a completed sheet to the judge's records on ``(qid, answer_hash)`` and score
    agreement. Raises ``AgreementError`` on answer drift, an unknown qid, or a bad verdict.
    """
    by_qid = {r["qid"]: r for r in records}
    confusion = {(h, j): 0 for h in _VERDICTS for j in _VERDICTS}
    disagreements: list[Disagreement] = []
    n = refused = ungraded = 0

    for row in sheet_rows:
        qid = row["qid"]
        record = by_qid.get(qid)
        if record is None:
            raise AgreementError(f"sheet row {qid!r} has no matching judge record")

        # The answer must be exactly what was graded. Check the sheet's own hash against
        # its text AND against the judge's record — any drift invalidates the join.
        sheet_answer = row.get("generated_answer", "")
        if answer_hash(sheet_answer) != row.get("answer_hash"):
            raise AgreementError(f"{qid}: sheet answer text doesn't match its own answer_hash")
        if row.get("answer_hash") != record["answer_hash"]:
            raise AgreementError(
                f"{qid}: answer text differs from the judged answer (answer_hash mismatch)"
            )

        human = (row.get("human_verdict") or "").strip().lower()
        if record.get("refused"):
            refused += 1
            continue  # deterministic; not part of the judge-vs-human comparison
        if human == "":
            ungraded += 1
            continue
        if human not in _VERDICTS:
            raise AgreementError(
                f"{qid}: human_verdict must be one of {_VERDICTS} (or blank), got {human!r}"
            )

        judge = record["correctness"]
        confusion[(human, judge)] += 1
        n += 1
        if human != judge:
            disagreements.append(
                Disagreement(qid, human, judge, record.get("correctness_reason") or "")
            )

    po, kappa = _cohen_kappa(confusion, n)
    return AgreementReport(
        n=n,
        agreement=round(po, 6),
        kappa=round(kappa, 6),
        confusion=confusion,
        disagreements=disagreements,
        refused=refused,
        ungraded=ungraded,
    )


def format_agreement(report: AgreementReport) -> str:
    """Render the agreement report: headline, 2x2 confusion matrix, and disagreements."""
    extra = ""
    if report.refused or report.ungraded:
        extra = f"  (+{report.refused} refused, {report.ungraded} ungraded)"
    lines = [
        f"judge vs human agreement over {report.n} non-refusal question(s){extra}",
        f"  agreement (po): {report.agreement:.4f}    Cohen's kappa: {report.kappa:.4f}",
        "",
        "confusion (rows = human, cols = judge):",
        f"{'':>12}{'judge:correct':>16}{'judge:incorrect':>18}",
    ]
    for h in _VERDICTS:
        cells = "".join(f"{report.confusion[(h, j)]:>16}" for j in _VERDICTS)
        lines.append(f"human:{h:<6}{cells}")
    if report.disagreements:
        lines.append("")
        lines.append("disagreements (human -> judge; judge's reason):")
        for d in report.disagreements:
            lines.append(f"  {d.qid}: {d.human} -> {d.judge}  |  {d.judge_reason}")
    return "\n".join(lines)
