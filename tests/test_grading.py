"""Blind hand-grading export + judge-agreement.

Offline: builds Tier 2 records by hand and round-trips them through the CSV. Proves the
sheet never leaks the judge's verdicts, refusals are pre-filled, and the agreement join
fails when the answer text has drifted from what was graded.
"""

from __future__ import annotations

import pytest

from ledgion.eval.grading import (
    AgreementError,
    export_grading_sheet,
    judge_agreement,
    read_grading_sheet,
)
from ledgion.eval.tier2 import answer_hash


def _rec(qid, answer_type, text, *, refused=False, correctness=None, reason="", uncited=False):
    return {
        "qid": qid,
        "answer_type": answer_type,
        "question": f"question {qid}",
        "gold_answer": f"gold {qid}",
        "answer_text": text,
        "answer_hash": answer_hash(text),
        "cited_pages": [["AMCOR_2023_10K", 50]] if not (refused or uncited) else [],
        "citation_validity": 1.0,
        "refused": refused,
        "uncited": uncited,
        "faithfulness": None if refused else (0.0 if uncited else 0.9),
        "correctness": correctness,
        "correctness_reason": reason,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def test_grading_sheet_omits_judge_verdicts_and_prefills_refusals(tmp_path):
    records = [
        _rec("q1", "numeric", "Net sales $14,694M", correctness="correct", reason="figure matches"),
        _rec("q2", "prose", "The filings do not contain the answer.", refused=True),
    ]
    out = tmp_path / "sheet.csv"
    to_grade = export_grading_sheet(records, out)

    assert to_grade == 1  # only the non-refusal needs a human
    text = out.read_text(encoding="utf-8")
    assert "figure matches" not in text  # the judge's reason must not leak
    rows = read_grading_sheet(out)
    assert all("correctness" not in col for col in rows[0])  # no judge-verdict column
    by_qid = {r["qid"]: r for r in rows}
    assert by_qid["q1"]["human_verdict"] == ""  # blank for the grader
    assert by_qid["q2"]["human_verdict"] == "refused"  # deterministic, pre-filled


def test_judge_agreement_confusion_kappa_and_disagreements(tmp_path):
    records = [
        _rec("q1", "numeric", "A", correctness="correct", reason="r1"),
        _rec("q2", "numeric", "B", correctness="incorrect", reason="wrong period"),
        _rec("q3", "prose", "C", correctness="correct", reason="r3"),
        _rec("q4", "prose", "D", refused=True),
    ]
    out = tmp_path / "s.csv"
    export_grading_sheet(records, out)
    rows = read_grading_sheet(out)
    human = {"q1": "correct", "q2": "correct", "q3": "correct"}  # q2 is the disagreement
    for r in rows:
        if r["qid"] in human:
            r["human_verdict"] = human[r["qid"]]

    report = judge_agreement(rows, records)
    assert report.n == 3 and report.refused == 1
    assert report.confusion[("correct", "correct")] == 2
    assert report.confusion[("correct", "incorrect")] == 1
    assert report.agreement == round(2 / 3, 6)
    assert len(report.disagreements) == 1
    d = report.disagreements[0]
    assert (d.qid, d.human, d.judge) == ("q2", "correct", "incorrect")
    assert d.judge_reason == "wrong period"


def test_agreement_fails_on_answer_hash_mismatch(tmp_path):
    records = [_rec("q1", "numeric", "original answer", correctness="correct")]
    out = tmp_path / "s.csv"
    export_grading_sheet(records, out)
    rows = read_grading_sheet(out)
    rows[0]["human_verdict"] = "correct"

    # The judge re-ran and the answer changed, but the sheet was graded against the old one.
    records[0]["answer_text"] = "different answer"
    records[0]["answer_hash"] = answer_hash("different answer")

    with pytest.raises(AgreementError):
        judge_agreement(rows, records)


def test_agreement_rejects_invalid_human_verdict(tmp_path):
    records = [_rec("q1", "numeric", "A", correctness="correct")]
    out = tmp_path / "s.csv"
    export_grading_sheet(records, out)
    rows = read_grading_sheet(out)
    rows[0]["human_verdict"] = "sort-of"

    with pytest.raises(AgreementError):
        judge_agreement(rows, records)
