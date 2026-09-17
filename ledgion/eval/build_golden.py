"""Build the golden dev set (``golden/dev.jsonl``) from cached FinanceBench metadata.

Emits one JSON row per question, for exactly the documents we have PDFs for::

    uv run python -m ledgion.eval.build_golden

Each row::

    {"qid": ..., "question": ..., "answer": ...,
     "answer_type": "numeric|prose", "doc_id": ...,
     "evidence_pages": [52], "evidence_text": ...}

The selected working set is **derived from the PDFs on disk** (``data/pdfs/*.pdf``),
not a static list: a question is included iff its filing has a matching PDF. (The
spike referenced a ``selected.json``; that artifact is not in the repo, and the
PDFs present are the ground truth of what we can actually evaluate against.)

``evidence_pages`` come from FinanceBench's ``evidence_page_num`` passed through
``to_one_based`` — the single page-convention conversion in the project. All
distinct evidence pages for a question are kept (sorted); ``evidence_text`` joins
every evidence snippet so the verifier has the full labelled text to match on.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from ledgion.config import REPO_ROOT, load_config
from ledgion.eval.golden import classify_answer_type, to_one_based

OPEN_SOURCE_FILE = "financebench_open_source.jsonl"
GOLDEN_PATH = REPO_ROOT / "golden" / "dev.jsonl"


def _resolve(path: Path) -> Path:
    """Resolve a (possibly relative) config path against the repo root."""
    return path if path.is_absolute() else REPO_ROOT / path


def _selected_doc_ids(pdf_dir: Path) -> set[str]:
    """Doc ids we can evaluate = the stems of the PDFs present in ``pdf_dir``."""
    return {pdf.stem for pdf in pdf_dir.glob("*.pdf")}


def build_rows() -> list[dict]:
    """Read FinanceBench metadata and return golden rows for the selected docs."""
    cfg = load_config()
    fb_dir = _resolve(cfg.paths.financebench_dir)
    pdf_dir = _resolve(cfg.paths.pdf_dir)
    selected = _selected_doc_ids(pdf_dir)
    if not selected:
        raise SystemExit(f"No PDFs found in {pdf_dir}; nothing to build.")

    rows: list[dict] = []
    with (fb_dir / OPEN_SOURCE_FILE).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["doc_name"] not in selected:
                continue
            evidence = record["evidence"]
            pages = sorted({to_one_based(e["evidence_page_num"]) for e in evidence})
            evidence_text = "\n\n".join(e["evidence_text"] for e in evidence)
            rows.append(
                {
                    "qid": record["financebench_id"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "answer_type": classify_answer_type(record["question"]),
                    "doc_id": record["doc_name"],
                    "evidence_pages": pages,
                    "evidence_text": evidence_text,
                }
            )

    # Stable, reviewable order → clean git diffs when the set is regenerated.
    rows.sort(key=lambda row: (row["doc_id"], row["qid"]))
    return rows


def _write(rows: list[dict]) -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": LF endings so the committed golden set is byte-stable across
    # Windows (dev) and the Linux CI runner. ensure_ascii=False keeps ☑ / non-
    # breaking spaces from 10-K text readable rather than \uXXXX-escaped.
    with GOLDEN_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_split(rows: list[dict]) -> None:
    """Print every row's question + answer_type (grouped) for user review.

    answer_type is classified by *question intent* (does answering require
    pulling specific figures?), so the question — not the answer — is printed
    next to each label for review.
    """
    per_doc = Counter(row["doc_id"] for row in rows)
    numeric = [row for row in rows if row["answer_type"] == "numeric"]
    prose = [row for row in rows if row["answer_type"] == "prose"]

    print(f"\nWrote {len(rows)} rows to {GOLDEN_PATH.relative_to(REPO_ROOT).as_posix()}")
    print("\nquestions per doc:")
    for doc_id in sorted(per_doc):
        print(f"  {doc_id:<28} {per_doc[doc_id]}")

    print(f"\nanswer_type split (by question intent):  numeric={len(numeric)}  prose={len(prose)}")
    for label, group in (("NUMERIC", numeric), ("PROSE", prose)):
        print(f"\n== {label} ({len(group)}) ==")
        for row in group:
            print(f"  {row['qid']}  [{row['doc_id']}]")
            print(f"      {row['question']}")


def main() -> None:
    if sys.platform == "win32":
        # 10-K answers contain ☑ and non-breaking spaces that Windows' cp1252
        # console cannot encode; force UTF-8 before printing any filing text.
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    rows = build_rows()
    _write(rows)
    _print_split(rows)


if __name__ == "__main__":
    main()
