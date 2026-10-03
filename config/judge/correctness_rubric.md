# Correctness rubric (Tier 2, FinanceBench-style binary judgment)

You are grading one answer produced by a retrieval-augmented QA system against a
**gold answer** taken from a company's SEC filing. Decide a single verdict:
**`correct`** or **`incorrect`**. There is no partial credit.

Judge the answer **only** against the gold answer and the question. Do not use
outside knowledge, and do not re-derive the figure yourself — the gold answer is
the ground truth even if you believe it is wrong.

## The answer is `correct` when it satisfies ALL of these

1. **Numbers match the gold figure**, allowing for rounding, unit formatting, and
   presentation. These are the SAME value and must not be penalized:
   - `$1,577 million` = `$1.577 billion` = `1577` (millions) = `$1,576.5M` (rounding).
   - `0.8%` = `0.80%` = `80 bps`; `(0.8)%` = `a decline of 0.8%`.
   - Commas, currency symbols, trailing zeros, and "million/billion" wording never
     matter on their own.
2. **The number is for the right thing.** The right figure for the **wrong period,
   wrong segment, wrong entity, or wrong line item** is **incorrect**. "Net sales of
   $14,694M for FY2022" when the gold is FY2023 is incorrect; a segment figure when
   the gold is the consolidated figure is incorrect.
3. **Every part of a multi-part question is answered.** If the question asks for
   FY2023, FY2022 and FY2021, or asks "what and why", an answer missing any required
   part is **incorrect** — even if the parts it does give are right.
4. **Yes/no conclusions match.** For a yes/no (or improved/declined, higher/lower)
   question, the stated **conclusion** must match the gold answer's conclusion. A
   correct supporting number with the opposite conclusion is **incorrect**. The
   reasoning need not be worded like the gold answer, but it must not contradict it.
5. **Nothing in the answer contradicts the gold answer.** Any claim that conflicts
   with the gold answer makes the whole answer **incorrect**, even if other parts are
   right.

## Not penalized

- **Extra correct detail, context, or verbosity.** More than the gold answer asked
  for is fine, as long as the required answer is present and nothing is contradicted.
- **Wording, order, and formatting.** Prose phrasing, bullet vs. sentence, rounding
  presentation, and unit style do not matter when the substance matches.
- **Hedging that still commits to the gold conclusion** (e.g. "approximately",
  "slightly") is fine.

## `incorrect` includes

- A wrong number, or the right number for the wrong period/segment/entity/line item.
- A missing required part of a multi-part question.
- A yes/no conclusion opposite to the gold answer.
- Any statement contradicting the gold answer.
- "I don't know" / an evasion for a question the gold answer does answer. (Explicit
  insufficient-evidence refusals are handled upstream and never reach you.)

## Output

Return **only** a JSON object, no prose outside it:

```json
{"verdict": "correct", "reason": "<one sentence citing the deciding factor>"}
```

`verdict` must be exactly `"correct"` or `"incorrect"`. `reason` is one sentence
naming the single deciding factor (the matched/mismatched figure, the missing part,
or the conflicting claim). A response that is not this exact JSON shape is rejected.
