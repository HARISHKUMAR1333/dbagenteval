"""Compare what the agent produced with the result of a reference query.

A case's `reference_sql` is a query known to be correct. It is run when the case is
evaluated, so the expected result is always current, and three checks use its rows:

  * `reference_rows` (no LLM): the agent's result rows are the reference rows.
  * `reference_numbers` (no LLM): used when the agent gives no rows, only an answer. The
    answer must state a value from the reference result and no number outside it.
  * `reference_answer` (LLM judge): the answer says what the reference result says.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dbagenteval.answer import Judge, ask_judge, states_value, ungrounded_numbers
from dbagenteval.case import Case
from dbagenteval.checks import _short
from dbagenteval.compare import rows_match
from dbagenteval.scorecard import FAIL, PASS, SKIP, CheckResult
from dbagenteval.trace import Stage, Trace

Rows = list[dict[str, Any]]

# An executor runs one SQL query on your database and returns its rows as dicts.
Executor = Callable[[str], Rows]

REFERENCE_PROMPT = """\
You are checking a data assistant's answer against the correct result for the question.

QUESTION:
{question}

CORRECT RESULT ROWS (JSON), from a reference query known to be right:
{rows}

ANSWER:
{answer}

Does the ANSWER give the correct result? Pass when the figures, names and statuses it
states agree with the CORRECT RESULT ROWS and it gives what the question asked for,
allowing rounding, unit formatting, and a summary that lists only some of many rows.
Fail when a stated figure or name differs from the correct result, when the answer does
not give the result at all (it refuses, or says nothing was found although rows exist),
or when it claims a result although the correct result has no rows. If a note says the
rows are cut short, do not fail a claim merely because it may concern a part not shown.

"""


def reference_rows(case: Case, execute: Executor | None) -> Rows | None:
    """Run the case's reference query, or return None when it has none.

    A reference query that cannot be run is a problem with the case or the setup, not
    with the agent, so it raises instead of producing a failed check.
    """
    if case.reference_sql is None:
        return None
    if execute is None:
        raise ValueError(
            "the case has `reference_sql` but no `execute` function was given to run it"
        )
    try:
        rows = execute(case.reference_sql)
    except Exception as e:
        raise RuntimeError(f"the reference SQL failed to run: {type(e).__name__}: {e}") from e
    try:
        Stage("execute", name="reference", rows=rows)  # validates the shape
    except ValueError as e:
        raise ValueError(f"the `execute` function must return a list of dicts: {e}") from e
    return rows


def reference_checks(
    trace: Trace, case: Case, reference: Rows, judge: Judge | None = None
) -> list[CheckResult]:
    """The checks that compare a trace with the reference result."""
    execute = trace.last_of("execute")
    answer = trace.last_of("answer")
    answer_text = answer.output if answer is not None and isinstance(answer.output, str) else ""

    results = []
    if execute is not None and execute.rows is not None:
        results.append(_check_rows(execute, case, reference))
    elif execute is not None and execute.error:
        results.append(
            CheckResult("reference_rows", execute.name, SKIP, "stage raised, no rows to compare")
        )
    elif answer is not None and answer_text.strip():
        results.append(_check_numbers(answer.name, answer_text, trace.question, reference))
    else:
        results.append(
            CheckResult(
                "reference_rows",
                "execute",
                FAIL,
                "the trace has neither result rows nor an answer to compare with the reference",
            )
        )

    if judge is not None and answer is not None and answer_text.strip():
        passed, reason = ask_judge(judge, REFERENCE_PROMPT, trace.question, answer_text, reference)
        status = PASS if passed else FAIL
        results.append(CheckResult("reference_answer", answer.name, status, _short(reason)))
    return results


def _check_rows(execute: Stage, case: Case, reference: Rows) -> CheckResult:
    rows = execute.rows or []
    if rows_match(reference, rows, case.ordered, case.precision):
        return CheckResult("reference_rows", execute.name, PASS, f"{len(rows)} row(s)")
    return CheckResult(
        "reference_rows",
        execute.name,
        FAIL,
        f"got {len(rows)} row(s), first: {_short(rows[:1])}; "
        f"the reference query gives {len(reference)} row(s), first: {_short(reference[:1])}",
    )


def _check_numbers(stage: str, answer: str, question: str, reference: Rows) -> CheckResult:
    """Without the agent's rows, compare the answer text with the reference result.

    A heuristic: it cannot tell whether a long answer covers every row, only that what
    it states comes from the reference result. Give the agent's rows in the trace, or
    use a judge, for a stronger check.
    """
    name = "reference_numbers"
    ungrounded = ungrounded_numbers(answer, question, reference)
    if ungrounded:
        return CheckResult(name, stage, FAIL, f"not in the reference result: {ungrounded}")
    values = [value for row in reference for value in row.values() if value is not None]
    if len(values) == 1 and not states_value(answer, values[0]):
        return CheckResult(
            name, stage, FAIL, f"the answer does not state the reference value {values[0]!r}"
        )
    if values and not any(states_value(answer, value) for value in values):
        return CheckResult(
            name, stage, FAIL, "the answer states no value from the reference result"
        )
    return CheckResult(name, stage, PASS)
