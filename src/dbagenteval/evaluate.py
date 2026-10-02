"""Score one trace against one case."""

from __future__ import annotations

from collections.abc import Iterable

from dbagenteval.answer import Faithful, Judge, Relevant
from dbagenteval.case import Case
from dbagenteval.checks import Check, check_expected, check_no_error, check_rows
from dbagenteval.reference import Executor, reference_checks, reference_rows
from dbagenteval.scorecard import Scorecard
from dbagenteval.trace import Trace


def evaluate(
    trace: Trace,
    case: Case | None = None,
    rules: Iterable[Check] = (),
    judge: Judge | None = None,
    execute: Executor | None = None,
) -> Scorecard:
    """Check a trace and return its scorecard.

    Four groups of checks run, in this order:
      1. every stage completed without an error;
      2. each expectation in the case matches its stage (and `rows` matches the result);
      3. when the case has `reference_sql`: it is run with `execute`, and the agent's
         result is compared with its rows (and, when `judge` is given, the answer too);
      4. each of `rules`, plus the faithfulness and relevance judges when `judge` is given.

    Without a case, only the first and fourth groups run.
    """
    case = case or Case(question=trace.question or "(no question)")
    results = [check_no_error(stage) for stage in trace.stages]

    for name, expected in case.expect.items():
        if name != "rows":
            results.append(check_expected(trace, name, expected))
    rows_result = check_rows(trace, case)
    if rows_result is not None:
        results.append(rows_result)
    reference = reference_rows(case, execute)
    if reference is not None:
        results.extend(reference_checks(trace, case, reference, judge))

    checks = list(rules)
    if judge is not None:
        checks += [Faithful(judge), Relevant(judge)]
    results.extend(check.run(trace, case) for check in checks)

    return Scorecard(
        question=trace.question,
        results=results,
        stage_order=[s.name for s in trace.stages],
        case_id=case.id,
        tags=list(case.tags),
    )
