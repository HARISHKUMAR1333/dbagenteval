"""The check interface, and the built-in checks that compare a trace with a case."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from dbagenteval.case import Case
from dbagenteval.compare import canonical, match_any
from dbagenteval.scorecard import FAIL, PASS, SKIP, CheckResult
from dbagenteval.trace import Stage, Trace


class Check(ABC):
    """One thing to verify about a trace. Subclass it to write your own rule.

    `run` returns a CheckResult attributed to the stage the check looked at; return a
    SKIP result when the check does not apply to this trace.
    """

    name: str = ""

    @abstractmethod
    def run(self, trace: Trace, case: Case) -> CheckResult: ...

    def result(
        self, stage: str, status: str, detail: str = "", score: float | None = None
    ) -> CheckResult:
        return CheckResult(self.name or type(self).__name__, stage, status, detail, score)


def _short(value: Any, limit: int = 200) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _as_items(output: Any) -> list[str]:
    """A stage's output as a list of names (retrieved columns, an allowed table...)."""
    if output is None:
        return []
    if isinstance(output, str) or not isinstance(output, Iterable):
        return [str(output)]
    items = []
    for item in output:
        if isinstance(item, dict) and "name" in item:
            item = item["name"]
        items.append(str(item))
    return items


def _same(actual: Any, expected: Any) -> bool:
    """Equal, ignoring case and outer spaces in text and the written form of numbers
    (3, 3.0 and "3" are one value). Booleans and containers must be equal exactly."""
    scalar = (str, int, float, Decimal)
    if (
        isinstance(actual, bool)
        or isinstance(expected, bool)
        or not isinstance(actual, scalar)
        or not isinstance(expected, scalar)
    ):
        return bool(actual == expected)
    return canonical(actual).casefold() == canonical(expected).casefold()


def check_no_error(stage: Stage) -> CheckResult:
    """Did the stage complete without raising?"""
    if stage.error:
        return CheckResult("no_error", stage.name, FAIL, _short(stage.error))
    return CheckResult("no_error", stage.name, PASS)


def check_expected(trace: Trace, name: str, expected: Any) -> CheckResult:
    """Compare one stage's output with the case's expected value.

    A retrieve stage passes when every expected item was retrieved (recall 1.0). Any
    other stage passes when its output equals the expected value, or any one of them
    when a list is given.
    """
    stage = trace.stage(name)
    if stage is None:
        return CheckResult("expected", name, FAIL, "stage missing from trace")

    if stage.kind == "retrieve":
        wanted = _as_items(expected)
        got = {i.strip().casefold() for i in _as_items(stage.output)}
        missing = [w for w in wanted if w.strip().casefold() not in got]
        recall = (len(wanted) - len(missing)) / len(wanted) if wanted else 1.0
        if missing:
            return CheckResult("recall", name, FAIL, f"not retrieved: {missing}", recall)
        return CheckResult(
            "recall", name, PASS, f"{len(wanted)} expected of {len(got)} retrieved", recall
        )

    options = expected if isinstance(expected, list) else [expected]
    if _same(stage.output, expected) or any(_same(stage.output, o) for o in options):
        return CheckResult("expected", name, PASS, _short(stage.output))
    wanted_text = " or ".join(repr(o) for o in options)
    return CheckResult(
        "expected", name, FAIL, f"got {_short(repr(stage.output))}, expected {_short(wanted_text)}"
    )


def check_rows(trace: Trace, case: Case) -> CheckResult | None:
    """Compare the executed result rows with the case's accepted answers."""
    accepted = case.accepted_answers
    if accepted is None:
        return None
    stage = trace.last_of("execute")
    if stage is None:
        return CheckResult("rows_match", "execute", FAIL, "no execute stage in trace")
    if stage.rows is None:
        if stage.error:
            return CheckResult("rows_match", stage.name, SKIP, "stage raised, no rows to compare")
        return CheckResult("rows_match", stage.name, FAIL, "the execute stage recorded no rows")

    matched = match_any(accepted, stage.rows, case.ordered, case.precision)
    if matched is not None:
        detail = f"{len(stage.rows)} row(s)"
        if len(accepted) > 1:
            detail += f", matched accepted answer {matched + 1} of {len(accepted)}"
        return CheckResult("rows_match", stage.name, PASS, detail)
    return CheckResult(
        "rows_match",
        stage.name,
        FAIL,
        f"got {len(stage.rows)} row(s), first: {_short(stage.rows[:1])}; "
        f"expected {len(accepted[0])} row(s), first: {_short(accepted[0][:1])}",
    )
