"""Checks on the final answer text: is it grounded in the rows, and does it answer the
question?"""

from __future__ import annotations

import json
import re
from bisect import bisect_left
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

from dbagenteval.case import Case
from dbagenteval.checks import Check, _short
from dbagenteval.scorecard import FAIL, PASS, SKIP, CheckResult
from dbagenteval.trace import Stage, Trace

# A judge is any function that takes a prompt and returns the model's text reply.
Judge = Callable[[str], str]

# Every run of digits counts, wherever it sits ("Rs.500", "No.42", "Q3"): the same
# pattern reads the rows and the question, so a number glued to a word is grounded the
# same way it was written. Alternatives, in order: a leading-dot decimal (.75), a number
# with thousands separators in any grouping (1,234,567 / 1,23,456), a plain number.
_NUMBER = re.compile(r"(?<![\w.])\.\d+|\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_LIST_MARKER = re.compile(r"^[ \t]*(\d+)[.)][ \t]+", re.MULTILINE)
_PERCENT = re.compile(r"\s?(?:%|percent\b)", re.IGNORECASE)
# "15k", "1.5M", "2 million": the figure meant is the number times the unit.
_MAGNITUDE = re.compile(
    r"(k|m|mn|bn)\b|\s?(thousand|million|billion|lakhs?|crores?)\b", re.IGNORECASE
)
_MAGNITUDES = {
    "k": 3, "thousand": 3, "lakh": 5, "lakhs": 5, "m": 6, "mn": 6, "million": 6,
    "crore": 7, "crores": 7, "bn": 9, "billion": 9,
}  # fmt: skip
# Values beyond this many digits are not figures anyone quotes; skipping them keeps the
# arithmetic below from overflowing.
_MAX_DIGITS = 60


def _find_answer(check: Check, trace: Trace, name: str | None) -> Stage | CheckResult:
    """The answer stage to check, or the SKIP / FAIL result explaining why there is none."""
    if name is None:
        stage = trace.last_of("answer")
        return stage if stage is not None else check.result("answer", SKIP, "no answer text")
    stage = trace.stage(name)
    if stage is not None:
        return stage
    answers = [s.name for s in trace.stages if s.kind == "answer"]
    if answers:  # there is an answer, just not under this name: likely a typo
        return check.result(name, FAIL, f"stage {name!r} is not in the trace (found {answers})")
    return check.result(name, SKIP, "no answer text")


def _numbers(text: str) -> list[tuple[re.Match[str], Decimal, int]]:
    """Numbers in `text` as (match, value, decimal places)."""
    found = []
    for match in _NUMBER.finditer(text):
        token = match.group().replace(",", "")
        if len(token) > _MAX_DIGITS:
            continue
        places = len(token.split(".")[1]) if "." in token else 0
        found.append((match, Decimal(token), places))
    return found


def _values(text: str) -> list[Decimal]:
    return [value for _, value, _ in _numbers(text)]


def _row_numbers(rows: list[dict[str, Any]]) -> list[Decimal]:
    values: list[Decimal] = []
    for row in rows:
        for cell in row.values():
            if cell is None or isinstance(cell, bool):
                continue
            if isinstance(cell, int | float | Decimal):
                try:
                    number = abs(Decimal(str(cell)))
                except (InvalidOperation, ValueError):  # ValueError: an enormous int
                    continue
                if number.is_finite() and abs(number.adjusted()) <= _MAX_DIGITS:
                    values.append(number)
            else:
                values.extend(_values(str(cell)))
    return values


def _strip_list_markers(text: str) -> str:
    """Remove "1.", "2." list numbering. Only a real list is stripped: at least two
    markers counting up from 1, so "17. That is the total" keeps its 17."""
    markers = [int(m.group(1)) for m in _LIST_MARKER.finditer(text)]
    if len(markers) < 2 or markers != list(range(1, len(markers) + 1)):
        return text
    return _LIST_MARKER.sub("", text)


def _any_within(sorted_values: list[Decimal], low: Decimal, high: Decimal) -> bool:
    index = bisect_left(sorted_values, low)
    return index < len(sorted_values) and sorted_values[index] <= high


def _figures(text: str) -> list[tuple[str, Decimal, Decimal, bool]]:
    """Each number an answer states, as (text, low, high, is a percentage): the range of
    values it can stand for once rounding and a unit ("1.2k") are allowed for."""
    text = _strip_list_markers(text)
    figures = []
    for match, value, places in _numbers(text):
        unit = _MAGNITUDE.match(text, match.end())
        scale = Decimal(1)
        if unit:
            scale = Decimal(10) ** _MAGNITUDES[(unit.group(1) or unit.group(2)).lower()]
        # Half a unit in the last shown place: 12.5 can stand for 12.4567.
        tolerance = Decimal(5).scaleb(-places - 1)
        figures.append(
            (
                match.group() + (unit.group() if unit else ""),
                (value - tolerance) * scale,
                (value + tolerance) * scale,
                _PERCENT.match(text, match.end()) is not None,
            )
        )
    return figures


def states_value(answer: str, value: Any) -> bool:
    """True when the answer text states this result value: a number within rounding, or
    a piece of text as written (ignoring case)."""
    if value is None or isinstance(value, bool):
        return False
    numbers = _row_numbers([{"value": value}]) if isinstance(value, int | float | Decimal) else []
    if not numbers:
        text = str(value).strip().casefold()
        return len(text) >= 2 and text in answer.casefold()
    return any(
        low <= numbers[0] <= high or (percent and low <= numbers[0] * 100 <= high)
        for _, low, high, percent in _figures(answer)
    )


def ungrounded_numbers(answer: str, question: str, rows: list[dict[str, Any]]) -> list[str]:
    """The numbers in the answer that appear neither in `rows`, nor in the question, nor
    as the row count."""
    allowed = sorted([*_row_numbers(rows), *_values(question), Decimal(len(rows))])
    return [
        text
        for text, low, high, percent in _figures(answer)
        if not _any_within(allowed, low, high)
        and not (percent and _any_within(allowed, low / 100, high / 100))
    ]


def _rows_json(rows: list[dict[str, Any]], max_chars: int) -> str:
    """Rows as JSON for a judge prompt. When they do not fit, whole rows are dropped
    from the end and the judge is told, so it never reads a cut-off row as complete."""
    full = json.dumps(rows, default=str)
    if len(full) <= max_chars:
        return full
    kept: list[str] = []
    used = 0
    for row in rows:
        encoded = json.dumps(row, default=str)
        if used + len(encoded) > max_chars:
            break
        kept.append(encoded)
        used += len(encoded) + 2
    if not kept:  # even the first row is too long: show what fits of it
        return f"{full[:max_chars]}\n(the first of {len(rows)} rows, cut short)"
    return f"[{', '.join(kept)}]\n(showing the first {len(kept)} of {len(rows)} rows)"


class NumbersGrounded(Check):
    """Every number in the answer appears in the result rows.

    A number also counts as grounded when it appears in the question ("top 5"), equals
    the row count, or is a result value shown rounded, as a percentage, or with a unit
    ("1.2k" for 1200). List numbering ("1.", "2.") is ignored. Skipped when no query ran.

    This is a heuristic with no LLM. It catches invented figures. It will also flag a
    correct figure the agent derived itself (a total it summed), and it compares
    magnitudes only, so it does not notice a wrong sign. Use a judge check for those.
    """

    name = "numbers_grounded"

    def __init__(self, stage: str | None = None) -> None:
        self.stage = stage

    def run(self, trace: Trace, case: Case) -> CheckResult:
        stage = _find_answer(self, trace, self.stage)
        if isinstance(stage, CheckResult):
            return stage
        if not isinstance(stage.output, str):
            return self.result(stage.name, SKIP, "no answer text")
        execute = trace.last_of("execute")
        if execute is None or execute.rows is None:
            return self.result(stage.name, SKIP, "no query result to ground the answer in")

        ungrounded = ungrounded_numbers(stage.output, trace.question, execute.rows)
        if ungrounded:
            return self.result(stage.name, FAIL, f"not in the result rows: {ungrounded}")
        return self.result(stage.name, PASS)


_VERDICT_FORMAT = (
    'Reply with only a JSON object: {"pass": true or false, "reason": "one short sentence"}.'
)

FAITHFUL_PROMPT = """\
You are checking a data assistant's answer against the query result it was written from.

QUESTION:
{question}

QUERY RESULT ROWS (JSON):
{rows}

ANSWER:
{answer}

Does every factual claim in the ANSWER follow from the QUERY RESULT ROWS? Pass when all
figures, names and statuses in the answer are supported by the rows, allowing rounding,
unit formatting and simple arithmetic over the rows. Fail when the answer states
anything the rows do not support, or contradicts them. If there are no rows, the answer
passes only when it says no matching data was found. If a note says the rows are cut
short, do not fail a claim merely because it may concern a part that is not shown.

"""

RELEVANT_PROMPT = """\
You are checking whether a data assistant's answer addresses the question it was asked.

QUESTION:
{question}

ANSWER:
{answer}

Does the ANSWER directly address what the QUESTION asks for? Pass when it gives what was
asked (or clearly says no matching data exists). Fail when it answers a different
question, leaves out part of what was asked, or avoids answering. Judge only whether it
addresses the question, not whether its figures are correct.

"""


class JudgeCheck(Check):
    """A check decided by an LLM judge. Subclass it and set `name` and `prompt`.

    `{question}`, `{answer}` and `{rows}` in `prompt` are replaced with the trace's
    values; other braces are left alone. The judge must reply with a JSON object holding
    a boolean `pass` and a `reason`; anything else raises, because a judge that cannot be
    read must not be mistaken for a pass or a fail.

    Set `needs_rows` when the check is about the query result: it is then skipped for
    answers that had no query (a greeting, a refusal).
    """

    prompt: str = ""
    needs_rows: bool = False

    def __init__(self, judge: Judge, stage: str | None = None, max_rows_chars: int = 20000) -> None:
        self.judge = judge
        self.stage = stage
        self.max_rows_chars = max_rows_chars

    def run(self, trace: Trace, case: Case) -> CheckResult:
        stage = _find_answer(self, trace, self.stage)
        if isinstance(stage, CheckResult):
            return stage
        if not isinstance(stage.output, str) or not stage.output.strip():
            return self.result(stage.name, SKIP, "no answer text")
        execute = trace.last_of("execute")
        rows = execute.rows if execute is not None else None
        if self.needs_rows and rows is None:
            return self.result(stage.name, SKIP, "no query result to judge the answer against")

        passed, reason = ask_judge(
            self.judge, self.prompt, trace.question, stage.output, rows or [], self.max_rows_chars
        )
        return self.result(stage.name, PASS if passed else FAIL, _short(reason))


def ask_judge(
    judge: Judge,
    template: str,
    question: str,
    answer: str,
    rows: list[dict[str, Any]],
    max_rows_chars: int = 20000,
) -> tuple[bool, str]:
    """Fill the template, call the judge, and return (passed, reason)."""
    # One pass over the template, so text inserted for one placeholder is never scanned
    # for another.
    values = {
        "{rows}": _rows_json(rows, max_rows_chars),
        "{question}": question,
        "{answer}": answer,
    }
    prompt = re.sub(r"\{(?:rows|question|answer)\}", lambda m: values[m.group()], template)
    reply = judge(prompt + _VERDICT_FORMAT)
    if not isinstance(reply, str):
        raise TypeError(
            "the judge must be a plain (not async) function returning the reply text, "
            f"got {type(reply).__name__}"
        )
    return _parse_verdict(reply)


class Faithful(JudgeCheck):
    """The judge confirms every claim in the answer is supported by the result rows."""

    name = "faithful"
    prompt = FAITHFUL_PROMPT
    needs_rows = True


class Relevant(JudgeCheck):
    """The judge confirms the answer addresses the question that was asked."""

    name = "relevant"
    prompt = RELEVANT_PROMPT


def _parse_verdict(reply: str) -> tuple[bool, str]:
    """The verdict from the last JSON object in the reply that has a boolean `pass`.
    Judges often wrap the object in prose or a code fence, and one that reasons first
    states its conclusion last."""
    decoder = json.JSONDecoder()
    verdict: tuple[bool, str] | None = None
    position = reply.find("{")
    while position != -1:
        try:
            data, end = decoder.raw_decode(reply, position)
        except (json.JSONDecodeError, RecursionError):
            position = reply.find("{", position + 1)
            continue
        if isinstance(data, dict) and isinstance(data.get("pass"), bool):
            verdict = (data["pass"], str(data.get("reason", "")))
        position = reply.find("{", end)  # past this object, so its insides are not re-read
    if verdict is None:
        raise ValueError(f"judge reply has no JSON object with a boolean 'pass': {_short(reply)!r}")
    return verdict
