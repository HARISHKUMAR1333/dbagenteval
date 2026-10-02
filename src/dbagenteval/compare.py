"""Decide whether two result sets are the same answer.

Checking only that a query ran is not enough: a query can run without error and still
return the wrong figure. The rows have to be compared, and three differences between a
hand-written expected result and a real one must not count as a mismatch:

  * Column names. The expected result may call a column `count` where the query calls it
    `total_order_count`. A column is compared with the result column of the same name
    when one exists, and otherwise with any column holding the same values.
  * How a value is represented. A count may arrive as the integer 3, the float 3.0, a
    Decimal, or the text "3.00", depending on the database driver. A missing value may be
    null or an empty string, and a flag may be a boolean or 0/1.
  * Extra columns. The expected result lists only the columns that matter; the real
    result may hold more.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

Rows = list[dict[str, Any]]

DEFAULT_PRECISION = 2

# Pairings tried before giving up on matching unnamed columns by value. Only a result
# with many look-alike columns in the wrong arrangement gets near it, and that result is
# reported as not matching, which it almost certainly is.
_SEARCH_BUDGET = 20_000

# Text is read as a number only when it is written as a plain one. "01234" (a postcode),
# "1E5" and "1_000" stay text, so they never equal the numbers they resemble.
_PLAIN_NUMBER = re.compile(r"-?(0|[1-9]\d*)(\.\d+)?")


def _as_number(value: Any) -> Decimal | None:
    """The value as a finite number, or None when it is not one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if not _PLAIN_NUMBER.fullmatch(value.strip()):
            return None
    elif not isinstance(value, int | float | Decimal):
        return None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):  # ValueError: an int too long to print
        return None
    return number if number.is_finite() else None


def canonical(value: Any, precision: int = DEFAULT_PRECISION) -> str:
    """The text two cells are compared by.

    Values that mean the same thing get the same text: a number in any type, rounded
    half-up to `precision` decimal places with trailing zeros dropped; a missing value,
    whether null or blank; a flag, whether a boolean or 0/1. Anything else is its own
    text with outer spaces removed.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    number = _as_number(value)
    if number is not None:
        try:
            rounded = number.quantize(Decimal(10) ** -precision, rounding=ROUND_HALF_UP)
        except InvalidOperation:  # more digits than Decimal arithmetic carries
            return str(value).strip()
        if rounded == 0:
            return "0"  # never "-0" or "0.00"
        text = f"{rounded:f}"
        return text.rstrip("0").rstrip(".") if "." in text else text
    try:
        return str(value).strip()
    except ValueError:  # an int too long for Python to print
        return f"{value:#x}"


def _column_major(rows: Rows, precision: int) -> dict[str, list[str]]:
    columns: list[str] = []
    for row in rows:
        columns.extend(c for c in row if c not in columns)
    return {c: [canonical(row.get(c), precision) for row in rows] for c in columns}


def rows_match(
    expected: Rows, actual: Rows, ordered: bool = False, precision: int = DEFAULT_PRECISION
) -> bool:
    """True when every expected column is found in `actual` with the same values in the
    same rows.

    An expected column whose name also exists in `actual` (ignoring case) is compared
    with that column. Any other expected column may match any remaining actual column
    with the same values, so `{count: 3}` matches a result column named `delivered`.
    Extra actual columns are allowed. Row order matters only when `ordered` is set.
    """
    if len(expected) != len(actual):
        return False
    if not expected:
        return True

    exp_cols = _column_major(expected, precision)
    act_cols = _column_major(actual, precision)
    by_name = {name.casefold(): name for name in reversed(act_cols)}

    def signature(values: list[str]) -> tuple[str, ...]:
        return tuple(values) if ordered else tuple(sorted(values))

    # Which actual columns could stand in for each expected column.
    named = {by_name[n.casefold()] for n in exp_cols if n.casefold() in by_name}
    candidates: dict[str, list[str]] = {}
    for name, values in exp_cols.items():
        if name.casefold() in by_name:
            pool = [by_name[name.casefold()]]
        else:
            pool = [a for a in act_cols if a not in named]
        candidates[name] = [a for a in pool if signature(act_cols[a]) == signature(values)]
    if not all(candidates.values()):
        return False

    # Most constrained first, so a wrong pairing is found before the search branches.
    names = sorted(exp_cols, key=lambda n: len(candidates[n]))

    def row_tuples(columns: list[list[str]]) -> list[tuple[str, ...]]:
        tuples = list(zip(*columns, strict=True))
        return tuples if ordered else sorted(tuples)

    # Columns can match one by one and still pair up wrong across rows (A=1,B=2 vs
    # A=2,B=1), so the chosen columns are compared together, row by row. Checking after
    # every choice, not only at the end, keeps look-alike columns from blowing up.
    targets = [row_tuples([exp_cols[n] for n in names[: i + 1]]) for i in range(len(names))]

    budget = _SEARCH_BUDGET

    def assign(i: int, chosen: list[str]) -> bool:
        nonlocal budget
        if i == len(names):
            return True
        tried: set[tuple[str, ...]] = set()
        for candidate in candidates[names[i]]:
            values = tuple(act_cols[candidate])
            if candidate in chosen or values in tried:
                continue
            budget -= 1
            if budget < 0:
                return False
            tried.add(values)  # an identical column would fail the same way
            picked = [*chosen, candidate]
            if row_tuples([act_cols[c] for c in picked]) == targets[i] and assign(i + 1, picked):
                return True
        return False

    return assign(0, [])


def match_any(
    accepted: list[Rows], actual: Rows, ordered: bool = False, precision: int = DEFAULT_PRECISION
) -> int | None:
    """Index of the first accepted answer that `actual` matches, or None."""
    for i, answer in enumerate(accepted):
        if rows_match(answer, actual, ordered, precision):
            return i
    return None
