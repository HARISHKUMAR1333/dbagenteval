"""The case: one test question and the correct values a person wrote down for it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dbagenteval.compare import DEFAULT_PRECISION

Rows = list[dict[str, Any]]

_CASE_FIELDS = {"question", "expect", "reference_sql", "id", "tags", "ordered", "precision"}


@dataclass
class Case:
    """One test case.

    `expect` maps a stage name to the value that stage should produce. The special key
    `rows` holds the correct result rows: either one answer (a list of row dicts) or
    several accepted answers (a list of those lists). Give only what you know; stages
    without an expectation are checked by rules alone.

    `reference_sql` is a query known to be correct. It is run when the case is evaluated
    and the agent's result is compared with what it returns, so the case stays right as
    the data changes. Use it instead of `rows` on a live database.

    `ordered` says whether row order matters when comparing rows (a Top-N question).
    `precision` is how many decimal places numbers are rounded to before comparing;
    raise it for rates and other small fractions.
    """

    question: str
    expect: dict[str, Any] = field(default_factory=dict)
    reference_sql: str | None = None
    id: str = ""
    tags: list[str] = field(default_factory=list)
    ordered: bool = False
    precision: int = DEFAULT_PRECISION

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("'question' must be a non-empty string")
        if not isinstance(self.expect, dict):
            raise ValueError("'expect' must be a mapping of stage name to expected value")
        if not all(isinstance(k, str) for k in self.expect):
            raise ValueError("'expect' keys must be stage names (strings)")
        if self.reference_sql is not None and (
            not isinstance(self.reference_sql, str) or not self.reference_sql.strip()
        ):
            raise ValueError("'reference_sql' must be a non-empty SQL string")
        if isinstance(self.tags, str):
            self.tags = [self.tags]
        if not isinstance(self.tags, list) or not all(isinstance(t, str) for t in self.tags):
            raise ValueError("'tags' must be a list of strings")
        if not isinstance(self.ordered, bool):
            raise ValueError("'ordered' must be true or false")
        if isinstance(self.precision, bool) or not isinstance(self.precision, int):
            raise ValueError("'precision' must be a whole number of decimal places")
        if self.precision < 0:
            raise ValueError("'precision' cannot be negative")
        self.id = str(self.id) if self.id is not None else ""
        if "rows" in self.expect:
            _accepted_answers(self.expect["rows"])  # validates the shape

    @property
    def key(self) -> str:
        """What identifies this case across runs: its id, or the question."""
        return self.id or self.question

    @property
    def accepted_answers(self) -> list[Rows] | None:
        """Every acceptable result set, or None when the case says nothing about rows."""
        if "rows" not in self.expect:
            return None
        return _accepted_answers(self.expect["rows"])

    @classmethod
    def from_dict(cls, data: Any) -> Case:
        if not isinstance(data, dict):
            raise ValueError("a case must be a mapping")
        unknown = sorted(set(data) - _CASE_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown field(s) {unknown}; allowed: {', '.join(sorted(_CASE_FIELDS))}"
            )
        if "question" not in data:
            raise ValueError("missing 'question'")
        return cls(
            question=data["question"],
            expect={} if data.get("expect") is None else data["expect"],
            reference_sql=data.get("reference_sql"),
            id=data.get("id", ""),
            tags=[] if data.get("tags") is None else data["tags"],
            ordered=data.get("ordered", False),
            precision=data.get("precision", DEFAULT_PRECISION),
        )


def _accepted_answers(value: Any) -> list[Rows]:
    """Normalise `expect.rows` to a list of accepted answers.

    A list of row dicts is one answer; a list of lists is several. An empty list is one
    answer: "no rows".
    """
    if not isinstance(value, list):
        raise ValueError("'expect.rows' must be a list of rows, or a list of accepted answers")
    if all(isinstance(item, dict) for item in value):
        answers = [value]
    elif all(isinstance(item, list) for item in value):
        answers = value
    else:
        raise ValueError(
            "'expect.rows' mixes rows and lists; give either one list of rows or a list "
            "of accepted answers"
        )
    for answer in answers:
        for row in answer:
            if not isinstance(row, dict) or not row:
                raise ValueError("each row in 'expect.rows' must be a non-empty mapping")
            for column in row:
                if not isinstance(column, str):
                    raise ValueError(
                        f"column name {column!r} in 'expect.rows' must be text; "
                        f'write it in quotes, as "{column}"'
                    )
    return answers
