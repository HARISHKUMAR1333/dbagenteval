"""The trace: a record of what an agent did for one question, stage by stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

STAGE_KINDS = ("route", "plan", "retrieve", "generate", "execute", "answer")

# `expect.rows` in a case means "the result rows", so no stage may take that name.
RESERVED_STAGE_NAMES = ("rows",)


@dataclass
class Stage:
    """One step of the agent's pipeline.

    `kind` says what sort of step it is; `name` tells two steps of the same kind apart
    (a classifier and a view picker are both `route`) and defaults to the kind.
    """

    kind: str
    name: str = ""
    output: Any = None
    rows: list[dict[str, Any]] | None = None
    error: str | None = None
    latency_ms: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in STAGE_KINDS:
            raise ValueError(
                f"unknown stage kind {self.kind!r}; expected one of {', '.join(STAGE_KINDS)}"
            )
        self.name = self.name or self.kind
        if self.name in RESERVED_STAGE_NAMES:
            raise ValueError(f"stage name {self.name!r} is reserved")
        if self.rows is not None and not (
            isinstance(self.rows, list)
            and all(isinstance(r, dict) and all(isinstance(k, str) for k in r) for r in self.rows)
        ):
            raise ValueError(
                f"stage {self.name!r}: rows must be a list of dicts (column name -> value); "
                "convert cursor tuples with dict(zip(columns, row))"
            )


@dataclass
class Trace:
    """Everything the agent did for one question, in the order it happened."""

    question: str
    stages: list[Stage] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stages = [_as_stage(s) for s in self.stages]
        names = [s.name for s in self.stages]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate stage name(s) {duplicates}; give stages of the same kind distinct names"
            )

    def stage(self, name: str) -> Stage | None:
        """The stage with this name, or None."""
        return next((s for s in self.stages if s.name == name), None)

    def last_of(self, kind: str) -> Stage | None:
        """The last stage of this kind, or None."""
        return next((s for s in reversed(self.stages) if s.kind == kind), None)

    @classmethod
    def simple(
        cls,
        question: str,
        answer: str | None = None,
        sql: str | None = None,
        rows: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> Trace:
        """A trace for an agent that exposes only its end results: the SQL it wrote, the
        rows it got back (or the error), and its final answer. Leave out what the agent
        does not give you."""
        stages = []
        if sql is not None:
            stages.append(Stage("generate", output=sql))
        if rows is not None or error is not None:
            stages.append(Stage("execute", rows=rows, error=error))
        if answer is not None:
            stages.append(Stage("answer", output=answer))
        return cls(question, stages)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        unknown = sorted(set(data) - {"question", "stages", "meta"})
        if unknown:
            raise ValueError(f"unknown trace field(s) {unknown}")
        if "question" not in data:
            raise ValueError("trace is missing 'question'")
        return cls(
            question=data["question"],
            stages=list(data.get("stages") or []),
            meta=dict(data.get("meta") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_stage(value: Any) -> Stage:
    if isinstance(value, Stage):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"a stage must be a Stage or a dict, got {type(value).__name__}")
    try:
        return Stage(**value)
    except TypeError as e:
        raise ValueError(f"bad stage {value!r}: {e}") from e
