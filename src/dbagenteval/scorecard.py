"""The scorecard: what came out of checking one trace against one case."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PASS = "pass"
FAIL = "fail"
SKIP = "skip"


@dataclass
class CheckResult:
    """The outcome of one check, attributed to the stage it looked at."""

    check: str
    stage: str
    status: str  # PASS | FAIL | SKIP
    detail: str = ""
    score: float | None = None

    def __post_init__(self) -> None:
        # A status the verdict does not recognise would be ignored, turning a failed
        # check into a pass.
        if self.status not in (PASS, FAIL, SKIP):
            raise ValueError(
                f"check {self.check!r} returned status {self.status!r}; "
                f"use the PASS, FAIL or SKIP constants from dbagenteval.scorecard"
            )


@dataclass
class Scorecard:
    """Every check result for one question, plus the verdict.

    `stage_order` is the order the stages ran in, so the first failing stage can be
    named: a wrong answer usually has one root cause, and it is the earliest failure.
    """

    question: str
    results: list[CheckResult] = field(default_factory=list)
    stage_order: list[str] = field(default_factory=list)
    case_id: str = ""
    tags: list[str] = field(default_factory=list)
    error: str | None = None  # the agent or the evaluation raised; nothing was scored

    @property
    def verdict(self) -> str:
        """ERROR if the run raised, FAIL if any check failed, PASS if at least one check
        passed and none failed, SKIP if nothing could be checked."""
        if self.error:
            return "ERROR"
        statuses = {r.status for r in self.results}
        if FAIL in statuses:
            return "FAIL"
        return "PASS" if PASS in statuses else "SKIP"

    @property
    def failed(self) -> CheckResult | None:
        """The failed check on the earliest stage, or None."""
        failures = [r for r in self.results if r.status == FAIL]
        if not failures:
            return None

        def position(result: CheckResult) -> int:
            # A stage the trace never reached sorts after the ones it did.
            if result.stage in self.stage_order:
                return self.stage_order.index(result.stage)
            return len(self.stage_order)

        return min(failures, key=position)

    @property
    def failed_stage(self) -> str | None:
        failed = self.failed
        return failed.stage if failed else None

    def stages(self) -> dict[str, str]:
        """Each stage's overall status, in run order."""
        names = list(self.stage_order)
        names.extend(r.stage for r in self.results if r.stage not in names)
        out: dict[str, str] = {}
        for name in names:
            statuses = {r.status for r in self.results if r.stage == name}
            out[name] = FAIL if FAIL in statuses else PASS if PASS in statuses else SKIP
        return out

    def to_dict(self) -> dict[str, Any]:
        failed = self.failed
        return {
            "id": self.case_id,
            "question": self.question,
            "tags": list(self.tags),
            "verdict": self.verdict,
            "failed_stage": failed.stage if failed else None,
            "failed_check": failed.check if failed else None,
            "error": self.error,
            "results": [
                {
                    "check": r.check,
                    "stage": r.stage,
                    "status": r.status,
                    "detail": r.detail,
                    "score": r.score,
                }
                for r in self.results
            ],
        }

    def summary(self) -> str:
        """A plain-text card, one line per check, in the order the stages ran."""
        lines = [f"Q: {self.question}"]
        if self.error:
            lines.append(f"  error: {self.error}")
        for stage in self.stages():
            for r in (r for r in self.results if r.stage == stage):
                line = f"  {r.stage:<12} {r.check:<18} {r.status.upper():<5} {r.detail}"
                lines.append(line.rstrip())
        verdict = self.verdict
        failed = self.failed
        if failed:
            verdict += f"  (first failing stage: {failed.stage})"
        lines.append(f"  VERDICT      {verdict}")
        return "\n".join(lines)
