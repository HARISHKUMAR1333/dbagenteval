"""Run an agent over a set of cases and collect the scorecards."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from dbagenteval.answer import Judge
from dbagenteval.case import Case
from dbagenteval.checks import Check
from dbagenteval.evaluate import evaluate
from dbagenteval.reference import Executor
from dbagenteval.scorecard import Scorecard
from dbagenteval.trace import Trace

# An agent takes the question and returns its trace (or a dict shaped like one), or just
# its final answer as text. It may be a plain function or an async one.
AgentResult = Trace | dict[str, Any] | str
Agent = Callable[[str], "AgentResult | Awaitable[AgentResult]"]


@dataclass
class SuiteResult:
    """The scorecards from one run, in case order."""

    cards: list[Scorecard] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.cards) and all(c.verdict == "PASS" for c in self.cards)

    def summary(self) -> dict[str, Any]:
        """Counts overall, per tag, and per first failing stage."""
        verdicts = [c.verdict for c in self.cards]
        total = len(verdicts)
        by_tag: dict[str, dict[str, int]] = {}
        failed_stages: dict[str, int] = {}
        for card in self.cards:
            for tag in card.tags:
                counts = by_tag.setdefault(tag, {"total": 0, "passed": 0})
                counts["total"] += 1
                counts["passed"] += card.verdict == "PASS"
            if card.failed_stage:
                failed_stages[card.failed_stage] = failed_stages.get(card.failed_stage, 0) + 1
        return {
            "total": total,
            "passed": verdicts.count("PASS"),
            "failed": verdicts.count("FAIL"),
            "errors": verdicts.count("ERROR"),
            "skipped": verdicts.count("SKIP"),
            "pass_rate": verdicts.count("PASS") / total if total else 0.0,
            "by_tag": by_tag,
            "failed_stages": failed_stages,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary(), "cases": [c.to_dict() for c in self.cards]}


async def _call_agent(agent: Agent, question: str) -> Trace:
    if inspect.iscoroutinefunction(agent):
        result = await agent(question)
    else:
        # A blocking agent runs in a thread so it cannot stall the other cases.
        result = await asyncio.to_thread(agent, question)
        if inspect.isawaitable(result):
            result = await result
    if isinstance(result, Trace):
        return result
    if isinstance(result, dict):
        return Trace.from_dict(result)
    if isinstance(result, str):
        return Trace.simple(question, answer=result)
    raise TypeError(
        f"the agent must return a Trace, a dict or the answer text, got {type(result).__name__}"
    )


async def arun_suite(
    cases: Iterable[Case],
    agent: Agent,
    *,
    rules: Iterable[Check] = (),
    judge: Judge | None = None,
    execute: Executor | None = None,
    concurrency: int = 4,
) -> SuiteResult:
    """Run every case through the agent and score it.

    `execute` runs a case's `reference_sql`; it is required when any case has one. A
    case whose agent call or evaluation raises gets an ERROR verdict carrying the
    message; the rest of the suite still runs. At most `concurrency` cases run at once.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    rules = list(rules)
    limit = asyncio.Semaphore(concurrency)

    async def run_one(case: Case) -> Scorecard:
        async with limit:
            step = "agent"
            try:
                trace = await _call_agent(agent, case.question)
                step = "evaluation"
                # Rules and judges are blocking calls; keep them off the event loop.
                return await asyncio.to_thread(evaluate, trace, case, rules, judge, execute)
            except Exception as e:  # noqa: BLE001 — one broken case must not stop the run
                return Scorecard(
                    question=case.question,
                    case_id=case.id,
                    tags=list(case.tags),
                    error=f"{step} raised {type(e).__name__}: {e}",
                )

    return SuiteResult(list(await asyncio.gather(*(run_one(c) for c in cases))))


def run_suite(
    cases: Iterable[Case],
    agent: Agent,
    *,
    rules: Iterable[Check] = (),
    judge: Judge | None = None,
    execute: Executor | None = None,
    concurrency: int = 4,
) -> SuiteResult:
    """Blocking form of `arun_suite`, for scripts and the CLI."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "run_suite() cannot be called while an event loop is running (in a notebook "
            "or async code); use `await arun_suite(...)` instead"
        )
    return asyncio.run(
        arun_suite(cases, agent, rules=rules, judge=judge, execute=execute, concurrency=concurrency)
    )


def check_question(
    agent: Agent,
    question: str,
    reference_sql: str,
    *,
    execute: Executor,
    rules: Iterable[Check] = (),
    judge: Judge | None = None,
) -> Scorecard:
    """Ask the agent one question and compare what it returns with a reference query.

    `agent` takes the question and returns its final answer as text, or a Trace when you
    also have its SQL and rows. `execute` runs SQL on your database and returns rows as
    dicts; it runs `reference_sql` to get the correct result.

    Without a judge the comparison uses no LLM: the agent's rows must be the reference
    rows, or, when the agent gives only an answer, the answer must state the reference
    values and no others. With a judge, the judge also decides whether the answer agrees
    with the reference result.
    """
    case = Case(question=question, reference_sql=reference_sql)
    result = run_suite([case], agent, rules=rules, judge=judge, execute=execute, concurrency=1)
    return result.cards[0]
