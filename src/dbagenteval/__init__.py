"""dbagenteval — stage-by-stage evaluation for natural-language-to-SQL agents.

Example:

    from dbagenteval import Trace, Case, evaluate
    from dbagenteval.rules import ReadOnly, TenantFilter

    card = evaluate(trace, case, rules=[ReadOnly(), TenantFilter("tenant_id")])
    card.verdict        # "PASS" | "FAIL"
    card.failed_stage   # the first stage that failed, or None
"""

__version__ = "0.1.0"

from dbagenteval.answer import Faithful, Judge, JudgeCheck, NumbersGrounded, Relevant
from dbagenteval.case import Case
from dbagenteval.checks import Check
from dbagenteval.dataset import DatasetError, filter_cases, load_cases
from dbagenteval.evaluate import evaluate
from dbagenteval.reference import Executor
from dbagenteval.runner import SuiteResult, arun_suite, check_question, run_suite
from dbagenteval.scorecard import CheckResult, Scorecard
from dbagenteval.trace import Stage, Trace

__all__ = [
    "Case",
    "Check",
    "CheckResult",
    "DatasetError",
    "Executor",
    "Faithful",
    "Judge",
    "JudgeCheck",
    "NumbersGrounded",
    "Relevant",
    "Scorecard",
    "Stage",
    "SuiteResult",
    "Trace",
    "arun_suite",
    "check_question",
    "evaluate",
    "filter_cases",
    "load_cases",
    "run_suite",
]
