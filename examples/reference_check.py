"""Compare an agent's result with a reference query run on the database.

`check_question` takes the agent function, the question and a SQL query known to be
correct. It runs the reference query through `execute` to get the correct result, asks
the agent the question, and compares the two.

Four small agents answer the same question here, to show each outcome:

  1. returns only its answer text, and the answer is right;
  2. returns only its answer text, and the number is wrong;
  3. writes SQL that forgets the store filter, so its rows differ from the reference;
  4. writes correct SQL.

An agent that only writes SQL is run through the same `run_sql` function, and its rows
go into the trace: the package runs the reference query, never the agent's own SQL.

Run it from the repository root:

    python -m examples.reference_check

That compares without an LLM. To also let an LLM judge the answer against the reference
result, pass your judge function (it takes a prompt and returns the reply text):

    python -m examples.reference_check my_module:my_judge
"""

import importlib
import sys

from dbagenteval import Scorecard, Trace, check_question
from examples.toy_agent import run_sql

QUESTION = "How many orders are delivered?"

# The correct query for the question, written once by a person.
REFERENCE_SQL = "SELECT COUNT(*) FROM orders WHERE store_id = 's1' AND status = 'DELIVERED'"


def answer_only(question: str) -> str:
    """An agent that exposes nothing but its final answer."""
    return "There are 3 delivered orders."


def answer_only_wrong(question: str) -> str:
    return "There are 4 delivered orders."


def sql_without_store_filter(question: str) -> Trace:
    """An agent that writes SQL. The SQL counts every store's orders, not only this one's."""
    sql = "SELECT COUNT(*) AS delivered FROM orders WHERE status = 'DELIVERED'"
    rows = run_sql(sql)
    answer = f"There are {rows[0]['delivered']} delivered orders."
    return Trace.simple(question, answer=answer, sql=sql, rows=rows)


def sql_correct(question: str) -> Trace:
    sql = "SELECT COUNT(*) AS delivered FROM orders WHERE store_id = 's1' AND status = 'DELIVERED'"
    rows = run_sql(sql)
    answer = f"There are {rows[0]['delivered']} delivered orders."
    return Trace.simple(question, answer=answer, sql=sql, rows=rows)


AGENTS = [answer_only, answer_only_wrong, sql_without_store_filter, sql_correct]


def main(judge=None) -> list[Scorecard]:
    """Check each agent against the reference query and print its scorecard."""
    cards = []
    for agent in AGENTS:
        card = check_question(agent, QUESTION, REFERENCE_SQL, execute=run_sql, judge=judge)
        print(f"--- {agent.__name__}")
        print(card.summary(), end="\n\n")
        cards.append(card)
    return cards


if __name__ == "__main__":
    judge = None
    if len(sys.argv) > 1:
        module_name, _, function_name = sys.argv[1].partition(":")
        judge = getattr(importlib.import_module(module_name), function_name)
    main(judge)
