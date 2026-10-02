from pathlib import Path

import pytest

from dbagenteval import Case, Trace, check_question, evaluate, load_cases, run_suite

REPO = Path(__file__).resolve().parent.parent
REFERENCE = "SELECT category, SUM(amount) AS total FROM orders GROUP BY category"
TOTALS = [{"category": "Electronics", "total": 20000}, {"category": "Clothing", "total": 8000}]
COUNT = [{"delivered": 3}]


def executor(rows):
    """An executor that returns fixed rows and remembers the SQL it was asked to run."""
    seen = []

    def execute(sql):
        seen.append(sql)
        return rows

    execute.seen = seen
    return execute


def checks(card):
    return {r.check: r.status for r in card.results}


def case(sql=REFERENCE, **fields):
    return Case(question="What is the total by category?", reference_sql=sql, **fields)


# ---- the agent gives its rows: compared with the reference rows, no LLM ----------------


def test_rows_equal_to_the_reference_rows_pass():
    execute = executor(TOTALS)
    agent_rows = [{"cat": "Clothing", "sum": 8000.0}, {"cat": "Electronics", "sum": 20000.0}]
    card = evaluate(Trace.simple("q", rows=agent_rows), case(), execute=execute)
    assert card.verdict == "PASS"
    assert checks(card)["reference_rows"] == "pass"
    assert execute.seen == [REFERENCE]


def test_rows_different_from_the_reference_fail_at_the_execute_stage():
    wrong = [{"cat": "Clothing", "sum": 8000}, {"cat": "Electronics", "sum": 99000}]
    card = evaluate(Trace.simple("q", rows=wrong), case(), execute=executor(TOTALS))
    assert (card.verdict, card.failed_stage, card.failed.check) == (
        "FAIL",
        "execute",
        "reference_rows",
    )
    assert "the reference query gives 2 row(s)" in card.failed.detail


def test_ordered_and_precision_apply_to_the_reference_comparison():
    reversed_rows = list(reversed(TOTALS))
    trace = Trace.simple("q", rows=reversed_rows)
    assert evaluate(trace, case(), execute=executor(TOTALS)).verdict == "PASS"
    assert evaluate(trace, case(ordered=True), execute=executor(TOTALS)).verdict == "FAIL"
    rate = Trace.simple("q", rows=[{"rate": 0.004}])
    assert evaluate(rate, case(), execute=executor([{"rate": 0.001}])).verdict == "PASS"
    assert evaluate(rate, case(precision=3), execute=executor([{"rate": 0.001}])).verdict == "FAIL"


def test_reference_comparison_is_skipped_when_the_query_stage_raised():
    trace = Trace.simple("q", error="relation does not exist")
    card = evaluate(trace, case(), execute=executor(TOTALS))
    assert checks(card) == {"no_error": "fail", "reference_rows": "skip"}


# ---- the agent gives only an answer: its numbers are compared, no LLM ------------------


@pytest.mark.parametrize(
    ("answer", "reference", "expected"),
    [
        ("There are 3 delivered orders.", COUNT, "pass"),
        ("There are 4 delivered orders.", COUNT, "fail"),  # a number not in the result
        ("Sorry, I cannot answer that.", COUNT, "fail"),  # the value is never stated
        ("Electronics leads with 20,000; Clothing has 8,000.", TOTALS, "pass"),
        ("Electronics leads with 20,000.", TOTALS, "pass"),  # a summary of some rows
        ("Electronics leads with 21,000.", TOTALS, "fail"),
        ("Here is the breakdown you asked for.", TOTALS, "fail"),  # states nothing
        ("The top category is Electronics.", [{"category": "Electronics"}], "pass"),
        ("The top category is Toys.", [{"category": "Electronics"}], "fail"),
        ("No matching orders were found.", [], "pass"),
        ("There are 12 matching orders.", [], "fail"),
    ],
)
def test_answer_text_against_the_reference_result(answer, reference, expected):
    card = evaluate(Trace.simple("q", answer=answer), case(), execute=executor(reference))
    assert checks(card)["reference_numbers"] == expected
    assert card.failed_stage == ("answer" if expected == "fail" else None)


def test_a_trace_with_nothing_to_compare_fails():
    card = evaluate(Trace("q", [{"kind": "route", "output": "x"}]), case(), execute=executor(COUNT))
    assert card.verdict == "FAIL"
    assert "neither result rows nor an answer" in card.failed.detail


# ---- with a judge ----------------------------------------------------------------------


def test_judge_compares_the_answer_with_the_reference_result():
    prompts = []

    def judge(prompt):
        prompts.append(prompt)
        agrees = "CORRECT RESULT ROWS" not in prompt
        return f'{{"pass": {str(agrees).lower()}, "reason": "the total is wrong"}}'

    trace = Trace.simple("q", answer="Electronics has 20,000.", rows=TOTALS)
    card = evaluate(trace, case(), judge=judge, execute=executor(TOTALS))
    assert checks(card) == {
        "no_error": "pass",
        "reference_rows": "pass",  # the rows are right...
        "reference_answer": "fail",  # ...and the judge still rules on the answer
        "faithful": "pass",
        "relevant": "pass",
    }
    reference_prompt = next(p for p in prompts if "CORRECT RESULT ROWS" in p)
    assert '"total": 8000' in reference_prompt and "Electronics has 20,000." in reference_prompt
    assert card.failed.detail == "the total is wrong"


def test_without_a_judge_no_llm_check_runs():
    trace = Trace.simple("q", answer="Electronics has 20,000.", rows=TOTALS)
    assert checks(evaluate(trace, case(), execute=executor(TOTALS))) == {
        "no_error": "pass",
        "reference_rows": "pass",
    }


# ---- setup problems are errors, not agent failures -------------------------------------


def test_reference_sql_without_an_executor_is_an_error():
    with pytest.raises(ValueError, match="no `execute` function"):
        evaluate(Trace.simple("q", rows=COUNT), case())
    result = run_suite([case()], lambda question: Trace.simple(question, rows=COUNT))
    assert result.cards[0].verdict == "ERROR"
    assert "no `execute` function" in result.cards[0].error


def test_a_reference_query_that_fails_is_an_error_naming_the_cause():
    def broken(sql):
        raise KeyError("no such table: orders")

    with pytest.raises(RuntimeError, match="reference SQL failed to run: KeyError"):
        evaluate(Trace.simple("q", rows=COUNT), case(), execute=broken)


def test_an_executor_returning_tuples_is_told_what_to_return():
    with pytest.raises(ValueError, match="must return a list of dicts"):
        evaluate(Trace.simple("q", rows=COUNT), case(), execute=lambda sql: [(3,)])


def test_a_case_without_reference_sql_never_calls_the_executor():
    execute = executor(COUNT)
    evaluate(Trace.simple("q", rows=COUNT), Case(question="q"), execute=execute)
    assert execute.seen == []


# ---- the one-question helper -----------------------------------------------------------


def test_check_question_with_an_agent_that_returns_only_its_answer():
    card = check_question(
        lambda question: "There are 3 delivered orders.",
        "How many orders are delivered?",
        "SELECT COUNT(*) FROM orders WHERE status = 'DELIVERED'",
        execute=executor(COUNT),
    )
    assert (card.verdict, checks(card)) == (
        "PASS",
        {"no_error": "pass", "reference_numbers": "pass"},
    )


def test_check_question_with_an_async_agent_that_returns_a_trace():
    async def agent(question):
        return Trace.simple(question, sql="SELECT 1", rows=[{"n": 4}], answer="There are 4.")

    card = check_question(
        agent, "How many?", "SELECT COUNT(*) FROM orders", execute=executor(COUNT)
    )
    assert (card.verdict, card.failed.check) == ("FAIL", "reference_rows")


def test_trace_simple_builds_only_the_stages_it_is_given():
    full = Trace.simple("q", answer="3", sql="SELECT 1", rows=[{"n": 3}])
    assert [s.kind for s in full.stages] == ["generate", "execute", "answer"]
    assert [s.kind for s in Trace.simple("q", answer="3").stages] == ["answer"]
    assert Trace.simple("q", error="boom").stages[0].error == "boom"


# ---- in a dataset ----------------------------------------------------------------------


def test_reference_sql_loads_from_yaml_and_is_validated(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text("- question: one\n  reference_sql: SELECT COUNT(*) FROM orders\n")
    assert load_cases(path)[0].reference_sql == "SELECT COUNT(*) FROM orders"
    for bad in ("", 5, ["SELECT 1"]):
        with pytest.raises(ValueError, match="reference_sql"):
            Case(question="q", reference_sql=bad)


def test_the_reference_example_runs_and_shows_each_outcome(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(REPO))
    from examples.reference_check import main

    cards = main()
    assert [card.verdict for card in cards] == ["PASS", "FAIL", "FAIL", "PASS"]
    assert [card.failed.check for card in cards if card.failed] == [
        "reference_numbers",
        "reference_rows",
    ]
    assert "--- sql_without_store_filter" in capsys.readouterr().out
