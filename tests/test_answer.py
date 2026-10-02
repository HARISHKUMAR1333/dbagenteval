from decimal import Decimal

import pytest

from dbagenteval import Case, Faithful, JudgeCheck, NumbersGrounded, Relevant, Trace
from dbagenteval.answer import _parse_verdict, _rows_json

CASE = Case(question="q")


def grounded(answer, rows, question="How are orders doing?"):
    trace = Trace(
        question,
        [{"kind": "execute", "rows": rows}, {"kind": "answer", "output": answer}],
    )
    return NumbersGrounded().run(trace, CASE)


def test_numbers_from_the_rows_pass():
    rows = [{"category": "Electronics", "total": 123456.0, "return_rate": 0.0457}]
    answer = "Electronics sold $123,456 with a return rate of 4.57%."
    assert grounded(answer, rows).status == "pass"
    # Thousands separators in any grouping, such as the Indian 1,23,456.
    assert grounded("Electronics sold 1,23,456.", rows).status == "pass"


def test_invented_number_fails_and_is_named():
    result = grounded("There are 17 pending orders.", [{"count": 15}])
    assert result.status == "fail"
    assert "17" in result.detail


def test_rounded_values_pass_but_wrong_rounding_fails():
    rows = [{"avg_hours": 12.4567}]
    assert grounded("The average is 12.5 hours.", rows).status == "pass"
    assert grounded("The average is 12.9 hours.", rows).status == "fail"


def test_question_numbers_row_count_and_list_markers_are_allowed():
    rows = [{"p": "A", "n": 40}, {"p": "B", "n": 30}]
    answer = "Here are the top 5 categories; 2 were found:\n1. A — 40\n2. B — 30"
    assert grounded(answer, rows, question="Show the top 5 categories").status == "pass"


def test_numbers_inside_text_cells_count():
    rows = [{"placed_on": "2026-09-01", "order_no": "260029370955"}]
    assert grounded("Order 260029370955 was placed on 1 September 2026.", rows).status == "pass"


def test_no_rows_means_any_figure_is_invented():
    assert grounded("No matching records were found.", []).status == "pass"
    assert grounded("There are 12 matching records.", []).status == "fail"


def test_skipped_without_an_answer_or_without_a_query():
    trace = Trace("q", [{"kind": "execute", "rows": []}])
    assert NumbersGrounded().run(trace, CASE).status == "skip"
    # A greeting has no query result, so its numbers cannot be judged against one.
    greeting = Trace("hi", [{"kind": "answer", "output": "I'm available 24/7 to help."}])
    assert NumbersGrounded().run(greeting, CASE).status == "skip"


@pytest.mark.parametrize(
    "answer",
    [
        "17. That is the number of pending orders.",
        "17) pending orders",
        "The total is Rs.99999.",
        "The total is USD99999.",
        "Order No.99999 is pending",
        "Sales grew x99999",
        "The rate is .75",
    ],
)
def test_numbers_glued_to_words_or_punctuation_are_still_checked(answer):
    assert grounded(answer, [{"count": 15}]).status == "fail"


def test_number_formats_that_are_grounded():
    assert grounded("The rate is .75", [{"rate": 0.75}]).status == "pass"
    assert grounded("About 45 percent were returned.", [{"rate": 0.45}]).status == "pass"
    assert grounded("That is 45 % of orders.", [{"rate": 0.45}]).status == "pass"
    assert grounded("Total: Rs.15 only", [{"count": 15}]).status == "pass"
    # "3, 4" is two numbers, not 34 or a thousands group.
    assert grounded("Stores 3, 4 are open.", [{"a": 3, "b": 4}]).status == "pass"


def answer_trace(rows):
    return Trace("q", [{"kind": "execute", "rows": rows}, {"kind": "answer", "output": "15."}])


def test_judge_verdict_is_read_from_json_inside_the_reply():
    reply = 'Sure.\n```json\n{"pass": false, "reason": "17 is not in the rows"}\n```'
    result = Faithful(lambda prompt: reply).run(answer_trace([{"n": 15}]), CASE)
    assert result.status == "fail"
    assert result.detail == "17 is not in the rows"


@pytest.mark.parametrize("reply", ["yes it is fine", '{"pass": "yes"}', "{not json}"])
def test_unreadable_judge_reply_raises(reply):
    with pytest.raises(ValueError):
        Faithful(lambda prompt: reply).run(answer_trace([]), CASE)


def test_judge_is_told_when_rows_do_not_fit():
    prompts = []

    def judge(prompt):
        prompts.append(prompt)
        return '{"pass": true, "reason": ""}'

    rows = [{"n": i} for i in range(100)]
    Faithful(judge, max_rows_chars=60).run(answer_trace(rows), CASE)
    assert "of 100 rows" in prompts[0]
    assert '{"n": 0}' in prompts[0]


def test_judge_reply_may_carry_other_braces_before_the_verdict():
    reply = 'Checking {rows}... {"note": 1} then {"pass": true, "reason": "ok {fine}"}'
    result = Faithful(lambda prompt: reply).run(answer_trace([{"n": 15}]), CASE)
    assert (result.status, result.detail) == ("pass", "ok {fine}")


def test_async_judge_is_rejected_with_a_clear_message():
    async def judge(prompt):
        return '{"pass": true}'

    def call(prompt):
        coroutine = judge(prompt)
        coroutine.close()  # never awaited on purpose
        return coroutine

    with pytest.raises(TypeError, match="plain .* function"):
        Faithful(call).run(answer_trace([]), CASE)


def test_faithful_is_skipped_without_a_query_but_relevant_still_runs():
    greeting = Trace("hi", [{"kind": "answer", "output": "Hello!"}])
    judge = lambda prompt: '{"pass": true, "reason": "fine"}'  # noqa: E731
    assert Faithful(judge).run(greeting, CASE).status == "skip"
    assert Relevant(judge).run(greeting, CASE).status == "pass"


def test_custom_judge_prompt_may_contain_json_braces():
    class Polite(JudgeCheck):
        name = "polite"
        prompt = 'Is this polite? {answer}\nExample reply: {"pass": true}\n'

    seen = []

    def judge(prompt):
        seen.append(prompt)
        return '{"pass": true, "reason": "yes"}'

    assert Polite(judge).run(answer_trace([]), CASE).status == "pass"
    assert "Is this polite? 15." in seen[0]


@pytest.mark.parametrize(
    ("answer", "rows", "expected"),
    [
        ("There are 15k orders.", [{"n": 15}], "fail"),
        ("Revenue was 15 million.", [{"n": 15}], "fail"),
        ("$1.5M in sales", [{"n": 1.5}], "fail"),
        ("About 1.2k orders", [{"n": 1200}], "pass"),
        ("Revenue was 1.5 million", [{"n": 1500000}], "pass"),
        ("2 lakh rupees", [{"n": 200000}], "pass"),
    ],
)
def test_a_unit_after_a_number_scales_it(answer, rows, expected):
    assert grounded(answer, rows).status == expected


def test_a_mistyped_answer_stage_fails_instead_of_never_running():
    trace = answer_trace([{"n": 15}])
    assert NumbersGrounded(stage="answr").run(trace, CASE).status == "fail"
    judge = lambda prompt: '{"pass": true, "reason": ""}'  # noqa: E731
    assert Relevant(judge, stage="answr").run(trace, CASE).status == "fail"
    no_answer = Trace("q", [{"kind": "route", "output": "x"}])
    assert NumbersGrounded(stage="reply").run(no_answer, CASE).status == "skip"


def test_the_final_verdict_wins_and_a_nested_object_is_not_a_verdict():
    reasoned = 'First: {"pass": true, "reason": "looks ok"}. Final: {"pass": false, "reason": "no"}'
    assert _parse_verdict(reasoned) == (False, "no")
    nested = '{"pass": false, "reason": "x", "extra": {"pass": true}}'
    assert _parse_verdict(nested) == (False, "x")


def test_one_row_longer_than_the_prompt_budget_is_cut_not_dropped():
    text = _rows_json([{"a": "x" * 30000}], 200)
    assert text.startswith('[{"a": "xxxx') and "cut short" in text


def test_extreme_cell_values_do_not_crash_the_number_check():
    assert grounded("There are 3.", [{"n": 10**5000, "m": 3}]).status == "pass"
    assert grounded("That is 5%.", [{"n": Decimal("9E+999999"), "m": 0.05}]).status == "pass"
