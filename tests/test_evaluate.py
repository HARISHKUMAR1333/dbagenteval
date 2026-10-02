import pytest

from dbagenteval import Case, Check, CheckResult, Scorecard, Stage, Trace, evaluate
from dbagenteval.rules import TenantFilter


def make_trace(**overrides):
    stages = {
        "classify": {"kind": "route", "name": "classify", "output": "data_question"},
        "view": {"kind": "route", "name": "view", "output": "orders_view"},
        "retrieve": {"kind": "retrieve", "output": ["pending_count", "customer_name"]},
        "generate": {
            "kind": "generate",
            "output": "SELECT pending_count FROM orders_view WHERE store_id = 's1'",
        },
        "execute": {"kind": "execute", "rows": [{"pending_count": 15}]},
        "answer": {"kind": "answer", "output": "There are 15 pending orders."},
    }
    for name, changes in overrides.items():
        if changes is None:
            del stages[name]
        else:
            stages[name] = {**stages[name], **changes}
    return Trace("How many orders are pending?", list(stages.values()))


CASE = Case(
    question="How many orders are pending?",
    expect={
        "classify": "data_question",
        "view": "orders_view",
        "retrieve": ["pending_count"],
        "rows": [{"count": 15}],
    },
)


def test_everything_right_passes():
    card = evaluate(make_trace(), CASE, rules=[TenantFilter("store_id")])
    assert card.verdict == "PASS"
    assert card.failed_stage is None
    assert set(card.stages().values()) == {"pass"}


def test_first_failing_stage_is_the_earliest_one():
    trace = make_trace(view={"output": "wrong_view"}, execute={"rows": [{"pending_count": 9}]})
    card = evaluate(trace, CASE)
    assert card.verdict == "FAIL"
    assert card.failed_stage == "view"
    assert card.stages()["execute"] == "fail"


def test_right_answer_with_unsafe_sql_still_fails():
    trace = make_trace(generate={"output": "SELECT pending_count FROM orders_view"})
    card = evaluate(trace, CASE, rules=[TenantFilter("store_id")])
    assert card.verdict == "FAIL"
    assert card.failed_stage == "generate"
    assert card.failed.check == "tenant_filter"


def test_expected_stage_missing_from_trace_fails():
    card = evaluate(make_trace(view=None), CASE)
    assert card.failed_stage == "view"
    assert "missing" in card.failed.detail


def test_retrieval_reports_missing_items_and_recall():
    case = Case(question="q", expect={"retrieve": ["pending_count", "coupon_code"]})
    card = evaluate(make_trace(), case)
    result = next(r for r in card.results if r.check == "recall")
    assert result.status == "fail"
    assert result.score == 0.5
    assert "coupon_code" in result.detail


def test_expected_list_means_any_of_and_text_ignores_case():
    case = Case(question="q", expect={"view": ["sales_view", "ORDERS_VIEW"]})
    assert evaluate(make_trace(), case).verdict == "PASS"


def test_stage_error_fails_that_stage_and_skips_row_comparison():
    trace = make_trace(execute={"rows": None, "error": "relation does not exist"}, answer=None)
    card = evaluate(trace, CASE)
    assert card.failed_stage == "execute"
    assert card.failed.check == "no_error"
    rows = next(r for r in card.results if r.check == "rows_match")
    assert rows.status == "skip"


def test_without_a_case_only_stage_errors_and_rules_run():
    card = evaluate(make_trace())
    assert card.verdict == "PASS"
    assert {r.check for r in card.results} == {"no_error"}


def test_judge_adds_faithful_and_relevant_checks():
    prompts = []

    def judge(prompt):
        prompts.append(prompt)
        return '{"pass": true, "reason": "supported"}'

    card = evaluate(make_trace(), CASE, judge=judge)
    assert {"faithful", "relevant"} <= {r.check for r in card.results}
    assert len(prompts) == 2
    assert '"pending_count": 15' in prompts[0]


def test_trace_rejects_bad_stages():
    with pytest.raises(ValueError, match="unknown stage kind"):
        Stage(kind="think")
    with pytest.raises(ValueError, match="duplicate stage name"):
        Trace("q", [{"kind": "route"}, {"kind": "route"}])
    with pytest.raises(ValueError, match="bad stage"):
        Trace("q", [{"kind": "route", "result": 1}])


def test_case_rejects_bad_rows():
    with pytest.raises(ValueError, match="mixes rows and lists"):
        Case(question="q", expect={"rows": [{"n": 1}, [{"n": 2}]]})
    assert Case(question="q", expect={"rows": [[{"n": 1}], [{"n": 2}]]}).accepted_answers == [
        [{"n": 1}],
        [{"n": 2}],
    ]


def test_rows_must_be_dicts_and_the_error_says_how_to_fix_it():
    with pytest.raises(ValueError, match="list of dicts"):
        Trace("q", [{"kind": "execute", "rows": [(15,)]}])
    with pytest.raises(ValueError, match="list of dicts"):
        Stage(kind="execute", rows={"n": 1})
    with pytest.raises(ValueError, match="list of dicts"):
        Stage(kind="execute", rows=[{1: "a"}])
    # YAML reads `2023:` and `yes:` as a number and a boolean, not as column names.
    for rows in ([{2023: 100}], [{True: 3}]):
        with pytest.raises(ValueError, match="must be text"):
            Case(question="q", expect={"rows": rows})


def test_reserved_stage_name_and_trace_from_dict_validation():
    with pytest.raises(ValueError, match="reserved"):
        Stage(kind="execute", name="rows")
    with pytest.raises(ValueError, match="unknown trace field"):
        Trace.from_dict({"question": "q", "steps": []})
    with pytest.raises(ValueError, match="missing 'question'"):
        Trace.from_dict({"stages": []})


def test_case_validates_its_fields():
    for bad in ({"ordered": "yes"}, {"tags": [1]}, {"expect": ["x"]}, {"precision": 1.5}):
        with pytest.raises(ValueError):
            Case(question="q", **bad)
    with pytest.raises(ValueError, match="unknown field"):
        Case.from_dict({"question": "q", "expects": {}})


def test_ordered_and_precision_from_the_case_reach_the_row_comparison():
    trace = Trace(
        "q", [{"kind": "execute", "rows": [{"p": "B", "r": 0.004}, {"p": "A", "r": 0.2}]}]
    )
    rows = [{"p": "A", "r": 0.2}, {"p": "B", "r": 0.004}]
    assert evaluate(trace, Case(question="q", expect={"rows": rows})).verdict == "PASS"
    assert (
        evaluate(trace, Case(question="q", expect={"rows": rows}, ordered=True)).verdict == "FAIL"
    )
    wrong = [{"p": "A", "r": 0.2}, {"p": "B", "r": 0.001}]
    assert evaluate(trace, Case(question="q", expect={"rows": wrong})).verdict == "PASS"
    assert (
        evaluate(trace, Case(question="q", expect={"rows": wrong}, precision=3)).verdict == "FAIL"
    )


def test_row_expectation_without_usable_rows_fails_with_the_reason():
    case = Case(question="q", expect={"rows": [{"n": 1}]})
    no_execute = evaluate(Trace("q", [{"kind": "answer", "output": "1"}]), case)
    assert (no_execute.failed.stage, no_execute.failed.detail) == (
        "execute",
        "no execute stage in trace",
    )
    no_rows = evaluate(Trace("q", [{"kind": "execute"}]), case)
    assert "recorded no rows" in no_rows.failed.detail
    several = Case(question="q", expect={"rows": [[{"n": 1}], [{"n": 2}]]})
    card = evaluate(Trace("q", [{"kind": "execute", "rows": [{"n": 2}]}]), several)
    assert "matched accepted answer 2 of 2" in card.results[-1].detail


def test_expected_values_compare_whole_lists_and_numbers_written_differently():
    plan = Trace("q", [{"kind": "plan", "output": ["a", "b"]}, {"kind": "route", "output": 3}])
    assert evaluate(plan, Case(question="q", expect={"plan": ["a", "b"]})).verdict == "PASS"
    assert evaluate(plan, Case(question="q", expect={"route": "3"})).verdict == "PASS"
    card = evaluate(plan, Case(question="q", expect={"route": False}))
    assert card.failed.detail == "got 3, expected False"


def test_a_stage_that_never_ran_is_blamed_after_the_ones_that_did():
    # The adapter emits no `view` stage and the rows are wrong: the wrong rows are the
    # real failure, so that is the stage named.
    trace = make_trace(view=None, execute={"rows": [{"pending_count": 9}]})
    assert evaluate(trace, CASE).failed_stage == "execute"


def test_an_unknown_check_status_cannot_pass_as_a_pass():
    class Shouty(Check):
        def run(self, trace, case):
            return self.result("answer", "FAIL", "wrong status spelling")

    with pytest.raises(ValueError, match="PASS, FAIL or SKIP"):
        evaluate(make_trace(), rules=[Shouty()])
    with pytest.raises(ValueError):
        CheckResult("check", "stage", "failed")


def test_verdict_is_skip_when_nothing_could_be_checked():
    card = evaluate(Trace("", []))
    assert (card.verdict, card.failed_stage, card.results) == ("SKIP", None, [])
    assert Scorecard("q", error="agent raised").verdict == "ERROR"


def test_summary_lists_checks_in_stage_order_and_names_the_failing_stage():
    trace = make_trace(generate={"output": "SELECT pending_count FROM orders_view"})
    text = evaluate(trace, CASE, rules=[TenantFilter("store_id")]).summary()
    lines = text.splitlines()
    assert lines[0] == "Q: How many orders are pending?"
    assert lines[-1] == "  VERDICT      FAIL  (first failing stage: generate)"
    assert text.index("classify") < text.index("tenant_filter") < text.index("rows_match")
