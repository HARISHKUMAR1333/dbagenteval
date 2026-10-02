import json
from pathlib import Path

import pytest

from dbagenteval import Case, Trace, arun_suite, run_suite
from dbagenteval.cli import main
from dbagenteval.report import diff_reports, diff_to_markdown, to_markdown

REPO = Path(__file__).resolve().parent.parent


def sync_agent(question):
    return Trace(question, [{"kind": "route", "name": "classify", "output": "data_question"}])


async def async_agent(question):
    return {"question": question, "stages": [{"kind": "route", "name": "classify", "output": "x"}]}


def broken_agent(question):
    raise RuntimeError("database is down")


CASES = [
    Case(question="one", tags=["easy"], expect={"classify": "data_question"}),
    Case(question="two", tags=["easy", "orders"], expect={"classify": "data_question"}),
]


def test_run_suite_with_a_plain_function():
    result = run_suite(CASES, sync_agent)
    assert result.passed
    assert result.summary()["by_tag"] == {
        "easy": {"total": 2, "passed": 2},
        "orders": {"total": 1, "passed": 1},
    }


def test_run_suite_with_an_async_agent_returning_dicts():
    result = run_suite(CASES, async_agent)
    summary = result.summary()
    assert not result.passed
    assert summary["failed"] == 2
    assert summary["failed_stages"] == {"classify": 2}


def test_agent_exception_becomes_an_error_verdict_and_the_run_continues():
    result = run_suite(CASES, broken_agent)
    assert [c.verdict for c in result.cards] == ["ERROR", "ERROR"]
    assert "database is down" in result.cards[0].error
    assert result.summary()["errors"] == 2


def test_unreadable_judge_becomes_an_error_not_a_pass():
    result = run_suite(
        [Case(question="q")],
        lambda q: Trace(q, [{"kind": "answer", "output": "42"}]),
        judge=lambda prompt: "no json here",
    )
    assert result.cards[0].verdict == "ERROR"
    assert "evaluation raised" in result.cards[0].error


def test_markdown_report_and_diff():
    before = run_suite(CASES, sync_agent).to_dict()
    after = run_suite(CASES, async_agent).to_dict()
    markdown = to_markdown(after)
    assert "**0 of 2 passed (0%)**" in markdown
    assert "| one | FAIL | classify | expected |" in markdown

    changes = diff_reports(before, after)
    assert [r["case"] for r in changes["regressions"]] == ["one", "two"]
    assert diff_reports(after, before)["fixes"][0]["case"] == "one"


@pytest.fixture
def in_repo(monkeypatch):
    monkeypatch.chdir(REPO)


def test_cli_runs_the_example_and_reports_the_planted_failure(in_repo, tmp_path, capsys):
    report_path = tmp_path / "report.json"
    code = main([
        "run", "examples/cases.yaml",
        "--agent", "examples.toy_agent:agent",
        "--rules", "examples.toy_agent:RULES",
        "--execute", "examples.toy_agent:run_sql",
        "--json", str(report_path),
        "--md", str(tmp_path / "report.md"),
    ])  # fmt: skip
    out = capsys.readouterr().out
    assert code == 1
    assert "4 of 5 passed" in out
    assert "generate / tenant_filter" in out

    report = json.loads(report_path.read_text())
    failed = [c for c in report["cases"] if c["verdict"] != "PASS"]
    assert [(c["id"], c["failed_stage"], c["failed_check"]) for c in failed] == [
        ("cancelled-count", "generate", "tenant_filter")
    ]
    assert (tmp_path / "report.md").read_text().startswith("# dbagenteval report")

    assert main(["diff", str(report_path), str(report_path)]) == 0


def test_cli_tag_filter_and_validate(in_repo, capsys):
    assert main(["validate", "examples/cases.yaml"]) == 0
    code = main([
        "run", "examples/cases.yaml",
        "--agent", "examples.toy_agent:agent",
        "--tags", "safety,smalltalk",
    ])  # fmt: skip
    assert code == 0
    assert "2 of 2 passed" in capsys.readouterr().out


def test_cli_usage_problems_exit_with_2(in_repo, tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- tags: [x]\n")
    assert main(["validate", str(bad)]) == 2
    assert main(["run", "examples/cases.yaml", "--agent", "nope.module:agent"]) == 2
    assert main(["run", "examples/cases.yaml", "--agent", "examples.toy_agent:missing"]) == 2
    assert main(["run", "examples/cases.yaml", "--agent", "x:y", "--tags", "nonexistent"]) == 2
    assert main(["diff", str(bad), str(bad)]) == 2
    assert "missing 'question'" in capsys.readouterr().err


def test_run_suite_inside_a_running_loop_says_to_use_arun_suite():
    import asyncio

    async def main():
        with pytest.raises(RuntimeError, match="arun_suite"):
            run_suite(CASES, sync_agent)
        return await arun_suite(CASES, sync_agent)

    assert asyncio.run(main()).passed


def test_concurrency_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        run_suite(CASES, sync_agent, concurrency=0)


def test_agent_returning_the_wrong_type_is_an_error_verdict():
    result = run_suite(CASES[:1], lambda question: 42)
    assert result.cards[0].verdict == "ERROR"
    assert "must return a Trace, a dict or the answer text" in result.cards[0].error


def test_markdown_escapes_table_characters():
    report = run_suite([Case(question="a | b", expect={"classify": "x|y"})], sync_agent).to_dict()
    assert "| a \\| b | FAIL |" in to_markdown(report)


def test_diff_lists_added_and_removed_cases():
    before = run_suite(CASES, sync_agent).to_dict()
    after = run_suite([CASES[0], Case(question="three")], sync_agent).to_dict()
    changes = diff_reports(before, after)
    assert [c["case"] for c in changes["added"]] == ["three"]
    assert [c["case"] for c in changes["removed"]] == ["two"]
    text = diff_to_markdown(diff_reports(run_suite(CASES, async_agent).to_dict(), before))
    assert "- one (FAIL -> PASS)" in text
    text.encode("ascii")  # safe on any console encoding


def write_module(tmp_path, monkeypatch, name, source):
    (tmp_path / f"{name}.py").write_text(source)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))


AGENTS = """
from dbagenteval import Trace
from dbagenteval.rules import ReadOnly

def ok(question):
    return Trace(question, [{"kind": "route", "name": "classify", "output": "data_question"},
                            {"kind": "answer", "output": "42"}])

def wrong(question):
    return Trace(question, [{"kind": "route", "name": "classify", "output": "other"}])

def empty(question):
    return Trace(question, [])

def judge(prompt):
    return '{"pass": true, "reason": "fine"}'

RULES = [ReadOnly()]
ONE_RULE = ReadOnly()
NOT_A_FUNCTION = 5
"""
DATASET = "- question: one\n  expect: {classify: data_question}\n"


def test_cli_diff_exits_1_on_a_regression(tmp_path, monkeypatch, capsys):
    write_module(tmp_path, monkeypatch, "cli_agents_a", AGENTS)
    (tmp_path / "cases.yaml").write_text(DATASET)
    assert main(["run", "cases.yaml", "--agent", "cli_agents_a:ok", "--json", "good.json"]) == 0
    assert main(["run", "cases.yaml", "--agent", "cli_agents_a:wrong", "--json", "bad.json"]) == 1
    capsys.readouterr()
    assert main(["diff", "good.json", "bad.json"]) == 1
    assert "1 regressions" in capsys.readouterr().out
    assert main(["diff", "bad.json", "good.json"]) == 0


def test_cli_judge_and_concurrency_options(tmp_path, monkeypatch, capsys):
    write_module(tmp_path, monkeypatch, "cli_agents_b", AGENTS)
    (tmp_path / "cases.yaml").write_text(DATASET)
    args = ["run", "cases.yaml", "--agent", "cli_agents_b:ok", "--json", "out.json"]
    assert main([*args, "--judge", "cli_agents_b:judge", "--concurrency", "1"]) == 0
    checks = {
        r["check"]: r["status"]
        for r in json.loads(Path("out.json").read_text())["cases"][0]["results"]
    }
    # No query ran, so there is nothing for `faithful` to be faithful to.
    assert (checks["relevant"], checks["faithful"]) == ("pass", "skip")


def test_cli_reports_cases_with_nothing_to_check(tmp_path, monkeypatch, capsys):
    write_module(tmp_path, monkeypatch, "cli_agents_c", AGENTS)
    (tmp_path / "cases.yaml").write_text("- question: one\n")
    assert main(["run", "cases.yaml", "--agent", "cli_agents_c:empty"]) == 1
    assert "1 with nothing to check" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--rules", "cli_agents_d:ONE_RULE"], "must be a list of checks"),
        (["--judge", "cli_agents_d:NOT_A_FUNCTION"], "is not a function"),
        (["--concurrency", "0"], "at least 1"),
        (["--json", "missing_dir/out.json"], "cannot write"),
    ],
)
def test_cli_bad_options_exit_2_before_running(tmp_path, monkeypatch, capsys, extra, message):
    write_module(tmp_path, monkeypatch, "cli_agents_d", AGENTS)
    (tmp_path / "cases.yaml").write_text(DATASET)
    assert main(["run", "cases.yaml", "--agent", "cli_agents_d:ok", *extra]) == 2
    assert message in capsys.readouterr().err


def test_cli_module_that_fails_to_import_exits_2(tmp_path, monkeypatch, capsys):
    write_module(tmp_path, monkeypatch, "cli_broken", "def agent(:\n")
    (tmp_path / "cases.yaml").write_text(DATASET)
    assert main(["run", "cases.yaml", "--agent", "cli_broken:agent"]) == 2
    assert "cannot import 'cli_broken': SyntaxError" in capsys.readouterr().err


@pytest.mark.parametrize(
    "content",
    ['{"cases": 5}', '{"cases": [{}]}', "[1, 2]", "not json", b"\xff\xfe\x00"],
)
def test_cli_diff_rejects_files_that_are_not_reports(tmp_path, capsys, content):
    path = tmp_path / "report.json"
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    assert main(["diff", str(path), str(path)]) == 2


def test_cli_needs_an_executor_for_cases_with_reference_sql(in_repo, capsys):
    args = ["run", "examples/cases.yaml", "--agent", "examples.toy_agent:agent"]
    assert main(args) == 2
    assert "--execute" in capsys.readouterr().err
    # ...but not when the selected cases have no reference query.
    assert main([*args, "--tags", "safety"]) == 0
