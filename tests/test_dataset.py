import json

import pytest

from dbagenteval import DatasetError, filter_cases, load_cases

GOOD_YAML = """\
# a comment
- id: one
  question: "How many orders?"
  tags: [easy]
  expect:
    classify: data_question
    rows: [{count: 3}]

- question: "hi"
  tags: smalltalk
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_yaml_cases(tmp_path):
    cases = load_cases(write(tmp_path, "cases.yaml", GOOD_YAML))
    assert [c.key for c in cases] == ["one", "hi"]
    assert cases[0].accepted_answers == [[{"count": 3}]]
    assert cases[1].tags == ["smalltalk"]


def test_loads_cases_under_a_cases_key(tmp_path):
    text = "cases:\n  - question: one\n  - question: two\n"
    assert len(load_cases(write(tmp_path, "cases.yml", text))) == 2


def test_loads_jsonl(tmp_path):
    text = '{"question": "one", "expect": {"rows": []}}\n\n{"question": "two"}\n'
    cases = load_cases(write(tmp_path, "cases.jsonl", text))
    assert cases[0].accepted_answers == [[]]
    assert cases[1].expect == {}


def test_errors_carry_file_and_line_and_are_all_reported(tmp_path):
    text = (
        "- question: fine\n"
        "- question: typo in key\n"
        "  expects: {classify: data_question}\n"
        "- tags: [easy]\n"
        "- question: fine\n"
    )
    path = write(tmp_path, "cases.yaml", text)
    with pytest.raises(DatasetError) as error:
        load_cases(path)
    problems = error.value.problems
    assert len(problems) == 3
    assert problems[0].startswith(f"{path}:2: ") and "expects" in problems[0]
    assert problems[1].startswith(f"{path}:4: ") and "question" in problems[1]
    assert problems[2].startswith(f"{path}:5: ") and "duplicate" in problems[2]


def test_invalid_yaml_and_jsonl_report_the_line(tmp_path):
    with pytest.raises(DatasetError, match=r"cases\.yaml:\d+: invalid YAML"):
        load_cases(write(tmp_path, "cases.yaml", "- question: a\n  bad: [unclosed\n"))
    with pytest.raises(DatasetError, match=r"cases\.jsonl:2: invalid JSON"):
        load_cases(write(tmp_path, "cases.jsonl", '{"question": "a"}\n{oops}\n'))


def test_rejects_empty_missing_and_unknown_files(tmp_path):
    with pytest.raises(DatasetError, match="expected a list"):
        load_cases(write(tmp_path, "empty.yaml", ""))
    with pytest.raises(DatasetError, match="cannot read"):
        load_cases(tmp_path / "missing.yaml")
    with pytest.raises(DatasetError, match="unsupported file type"):
        load_cases(write(tmp_path, "cases.csv", "question\n"))


def test_filter_cases_by_tag(tmp_path):
    cases = load_cases(write(tmp_path, "cases.yaml", GOOD_YAML))
    assert [c.key for c in filter_cases(cases, ["smalltalk"])] == ["hi"]
    assert len(filter_cases(cases, [])) == 2


def test_loads_json_including_tab_indented_files(tmp_path):
    cases = [{"question": "one", "expect": {"rows": [{"n": 1}]}}, {"question": "two"}]
    path = write(tmp_path, "cases.json", json.dumps(cases, indent="\t"))
    assert [c.key for c in load_cases(path)] == ["one", "two"]
    wrapped = write(tmp_path, "wrapped.json", json.dumps({"cases": cases}))
    assert len(load_cases(wrapped)) == 2


def test_json_must_be_real_json_and_problems_name_the_case(tmp_path):
    with pytest.raises(DatasetError, match=r"cases\.json:1: invalid JSON"):
        load_cases(write(tmp_path, "cases.json", "[{question: a}]"))
    with pytest.raises(DatasetError, match=r"cases\.json: case 2: missing 'question'"):
        load_cases(write(tmp_path, "cases.json", '[{"question": "a"}, {"tags": []}]'))


def test_duplicate_yaml_keys_are_reported_instead_of_dropping_expectations(tmp_path):
    text = (
        "- question: one\n"
        "  expect:\n"
        "    classify: data_question\n"
        "    rows: [{n: 1}]\n"
        "  expect:\n"
        "    classify: other\n"
        "- question: two\n"
        "  expect: {classify: a, classify: b}\n"
    )
    path = write(tmp_path, "cases.yaml", text)
    with pytest.raises(DatasetError) as error:
        load_cases(path)
    assert error.value.problems == [
        f"{path}:5: duplicate key 'expect'",
        f"{path}:8: duplicate key 'classify'",
    ]


def test_byte_order_mark_and_unusual_line_breaks_are_accepted(tmp_path):
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + '{"question": "a\u2028b"}\n'.encode())
    assert load_cases(path)[0].question == "a\u2028b"


def test_a_file_that_is_not_utf8_is_reported_not_raised(tmp_path):
    path = tmp_path / "latin.yaml"
    path.write_bytes("- question: caf\xe9\n".encode("latin-1"))
    with pytest.raises(DatasetError, match="not a UTF-8 text file"):
        load_cases(path)
