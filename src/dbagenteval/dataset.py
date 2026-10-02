"""Load test cases from a YAML, JSON or JSONL file, with errors that say where."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from dbagenteval.case import Case


class DatasetError(ValueError):
    """A dataset file could not be loaded. `problems` lists every issue found."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(problems))


def load_cases(path: str | Path) -> list[Case]:
    """Read every case in the file, or raise DatasetError listing all problems.

    `.yaml`/`.yml` and `.json` hold a list of cases (or a mapping with a `cases` list);
    `.jsonl` holds one case per line.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")  # tolerates a byte-order mark
    except OSError as e:
        raise DatasetError([f"{path}: cannot read file ({e.strerror or e})"]) from e
    except UnicodeDecodeError as e:
        raise DatasetError([f"{path}: not a UTF-8 text file ({e.reason})"]) from e

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        located = _read_jsonl(path, text)
    elif suffix == ".json":
        located = _read_json(path, text)
    elif suffix in (".yaml", ".yml"):
        located = _read_yaml(path, text)
    else:
        raise DatasetError([f"{path}: unsupported file type; use .yaml, .yml, .json or .jsonl"])

    cases: list[Case] = []
    problems: list[str] = []
    seen: dict[str, str] = {}
    for where, raw in located:
        try:
            case = Case.from_dict(raw)
        except ValueError as e:
            problems.append(f"{where}: {e}")
            continue
        if case.key in seen:
            problems.append(f"{where}: duplicate case {case.key!r} (first at {seen[case.key]})")
            continue
        seen[case.key] = where
        cases.append(case)
    if problems:
        raise DatasetError(problems)
    if not cases:
        raise DatasetError([f"{path}: no cases found"])
    return cases


def filter_cases(cases: Iterable[Case], tags: Iterable[str]) -> list[Case]:
    """Cases carrying at least one of `tags`; all cases when `tags` is empty."""
    wanted = set(tags)
    return [c for c in cases if not wanted or wanted & set(c.tags)]


def _case_list(path: Path, data: Any) -> list[Any]:
    if isinstance(data, dict) and "cases" in data:
        data = data["cases"]
    if not isinstance(data, list):
        raise DatasetError([f"{path}: expected a list of cases, or a mapping with a 'cases' list"])
    return data


def _read_jsonl(path: Path, text: str) -> list[tuple[str, Any]]:
    located: list[tuple[str, Any]] = []
    problems: list[str] = []
    # Split on "\n" only: JSON strings may hold other line-break characters unescaped.
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            located.append((f"{path}:{number}", json.loads(line)))
        except json.JSONDecodeError as e:
            problems.append(f"{path}:{number}: invalid JSON ({e.msg})")
    if problems:
        raise DatasetError(problems)
    return located


def _read_json(path: Path, text: str) -> list[tuple[str, Any]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise DatasetError([f"{path}:{e.lineno}: invalid JSON ({e.msg})"]) from e
    return [(f"{path}: case {n}", raw) for n, raw in enumerate(_case_list(path, data), start=1)]


def _read_yaml(path: Path, text: str) -> list[tuple[str, Any]]:
    loader = yaml.SafeLoader(text)
    try:
        root = loader.get_single_node()
        duplicates = _duplicate_keys(root, path)
        if duplicates:
            raise DatasetError(duplicates)
        data = loader.construct_document(root) if root is not None else None
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        line = f":{mark.line + 1}" if mark is not None else ""
        problem = getattr(e, "problem", None) or str(e)
        raise DatasetError([f"{path}{line}: invalid YAML ({problem})"]) from e
    finally:
        loader.dispose()

    cases = _case_list(path, data)
    sequence: Any = root
    if isinstance(root, yaml.MappingNode):
        sequence = next(value for key, value in root.value if key.value == "cases")
    lines = [node.start_mark.line + 1 for node in sequence.value]
    return [(f"{path}:{line}", raw) for line, raw in zip(lines, cases, strict=True)]


def _duplicate_keys(node: yaml.Node | None, path: Path) -> list[str]:
    """YAML keeps the last of two equal keys without complaint, which would silently
    drop expectations (a second `expect:` replacing the first)."""
    problems: list[str] = []
    if isinstance(node, yaml.MappingNode):
        seen: set[str] = set()
        for key, value in node.value:
            if isinstance(key, yaml.ScalarNode) and key.value != "<<":
                if key.value in seen:
                    problems.append(
                        f"{path}:{key.start_mark.line + 1}: duplicate key {key.value!r}"
                    )
                seen.add(key.value)
            problems += _duplicate_keys(value, path)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            problems += _duplicate_keys(item, path)
    return problems
