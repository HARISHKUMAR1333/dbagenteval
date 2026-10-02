"""Command line: validate a dataset, run an agent over it, compare two runs.

Exit codes: 0 all good, 1 something did not pass (or regressed), 2 the command could
not run (bad dataset, bad arguments, unreadable report).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dbagenteval import __version__
from dbagenteval.checks import Check
from dbagenteval.dataset import DatasetError, filter_cases, load_cases
from dbagenteval.report import diff_reports, diff_to_markdown, to_markdown
from dbagenteval.runner import run_suite


class CliError(Exception):
    """A problem with how the command was called; printed without a traceback."""


def _load_object(spec: str) -> Any:
    """Import `module:attribute`, looking in the current directory first."""
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise CliError(f"expected module:attribute, got {spec!r}")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        module = importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001 — the user's module may fail in any way on import
        raise CliError(f"cannot import {module_name!r}: {type(e).__name__}: {e}") from e
    try:
        return getattr(module, attribute)
    except AttributeError:
        raise CliError(f"module {module_name!r} has no attribute {attribute!r}") from None


def _load_callable(spec: str, what: str) -> Any:
    loaded = _load_object(spec)
    if not callable(loaded):
        raise CliError(f"{spec} is not a function; --{what} needs one")
    return loaded


def _load_rules(spec: str) -> list[Check]:
    rules = _load_object(spec)
    if not isinstance(rules, list | tuple) or not all(isinstance(r, Check) for r in rules):
        raise CliError(f"{spec} must be a list of checks, such as [ReadOnly(), ...]")
    return list(rules)


def _read_report(path: str) -> dict[str, Any]:
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:  # ValueError covers bad JSON and bad encoding
        raise CliError(f"cannot read report {path}: {e}") from e
    if not isinstance(report, dict):
        raise CliError(f"{path} is not a dbagenteval JSON report")
    cases = report.get("cases")
    if not isinstance(cases, list) or not all(
        isinstance(c, dict) and "verdict" in c and (c.get("id") or c.get("question")) for c in cases
    ):
        raise CliError(f"{path} is not a dbagenteval JSON report")
    return report


def _write(path: str, text: str) -> None:
    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError as e:
        raise CliError(f"cannot write {path}: {e.strerror or e}") from e


def _cmd_validate(args: argparse.Namespace) -> int:
    cases = load_cases(args.dataset)
    print(f"{args.dataset}: {len(cases)} case(s) OK")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(args.dataset)
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    selected = filter_cases(cases, tags)
    if not selected:
        raise CliError(f"no cases match tags {tags}")
    if args.concurrency < 1:
        raise CliError("--concurrency must be at least 1")
    agent = _load_callable(args.agent, "agent")
    rules = _load_rules(args.rules) if args.rules else []
    judge = _load_callable(args.judge, "judge") if args.judge else None
    execute = _load_callable(args.execute, "execute") if args.execute else None
    if execute is None and any(case.reference_sql for case in selected):
        raise CliError(
            "some cases have `reference_sql`; pass --execute module:function to run it "
            "(a function taking SQL and returning rows as dicts)"
        )
    # Fail on an unwritable report path now, not after the whole suite has run.
    for path in (args.json, args.md):
        if path:
            _write(path, "")

    result = run_suite(
        selected, agent, rules=rules, judge=judge, execute=execute, concurrency=args.concurrency
    )

    for card in result.cards:
        line = f"{card.verdict:<5} {card.case_id or card.question}"
        failed = card.failed
        if failed:
            line += f"\n      {failed.stage} / {failed.check}: {failed.detail}"
        elif card.error:
            line += f"\n      {card.error}"
        print(line)
    summary = result.summary()
    print(
        f"\n{summary['passed']} of {summary['total']} passed ({summary['pass_rate']:.0%})"
        f", {summary['failed']} failed, {summary['errors']} errors"
        + (f", {summary['skipped']} with nothing to check" if summary["skipped"] else "")
    )
    for stage, count in sorted(summary["failed_stages"].items(), key=lambda kv: -kv[1]):
        print(f"  first failing stage {stage}: {count}")

    report = result.to_dict()
    if args.json:
        _write(args.json, json.dumps(report, indent=2, default=str))
    if args.md:
        _write(args.md, to_markdown(report))
    return 0 if result.passed else 1


def _cmd_diff(args: argparse.Namespace) -> int:
    changes = diff_reports(_read_report(args.old), _read_report(args.new))
    print(diff_to_markdown(changes), end="")
    return 1 if changes["regressions"] else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dbagenteval", description="Stage-by-stage evaluation for NL-to-SQL agents."
    )
    parser.add_argument("--version", action="version", version=f"dbagenteval {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="check a dataset file for mistakes")
    validate.add_argument("dataset")
    validate.set_defaults(func=_cmd_validate)

    run = commands.add_parser("run", help="run an agent over a dataset and score it")
    run.add_argument("dataset")
    run.add_argument(
        "--agent", required=True, help="module:function taking a question, returning a trace"
    )
    run.add_argument("--rules", help="module:attribute holding a list of checks")
    run.add_argument("--judge", help="module:function taking a prompt, returning the reply text")
    run.add_argument(
        "--execute", help="module:function running SQL and returning rows, for reference_sql"
    )
    run.add_argument("--tags", help="comma-separated; run only cases with any of these tags")
    run.add_argument("--concurrency", type=int, default=4, help="cases run at once (default 4)")
    run.add_argument("--json", metavar="PATH", help="write the full report as JSON")
    run.add_argument("--md", metavar="PATH", help="write the report as Markdown")
    run.set_defaults(func=_cmd_run)

    diff = commands.add_parser("diff", help="compare two JSON reports; exit 1 on regressions")
    diff.add_argument("old")
    diff.add_argument("new")
    diff.set_defaults(func=_cmd_diff)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # sqlglot warns on stderr for each statement it falls back on; the rule that sees
    # that statement already reports it as a failure.
    logging.getLogger("sqlglot").setLevel(logging.ERROR)
    # Questions and answers can hold any character; a console that cannot show one
    # should print a placeholder, not crash.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        return int(args.func(args))
    except DatasetError as e:
        for problem in e.problems:
            print(problem, file=sys.stderr)
        return 2
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
