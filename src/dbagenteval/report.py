"""Render a run as Markdown, and compare two runs.

Both work on the plain dict from `SuiteResult.to_dict()`, so a report saved as JSON by
an earlier run can be read back and compared without re-running anything.
"""

from __future__ import annotations

from typing import Any


def _cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def _key(case: dict[str, Any]) -> str:
    return str(case.get("id") or case["question"])


def _failure_detail(case: dict[str, Any]) -> str:
    if case.get("error"):
        return str(case["error"])
    for result in case.get("results", []):
        if (
            result["status"] == "fail"
            and result["stage"] == case.get("failed_stage")
            and result["check"] == case.get("failed_check")
        ):
            return str(result["detail"])
    return ""


def to_markdown(report: dict[str, Any]) -> str:
    """A report for a PR comment or CI log: headline, breakdowns, then each failure."""
    s = report["summary"]
    lines = [
        "# dbagenteval report",
        "",
        f"**{s['passed']} of {s['total']} passed ({s['pass_rate']:.0%})**"
        f" · {s['failed']} failed · {s['errors']} errors · {s['skipped']} skipped",
    ]
    if s["by_tag"]:
        lines += ["", "## By tag", "", "| Tag | Passed | Total |", "|---|---|---|"]
        lines += [
            f"| {_cell(tag)} | {c['passed']} | {c['total']} |"
            for tag, c in sorted(s["by_tag"].items())
        ]
    if s["failed_stages"]:
        lines += ["", "## First failing stage", "", "| Stage | Cases |", "|---|---|"]
        lines += [
            f"| {_cell(stage)} | {count} |"
            for stage, count in sorted(s["failed_stages"].items(), key=lambda kv: -kv[1])
        ]
    not_passed = [c for c in report["cases"] if c["verdict"] != "PASS"]
    if not_passed:
        lines += [
            "",
            "## Not passed",
            "",
            "| Case | Verdict | Stage | Check | Detail |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {_cell(_key(c))} | {c['verdict']} | {_cell(c.get('failed_stage'))} "
            f"| {_cell(c.get('failed_check'))} | {_cell(_failure_detail(c))} |"
            for c in not_passed
        ]
    return "\n".join(lines) + "\n"


def diff_reports(old: dict[str, Any], new: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """What changed between two runs, matching cases by id (or question).

    `regressions` passed before and do not now; `fixes` are the reverse; `added` and
    `removed` are cases present in only one run.
    """
    before = {_key(c): c for c in old["cases"]}
    after = {_key(c): c for c in new["cases"]}
    changes: dict[str, list[dict[str, Any]]] = {
        "regressions": [],
        "fixes": [],
        "added": [],
        "removed": [],
    }
    for key, case in after.items():
        if key not in before:
            changes["added"].append({"case": key, "verdict": case["verdict"]})
            continue
        was, now = before[key]["verdict"], case["verdict"]
        entry = {
            "case": key,
            "was": was,
            "now": now,
            "failed_stage": case.get("failed_stage"),
            "detail": _failure_detail(case),
        }
        if was == "PASS" and now != "PASS":
            changes["regressions"].append(entry)
        elif was != "PASS" and now == "PASS":
            changes["fixes"].append(entry)
    changes["removed"] = [
        {"case": key, "verdict": case["verdict"]}
        for key, case in before.items()
        if key not in after
    ]
    return changes


def diff_to_markdown(changes: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# dbagenteval diff",
        "",
        f"**{len(changes['regressions'])} regressions, {len(changes['fixes'])} fixes"
        f", {len(changes['added'])} added, {len(changes['removed'])} removed**",
    ]
    if changes["regressions"]:
        lines += ["", "## Regressions", "", "| Case | Now | Stage | Detail |", "|---|---|---|---|"]
        lines += [
            f"| {_cell(r['case'])} | {r['now']} | {_cell(r['failed_stage'])} "
            f"| {_cell(r['detail'])} |"
            for r in changes["regressions"]
        ]
    if changes["fixes"]:
        lines += ["", "## Fixes", ""]
        lines += [f"- {r['case']} ({r['was']} -> PASS)" for r in changes["fixes"]]
    for title in ("added", "removed"):
        if changes[title]:
            lines += ["", f"## {title.capitalize()}", ""]
            lines += [f"- {r['case']} ({r['verdict']})" for r in changes[title]]
    return "\n".join(lines) + "\n"
