"""Run and report the deterministic verification gate for the focused suite.

The gate deliberately reports to stdout instead of writing a report artifact:
verification must leave a clean checkout unchanged.  The final JSON record is
intended to be retained in CI or an issue closure comment when needed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from shlex import join as shell_join

ROOT = Path(__file__).resolve().parents[1]
_WARNING_RE = re.compile(
    r"warning|deprecated|chunk.?size|missing live|development server", re.IGNORECASE
)
_PYTEST_COUNT_RE = re.compile(
    r"(?P<count>\d+) (?P<kind>passed|failed|skipped|xfailed|xpassed|error|errors)"
)


@dataclass(frozen=True)
class StepResult:
    name: str
    command: str
    exit_code: int
    duration_seconds: float
    output: str

    @property
    def warnings(self) -> list[str]:
        return [line.strip() for line in self.output.splitlines() if _WARNING_RE.search(line)]

    @property
    def pytest_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for match in _PYTEST_COUNT_RE.finditer(self.output):
            kind = match.group("kind")
            if kind == "errors":
                kind = "error"
            counts[kind] = int(match.group("count"))
        return counts


def _git_status() -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commands() -> list[tuple[str, list[str]]]:
    return [
        ("python lint", ["uv", "run", "ruff", "check", "."]),
        ("python formatting", ["uv", "run", "ruff", "format", "--check", "."]),
        ("python typing", ["uv", "run", "mypy", "src"]),
        ("backend tests", ["uv", "run", "pytest"]),
        ("reports browser tests", ["npm", "--prefix", "web", "run", "e2e-reports"]),
    ]


def _run_step(name: str, command: list[str]) -> StepResult:
    rendered = shell_join(command)
    print(f"[verify] {name}: {rendered}", flush=True)
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    result = StepResult(
        name=name,
        command=rendered,
        exit_code=completed.returncode,
        duration_seconds=round(time.monotonic() - started, 3),
        output=output,
    )
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    print(f"[verify] {name}: exit {result.exit_code}", flush=True)
    return result


def _step_record(result: StepResult) -> dict[str, object]:
    record: dict[str, object] = {
        "name": result.name,
        "command": result.command,
        "exit_code": result.exit_code,
        "duration_seconds": result.duration_seconds,
        "warnings": result.warnings,
    }
    if result.pytest_counts:
        record["pytest_counts"] = result.pytest_counts
    return record


def main() -> int:
    command = os.environ.get("VERIFY_COMMAND", "uv run python scripts/verify_gate.py")
    before_status = _git_status()
    commit = _git_commit()
    results: list[StepResult] = []
    exit_code = 0

    if before_status:
        print("[verify] refusing to run: checkout is not clean", file=sys.stderr)
        exit_code = 2
    else:
        try:
            for name, step_command in _commands():
                result = _run_step(name, step_command)
                results.append(result)
                if result.exit_code:
                    exit_code = result.exit_code
                    break
        except (OSError, RuntimeError) as error:
            print(f"[verify] {error}", file=sys.stderr)
            exit_code = 2

    after_status = _git_status()
    unchanged = before_status == after_status
    if not unchanged and exit_code == 0:
        print("[verify] verification changed the checkout", file=sys.stderr)
        exit_code = 2

    test_counts: dict[str, int] = {}
    warnings: list[str] = []
    for result in results:
        for kind, count in result.pytest_counts.items():
            test_counts[kind] = test_counts.get(kind, 0) + count
        warnings.extend(result.warnings)

    report = {
        "gate": "real-bench deterministic verification",
        "evidence_layer": "deterministic_implementation",
        "artifact_root": str(ROOT),
        "evidence_boundary": (
            "proves the focused endpoint benchmark implementation on frozen inputs; does not "
            "prove live model quality, target-hardware performance, operator declarations, "
            "or engine validation"
        ),
        "commit": commit,
        "command": command,
        "clean_checkout": not before_status,
        "worktree_unchanged": unchanged,
        "exit_code": exit_code,
        "test_counts": test_counts,
        "warnings": warnings,
        "steps": [_step_record(result) for result in results],
    }
    print("VERIFY_REPORT=" + json.dumps(report, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
