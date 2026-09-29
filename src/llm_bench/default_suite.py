"""Complete default-suite orchestration over the existing ordinary workload runners."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .humaneval import execute_coding
from .ordinary_runs import (
    DEFAULT_SUITE_UNITS,
    DefaultSuiteWorkload,
    RunCancelled,
    Submission,
    cancellation_path,
    cancellation_requested,
    create_run_directory,
    execute_fresh_context,
    execute_growing_session,
    finish_run,
    finished_on_exit,
    read_result,
    run_secrets,
    sanitize_files,
    save,
)
from .ruler_runs import execute_ruler

SUMMARY_FIELDS = ("fresh_context_curve", "growing_session_curve", "coding", "ruler")


def _phase_id(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:32]


def _warmup_seconds(directory: Path) -> float:
    elapsed = 0.0
    for path in directory.rglob("execution.json"):
        if "warmup" not in path.parts:
            continue
        try:
            value = json.loads(path.read_text()).get("elapsed_seconds")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(value, (int, float)):
            elapsed += value
    return round(elapsed, 6)


def _phase_accounted(name: str, result: dict[str, Any]) -> int:
    coverage = result.get("coverage", {}).get(name, {})
    if name == "coding":
        return sum(row.get("outcome") != "not_attempted" for row in result.get("outcomes", []))
    for field in ("accounted", "attempted", "graded"):
        value = coverage.get(field)
        if isinstance(value, int):
            return value
    return 0


def refresh_progress(directory: Path, result: dict[str, Any]) -> None:
    """Attach the active phase's retained snapshot without altering run evidence."""
    current = result.get("phase_progress", {}).get("current")
    if current not in DEFAULT_SUITE_UNITS:
        return
    phases = result.get("phase_progress", {}).get("phases", [])
    active = next((phase for phase in phases if phase.get("name") == current), None)
    if active is None:
        return
    child_summary = directory / "phases" / current / _phase_id(current) / "summary.json"
    try:
        snapshot = json.loads(child_summary.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(snapshot, dict):
        return
    active["live_coverage"] = snapshot.get("coverage", {})
    try:
        outcomes = json.loads(child_summary.with_name("outcomes.json").read_text())
    except (OSError, ValueError):
        outcomes = snapshot.get("outcomes", [])
    if not isinstance(outcomes, list) or not all(isinstance(row, dict) for row in outcomes):
        return
    active["live_outcomes"] = outcomes
    pending = {"coding": "not_attempted", "long_context": "missing"}.get(current)
    active["accounted_units"] = sum(row.get("outcome") != pending for row in outcomes)


@finished_on_exit
def execute_default_suite(
    root: Path,
    submission: Submission,
    workload: DefaultSuiteWorkload,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the four fixed phases once and retain one independently readable result."""
    run_id, directory = create_run_directory(root, run_id)
    marker = cancellation_path(directory)
    retained_workload = {
        **workload.model_dump(),
        "evidence": "run",
        "expected_coverage": DEFAULT_SUITE_UNITS,
    }
    save(directory / "setup.json", submission.setup.model_dump())
    save(directory / "workload.json", retained_workload)
    phase_progress = [
        {
            "name": name,
            "status": "pending",
            "expected_units": expected,
            "accounted_units": 0,
            "elapsed_seconds": 0.0,
        }
        for name, expected in DEFAULT_SUITE_UNITS.items()
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "evidence": "run",
        "status": "partial",
        "default_workload_complete": False,
        "families": {"performance": "partial", "coding": "partial", "long_context": "partial"},
        "coverage": {
            name: {"expected": expected, "accounted": 0}
            for name, expected in DEFAULT_SUITE_UNITS.items()
        },
        "phase_progress": {"current": None, "completed": 0, "total": 4, "phases": phase_progress},
        "outcomes": [],
    }
    save(directory / "summary.json", summary)
    started = time.monotonic()

    phases: list[
        tuple[
            str,
            Any,
            Callable[[Path, Submission, Any, str | None, Path | None], dict[str, Any]],
        ]
    ] = [
        ("fresh_context", workload.fresh_context, execute_fresh_context),
        ("growing_session", workload.growing_session, execute_growing_session),
        ("coding", workload.coding, execute_coding),
        ("long_context", workload.long_context, execute_ruler),
    ]
    stopped_reason: str | None = None
    for index, (name, phase_workload, runner) in enumerate(phases):
        phase = phase_progress[index]
        if stopped_reason is not None or cancellation_requested(marker):
            stopped_reason = stopped_reason or "Cancelled"
            phase.update(
                status="skipped", diagnostic=f"Not attempted after {stopped_reason.lower()}"
            )
            continue

        phase_started = time.monotonic()
        phase_root = directory / "phases" / name
        phase_directory = phase_root / _phase_id(name)
        phase.update(
            status="running",
            started_at_seconds=round(phase_started - started, 6),
            result_path=str(phase_directory.relative_to(directory)),
        )
        summary["phase_progress"]["current"] = name
        save(directory / "summary.json", summary)
        phase_submission = Submission(setup=submission.setup, workload=phase_workload)
        try:
            result = runner(phase_root, phase_submission, phase_workload, _phase_id(name), marker)
            elapsed = time.monotonic() - phase_started
            phase.update(
                status=result["status"],
                accounted_units=_phase_accounted(name, result),
                elapsed_seconds=round(elapsed, 6),
                finished_at_seconds=round(time.monotonic() - started, 6),
                observed_timing={
                    "wall_seconds": round(elapsed, 6),
                    "warmup_seconds": _warmup_seconds(phase_directory),
                    "session_construction_seconds": result.get("construction", {}).get(
                        "construction_seconds", 0.0
                    ),
                },
            )
            if result.get("diagnostic"):
                phase["diagnostic"] = result["diagnostic"]
            native_coverage = result.get("coverage", {}).get(name, {})
            summary["coverage"][name] = {
                **native_coverage,
                "accounted": _phase_accounted(name, result),
            }
            summary["outcomes"].extend(
                {"phase": name, **outcome} for outcome in result.get("outcomes", [])
            )
            for field in SUMMARY_FIELDS:
                if field in result:
                    summary[field] = result[field]
            if name in {"fresh_context", "growing_session"} and "cache" in result:
                summary.setdefault("cache", {})[name] = result["cache"]
            if "warmups" in result:
                summary.setdefault("warmups", {})[name] = result["warmups"]
            if "construction" in result:
                summary["construction"] = result["construction"]
            if result.get("diagnostic") == "Endpoint became unreachable":
                stopped_reason = "Endpoint became unreachable"
            elif result.get("diagnostic") == "Cancelled" or cancellation_requested(marker):
                stopped_reason = "Cancelled"
        except (KeyboardInterrupt, RunCancelled):
            stopped_reason = "Cancelled"
            phase.update(
                status="partial",
                diagnostic="Cancelled",
                elapsed_seconds=round(time.monotonic() - phase_started, 6),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            phase.update(
                status="partial",
                diagnostic=str(exc),
                elapsed_seconds=round(time.monotonic() - phase_started, 6),
            )
        summary["phase_progress"]["completed"] = sum(
            item["status"] in {"completed", "partial"} for item in phase_progress
        )
        save(directory / "outcomes.json", summary["outcomes"])
        save(directory / "summary.json", summary)

    summary["phase_progress"]["current"] = None
    summary["phase_progress"]["completed"] = sum(
        item["status"] in {"completed", "partial"} for item in phase_progress
    )
    fresh_status = phase_progress[0]["status"]
    growing_status = phase_progress[1]["status"]
    summary["families"] = {
        "performance": (
            "completed" if fresh_status == growing_status == "completed" else "partial"
        ),
        "coding": "completed" if phase_progress[2]["status"] == "completed" else "partial",
        "long_context": ("completed" if phase_progress[3]["status"] == "completed" else "partial"),
    }
    complete = all(phase["status"] == "completed" for phase in phase_progress)
    summary["status"] = "completed" if complete else "partial"
    summary["default_workload_complete"] = complete
    summary["wall_seconds"] = round(time.monotonic() - started, 6)
    if stopped_reason:
        summary["diagnostic"] = stopped_reason
    save(directory / "outcomes.json", summary["outcomes"])
    save(directory / "summary.json", summary)
    sanitize_files(directory, run_secrets(submission))
    finish_run(directory)
    return read_result(root, run_id)
