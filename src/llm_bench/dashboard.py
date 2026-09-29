"""Read-only, rebuildable snapshot of retained ordinary benchmark runs."""

from __future__ import annotations

import json
import re
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ordinary_runs import read_result, run_lock_held

RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
OUTCOME_FIELDS = {
    "phase",
    "condition",
    "task",
    "task_id",
    "problem_id",
    "task_type",
    "request_id",
    "scenario",
    "actual_input_tokens",
    "actual_output_tokens",
    "outcome",
    "diagnostic",
    "input_tokens",
    "output_target",
    "output_tokens",
    "completion_tokens",
    "generated_tokens",
    "repetition",
    "checkpoint",
    "score",
    "grading_available",
    "metrics",
}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _lifecycle(directory: Path, status: str) -> dict[str, Any]:
    marker = _json(directory / "execution-state.json")
    if isinstance(marker, dict):
        if marker.get("state") == "finished":
            return {"state": "finished", "verified": True}
        if (
            marker.get("state") == "running"
            and marker.get("hostname") == socket.gethostname()
            and run_lock_held(directory)
        ):
            return {"state": "active", "verified": True}
    if status == "completed":
        return {"state": "finished", "verified": False}
    return {"state": "unverified", "verified": False}


def _observed_outcomes(directory: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    retained = result.get("outcomes", [])
    outcomes = list(retained) if isinstance(retained, list) else []
    if result["lifecycle"]["state"] == "finished":
        return outcomes
    if result.get("workload", {}).get("profile") == "default-suite":
        current = result.get("phase_progress", {}).get("current")
        if any(row.get("phase") == current for row in outcomes if isinstance(row, dict)):
            return outcomes
        for phase in result.get("phase_progress", {}).get("phases", []):
            if phase.get("name") == current:
                live = phase.get("live_outcomes", [])
                if isinstance(live, list):
                    outcomes.extend(
                        {"phase": current, **row} for row in live if isinstance(row, dict)
                    )
                break
    else:
        live = _json(directory / "outcomes.json")
        if isinstance(live, list) and all(isinstance(row, dict) for row in live):
            outcomes = live
    return outcomes


def _project_run(result: dict[str, Any]) -> dict[str, Any]:
    """Send live chart evidence once, without retained transcripts or duplicate rows."""
    result.pop("outcomes", None)
    result["observed_outcomes"] = [
        {key: value for key, value in row.items() if key in OUTCOME_FIELDS}
        for row in result["observed_outcomes"]
        if isinstance(row, dict)
    ]
    for phase in result.get("phase_progress", {}).get("phases", []):
        phase.pop("live_outcomes", None)
    if "phase_progress" not in result:
        workload = result.get("workload", {})
        profile = workload.get("profile", workload.get("scenario", "validation"))
        coverage: dict[str, Any] = next(iter(result.get("coverage", {}).values()), {})
        rows = result["observed_outcomes"]
        expected = coverage.get("expected", 1)
        accounted = sum(row.get("outcome") not in {"not_attempted", "missing"} for row in rows)
        active = result["lifecycle"]["state"] == "active"
        result["phase_progress"] = {
            "current": profile if active else None,
            "completed": 0 if active else 1,
            "total": 1,
            "phases": [
                {
                    "name": profile,
                    "status": "running" if active else result.get("status", "partial"),
                    "expected_units": expected,
                    "accounted_units": accounted,
                }
            ],
        }
    return result


def _diagnostic(exc: Exception) -> str:
    """A client-safe reason: never the raw text, which may hold absolute local paths."""
    if isinstance(exc, OSError):
        return "Run files are unreadable"
    if isinstance(exc, ValueError):
        return "Run record is malformed"
    return "Run record is incomplete"


def dashboard_snapshot(root: Path) -> dict[str, Any]:
    """Read persisted results only; incomplete evidence stays partial or unverified."""
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        directories = [
            path
            for path in root.iterdir()
            if path.is_dir() and not path.is_symlink() and RUN_ID.fullmatch(path.name)
        ]
    except FileNotFoundError:
        directories = []
    directories.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for directory in directories:
        try:
            result = read_result(root, directory.name)
            if not isinstance(result, dict) or not isinstance(result.get("workload"), dict):
                raise ValueError("Malformed run record")
            result["lifecycle"] = _lifecycle(directory, result.get("status", "partial"))
            result["observed_outcomes"] = _observed_outcomes(directory, result)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append({"run_id": directory.name, "diagnostic": _diagnostic(exc)})
            continue
        runs.append(_project_run(result))
    return {"generated_at": datetime.now(UTC).isoformat(), "runs": runs, "errors": errors}
