"""Rebuildable comparison index over the public ordinary result contract."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from .ordinary_runs import read_result
from .report_projection import compact_report


def parameter_identity(value: Any) -> str:
    """Compare exact JSON values without Python's boolean/numeric coercion."""
    return json.dumps(value, sort_keys=True, allow_nan=False)


def parameters(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Expose recorded setup/workload values, retaining lists as exact profiles."""
    fields: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            fields.update(parameters(item, path))
        else:
            fields[path] = item
    return fields


def rebuild_index(root: Path, *, compact: bool = False) -> dict[str, Any]:
    runs = []
    errors = []
    added_at: dict[str, float] = {}
    root.mkdir(parents=True, exist_ok=True)
    for directory in sorted(root.iterdir()):
        if not re.fullmatch(r"[0-9a-f]{32}", directory.name) or directory.is_symlink():
            continue
        if not directory.is_dir():
            continue
        try:
            run = read_result(root, directory.name)
            if compact:
                run = compact_report(run)
            if (
                run.get("schema_version") != 1
                or run.get("run_id") != directory.name
                or not isinstance(run.get("setup"), dict)
                or not isinstance(run.get("workload"), dict)
            ):
                raise ValueError("Invalid ordinary result")
            # The retained summary is authoritative. Never recalculate grades or timings.
            run["parameters"] = parameters(
                {key: run[key] for key in ("setup", "workload", "status", "evidence") if key in run}
            )
            runs.append(run)
            # Prefer directory creation time; setup time is stable on filesystems
            # without birth time, unlike directory mtime which changes during a run.
            created = getattr(directory.stat(), "st_birthtime", None)
            added_at[directory.name] = (
                created if created is not None else (directory / "setup.json").stat().st_mtime
            )
        except (OSError, ValueError, TypeError, AttributeError):
            errors.append({"run_id": directory.name, "error": "Unreadable ordinary run directory"})
    facets: dict[str, list[Any]] = {}
    for run in runs:
        for key, value in run["parameters"].items():
            values = facets.setdefault(key, [])
            if all(parameter_identity(value) != parameter_identity(item) for item in values):
                values.append(value)
    report = {
        "schema_version": 1,
        "runs": runs,
        "parameters": facets,
        "errors": errors,
        "added_at": added_at,
    }
    if compact:
        # Browser polling reads the directories without rewriting the full transcript index.
        return report
    # Concurrent readers can rebuild safely; no index is ever an authority for run data.
    with tempfile.NamedTemporaryFile(mode="w", dir=root, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.close()
            temporary.replace(root / "report-index.json")
        finally:
            temporary.unlink(missing_ok=True)
    return report


def comparison(
    root: Path,
    filters: list[str],
    sort: str = "run_id",
    descending: bool = False,
    run_ids: list[str] | None = None,
    *,
    compact: bool = False,
) -> dict[str, Any]:
    predicates = []
    for expression in filters:
        field, separator, value = expression.partition("=")
        if not separator or not field:
            raise ValueError('Filters must be parameter=JSON, e.g. status="partial"')
        try:
            predicates.append((field, parameter_identity(json.loads(value))))
        except ValueError as exc:
            raise ValueError("Filter value must be JSON") from exc
    report = rebuild_index(root, compact=compact)
    rows = report["runs"]
    if run_ids:
        rows = [row for row in rows if row["run_id"] in run_ids]
    for field, value in predicates:
        rows = [
            row
            for row in rows
            if field in row["parameters"] and parameter_identity(row["parameters"][field]) == value
        ]

    def sort_value(row: dict[str, Any]) -> tuple[int, Any]:
        value = row["run_id"] if sort == "run_id" else row["parameters"].get(sort)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (0, value)
        return (1, json.dumps(value, sort_keys=True))

    present = [row for row in rows if sort == "run_id" or row["parameters"].get(sort) is not None]
    absent = [row for row in rows if sort != "run_id" and row["parameters"].get(sort) is None]
    report["runs"] = sorted(present, key=sort_value, reverse=descending) + absent
    return report
