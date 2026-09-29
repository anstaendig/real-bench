"""Dashboard state is derived from retained run evidence, never guessed."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_default_suite import fresh_tool
from test_ordinary_runs import fresh_submission

from llm_bench.dashboard import dashboard_snapshot
from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_runs import (
    create_run_directory,
    finish_run,
    finished_on_exit,
    read_result,
    run_lock_held,
    save,
)


def _run(root: Path, profile: str = "fresh-context-speed") -> tuple[str, Path]:
    run_id, directory = create_run_directory(root)
    save(directory / "setup.json", {"declared": {"model": {"source": "test"}}})
    save(directory / "workload.json", {"profile": profile, "evidence": "run"})
    save(directory / "summary.json", {"run_id": run_id, "status": "partial", "outcomes": []})
    return run_id, directory


def test_active_standalone_uses_retained_outcomes_without_changing_result(tmp_path: Path) -> None:
    run_id, directory = _run(tmp_path)
    row = {"outcome": "success", "metrics": {"ttft_ms": 42}}
    save(directory / "outcomes.json", [row])
    original = read_result(tmp_path, run_id)

    snapshot = dashboard_snapshot(tmp_path)
    run = snapshot["runs"][0]
    assert run["lifecycle"] == {"state": "active", "verified": True}
    assert run["observed_outcomes"] == [row]
    assert "outcomes" not in run
    assert read_result(tmp_path, run_id) == original

    finish_run(directory)
    assert dashboard_snapshot(tmp_path)["runs"][0]["lifecycle"] == {
        "state": "finished",
        "verified": True,
    }


def test_stale_or_legacy_partial_is_unverified(tmp_path: Path) -> None:
    _, directory = _run(tmp_path)
    finish_run(directory)
    marker = json.loads((directory / "execution-state.json").read_text())
    marker["state"] = "running"
    save(directory / "execution-state.json", marker)
    assert dashboard_snapshot(tmp_path)["runs"][0]["lifecycle"]["state"] == "unverified"

    (directory / "execution-state.json").unlink()
    assert dashboard_snapshot(tmp_path)["runs"][0]["lifecycle"]["state"] == "unverified"


def test_suite_combines_finished_phase_and_current_retained_rows(tmp_path: Path) -> None:
    run_id, directory = _run(tmp_path, "default-suite")
    phases = [
        {"name": "fresh_context", "status": "completed", "accounted_units": 1},
        {"name": "growing_session", "status": "running", "accounted_units": 0},
    ]
    save(
        directory / "summary.json",
        {
            "run_id": run_id,
            "status": "partial",
            "outcomes": [{"phase": "fresh_context", "outcome": "success", "metrics": {}}],
            "phase_progress": {
                "current": "growing_session",
                "completed": 1,
                "total": 2,
                "phases": phases,
            },
        },
    )
    # The child identity is the fixed suite phase identity, not a fresh UUID.
    from llm_bench.default_suite import _phase_id

    child = directory / "phases" / "growing_session" / _phase_id("growing_session")
    child.mkdir(parents=True)
    save(child / "summary.json", {"coverage": {"growing_session": {"accounted": 1}}})
    save(child / "outcomes.json", [{"outcome": "failed", "metrics": {}}])

    result = dashboard_snapshot(tmp_path)["runs"][0]
    assert [row["phase"] for row in result["observed_outcomes"]] == [
        "fresh_context",
        "growing_session",
    ]
    assert result["phase_progress"]["phases"][1]["accounted_units"] == 1


def test_finished_summary_wins_over_stale_outcomes_file(tmp_path: Path) -> None:
    run_id, directory = _run(tmp_path)
    save(
        directory / "summary.json",
        {"run_id": run_id, "status": "partial", "outcomes": [{"outcome": "failed"}]},
    )
    save(directory / "outcomes.json", [{"outcome": "missing"}])
    finish_run(directory)
    assert dashboard_snapshot(tmp_path)["runs"][0]["observed_outcomes"] == [{"outcome": "failed"}]


def test_suite_does_not_double_count_a_phase_merged_into_parent(tmp_path: Path) -> None:
    run_id, directory = _run(tmp_path, "default-suite")
    from llm_bench.default_suite import _phase_id

    child = directory / "phases" / "fresh_context" / _phase_id("fresh_context")
    child.mkdir(parents=True)
    save(child / "summary.json", {"coverage": {}})
    save(child / "outcomes.json", [{"outcome": "success", "metrics": {}}])
    save(
        directory / "summary.json",
        {
            "run_id": run_id,
            "status": "partial",
            "outcomes": [{"phase": "fresh_context", "outcome": "success", "metrics": {}}],
            "phase_progress": {
                "current": "fresh_context",
                "completed": 1,
                "total": 1,
                "phases": [{"name": "fresh_context", "status": "completed"}],
            },
        },
    )
    assert len(dashboard_snapshot(tmp_path)["runs"][0]["observed_outcomes"]) == 1


def test_unreadable_run_is_reported(tmp_path: Path) -> None:
    run_id, directory = _run(tmp_path)
    save(directory / "setup.json", [])
    save(directory / "workload.json", [])
    snapshot = dashboard_snapshot(tmp_path)
    assert snapshot["runs"] == []
    assert snapshot["errors"][0]["run_id"] == run_id


def test_process_death_releases_active_lock(tmp_path: Path) -> None:
    script = (
        "from pathlib import Path; "
        "from llm_bench.ordinary_runs import create_run_directory, save; "
        f"run_id, directory = create_run_directory(Path({str(tmp_path)!r})); "
        "save(directory / 'setup.json', {}); "
        "save(directory / 'workload.json', {'profile': 'fresh-context-speed'}); "
        "save(directory / 'summary.json', {'run_id': run_id, 'status': 'partial', 'outcomes': []})"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    assert dashboard_snapshot(tmp_path)["runs"][0]["lifecycle"] == {
        "state": "unverified",
        "verified": False,
    }


def test_runner_exception_releases_its_lock(tmp_path: Path) -> None:
    @finished_on_exit
    def fail() -> None:
        _, directory = _run(tmp_path)
        assert run_lock_held(directory)
        raise RuntimeError("unhandled test failure")

    with pytest.raises(RuntimeError, match="unhandled test failure"):
        fail()
    directory = next(tmp_path.iterdir())
    assert not run_lock_held(directory)
    assert dashboard_snapshot(tmp_path)["runs"][0]["lifecycle"]["state"] == "finished"


@pytest.mark.usefixtures("external_tool_process_adapter")
def test_fresh_run_exposes_completed_condition_before_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[dict] = []

    def external(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if (
            "--output-artifact-dir" in command
            and "fresh-context-4096-1024" in str(command)
            and not observed
        ):
            run = dashboard_snapshot(tmp_path)["runs"][0]
            observed.append(run)
        return fresh_tool(command)

    monkeypatch.setattr(subprocess, "run", external)
    response = TestClient(create_app(tmp_path)).post("/api/runs", json=fresh_submission())
    assert response.status_code == 201, response.text
    assert observed[0]["lifecycle"]["state"] == "active"
    assert len(observed[0]["observed_outcomes"]) == 5
    assert any(row["metrics"].get("ttft") for row in observed[0]["observed_outcomes"])
