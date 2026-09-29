"""Actual child processes exercise the shared external-tool lifecycle boundary."""

import json
import os
import signal
import sys
import time
from pathlib import Path

import psutil
import pytest

from llm_bench.ordinary_runs import RunCancelled, execute_tool


def _stop_test_child(pid_file: Path) -> None:
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group ownership")
def test_cancellation_preserves_logs_and_stops_resistant_descendant(tmp_path):
    marker = tmp_path / "cancel.json"
    pid_file = tmp_path / "child.pid"
    ready = tmp_path / "child.ready"
    child = (
        "import signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).touch(); time.sleep(60)"
    )
    script = tmp_path / "owned.py"
    script.write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        f"Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        f"while not Path({str(ready)!r}).exists(): time.sleep(0.01)\n"
        "print('retained before cancellation', flush=True)\n"
        "print('diagnostic before cancellation', file=sys.stderr, flush=True)\n"
        f"Path({str(marker)!r}).touch()\n"
        "time.sleep(60)\n"
    )
    try:
        with pytest.raises(RunCancelled):
            execute_tool([sys.executable, str(script)], tmp_path, 1, 900, cancel_marker=marker)
        child_pid = int(pid_file.read_text())
        deadline = time.monotonic() + 3
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.02)
        assert not psutil.pid_exists(child_pid) or (
            psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        ), "Owned descendant continues after cancellation"
        execution = json.loads((tmp_path / "execution.json").read_text())
        assert execution["interrupted"] is True
        assert "retained before cancellation" in execution["stdout"]
        assert "diagnostic before cancellation" in execution["stderr"]
    finally:
        _stop_test_child(pid_file)


def test_process_deadline_retains_output_and_effective_timeout(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    import llm_bench.ordinary_runs as runs

    ticks = iter([0.0, 122.0])
    monkeypatch.setattr(runs, "time", SimpleNamespace(monotonic=lambda: next(ticks, 122.0)))
    command = [
        sys.executable,
        "-c",
        "import time; print('work before deadline', flush=True); time.sleep(60)",
    ]
    with pytest.raises(subprocess.TimeoutExpired):
        execute_tool(command, tmp_path, 1, 1)
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert execution["timed_out"] is True
    assert execution["timeout_seconds"] == 121
    assert execution["elapsed_seconds"] == 122.0
    assert "work before deadline" in execution["stdout"]
    assert execution["exit_code"] is not None


@pytest.mark.skipif(os.name != "posix", reason="Executable fake Docker command")
def test_cancellation_removes_only_named_owned_grading_container(tmp_path, monkeypatch):
    marker = tmp_path / "cancel.json"
    removed = tmp_path / "removed.json"
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, time\nfrom pathlib import Path\n"
        "if sys.argv[1] == 'run':\n"
        "    print('grading started', flush=True)\n"
        f"    Path({str(marker)!r}).touch()\n"
        "    time.sleep(60)\n"
        "elif sys.argv[1] == 'rm':\n"
        f"    Path({str(removed)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(RunCancelled):
        execute_tool(
            ["docker", "run", "--rm", "--name", "real-bench-test-owned", "fake-image"],
            tmp_path,
            1,
            900,
            cancel_marker=marker,
        )
    assert json.loads(removed.read_text()) == ["rm", "--force", "real-bench-test-owned"]
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert "grading started" in execution["stdout"]
    assert execution["interrupted"] is True


@pytest.mark.skipif(os.name != "posix", reason="Executable fake upstream tool")
@pytest.mark.parametrize("completed_native", [False, True])
def test_api_cancellation_is_retained_and_readable_from_cli_and_view(
    tmp_path, monkeypatch, completed_native
):
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient
    from test_failure_runs import _write_native
    from test_ordinary_runs import submission
    from typer.testing import CliRunner

    from llm_bench.ordinary_api import create_app
    from llm_bench.ordinary_cli import app

    run_id = "129" + "0" * 29
    results = tmp_path / "results"
    pid_file = tmp_path / "tool.pid"
    retained = tmp_path / "completed-native"
    _write_native(retained, "completed-request")
    copy_native = (
        f"import shutil, sys; shutil.copytree({str(retained)!r}, "
        "sys.argv[sys.argv.index('--output-artifact-dir') + 1])\n"
        if completed_native
        else ""
    )
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import os, time\nfrom pathlib import Path\n"
        + copy_native
        + "print('diagnostic secret-124', flush=True)\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    client = TestClient(create_app(results))
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            client.post,
            "/api/runs",
            json=submission(),
            headers={"X-Real-Bench-Run-Id": run_id},
        )
        try:
            deadline = time.monotonic() + 10
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert pid_file.exists(), "Owned tool did not start"
            cancelled = client.post(f"/api/runs/{run_id}/cancel")
            assert cancelled.status_code == 200
            assert cancelled.json()["cancellation_requested"] is True
            response = pending.result(timeout=10)
            assert response.status_code == 201, response.text
            result = response.json()
            assert result["status"] == "partial"
            observed = result["outcomes"][0]
            assert observed["outcome"] == ("success" if completed_native else "cancelled")
            if completed_native:
                assert observed["metrics"]["ttft"]["value"] == 1.0
                assert observed["actual_output_tokens"] == 256
            else:
                assert observed["metrics"] == {}
            readback = client.get(f"/api/runs/{run_id}")
            assert readback.json() == result
            cli = CliRunner().invoke(app, ["result", run_id, "--results", str(results)])
            assert cli.exit_code == 0, cli.output
            assert json.loads(cli.output) == result
            view = client.get(f"/runs/{run_id}")
            assert "partial" in view.text
            assert observed["outcome"] in view.text
            evidence = (results / run_id / "execution.json").read_text()
            assert "diagnostic" in evidence and "secret-124" not in evidence
        finally:
            _stop_test_child(pid_file)
