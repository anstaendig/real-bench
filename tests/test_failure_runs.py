"""Public ordinary-run failure lifecycle regressions."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_humaneval_runs import coding_submission
from test_humaneval_runs import controlled_tools as coding_tools
from test_ordinary_runs import _growing_fake, fresh_submission, growing_submission
from test_ruler_runs import controlled_tools as ruler_tools
from test_ruler_runs import ruler_submission, source

from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_runs import CODING_OUTPUT_BUDGET, RunCancelled, execute_tool

pytestmark = pytest.mark.usefixtures("external_tool_process_adapter")


@pytest.mark.skipif(os.name != "posix", reason="SIGINT process-group behavior")
def test_keyboard_interrupt_persists_durable_cancellation(tmp_path):
    marker = tmp_path / "cancel.json"
    timer = threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGINT))
    timer.start()
    try:
        with pytest.raises(RunCancelled):
            execute_tool(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                tmp_path,
                1,
                900,
                cancel_marker=marker,
            )
    finally:
        timer.cancel()
    assert json.loads(marker.read_text())["reason"] == "keyboard_interrupt"


def _write_native(directory: Path, request_id: str, input_tokens: int = 4096) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    packet = json.dumps(
        {
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": 256},
            "choices": [{"delta": {"content": "done"}, "finish_reason": "length"}],
        }
    )
    raw = {
        "metadata": {"x_request_id": request_id},
        "status": 200,
        "responses": [
            {"packets": [{"name": "data", "value": packet}, {"name": "data", "value": "[DONE]"}]}
        ],
    }
    metrics = {
        "metadata": {"x_request_id": request_id},
        "metrics": {
            "time_to_first_token": {"value": 1.0, "unit": "ms"},
            "request_latency": {"value": 2.0, "unit": "ms"},
            "output_token_throughput_per_user": {"value": 3.0, "unit": "tokens/sec/user"},
            "usage_prompt_tokens": {"value": input_tokens, "unit": "tokens"},
            "usage_completion_tokens": {"value": 256, "unit": "tokens"},
        },
    }
    (directory / "profile_export_raw.jsonl").write_text(json.dumps(raw) + "\n")
    (directory / "profile_export.jsonl").write_text(json.dumps(metrics) + "\n")


def _fresh_preparation(command: list[str]) -> None:
    root = Path(command[command.index("--directory") + 1])
    spec = json.loads(Path(command[command.index("--spec") + 1]).read_text())
    condition = spec["conditions"][0]
    directory = root / condition["name"]
    (directory / "warmup").mkdir(parents=True)
    row = {
        "text": "fixture",
        "output_length": 256,
        "extra": {"messages": [{"role": "user", "content": "x"}]},
    }
    (directory / "warmup" / "input.jsonl").write_text(json.dumps(row) + "\n")
    (directory / "input.jsonl").write_text((json.dumps(row) + "\n") * 5)
    observations = [
        {
            "condition": condition["name"],
            "phase": phase,
            "repetition": index,
            "nonce": f"{phase}-{index}",
            "local_input_tokens": 4096,
        }
        for phase, count in (("warmup", 1), ("measured", 5))
        for index in range(count)
    ]
    (root / "preparation.json").write_text(json.dumps({"observations": observations}) + "\n")


def test_fresh_cancellation_retains_completed_requests_and_stops_later_work(tmp_path, monkeypatch):
    calls: list[str] = []

    def external(command, **kwargs):
        if any(str(part).endswith("performance_inputs.py") for part in command):
            _fresh_preparation(command)
            return subprocess.CompletedProcess(command, 0, "", "")
        native = Path(command[command.index("--output-artifact-dir") + 1])
        repetition = native.parent.name
        _write_native(native, repetition)
        calls.append(repetition)
        if repetition == "repetition-001":
            native.parents[2].joinpath("cancel.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", external)
    result = TestClient(create_app(tmp_path)).post("/api/runs", json=fresh_submission(5000)).json()
    assert calls == ["warmup", "repetition-000", "repetition-001"]
    measured = [row for row in result["outcomes"] if row["condition"] == "fresh-context-4096-256"]
    assert [row["outcome"] for row in measured] == [
        "success",
        "success",
        "missing",
        "missing",
        "missing",
    ]
    assert result["status"] == "partial" and result["diagnostic"] == "Cancelled"


def test_fresh_timeout_continues_to_later_scored_requests(tmp_path, monkeypatch):
    calls: list[str] = []

    def external(command, **kwargs):
        if any(str(part).endswith("performance_inputs.py") for part in command):
            _fresh_preparation(command)
            return subprocess.CompletedProcess(command, 0, "", "")
        native = Path(command[command.index("--output-artifact-dir") + 1])
        calls.append(native.parent.name)
        if native.parent.name == "repetition-001":
            raise subprocess.TimeoutExpired(command, 900)
        _write_native(native, native.parent.name)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", external)
    result = TestClient(create_app(tmp_path)).post("/api/runs", json=fresh_submission(5000)).json()
    assert calls == [
        "warmup",
        "repetition-000",
        "repetition-001",
        "repetition-002",
        "repetition-003",
        "repetition-004",
    ]
    rows = [row for row in result["outcomes"] if row["condition"] == "fresh-context-4096-256"]
    assert [row["outcome"] for row in rows] == [
        "success",
        "failed",
        "success",
        "success",
        "success",
    ]


def test_growing_outage_stops_later_sessions(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    baseline = _growing_fake(calls, {})

    def external(command, **kwargs):
        result = baseline(command, **kwargs)
        if "profile" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            native.joinpath("profile_export_raw.jsonl").write_text(
                json.dumps({"metadata": {"x_request_id": "outage"}, "status": 0, "responses": []})
                + "\n"
            )
        return result

    monkeypatch.setattr(subprocess, "run", external)
    result = (
        TestClient(create_app(tmp_path)).post("/api/runs", json=growing_submission(5000)).json()
    )
    profiles = [command for command in calls if "profile" in command]
    assert len(profiles) == 1
    assert result["status"] == "partial"
    assert result["diagnostic"] == "Endpoint became unreachable"


def test_humaneval_timeout_continues_to_later_tasks(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    baseline = coding_tools(calls)

    def external(command, **kwargs):
        if (
            "profile" in command
            and Path(command[command.index("--output-artifact-dir") + 1]).parent.name
            == "HumanEval_0"
        ):
            raise subprocess.TimeoutExpired(command, 900)
        return baseline(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", external)
    result = (
        TestClient(create_app(tmp_path)).post("/api/runs", json=coding_submission(tmp_path)).json()
    )
    assert result["outcomes"][0]["outcome"] == "generation_failed"
    assert result["outcomes"][1]["outcome"] in {"correct", "incorrect"}
    assert len([command for command in calls if "profile" in command]) == 163


def test_humaneval_malformed_or_missing_native_export_continues(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    baseline = coding_tools(calls)

    def external(command, **kwargs):
        if "profile" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            if native.parent.name == "HumanEval_0":
                return subprocess.CompletedProcess(command, 1, "", "tool error without raw export")
        return baseline(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", external)
    result = (
        TestClient(create_app(tmp_path)).post("/api/runs", json=coding_submission(tmp_path)).json()
    )
    assert result["outcomes"][0]["outcome"] == "generation_failed"
    assert result["outcomes"][1]["outcome"] in {"correct", "incorrect"}
    assert len([command for command in calls if "profile" in command]) == 163


def test_humaneval_cancellation_retains_completed_generation(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    baseline = coding_tools(calls)

    def external(command, **kwargs):
        result = baseline(command, **kwargs)
        if "profile" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            if native.parent.name == "HumanEval_0":
                native.parents[1].joinpath("cancel.json").write_text("{}")
        return result

    monkeypatch.setattr(subprocess, "run", external)
    result = (
        TestClient(create_app(tmp_path)).post("/api/runs", json=coding_submission(tmp_path)).json()
    )
    assert result["status"] == "partial" and result["diagnostic"] == "Cancelled"
    assert result["outcomes"][0]["outcome"] == "grading_infrastructure"
    assert result["coverage"]["coding"]["generated"] == 1


def test_ruler_timeout_continues_to_later_requests(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    baseline = ruler_tools()

    def external(command, **kwargs):
        if "profile" in command:
            calls.append(command)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(command, 900)
        return baseline(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", external)
    result = (
        TestClient(create_app(tmp_path))
        .post("/api/runs", json=ruler_submission(source(tmp_path)))
        .json()
    )
    assert result["outcomes"][0]["outcome"] == "failed"
    assert result["outcomes"][1]["outcome"] == "success"
    assert len(calls) == 117


def test_humaneval_known_output_reservation_exclusion_is_unsupported(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("known context exclusion invoked an external tool")

    payload = coding_submission(tmp_path)
    payload["setup"]["declared"]["requests"]["context_window"] = CODING_OUTPUT_BUDGET
    monkeypatch.setattr(subprocess, "run", forbidden)
    result = TestClient(create_app(tmp_path)).post("/api/runs", json=payload).json()
    assert result["status"] == "completed"
    assert result["coverage"]["coding"] == {
        "expected": 164,
        "attempted": 0,
        "generated": 0,
        "graded": 0,
    }
    assert {row["outcome"] for row in result["outcomes"]} == {"unsupported"}


@pytest.mark.parametrize("native_case", ["valid", "mismatched_input", "invalid_score"])
def test_ruler_cancelled_tool_validates_completed_native_result(tmp_path, monkeypatch, native_case):
    upstream = source(tmp_path / "results")
    if native_case == "invalid_score":
        constants = upstream / "scripts/eval/synthetic/constants.py"
        with constants.open("a") as stream:
            stream.write(
                "\nfor task in TASKS.values():\n    task['metric_fn'] = lambda *args: 101\n"
            )
    baseline = ruler_tools()
    requests = []

    def external(command, **kwargs):
        result = baseline(command, **kwargs)
        if "profile" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            requests.append(native.parent.name)
            if native_case == "mismatched_input":
                exported = native / "profile_export_raw.jsonl"
                raw = json.loads(exported.read_text())
                packet = raw["responses"][0]["packets"][0]
                payload = json.loads(packet["value"])
                payload["usage"]["prompt_tokens"] = 32769
                packet["value"] = json.dumps(payload)
                exported.write_text(json.dumps(raw) + "\n")
            raise RunCancelled("Cancelled after native export")
        return result

    monkeypatch.setattr(subprocess, "run", external)
    response = TestClient(create_app(tmp_path / "results")).post(
        "/api/runs", json=ruler_submission(upstream)
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert len(requests) == 1
    assert result["status"] == "partial"
    observed = result["outcomes"][0]
    assert observed["outcome"] == ("success" if native_case == "valid" else "cancelled")
    assert observed["score"] == (100.0 if native_case == "valid" else None)
    assert result["coverage"]["long_context"]["scored"] == int(native_case == "valid")
