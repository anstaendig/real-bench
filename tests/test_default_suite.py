"""One public submission executes and retains the complete default suite."""

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_humaneval_runs import controlled_tools as coding_tools
from test_humaneval_runs import stage_problems
from test_ordinary_runs import _growing_fake, submission
from test_ruler_runs import controlled_tools as ruler_tools
from test_ruler_runs import source as ruler_source
from typer.testing import CliRunner

from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_cli import app
from llm_bench.ordinary_runs import Submission, read_result, request_cancellation

pytestmark = pytest.mark.usefixtures("external_tool_process_adapter")


def default_submission(results: Path | None = None) -> dict:
    """The full submission; with ``results`` its inputs are staged under results/setup."""
    payload = submission()
    payload["setup"]["declared"]["requests"]["context_window"] = 262144
    payload["workload"] = {
        "profile": "default-suite",
        "fresh_context": {
            "profile": "fresh-context-speed",
            "scenario": "fresh-context",
            "checkpoints": [4096, 16384, 32768, 65536, 131072, 196608],
            "output_targets": [256, 1024],
            "repetitions": 5,
            "warmup_repetitions": 1,
            "request_timeout_seconds": 900,
            "cache_configuration": {"prefix_caching": True},
            "cache_observations": [],
        },
        "growing_session": {
            "profile": "growing-session-speed",
            "scenario": "growing-session",
            "checkpoints": [4096, 16384, 32768, 65536, 131072, 196608],
            "output_targets": [256, 1024],
            "repetitions": 5,
            "warmup_repetitions": 1,
            "request_timeout_seconds": 900,
            "cache_configuration": {"prefix_caching": True},
            "cache_observations": [],
        },
        "coding": {
            "profile": "humaneval-plus",
            "problems_path": str(
                stage_problems(results) if results else "/pinned/problems.jsonl.gz"
            ),
        },
        "long_context": {
            "profile": "sampled-ruler",
            "source_path": str(results / "setup" / "pinned-ruler") if results else "/pinned/ruler",
        },
    }
    return payload


def test_default_suite_accepts_full_submission_without_forecast():
    payload = default_submission()

    workload = Submission.model_validate(payload).workload

    assert workload.profile == "default-suite"
    assert "forecast" not in type(workload).model_fields


def test_default_suite_rejects_forecast_input():
    payload = default_submission()
    payload["workload"]["forecast"] = {
        "source": {
            "kind": "pilot",
            "reference": "pilot-results.md",
            "setup_sha256": "0" * 64,
            "limitations": "Reduced pilot.",
        },
        "planning_target_seconds": 10800,
        "phases": {
            "fresh_context": {"observed_units": 2, "observed_seconds": 20},
            "growing_session": {"observed_units": 2, "observed_seconds": 30},
            "coding": {"observed_units": 1, "observed_seconds": 4},
            "long_context": {"observed_units": 3, "observed_seconds": 9},
        },
    }

    with pytest.raises(ValueError, match="forecast"):
        Submission.model_validate(payload)


def fresh_tool(command):
    if any(str(part).endswith("performance_inputs.py") for part in command):
        specification = json.loads(Path(command[command.index("--spec") + 1]).read_text())
        directory = Path(command[command.index("--directory") + 1])
        prepared = []
        for condition in specification["conditions"]:
            condition_directory = directory / condition["name"]
            condition_directory.mkdir()
            for phase, count in (("warmup", 1), ("measured", 5)):
                destination = (
                    condition_directory / "warmup/input.jsonl"
                    if phase == "warmup"
                    else condition_directory / "input.jsonl"
                )
                destination.parent.mkdir(exist_ok=True)
                rows = []
                for repetition in range(count):
                    nonce = f"{condition['name']}-{phase}-{repetition}"
                    rows.append(
                        {
                            "text": "Messages supplied in extra.messages",
                            "output_length": condition["output_target"],
                            "extra": {
                                **specification["request_settings"],
                                "messages": [{"role": "user", "content": nonce}],
                            },
                        }
                    )
                    prepared.append(
                        {
                            "condition": condition["name"],
                            "phase": phase,
                            "repetition": repetition,
                            "nonce": nonce,
                            "local_input_tokens": condition["input_tokens"],
                        }
                    )
                destination.write_text("".join(json.dumps(row) + "\n" for row in rows))
        (directory / "preparation.json").write_text(json.dumps({"observations": prepared}))
        return subprocess.CompletedProcess(command, 0, "", "")

    native = Path(command[command.index("--output-artifact-dir") + 1])
    native.mkdir()
    condition_directory = native.parent
    condition_name = (
        condition_directory.parent.name
        if condition_directory.name.startswith("repetition-")
        else (
            condition_directory.parent.name
            if condition_directory.name == "warmup"
            else condition_directory.name
        )
    )
    input_tokens, output_target = (int(value) for value in condition_name.split("-")[-2:])
    request_id = f"{condition_name}-{condition_directory.name}"
    raw = {
        "metadata": {"x_request_id": request_id},
        "status": 200,
        "responses": [
            {
                "packets": [
                    {
                        "name": "data",
                        "value": json.dumps(
                            {
                                "choices": [
                                    {"delta": {"content": "result"}, "finish_reason": "length"}
                                ],
                                "usage": {
                                    "prompt_tokens": input_tokens,
                                    "completion_tokens": output_target,
                                },
                            }
                        ),
                    },
                    {"name": "data", "value": "[DONE]"},
                ]
            }
        ],
    }
    measured = {
        "metadata": {"x_request_id": request_id},
        "metrics": {
            "time_to_first_token": {"value": 1000.0, "unit": "ms"},
            "request_latency": {"value": 2000.0, "unit": "ms"},
            "output_token_throughput_per_user": {
                "value": 80.0,
                "unit": "tokens/sec/user",
            },
            "usage_prompt_tokens": {"value": input_tokens, "unit": "tokens"},
            "usage_completion_tokens": {"value": output_target, "unit": "tokens"},
        },
    }
    (native / "profile_export_raw.jsonl").write_text(json.dumps(raw) + "\n")
    (native / "profile_export.jsonl").write_text(json.dumps(measured) + "\n")
    return subprocess.CompletedProcess(command, 0, "", "")


@pytest.mark.parametrize(
    "mode", ["success", "performance_failed", "coding_failed", "coding_outage", "ruler_failed"]
)
def test_default_suite_is_one_public_retained_lifecycle(tmp_path, monkeypatch, mode):
    results = tmp_path / "results"
    source = ruler_source(results)
    payload = default_submission(results)
    payload["setup"]["declared"]["model"]["tokenizer"] = str(source / "tokenizer")
    payload["workload"]["long_context"]["source_path"] = str(source)
    calls = []
    coding = coding_tools(calls)
    growing = _growing_fake(calls, {})
    # The integrated quality check exercises the corrected compatibility path,
    # not just the exact-count case.  The controlled endpoint owns a count one
    # token above submitted-tokenizer preparation, still within each ceiling.
    ruler = ruler_tools(endpoint_prompt_delta=1 if mode == "success" else 0)
    progress_snapshots = []

    def failed_native(command, status):
        native = Path(command[command.index("--output-artifact-dir") + 1])
        native.mkdir()
        native.joinpath("profile_export_raw.jsonl").write_text(
            json.dumps({"metadata": {"x_request_id": mode}, "status": status, "responses": []})
            + "\n"
        )
        return subprocess.CompletedProcess(command, 0, "", f"controlled {mode}")

    def external(command, **kwargs):
        run_directories = list(results.glob("[0-9a-f]" * 32))
        if run_directories:
            progress = read_result(results, run_directories[0].name)["phase_progress"]
            active = next(
                (phase for phase in progress["phases"] if phase["name"] == progress["current"]),
                None,
            )
            if active and active.get("live_outcomes"):
                progress_snapshots.append(active)
        rendered = " ".join(str(part) for part in command)
        if any(str(part).endswith("ruler_pilot.py") for part in command):
            return ruler(command, **kwargs)
        if command[:2] == ["docker", "version"] or command[0] == "docker":
            return coding(command, **kwargs)
        if any(str(part).endswith("performance_inputs.py") for part in command):
            specification = json.loads(Path(command[command.index("--spec") + 1]).read_text())
            return (
                fresh_tool(command) if "conditions" in specification else growing(command, **kwargs)
            )
        if mode in {"coding_failed", "coding_outage"} and "HumanEval_0" in rendered:
            return failed_native(command, 500 if mode == "coding_failed" else 0)
        if "HumanEval_" in rendered:
            return coding(command, **kwargs)
        if mode == "ruler_failed" and "ruler-niah_single_1-32768-0" in rendered:
            return failed_native(command, 500)
        if "ruler-" in rendered:
            return ruler(command, **kwargs)
        if "growing-session-" in rendered:
            return growing(command, **kwargs)
        if mode == "performance_failed" and "fresh-context-4096-256/repetition-000/" in rendered:
            return failed_native(command, 500)
        return fresh_tool(command)

    monkeypatch.setattr(subprocess, "run", external)
    client = TestClient(create_app(results))
    response = client.post("/api/runs", json=payload)
    assert response.status_code == 201, response.text
    result = response.json()

    expected_statuses = {
        "success": ["completed", "completed", "completed", "completed"],
        "performance_failed": ["partial", "completed", "completed", "completed"],
        "coding_failed": ["completed", "completed", "partial", "completed"],
        "coding_outage": ["completed", "completed", "partial", "skipped"],
        "ruler_failed": ["completed", "completed", "completed", "partial"],
    }[mode]
    assert result["status"] == ("completed" if mode == "success" else "partial")
    assert result["default_workload_complete"] is (mode == "success")
    assert [phase["status"] for phase in result["phase_progress"]["phases"]] == expected_statuses
    assert result["families"]["performance"] == (
        "partial" if mode == "performance_failed" else "completed"
    )
    assert result["families"]["coding"] == (
        "partial" if mode in {"coding_failed", "coding_outage"} else "completed"
    )
    assert result["families"]["long_context"] == (
        "completed" if mode in {"success", "performance_failed", "coding_failed"} else "partial"
    )
    assert result["coverage"]["fresh_context"]["accounted"] == 60
    if mode == "performance_failed":
        failed = [
            row
            for row in result["outcomes"]
            if row["phase"] == "fresh_context" and row["outcome"] == "failed"
        ]
        assert len(failed) == 1
        assert failed[0]["metrics"] == {}
    assert result["coverage"]["growing_session"]["accounted"] == 60
    assert result["coverage"]["coding"]["accounted"] == (1 if mode == "coding_outage" else 164)
    assert result["coverage"]["long_context"]["accounted"] == (
        0 if mode == "coding_outage" else 117
    )
    assert len(result["outcomes"]) == (284 if mode == "coding_outage" else 401)
    assert ("long_context" in {row["phase"] for row in result["outcomes"]}) is (
        mode != "coding_outage"
    )
    assert all(
        "observed_timing" in phase
        for phase in result["phase_progress"]["phases"]
        if phase["status"] != "skipped"
    )
    assert any(snapshot["accounted_units"] > 0 for snapshot in progress_snapshots)
    assert result["construction"]["sizing_requests"] == 70
    assert "forecast" not in result
    assert "forecast_comparison" not in result
    assert "derived_forecast" not in result["workload"]
    assert result["wall_seconds"] > 0
    assert all(
        isinstance(phase.get("elapsed_seconds"), (int, float)) and phase["elapsed_seconds"] >= 0
        for phase in result["phase_progress"]["phases"]
        if phase["status"] != "skipped"
    )
    assert result["coverage"]["coding"]["graded"] == (
        0 if mode == "coding_outage" else 163 if mode == "coding_failed" else 164
    )
    if mode == "coding_outage":
        assert result["coding"]["accuracy"] is None
    else:
        assert 0.49 < result["coding"]["accuracy"] <= 0.5
    if mode != "coding_outage":
        assert result["ruler"]["quality_by_context"]["196608"]["scored"] == 39
    if mode == "success":
        assert result["coding"]["identity"]["status"] == "verified"
        assert result["coding"]["identity"]["counts"]["verified"] == 164
        assert result["ruler"]["input_accounting"]["compatibility_counts"] == {
            "exact": 0,
            "endpoint-authoritative": 117,
            "incompatible": 0,
            "unavailable": 0,
        }
    assert len(result["fresh_context_curve"]) == 12
    assert len(result["growing_session_curve"]) == 12

    directory = results / result["run_id"]
    assert len(list(directory.glob("phases/*/*/summary.json"))) == (
        3 if mode == "coding_outage" else 4
    )
    assert (
        json.loads((directory / "setup.json").read_text())["declared"]["model"][
            "quantized_artifact"
        ]
        == "model-q4"
    )
    assert client.get("/api/runs/" + result["run_id"]).json() == result
    view = client.get("/runs/" + result["run_id"]).text
    expected_heading = (
        "Complete default-suite results" if mode == "success" else "Partial default-suite results"
    )
    assert expected_heading in view
    cli = CliRunner().invoke(app, ["result", result["run_id"], "--results", str(results)])
    assert cli.exit_code == 0
    assert json.loads(cli.stdout) == result
    if mode == "success":
        report = client.get("/api/reports", params=[("run_id", result["run_id"])]).json()
        assert report["errors"] == []
        assert len(report["runs"]) == 1
        indexed = report["runs"][0]
        assert {key: value for key, value in indexed.items() if key != "parameters"} == result
        assert indexed["coding"]["identity"]["status"] == "verified"
        assert indexed["ruler"]["input_accounting"] == result["ruler"]["input_accounting"]
        rebuilt = CliRunner().invoke(app, ["rebuild-index", "--results", str(results)])
        assert rebuilt.exit_code == 0, rebuilt.output
        assert json.loads(rebuilt.stdout) == report


def test_partial_default_suite_result_view_is_explicit(tmp_path):
    run_id = "a" * 32
    directory = tmp_path / run_id
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps({"status": "partial", "default_workload_complete": False})
    )
    (directory / "setup.json").write_text(json.dumps({"declared": {}}))
    (directory / "workload.json").write_text(json.dumps({"profile": "default-suite"}))

    response = TestClient(create_app(tmp_path)).get("/runs/" + run_id)

    assert response.status_code == 200
    assert "Partial default-suite results" in response.text
    assert "Complete default-suite results" not in response.text


def test_historical_default_suite_result_with_forecast_remains_readable(tmp_path):
    """Old result directories retain their forecast fields and stay readable."""
    run_id = "b" * 32
    directory = tmp_path / run_id
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "status": "completed",
                "default_workload_complete": True,
                "wall_seconds": 17488.255,
                "forecast": {"total_estimated_seconds": 18899.967},
                "forecast_comparison": {
                    "estimated_seconds": 18899.967,
                    "observed_seconds": 17488.255,
                    "difference_seconds": -1411.712,
                    "planning_target_seconds": 10800,
                },
            }
        )
    )
    (directory / "setup.json").write_text(json.dumps({"declared": {}}))
    (directory / "workload.json").write_text(
        json.dumps(
            {
                "profile": "default-suite",
                "evidence": "run",
                "derived_forecast": {"total_estimated_seconds": 18899.967},
            }
        )
    )

    result = read_result(tmp_path, run_id)
    assert result["forecast"]["total_estimated_seconds"] == 18899.967
    assert result["forecast_comparison"]["planning_target_seconds"] == 10800
    assert result["workload"]["derived_forecast"]["total_estimated_seconds"] == 18899.967

    response = TestClient(create_app(tmp_path)).get("/api/runs/" + run_id)
    assert response.status_code == 200
    assert response.json() == result

    cli = CliRunner().invoke(app, ["result", run_id, "--results", str(tmp_path)])
    assert cli.exit_code == 0
    assert json.loads(cli.stdout) == result


def test_ordinary_phase_failure_continues_then_shared_cancellation_stops_suite(
    tmp_path, monkeypatch
):
    results = tmp_path / "results"
    payload = default_submission(results)
    calls = []

    def external(command, **kwargs):
        calls.append(command)
        assert any(str(part).endswith("performance_inputs.py") for part in command)
        specification = json.loads(Path(command[command.index("--spec") + 1]).read_text())
        if "conditions" in specification:
            return subprocess.CompletedProcess(command, 1, "", "controlled preparation failure")
        run_id = next(
            path.name for path in results.iterdir() if path.is_dir() and path.name != "setup"
        )
        request_cancellation(results, run_id)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", external)
    result = TestClient(create_app(results)).post("/api/runs", json=payload).json()

    assert result["status"] == "partial"
    assert result["diagnostic"] == "Cancelled"
    assert [phase["status"] for phase in result["phase_progress"]["phases"]] == [
        "partial",
        "partial",
        "skipped",
        "skipped",
    ]
    assert result["coverage"]["fresh_context"]["accounted"] == 60
    assert result["coverage"]["growing_session"]["accounted"] == 0
    assert len(calls) == 2
    assert (results / result["run_id"] / "cancel.json").is_file()


def test_endpoint_outage_stops_every_later_default_suite_phase(tmp_path, monkeypatch):
    results = tmp_path / "results"
    payload = default_submission(results)
    calls = []

    def external(command, **kwargs):
        calls.append(command)
        if any(str(part).endswith("performance_inputs.py") for part in command):
            return fresh_tool(command)
        native = Path(command[command.index("--output-artifact-dir") + 1])
        native.mkdir()
        native.joinpath("profile_export_raw.jsonl").write_text(
            json.dumps({"metadata": {"x_request_id": "outage"}, "status": 0, "responses": []})
            + "\n"
        )
        return subprocess.CompletedProcess(command, 0, "", "controlled endpoint outage")

    monkeypatch.setattr(subprocess, "run", external)
    result = TestClient(create_app(results)).post("/api/runs", json=payload).json()

    assert result["diagnostic"] == "Endpoint became unreachable"
    assert [phase["status"] for phase in result["phase_progress"]["phases"]] == [
        "partial",
        "skipped",
        "skipped",
        "skipped",
    ]
    assert len(calls) == 2
    assert result["coverage"]["fresh_context"]["accounted"] == 60
    assert all(row["phase"] == "fresh_context" for row in result["outcomes"])
