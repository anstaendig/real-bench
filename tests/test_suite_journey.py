"""The complete default suite driven end to end through the public commands.

Setup runs as ``bench setup`` (resumable, with unknown metadata left explicit),
finishes with the composed prepared suite submission and its launch command
without starting any benchmark, and launch runs as ``bench run`` against the
prepared directory: explicit reasoning and reviewed sampling, automatic final
prep/validation, the ordinary four-phase lifecycle, and cancellation, endpoint
outage, authentication, and independent result reads. Preparation and scored
runs are served by a controlled ``subprocess.run``: no network, no native
tool, no model weights, no Docker daemon.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import fake_preparation
from fastapi.testclient import TestClient
from test_default_suite import fresh_tool
from test_humaneval_runs import controlled_tools as coding_tools
from test_ordinary_runs import _growing_fake
from test_ruler_runs import controlled_tools as ruler_tools
from typer.testing import CliRunner

from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_cli import app

pytestmark = pytest.mark.usefixtures("external_tool_process_adapter")

RUN_CONFIG = json.dumps(
    {
        "reasoning_effort": "xhigh",
        "temperature": 0.5,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "presence_penalty": None,
        "repetition_penalty": None,
    }
)


def setup_answers(partial: bool = False) -> dict[str, Any]:
    """The full nonsecret answers, or the two the resume flow starts from."""
    if partial:
        return {
            "requests.base_url": "http://localhost:8080",
            "model.tokenizer": "org/model",
        }
    return {
        "hardware.accelerator": "Test GPU",
        "hardware.count": 1,
        "hardware.memory": "32 GiB",
        "hardware.os": "Linux",
        "model.tokenizer": "org/model",
        "model.template_overrides": {},
        "serving.launch_command": "serve --api-key secret-124",
        "serving.environment": {"API_KEY": "secret-124", "HF_TOKEN": "hf_sensitive_124"},
        "requests.base_url": "http://localhost:8080",
        "requests.model_id": "test-model",
        "requests.context_window": 262144,
        "requests.generation": {"temperature": 0.5},
        "requests.reasoning": {"chat_template_kwargs": {"enable_thinking": True}},
    }


def run_setup(directory: Path, answers: dict[str, Any]) -> Any:
    return CliRunner().invoke(
        app,
        [
            "setup",
            "--directory",
            str(directory),
            "--no-interactive",
            "--answers",
            json.dumps(answers),
        ],
    )


def summary_of(result: Any) -> dict[str, Any]:
    return json.loads(result.output[result.output.index("{") :])


def run_suite(
    summary: dict[str, Any],
    results: Path,
    run_config: str,
    run_id: str | None = None,
) -> Any:
    """Launch the composed suite exactly as the setup summary advertises it."""
    command = [
        "run",
        summary["prepared_submissions"]["default-suite"],
        "--results",
        str(results),
        "--setup-directory",
        str(results / "setup"),
        "--run-config",
        run_config,
    ]
    if run_id is not None:
        command += ["--run-id", run_id]
    return CliRunner().invoke(app, command)


def dispatcher(results: Path, mode: str = "success") -> tuple[Any, list[list[str]]]:
    """One controlled ``subprocess.run``: preparation, gates, and the four phases.

    ``mode`` is ``success``, ``cancel`` (write a cancellation marker into the
    running suite at the first growing request), or ``outage`` (report an
    unreachable endpoint at the first growing request). Docker is stateful (a
    pulled image stays present) so the quality launch gate verifies the grading
    image at launch; everything else routes to the per-phase controlled tools.
    """
    calls: list[list[str]] = []
    pulled: set[str] = set()
    coding = coding_tools(calls)
    growing = _growing_fake(calls, {})
    ruler = ruler_tools()

    def external(command, **kwargs):
        calls.append([str(part) for part in command])
        args = [str(part) for part in command]
        rendered = " ".join(args)
        if any(p.endswith("ruler_source_prepare.py") for p in args):
            return fake_preparation(command, **kwargs)
        if any(p.endswith("speed_prepare.py") for p in args):
            return fake_preparation(command, **kwargs)
        if args and args[0] == "git" and "-C" in args:
            return fake_preparation(command, **kwargs)
        if "-c" in args:
            return subprocess.CompletedProcess(command, 0, "0.12.0\n5.17.0\n", "")
        if args and args[0] == "docker":
            if args[1] == "version":
                return subprocess.CompletedProcess(command, 0, "", "controlled")
            if args[1] == "pull":
                pulled.add(args[-1])
                return subprocess.CompletedProcess(command, 0, "", f"pulled {args[-1]}")
            if args[1] == "image":
                return subprocess.CompletedProcess(
                    command, 0, "[]" if args[-1] not in pulled else "[{}]", ""
                )
            if "uname" in rendered:
                return subprocess.CompletedProcess(command, 0, "x86_64\n", "")
            return coding(command, **kwargs)
        if any(p.endswith("ruler_pilot.py") for p in args):
            return ruler(command, **kwargs)
        if any(p.endswith("performance_inputs.py") for p in args):
            specification = json.loads(Path(command[command.index("--spec") + 1]).read_text())
            if "conditions" in specification:
                return fresh_tool(command)
            return growing(command, **kwargs)
        if "HumanEval_" in rendered:
            return coding(command, **kwargs)
        if "ruler-" in rendered:
            return ruler(command, **kwargs)
        if "growing-session-" in rendered:
            if mode == "cancel":
                run_dir = next(results.glob("[0-9a-f]" * 32))
                (run_dir / "cancel.json").write_text(json.dumps({"requested": True}))
                return growing(command, **kwargs)
            if mode == "outage":
                native = Path(command[command.index("--output-artifact-dir") + 1])
                native.mkdir()
                (native / "profile_export_raw.jsonl").write_text(
                    json.dumps(
                        {"metadata": {"x_request_id": "outage"}, "status": 0, "responses": []}
                    )
                    + "\n"
                )
                return subprocess.CompletedProcess(command, 0, "", "controlled endpoint outage")
            return growing(command, **kwargs)
        return fresh_tool(command)

    return external, calls


def results_readables(result: dict[str, Any], results: Path) -> None:
    """Independent CLI and API reads agree on a retained suite result."""
    client = TestClient(create_app(results))
    assert client.get("/api/runs/" + result["run_id"]).json() == result
    view = client.get("/runs/" + result["run_id"]).text
    heading = (
        "Complete default-suite results"
        if result["status"] == "completed"
        else "Partial default-suite results"
    )
    assert heading in view
    cli = CliRunner().invoke(app, ["result", result["run_id"], "--results", str(results)])
    assert cli.exit_code == 0
    assert json.loads(cli.stdout) == result


def assert_launch_gates(result: dict[str, Any], results: Path) -> None:
    run_dir = results / result["run_id"]
    for name in (
        "speed-preparation.json",
        "quality-preparation.json",
        "long-context-preparation.json",
    ):
        assert json.loads((run_dir / name).read_text())["prepared"] is True
    # The test engine is not gated, so no live compatibility probe is retained.
    assert not (run_dir / "compatibility.json").exists()


def test_setup_resumes_and_surfaces_the_suite_without_running(tmp_path, monkeypatch):
    """Two setup steps: the first records progress and reports the gap, the
    second resumes and ends with the saved prepared suite submission and its
    exact launch command — never starting the benchmark."""
    monkeypatch.setattr(subprocess, "run", dispatcher(tmp_path / "results")[0])
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"

    first = run_setup(setup_dir, setup_answers(partial=True))
    assert first.exit_code != 0
    assert "execution-critical" in first.output
    for gap in ("model.template_overrides", "requests.generation", "requests.reasoning"):
        assert gap in first.output

    second = run_setup(setup_dir, setup_answers())
    assert second.exit_code == 0, second.output
    summary = summary_of(second)
    assert summary["resumed"] is True
    assert summary["critical_complete"] is True
    assert summary["default_suite_ready"] is True
    statuses = {entry["name"]: entry["status"] for entry in summary["preparations"]}
    assert statuses["performance_tools"] == "complete"
    assert statuses["tokenizer"] == "complete"
    assert statuses["datasets"] == "complete"
    assert statuses["grading_container"] == "complete"
    assert statuses["long_context"] == "complete"
    submissions = summary["prepared_submissions"]
    assert set(submissions) == {
        "fresh-context",
        "growing-session",
        "humaneval-plus",
        "sampled-ruler",
        "default-suite",
    }
    # The suite submission is saved, carries the nested fixed workloads, and
    # leaves run_config to the launch boundary.
    suite_path = Path(submissions["default-suite"])
    assert suite_path.is_file()
    payload = json.loads(suite_path.read_text())
    assert payload["workload"]["profile"] == "default-suite"
    assert set(payload["workload"]) == {
        "profile",
        "fresh_context",
        "growing_session",
        "coding",
        "long_context",
    }
    assert payload["run_config"] is None
    assert "secret-124" not in suite_path.read_text()
    # The exact launch command names the saved submission and the prepared
    # directory; no benchmark has been started.
    assert summary["launch_commands"]["default-suite"] == (
        f"bench run {suite_path} --setup-directory {setup_dir}"
    )
    assert {path.name for path in results.iterdir()} == {"setup"}


def test_setup_with_unknown_metadata_still_offers_the_suite(tmp_path, monkeypatch):
    """Descriptive unknowns stay explicit and launchable; the suite is still
    composed because every workload resource is prepared."""
    monkeypatch.setattr(subprocess, "run", dispatcher(tmp_path / "results")[0])
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    result = run_setup(setup_dir, setup_answers())
    assert result.exit_code == 0, result.output
    summary = summary_of(result)
    assert summary["critical_complete"] is True
    assert summary["unknown"] == [
        "hardware.cpu",
        "hardware.driver",
        "hardware.ram",
        "model.quantized_artifact",
        "model.revision",
        "model.source",
        "serving.engine",
        "serving.version",
    ]
    assert summary["default_suite_ready"] is True
    assert "default-suite" in summary["prepared_submissions"]


def test_prepared_suite_launches_through_the_public_cli(tmp_path, monkeypatch):
    """The composed suite launches from its exact advertised command with
    explicit reasoning and reviewed sampling, auto-finishes validation, runs
    the four-phase lifecycle, and retains independently readable results."""
    external, calls = dispatcher(tmp_path / "results")
    monkeypatch.setattr(subprocess, "run", external)
    monkeypatch.setenv("REAL_BENCH_API_KEY", "journey-key-777")
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"

    setup = run_setup(setup_dir, setup_answers())
    assert setup.exit_code == 0, setup.output
    summary = summary_of(setup)

    launch = run_suite(summary, results, RUN_CONFIG)
    assert launch.exit_code == 0, launch.output
    result = json.loads(launch.stdout)
    assert result["status"] == "completed"
    assert result["default_workload_complete"] is True
    assert [phase["status"] for phase in result["phase_progress"]["phases"]] == [
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    # The fixed default-suite coverage is preserved by the public launch.
    assert result["coverage"]["fresh_context"]["accounted"] == 60
    assert result["coverage"]["growing_session"]["accounted"] == 60
    assert result["coverage"]["coding"]["accounted"] == 164
    assert result["coverage"]["long_context"]["accounted"] == 117
    assert len(result["outcomes"]) == 401
    assert result["coding"]["identity"]["status"] == "verified"
    assert result["coding"]["accuracy"] == 0.5
    assert result["ruler"]["quality_by_context"]["196608"]["scored"] == 39
    # The retained workload is the composed suite.
    retained = json.loads((results / result["run_id"] / "workload.json").read_text())
    assert retained["profile"] == "default-suite"
    # The explicit per-launch reasoning and reviewed sampling were recorded.
    run_setup_record = json.loads((results / result["run_id"] / "setup.json").read_text())
    assert (
        run_setup_record["declared"]["requests"]["reasoning"]["chat_template_kwargs"][
            "reasoning_effort"
        ]
        == "xhigh"
    )
    growing_inputs = [
        path for path in (results / result["run_id"]).glob("phases/growing_session/**/input.jsonl")
    ]
    assert growing_inputs
    for path in growing_inputs:
        assert json.loads(path.read_text())["extra"]["chat_template_kwargs"] == {
            "enable_thinking": True,
            "reasoning_effort": "xhigh",
        }
    # Automatic final prep/validation ran and passed at launch.
    assert_launch_gates(result, results)
    # The runtime key is sent to the tools but never retained in any artifact.
    assert any("--api-key" in c and "journey-key-777" in c for c in calls)
    # The same per-argument predicates that guard the stop cases below detect
    # the later phases' scoring work here, proving those guards are effective.
    assert any(any("HumanEval_" in arg for arg in c) for c in calls)
    assert any(any("ruler-" in arg for arg in c) for c in calls)
    for path in (results / result["run_id"]).rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, ValueError):
            continue
        assert "journey-key-777" not in text, f"runtime key retained in {path.name}"
    assert "secret-124" not in json.dumps(run_setup_record)
    assert "journey-key-777" not in launch.output
    # Independent CLI and API reads agree.
    results_readables(result, results)


def test_cancellation_mid_suite_is_retained_and_stops_later_phases(tmp_path, monkeypatch):
    """A cancellation issued for the running suite retains the finished work,
    marks the interrupted phase partial, and skips every later phase."""
    external, calls = dispatcher(tmp_path / "results", mode="cancel")
    monkeypatch.setattr(subprocess, "run", external)
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    summary = summary_of(run_setup(setup_dir, setup_answers()))

    run_id = "c" * 32
    launch = run_suite(summary, results, RUN_CONFIG, run_id)
    assert launch.exit_code == 0, launch.output
    result = json.loads(launch.stdout)
    assert result["status"] == "partial"
    assert result["diagnostic"] == "Cancelled"
    assert [phase["status"] for phase in result["phase_progress"]["phases"]] == [
        "completed",
        "partial",
        "skipped",
        "skipped",
    ]
    assert result["coverage"]["fresh_context"]["accounted"] == 60
    # No coding or long-context scoring tool work was started after cancellation;
    # each predicate matches the marker as a substring of an individual argv
    # element (membership of the command list would require an element equal
    # to the marker, which never holds for a full path).
    assert not any(any("HumanEval_" in arg for arg in c) for c in calls)
    assert not any(any("ruler-" in arg for arg in c) for c in calls)
    assert (results / run_id / "cancel.json").is_file()
    results_readables(result, results)


def test_endpoint_outage_mid_suite_is_retained_and_stops_later_phases(tmp_path, monkeypatch):
    """An endpoint that goes unreachable mid-suite retains the finished phase,
    marks the interrupted phase partial, and skips every later phase."""
    external, calls = dispatcher(tmp_path / "results", mode="outage")
    monkeypatch.setattr(subprocess, "run", external)
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    summary = summary_of(run_setup(setup_dir, setup_answers()))

    launch = run_suite(summary, results, RUN_CONFIG)
    assert launch.exit_code == 0, launch.output
    result = json.loads(launch.stdout)
    assert result["status"] == "partial"
    assert result["diagnostic"] == "Endpoint became unreachable"
    assert [phase["status"] for phase in result["phase_progress"]["phases"]] == [
        "completed",
        "partial",
        "skipped",
        "skipped",
    ]
    assert result["coverage"]["fresh_context"]["accounted"] == 60
    # The outage stops the suite before the later phases' scoring tool work starts.
    assert not any(any("HumanEval_" in arg for arg in c) for c in calls)
    assert not any(any("ruler-" in arg for arg in c) for c in calls)


def test_interrupted_repreparation_does_not_advertise_the_old_suite(tmp_path, monkeypatch):
    """An interruption inside a re-preparation (after the configuration changed)
    removes the suite prepared for the previous configuration, so a resumable
    session never advertises a suite for a configuration it no longer matches."""
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    external, _ = dispatcher(results)

    def interrupting(command, **kwargs):
        args = [str(part) for part in command]
        if any(p.endswith("speed_prepare.py") for p in args):
            raise KeyboardInterrupt
        return external(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", external)
    first = run_setup(setup_dir, setup_answers())
    assert first.exit_code == 0, first.output
    summary = summary_of(first)
    suite = Path(summary["prepared_submissions"]["default-suite"])
    assert suite.is_file()
    # A changed tokenizer forces the speed re-preparation, which is interrupted
    # mid-flight: the old suite is removed instead of being advertised.
    monkeypatch.setattr(subprocess, "run", interrupting)
    answers = setup_answers()
    answers["model.tokenizer"] = "org/other"
    changed = run_setup(setup_dir, answers)
    assert changed.exit_code != 0
    assert not suite.is_file()
