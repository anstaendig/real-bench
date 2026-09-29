"""Resumable manual setup: guided entry, critical fields, unknowns, and revalidation.

Every case is driven through the public command boundary: the ``bench setup``
CLI (interactive TTY prompts or agent-driven ``--answers``) and the
``/api/setup`` endpoints. Manual entry and agent-driven use share one input and
state behavior.
"""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import fake_preparation
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_cli import app
from llm_bench.ordinary_runs import Setup, Submission
from llm_bench.setup_state import (
    CRITICAL_SETUP,
    INTEGER_FIELDS,
    OBJECT_FIELDS,
    UNKNOWN_FIELDS,
    load_state,
)

pytestmark = [
    pytest.mark.usefixtures("external_tool_process_adapter"),
    pytest.mark.usefixtures("preparation_processes"),
]

MARKER = '{\n  "state_file"'


def parse(result_output: str) -> dict[str, Any]:
    """Extract the JSON result block from setup command output."""
    obj, _ = json.JSONDecoder().raw_decode(result_output[result_output.index(MARKER) :])
    return obj


def full_setup(**overrides: Any) -> dict[str, Any]:
    """A complete setup record; descriptive fields accept unknown explicitly."""
    declared: dict[str, dict[str, Any]] = {
        "hardware": {
            "accelerator": "Test GPU",
            "count": 1,
            "memory": "32 GiB",
            "cpu": "CPU",
            "ram": "64 GiB",
            "os": "Linux",
            "driver": "610",
        },
        "model": {
            "source": "org/model",
            "revision": "abc",
            "quantized_artifact": "model-q4",
            "tokenizer": "org/model",
            "template_overrides": {},
        },
        "serving": {
            "engine": "test-engine",
            "version": "1",
            "launch_command": "serve --api-key secret-124",
            "environment": {"API_KEY": "secret-124", "HF_TOKEN": "hf_sensitive_124"},
        },
        "requests": {
            "base_url": "http://localhost:8080",
            "model_id": "test-model",
            "generation": {"temperature": 0.5},
            "reasoning": {"chat_template_kwargs": {"enable_thinking": True}},
            "context_window": 8192,
        },
        "benchmark": {
            "tool": "aiperf",
            "version": "0.12.0",
            "transformers_version": "5.17.0",
            "seed": 121,
            "settings": {"concurrency": 1},
        },
    }
    for path, value in overrides.items():
        category, field = path.split(".")
        declared[category][field] = value
    return {"declared": declared, "observed": {}}


def submission_with(setup: dict[str, Any]) -> dict[str, Any]:
    return {
        "setup": setup,
        "run_config": {
            "reasoning_effort": "xhigh",
            "temperature": 0.5,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "presence_penalty": None,
            "repetition_penalty": None,
        },
        "workload": {
            "scenario": "fresh-context",
            "input_tokens": 4096,
            "output_target": 256,
            "messages": [{"role": "user", "content": "Read the log."}],
        },
    }


def retained_pilot_tool(command, **kwargs):
    """Serve the retained native export so a validation run can complete."""
    if (
        "-c" in command
        or str(command[0]) == "docker"
        or any(str(p).endswith("speed_prepare.py") for p in command)
    ):
        return fake_preparation(command, **kwargs)
    dataset = Path(command[command.index("--input-file") + 1])
    sent = json.loads(dataset.read_text())
    assert sent["extra"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    }
    assert sent["extra"]["temperature"] == 0.5
    native = Path(command[command.index("--output-artifact-dir") + 1])
    native.mkdir(parents=True, exist_ok=True)
    source = Path("tests/fixtures/pilots/performance-121/native")
    for name in ("profile_export.jsonl", "profile_export_raw.jsonl"):
        (native / name).write_bytes(gzip.decompress((source / (name + ".gz")).read_bytes()))
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


# --- 1. Guided manual entry -------------------------------------------------


def test_manual_entry_records_answers_unknowns_and_preparation(tmp_path):
    """Guided entry through the public CLI boundary; answers save per prompt."""
    directory = tmp_path / "setup"
    runner = CliRunner()
    # Answers in prompt order; blanks record descriptive unknowns.
    answers = [
        "Test GPU",
        "1",
        "32 GiB",
        "",  # cpu -> unknown
        "",  # ram -> unknown
        "Linux",
        "",  # driver -> unknown
        "",  # source -> unknown
        "",  # revision -> unknown
        "",  # quantized_artifact -> unknown
        "org/model",
        "",  # template_overrides: no keys
        "",  # engine -> unknown
        "",  # version -> unknown
        "serve --api-key secret-124",
        "",  # environment: no keys
        "http://localhost:8080",
        "test-model",
        "8192",
        "temperature",
        "0.5",
        "",  # generation done
        "chat_template_kwargs",
        '{"enable_thinking": true}',
        "",  # reasoning done
    ]
    result = runner.invoke(
        app, ["setup", "--directory", str(directory)], input="\n".join(answers) + "\n"
    )
    assert result.exit_code == 0, result.output
    summary = parse(result.output)
    assert summary["resumed"] is False
    assert summary["critical_complete"] is True
    assert summary["missing_critical"] == []
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
    assert all(entry["needs_revalidation"] is False for entry in summary["preparations"])
    state = load_state(directory)
    assert state is not None
    assert state.provided["requests"]["base_url"] == "http://localhost:8080"
    assert state.provided["hardware"]["cpu"] == "unknown"
    assert state.setup["declared"]["serving"]["launch_command"] == "serve --api-key [REDACTED]"
    assert state.setup["declared"]["benchmark"]["seed"] == 121
    # No secret is stored in the setup state.
    persisted = (directory / "setup-state.json").read_text()
    assert "secret-124" not in persisted
    # The public output is redacted as well as the saved state.
    assert "secret-124" not in result.output
    assert "hf_sensitive_124" not in persisted


def test_manual_entry_rejects_invalid_critical_answers_before_persisting(tmp_path):
    """Invalid execution-critical values fail at the boundary and never persist."""
    directory = tmp_path / "setup"
    runner = CliRunner()
    for path, value in (
        ("requests.base_url", "ftp://model-server"),
        ("requests.context_window", 0),
        ("requests.context_window", "8192"),
        ("benchmark.seed", 0),
    ):
        result = runner.invoke(
            app,
            [
                "setup",
                "--directory",
                str(directory),
                "--no-interactive",
                "--answers",
                json.dumps({path: value}),
            ],
        )
        assert result.exit_code != 0, f"{path}={value} was accepted"
    # The rejected values never reached the saved record.
    assert load_state(directory) is None


def test_critical_unknowns_are_rejected_and_descriptive_unknowns_preserved():
    """Unknown is a valid value for descriptive metadata only."""
    for category, field in CRITICAL_SETUP:
        setup = full_setup()
        setup["declared"][category][field] = "unknown"
        with pytest.raises(ValueError, match="execution-critical"):
            Setup.model_validate(setup)
    for category, field in UNKNOWN_FIELDS:
        if (category, field) in CRITICAL_SETUP or (category, field) in OBJECT_FIELDS:
            continue
        if (category, field) in INTEGER_FIELDS:
            continue
        setup = full_setup()
        setup["declared"][category][field] = "unknown"
        record = Setup.model_validate(setup)
        assert record.declared[category][field] == "unknown", (
            f"{category}.{field} descriptive unknown must be preserved"
        )


# --- 2. Resumption and interruption -----------------------------------------


def test_interruption_saves_progress_and_resume_keeps_valid_answers(tmp_path):
    """Interruption leaves a resumable state; resume does not re-ask answered fields."""
    directory = tmp_path / "setup"
    runner = CliRunner()
    first_five = [
        "Test GPU",
        "1",
        "32 GiB",
        "",  # cpu -> unknown
        "",  # ram -> unknown
    ]
    interrupted = runner.invoke(
        app, ["setup", "--directory", str(directory)], input="\n".join(first_five) + "\n"
    )
    assert interrupted.exit_code == 130
    assert "interrupted" in interrupted.output.lower()
    state = load_state(directory)
    assert state is not None
    assert state.provided["hardware"] == {
        "accelerator": "Test GPU",
        "count": 1,
        "memory": "32 GiB",
        "cpu": "unknown",
        "ram": "unknown",
    }
    # Resume: the five answered fields are not re-asked, so the first prompt is
    # the next unanswered field (hardware.os), not hardware.accelerator.
    resume_answers = [
        "",  # os -> unknown
        "610",
        "",  # source -> unknown
        "",  # revision -> unknown
        "",  # quantized_artifact -> unknown
        "org/model",
        "",  # template_overrides: no keys
        "",  # engine -> unknown
        "1",
        "serve --api-key secret-124",
        "",  # environment: no keys
        "http://localhost:8080",
        "test-model",
        "8192",
        "temperature",
        "0.5",
        "",  # generation done
        "chat_template_kwargs",
        '{"enable_thinking": true}',
        "",  # reasoning done
    ]
    resumed = runner.invoke(
        app, ["setup", "--directory", str(directory)], input="\n".join(resume_answers) + "\n"
    )
    assert resumed.exit_code == 0, resumed.output
    summary = parse(resumed.output)
    assert summary["resumed"] is True
    assert summary["critical_complete"] is True
    assert summary["missing_critical"] == []
    assert "hardware.os (blank records unknown)" in resumed.output
    assert "hardware.accelerator" not in resumed.output
    assert "secret-124" not in resumed.output
    state = load_state(directory)
    assert state.provided["hardware"]["accelerator"] == "Test GPU"
    assert state.provided["requests"]["context_window"] == 8192
    assert state.setup["declared"]["serving"]["launch_command"] == "serve --api-key [REDACTED]"
    assert state.setup["declared"]["requests"]["reasoning"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_partial_answers_persist_without_interactivity(tmp_path):
    """Agent-driven partial input persists valid answers and reports the gap."""
    directory = tmp_path / "setup"
    runner = CliRunner()
    partial = json.dumps(
        {"requests.base_url": "http://localhost:8080", "model.tokenizer": "org/model"}
    )
    result = runner.invoke(
        app, ["setup", "--directory", str(directory), "--no-interactive", "--answers", partial]
    )
    assert result.exit_code != 0
    assert "execution-critical" in result.output
    # The missing critical objects are reported by name, not filled silently.
    for gap in (
        "model.template_overrides",
        "requests.generation",
        "requests.reasoning",
    ):
        assert gap in result.output
    # Valid answers persist even though the record is not yet launchable.
    state = load_state(directory)
    assert state is not None
    assert state.provided["requests"]["base_url"] == "http://localhost:8080"
    assert state.provided["model"]["tokenizer"] == "org/model"
    # Re-submitting the same answers resumes without wiping saved progress.
    again = runner.invoke(
        app, ["setup", "--directory", str(directory), "--no-interactive", "--answers", partial]
    )
    assert again.exit_code != 0
    assert "execution-critical" in again.output
    state = load_state(directory)
    assert state.provided["requests"]["base_url"] == "http://localhost:8080"
    assert state.provided["model"]["tokenizer"] == "org/model"


# --- 3. Public reads expose unknowns and honest preparation ------------------


def test_unknown_descriptive_fields_survive_public_result_reads(tmp_path, monkeypatch):
    """A run submitted from a record with explicit unknowns stays readable.

    Unknowns are visible through the API result read, the CLI result read, and
    the run directory; declarations remain distinct from sourced observations.
    """
    monkeypatch.setattr(subprocess, "run", retained_pilot_tool)
    directory = tmp_path / "setup"
    runner = CliRunner()
    # Descriptive unknowns plus the critical values needed to launch a run.
    answers = {
        "hardware.cpu": "unknown",
        "model.source": "unknown",
        "serving.launch_command": "unknown",
        "hardware.accelerator": "Test GPU",
        "hardware.memory": "32 GiB",
        "hardware.ram": "64 GiB",
        "hardware.os": "Linux",
        "hardware.driver": "610",
        "model.revision": "abc",
        "model.quantized_artifact": "model-q4",
        "model.tokenizer": "org/model",
        "model.template_overrides": {},
        "serving.engine": "test-engine",
        "serving.version": "1",
        "requests.base_url": "http://localhost:8080",
        "requests.model_id": "test-model",
        "requests.context_window": 8192,
        "requests.generation": {"temperature": 0.5},
        "requests.reasoning": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    result = runner.invoke(
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
    assert result.exit_code == 0, result.output
    state = load_state(directory)
    assert state is not None
    declared = state.setup["declared"]
    payload = submission_with({"declared": declared, "observed": {}})
    Submission.model_validate(payload)

    client = TestClient(create_app(tmp_path))
    submitted = client.post("/api/runs", json=payload)
    assert submitted.status_code == 201, submitted.text
    result = client.get("/api/runs/" + submitted.json()["run_id"]).json()
    assert result["status"] == "completed"
    # Unknowns are exposed in the public result read, distinct from observations.
    assert result["setup"]["declared"]["hardware"]["cpu"] == "unknown"
    assert result["setup"]["declared"]["model"]["source"] == "unknown"
    assert result["setup"]["declared"]["serving"]["launch_command"] == "unknown"
    assert "declared" in result["setup"] and "observed" in result["setup"]
    assert "secret-124" not in json.dumps(result)
    read = runner.invoke(app, ["result", submitted.json()["run_id"], "--results", str(tmp_path)])
    assert read.exit_code == 0, read.output
    assert json.loads(read.output) == result
    run_dir = tmp_path / submitted.json()["run_id"]
    retained = "".join(p.read_text() for p in run_dir.rglob("*") if p.is_file())
    assert "unknown" in retained
    assert "secret-124" not in retained


def test_setup_state_reported_through_api_and_never_complete(tmp_path):
    """The public setup API exposes unknowns and honest preparation status."""
    client = TestClient(create_app(tmp_path))
    # Four critical scalars are not enough: the critical objects are still
    # required, so the boundary reports the gap instead of reporting complete.
    response = client.post(
        "/api/setup",
        json={
            "answers": {
                "requests.base_url": "http://localhost:8080",
                "requests.model_id": "test-model",
                "requests.context_window": 8192,
                "model.tokenizer": "org/model",
            },
        },
    )
    assert response.status_code == 422, response.text
    for gap in ("model.template_overrides", "requests.generation", "requests.reasoning"):
        assert gap in response.text
    # Completing the critical objects makes the record launchable; the valid
    # answers from the first call were kept, so this resumes.
    response = client.post(
        "/api/setup",
        json={
            "answers": {
                "model.template_overrides": {},
                "requests.generation": {"temperature": 0.5},
                "requests.reasoning": {"chat_template_kwargs": {"enable_thinking": True}},
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["resumed"] is True
    assert body["critical_complete"] is True
    assert body["missing_critical"] == []
    assert body["unknown"]
    # The speed, quality, and long-context preparations complete under the
    # controlled tools; the endpoint compatibility check stays unimplemented and
    # never reports complete.
    statuses = {entry["name"]: entry["status"] for entry in body["preparations"]}
    assert statuses["performance_tools"] == "complete"
    assert statuses["tokenizer"] == "complete"
    assert statuses["datasets"] == "complete"
    assert statuses["grading_container"] == "complete"
    assert statuses["long_context"] == "complete"
    assert statuses["compatibility"] == "unimplemented"
    # Every workload is prepared, so the full default suite is ready too.
    assert body["speed_ready"] is True
    assert body["default_suite_ready"] is True
    assert set(body["prepared_submissions"]) == {
        "fresh-context",
        "growing-session",
        "humaneval-plus",
        "sampled-ruler",
        "default-suite",
    }
    assert "launch_commands" in body
    # Descriptive unknowns are preserved explicitly in the saved setup.
    assert body["setup"]["declared"]["hardware"]["accelerator"] == "unknown"
    assert body["setup"]["declared"]["serving"]["launch_command"] == "unknown"
    # The saved state is readable back through the public GET boundary.
    read = client.get("/api/setup")
    assert read.status_code == 200, read.text
    assert read.json()["critical_complete"] is True
    # A second posting into the same state directory resumes.
    again = client.post("/api/setup", json={"answers": {"requests.model_id": "test-model"}})
    assert again.status_code == 200
    assert again.json()["resumed"] is True


# --- 4. Changed configuration revalidation -----------------------------------


def test_changed_configuration_marks_only_affected_preparations(tmp_path):
    """Changing an input marks exactly the preparations that depend on it."""
    directory = tmp_path / "setup"
    runner = CliRunner()
    complete = {
        "requests.base_url": "http://localhost:8080",
        "requests.model_id": "test-model",
        "requests.context_window": 8192,
        "model.tokenizer": "org/model",
        "model.template_overrides": {},
        "requests.generation": {"temperature": 0.5},
        "requests.reasoning": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    first = runner.invoke(
        app,
        [
            "setup",
            "--directory",
            str(directory),
            "--no-interactive",
            "--answers",
            json.dumps(complete),
        ],
    )
    assert first.exit_code == 0, first.output
    summary = parse(first.output)
    names = {entry["name"] for entry in summary["preparations"]}
    assert names == {
        "datasets",
        "grading_container",
        "tokenizer",
        "compatibility",
        "performance_tools",
        "long_context",
    }
    state = load_state(directory)
    assert state is not None
    baseline = {name: entry.hash for name, entry in state.preparations.items()}
    assert all(entry.needs_revalidation is False for entry in state.preparations.values())

    # Change the endpoint: compatibility depends on it; tokenizer does not.
    changed = runner.invoke(
        app,
        [
            "setup",
            "--directory",
            str(directory),
            "--answers",
            json.dumps({"requests.base_url": "http://changed:9000"}),
        ],
    )
    assert changed.exit_code == 0, changed.output
    summary = parse(changed.output)
    flags = {entry["name"]: entry["needs_revalidation"] for entry in summary["preparations"]}
    assert flags == {
        "datasets": False,
        "grading_container": False,
        "tokenizer": False,
        "compatibility": True,
        "performance_tools": False,
        "long_context": False,
    }
    note = next(e for e in summary["preparations"] if e["name"] == "compatibility")
    assert "revalidate" in note["note"]
    state = load_state(directory)
    assert state.setup["declared"]["requests"]["base_url"] == "http://changed:9000"
    compatibility_after_url_change = state.preparations["compatibility"].hash

    # Change the tokenizer: the tokenizer preparation is affected (its hash
    # changes) and is re-prepared against the new source, so its revalidation
    # flag clears on completion; compatibility keeps the flag raised by the
    # endpoint change because it is unimplemented and never re-prepared.
    changed = runner.invoke(
        app,
        [
            "setup",
            "--directory",
            str(directory),
            "--answers",
            json.dumps({"model.tokenizer": "other/tokenizer"}),
        ],
    )
    assert changed.exit_code == 0, changed.output
    summary = parse(changed.output)
    flags = {entry["name"]: entry["needs_revalidation"] for entry in summary["preparations"]}
    assert flags == {
        "datasets": False,
        "grading_container": False,
        "tokenizer": False,
        "compatibility": True,
        "performance_tools": False,
        "long_context": False,
    }
    state = load_state(directory)
    assert state.preparations["tokenizer"].hash != baseline["tokenizer"]
    assert state.preparations["compatibility"].hash == compatibility_after_url_change
