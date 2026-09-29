"""Setup-to-speed launch: preparation, reuse, interruption, failure, invalidation.

Every case is driven through the public command boundary (the ``bench setup``
and ``bench run`` CLIs and the ``/api/setup`` / ``/api/runs`` endpoints) with
the speed preparation subprocesses (pinned-tools probe, tokenizer fetch) and the
scored run subprocesses (input preparation, aiperf) served by a controlled
``subprocess.run``: no network, no native tool, no model weights. The API uses
its own ``results/setup`` directory as the setup state, exactly as ``bench run``
serves a ``--setup-directory``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import DOCKER_VERSION_OUTPUT, fake_preparation, option_value
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from llm_bench import setup_state
from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_cli import app

pytestmark = [
    pytest.mark.xdist_group("slow-setup"),
    pytest.mark.usefixtures("preparation_processes", "external_tool_process_adapter"),
]

CHECKPOINTS = [4096, 16384, 32768, 65536, 131072, 196608]
OUTPUT_TARGETS = [256, 1024]
TOKENIZER = "org/model"
TOKENIZER_B = "org/other-model"
GATED_STDERR = (
    "The tokenizer source is gated and needs a human-only Hugging Face token. "
    "Create a token at https://huggingface.co/settings/tokens, then run "
    "export HF_TOKEN=<token> in the terminal that runs the benchmark, and "
    "re-run bench setup."
)


def setup_answers(tokenizer: str = TOKENIZER) -> dict[str, Any]:
    return {
        "hardware.accelerator": "Test GPU",
        "hardware.count": 1,
        "hardware.memory": "32 GiB",
        "hardware.cpu": "CPU",
        "hardware.ram": "64 GiB",
        "hardware.os": "Linux",
        "hardware.driver": "610",
        "model.source": "org/model",
        "model.revision": "abc",
        "model.quantized_artifact": "model-q4",
        "model.tokenizer": tokenizer,
        "model.template_overrides": {},
        "serving.engine": "test-engine",
        "serving.version": "1",
        "serving.launch_command": "serve --api-key secret-124",
        "serving.environment": {"API_KEY": "secret-124", "HF_TOKEN": "hf_sensitive_124"},
        "requests.base_url": "http://localhost:8080",
        "requests.model_id": "test-model",
        "requests.context_window": 262144,
        "requests.generation": {"temperature": 0.5},
        "requests.reasoning": {"chat_template_kwargs": {"enable_thinking": True}},
    }


def speed_workload(profile: str, scenario: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "scenario": scenario,
        "checkpoints": list(CHECKPOINTS),
        "output_targets": list(OUTPUT_TARGETS),
        "repetitions": 5,
        "warmup_repetitions": 1,
        "request_timeout_seconds": 900,
        "cache_configuration": {},
        "cache_observations": [],
    }


def declared(tokenizer: str = TOKENIZER) -> dict[str, Any]:
    return {
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
            "tokenizer": tokenizer,
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
            "context_window": 262144,
        },
        "benchmark": {
            "tool": "aiperf",
            "version": "0.12.0",
            "transformers_version": "5.17.0",
            "seed": 121,
            "settings": {"concurrency": 1},
        },
    }


def fresh_payload(tokenizer: str = TOKENIZER) -> dict[str, Any]:
    return {
        "setup": {"declared": declared(tokenizer), "observed": {}},
        "run_config": {
            "reasoning_effort": "high",
            "temperature": 0.5,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "presence_penalty": None,
            "repetition_penalty": None,
        },
        "workload": speed_workload("fresh-context-speed", "fresh-context"),
    }


def run_fake(
    fetch: Any | None = None,
    counts: dict[str, int] | None = None,
    *,
    probe: str = "0.12.0\n5.17.0\n",
    fetch_exit: int | None = None,
    fetch_stderr: str | None = None,
) -> Any:
    """Controlled ``subprocess.run`` for setup preparation and scored runs.

    The pinned-tools probe and the tokenizer fetch serve the preparation;
    ``fetch`` overrides the fetch, and ``fetch_exit``/``fetch_stderr`` fail
    every fetch (``fetch_exit`` 3 emulates the helper's gated-source report).
    The input preparer and aiperf serve a full fresh-context curve.
    """

    def external(command, **kwargs):
        # The pinned-tool commands all share the ``uv run --with ... python``
        # prefix, so the script name (or the probe's ``-c``) selects the tool.
        args = list(command)
        if any(str(part).endswith("ruler_source_prepare.py") for part in args):
            return fake_preparation(command, **kwargs)
        if any(str(part).endswith("speed_prepare.py") for part in args):
            source = option_value(args, "--source")
            if fetch is not None:
                return fetch(command, **kwargs)
            if counts is not None:
                counts["fetch"] = counts.get("fetch", 0) + 1
            if fetch_exit is not None:
                stderr = (
                    GATED_STDERR
                    if fetch_exit == 3 and fetch_stderr is None
                    else fetch_stderr or f"controlled fetch failure for {source}"
                )
                return subprocess.CompletedProcess(command, fetch_exit, "", stderr)
            return fake_preparation(command, **kwargs)
        if args and args[0] == "docker":
            if counts is not None:
                counts.setdefault("docker", []).append(list(args[1:]))
            if args[1] == "version":
                return subprocess.CompletedProcess(command, 0, DOCKER_VERSION_OUTPUT, "")
            if args[1] == "image":
                return subprocess.CompletedProcess(command, 0, "[{}]", "")
            if args[1] == "pull":
                return subprocess.CompletedProcess(command, 0, "", f"pulled {args[-1]}")
            if args[1] == "run" and "uname" in args:
                return subprocess.CompletedProcess(command, 0, "x86_64\n", "")
            raise LookupError(f"unhandled docker command: {command}")
        if any(str(part).endswith("performance_inputs.py") for part in args):
            spec = json.loads(Path(args[args.index("--spec") + 1]).read_text())
            if "conditions" in spec:
                return fresh_tool(command)
            return growing_tool(command)
        if args and args[0] == "git":
            return fake_preparation(command, **kwargs)
        if "-c" in args:
            return subprocess.CompletedProcess(command, 0, probe, "")
        native = Path(args[args.index("--output-artifact-dir") + 1])
        if "growing-session" in str(native.parents[1]):
            return aiperf_growing(command)
        return aiperf(command)

    return external


def fresh_tool(command) -> subprocess.CompletedProcess:
    """Serve the fresh-context input preparer: exact-checkpoint rows per phase."""
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
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "preparation.json").write_text(json.dumps({"observations": prepared}))
    return subprocess.CompletedProcess(command, 0, "", "")


def growing_tool(command) -> subprocess.CompletedProcess:
    """Serve the growing-session turn sizer: one row per (turn|warmup) spec."""
    specification = json.loads(Path(command[command.index("--spec") + 1]).read_text())
    directory = Path(command[command.index("--directory") + 1])
    history = list(specification["history"])
    structural = list(specification["structural"])
    messages = history + structural
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "input.jsonl").write_text(
        json.dumps(
            {
                "text": "Messages supplied in extra.messages",
                "output_length": specification["output_target"],
                "extra": {
                    **specification["request_settings"],
                    "messages": messages,
                },
            }
        )
        + "\n"
    )
    return subprocess.CompletedProcess(command, 0, "", "")


def aiperf_growing(command) -> subprocess.CompletedProcess:
    """Serve aiperf for a growing-session turn: a usable assistant response."""
    native = Path(command[command.index("--output-artifact-dir") + 1])
    native.mkdir(parents=True, exist_ok=True)
    input_file = Path(command[command.index("--input-file") + 1])
    row = json.loads(input_file.read_text().splitlines()[0])
    output_target = row["output_length"]
    name = native.parent.name
    checkpoint = int(name.split("-")[1]) if name.startswith("turn-") else 0
    request_id = f"{name}-req"
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
                                    {
                                        "delta": {
                                            "content": f"assistant-response-{name}",
                                            "reasoning_content": "thought",
                                        },
                                        "finish_reason": "length",
                                    }
                                ]
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
            "usage_prompt_tokens": {"value": checkpoint, "unit": "tokens"},
            "usage_completion_tokens": {"value": output_target, "unit": "tokens"},
        },
    }
    (native / "profile_export_raw.jsonl").write_text(json.dumps(raw) + "\n")
    (native / "profile_export.jsonl").write_text(json.dumps(measured) + "\n")
    return subprocess.CompletedProcess(command, 0, "", "")


def aiperf(command) -> subprocess.CompletedProcess:
    """Serve aiperf: one measured request per invocation with native exports."""
    native = Path(command[command.index("--output-artifact-dir") + 1])
    native.mkdir(parents=True, exist_ok=True)
    condition_name = native.parent.parent.name
    input_tokens, output_target = (int(value) for value in condition_name.split("-")[-2:])
    request_id = f"{condition_name}-{native.parent.name}"
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
            "output_token_throughput_per_user": {"value": 80.0, "unit": "tokens/sec/user"},
            "prefill_throughput_per_user": {"value": 900.0, "unit": "tokens/sec/user"},
            "usage_prompt_tokens": {"value": input_tokens, "unit": "tokens"},
            "usage_completion_tokens": {"value": output_target, "unit": "tokens"},
        },
    }
    (native / "profile_export_raw.jsonl").write_text(json.dumps(raw) + "\n")
    (native / "profile_export.jsonl").write_text(json.dumps(measured) + "\n")
    return subprocess.CompletedProcess(command, 0, "", "")


def run_setup_cli(directory: Path, answers: dict[str, Any]) -> Any:
    args = [
        "setup",
        "--directory",
        str(directory),
        "--no-interactive",
        "--answers",
        json.dumps(answers),
    ]
    return CliRunner().invoke(app, args)


def summary_of(result: Any) -> dict[str, Any]:
    """Parse the setup report the CLI echoes (``state_file`` is the first key)."""
    return json.loads(result.output[result.output.index("{") :])


def state(directory: Path) -> setup_state.SetupState:
    loaded = setup_state.load_state(directory)
    assert loaded is not None
    return loaded


def api(results: Path) -> TestClient:
    return TestClient(create_app(results))


# --- 1. Setup-to-speed execution and independent result reads -----------------


def test_setup_preparation_launches_speed_and_reads_independently(tmp_path, monkeypatch):
    """A prepared setup launches a speed profile through the controlled tools;
    the result is read back independently by the API and the CLI."""
    counts: dict[str, int] = {}
    monkeypatch.setattr(subprocess, "run", run_fake(counts=counts))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    result = run_setup_cli(setup_dir, setup_answers())
    assert result.exit_code == 0, result.output
    summary = summary_of(result)
    assert summary["speed_ready"] is True
    assert summary["default_suite_ready"] is True
    entries = {entry["name"]: entry for entry in summary["preparations"]}
    assert entries["performance_tools"]["status"] == "complete"
    assert entries["tokenizer"]["status"] == "complete"
    # The pinned tools are verified in the pinned environment and the
    # tokenizer is on disk with a sha256 manifest; no secret is echoed.
    evidence = json.loads((setup_dir / "speed" / "performance-tools.json").read_text())
    assert evidence["observed"] == {"aiperf": "0.12.0", "transformers": "5.17.0"}
    assert evidence["declared"] == {"aiperf": "0.12.0", "transformers": "5.17.0"}
    manifest = json.loads((setup_dir / "speed" / "tokenizer" / "manifest.json").read_text())
    assert manifest["source"] == TOKENIZER
    for name in manifest["files"]:
        token_file = setup_dir / "speed" / "tokenizer" / name
        assert token_file.is_file()
        assert not token_file.is_symlink()
    assert "secret-124" not in json.dumps(summary)
    submissions = summary["prepared_submissions"]
    assert set(submissions) == {
        "fresh-context",
        "growing-session",
        "humaneval-plus",
        "sampled-ruler",
        "default-suite",
    }
    for name in ("fresh-context", "growing-session"):
        text = Path(submissions[name]).read_text()
        payload = json.loads(text)
        assert payload["workload"]["profile"] in {"fresh-context-speed", "growing-session-speed"}
        assert payload["workload"]["checkpoints"] == CHECKPOINTS
        assert payload["workload"]["output_targets"] == OUTPUT_TARGETS
        assert payload["workload"]["repetitions"] == 5
        assert payload["workload"]["warmup_repetitions"] == 1
        assert payload["run_config"] is None
        assert "secret-124" not in text
    # The coding submission launches the pinned dataset from the prepared
    # copy and carries no secret either.
    coding_text = Path(submissions["humaneval-plus"]).read_text()
    coding_payload = json.loads(coding_text)
    assert coding_payload["workload"]["profile"] == "humaneval-plus"
    assert coding_payload["workload"]["problems_path"] == str(
        setup_dir / "quality" / "HumanEvalPlus-v0.1.10.jsonl.gz"
    )
    assert coding_payload["run_config"] is None
    assert "secret-124" not in coding_text

    # Launch the fresh-context speed profile through the API.
    client = api(results)
    submitted = client.post("/api/runs", json=fresh_payload())
    assert submitted.status_code == 201, submitted.text
    run = submitted.json()
    assert run["status"] == "completed"
    assert run["coverage"]["fresh_context"] == {"expected": 60, "accounted": 60, "successful": 60}
    assert len(run["outcomes"]) == 60
    assert len(run["fresh_context_curve"]) == 12
    # The launch sized against the verified local tokenizer, not the source.
    prepared_tokenizer = str(setup_dir / "speed" / "tokenizer")
    assert run["setup"]["declared"]["model"]["tokenizer"] == prepared_tokenizer
    gate = json.loads((results / run["run_id"] / "speed-preparation.json").read_text())
    assert gate["prepared"] is True
    # The independent reads agree: the API result read, the HTML view, and the
    # CLI result read all expose the same run.
    assert client.get("/api/runs/" + run["run_id"]).json() == run
    assert "Fresh-context speed curve" in client.get("/runs/" + run["run_id"]).text
    cli_read = CliRunner().invoke(app, ["result", run["run_id"], "--results", str(results)])
    assert cli_read.exit_code == 0, cli_read.output
    assert json.loads(cli_read.output) == run
    saved = client.get("/api/setup", params=[("directory", str(setup_dir))]).json()
    assert saved["speed_ready"] is True
    assert saved["prepared_submissions"] == submissions
    assert counts["fetch"] == 1, "the prepared tokenizer must not be re-fetched at launch"


def test_setup_preparation_launches_speed_through_cli(tmp_path, monkeypatch):
    """The prepared submission named by the setup report launches through the
    public ``bench run --setup-directory`` command."""
    monkeypatch.setattr(subprocess, "run", run_fake())
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    result = run_setup_cli(setup_dir, setup_answers())
    assert result.exit_code == 0, result.output
    summary = summary_of(result)
    run_config = (
        '{"reasoning_effort": "high", "temperature": 0.5, "top_p": null, "top_k": null,'
        ' "min_p": null, "presence_penalty": null, "repetition_penalty": null}'
    )
    launch = CliRunner().invoke(
        app,
        [
            "run",
            summary["prepared_submissions"]["fresh-context"],
            "--results",
            str(results),
            "--setup-directory",
            str(setup_dir),
            "--no-interactive",
            "--run-config",
            run_config,
        ],
    )
    assert launch.exit_code == 0, launch.output
    run = json.loads(launch.output[launch.output.index("{") :])
    assert run["status"] == "completed"
    gate = json.loads((results / run["run_id"] / "speed-preparation.json").read_text())
    assert gate["prepared"] is True


# --- 2. Reuse of verified downloads -------------------------------------------


def test_prepared_resources_are_reused_and_verified_on_launch(tmp_path, monkeypatch):
    """A second setup reuses the verified tokenizer instead of re-fetching; a
    tampered tokenizer is refused at launch until setup repairs it."""
    counts: dict[str, int] = {}
    monkeypatch.setattr(subprocess, "run", run_fake(counts=counts))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    assert run_setup_cli(setup_dir, setup_answers()).exit_code == 0
    manifest_path = setup_dir / "speed" / "tokenizer" / "manifest.json"
    # A marker a genuine re-fetch would overwrite.
    manifest = json.loads(manifest_path.read_text())
    manifest["fetched_at"] = "marked"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    second = run_setup_cli(setup_dir, setup_answers())
    assert second.exit_code == 0, second.output
    assert json.loads(manifest_path.read_text())["fetched_at"] == "marked"
    assert counts["fetch"] == 1, "a verified tokenizer must not be re-fetched"

    # Tamper the tokenizer: the launch must refuse and start no work; setup
    # must repair it.
    (setup_dir / "speed" / "tokenizer" / "tokenizer.json").write_text("tampered")
    client = api(results)
    refused = client.post("/api/runs", json=fresh_payload())
    assert refused.status_code == 422, refused.text
    assert "tokenizer" in refused.text
    assert {p.name for p in results.iterdir()} == {"setup"}  # no run work on a refused launch
    repaired = run_setup_cli(setup_dir, setup_answers())
    assert repaired.exit_code == 0
    assert json.loads(manifest_path.read_text())["fetched_at"] != "marked"
    assert counts["fetch"] == 2
    launched = client.post("/api/runs", json=fresh_payload())
    assert launched.status_code == 201, launched.text
    assert launched.json()["status"] == "completed"


# --- 3. Interrupted preparation -----------------------------------------------


def test_interrupted_preparation_resumes_after_save(tmp_path, monkeypatch):
    """A Ctrl-C during the tokenizer fetch saves the interrupted preparation;
    the session resumes and completes the same preparation."""
    interrupted = {"n": 0}

    def interrupt_once(command, **kwargs):
        if any(str(part).endswith("speed_prepare.py") for part in command):
            interrupted["n"] += 1
            if interrupted["n"] == 1:
                raise KeyboardInterrupt
            return fake_preparation(command, **kwargs)
        if "-c" in command:
            return subprocess.CompletedProcess(command, 0, "0.12.0\n5.17.0\n", "")
        return fake_preparation(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", interrupt_once)
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    first = run_setup_cli(setup_dir, setup_answers())
    assert first.exit_code == 130, first.output
    assert "Setup interrupted; saved progress remains" in first.output
    saved = state(setup_dir)
    assert saved.preparations["performance_tools"].status == "complete"
    assert saved.preparations["tokenizer"].status == "in-progress"
    assert not (setup_dir / "submissions").exists()
    # The interrupted state is reported honestly by the public read boundary.
    client = api(results)
    view = client.get("/api/setup", params=[("directory", str(setup_dir))]).json()
    assert view["speed_ready"] is False
    assert "prepared_submissions" not in view

    # Resume: the probe re-verifies, the fetch completes, and the submissions
    # appear.
    monkeypatch.setattr(subprocess, "run", run_fake())
    resumed = run_setup_cli(setup_dir, setup_answers())
    assert resumed.exit_code == 0, resumed.output
    assert state(setup_dir).preparations["tokenizer"].status == "complete"
    summary = summary_of(resumed)
    assert summary["speed_ready"] is True
    assert "fresh-context" in summary["prepared_submissions"]


# --- 4. Dependency failures ----------------------------------------------------


def test_dependency_failure_is_honest_and_refuses_launch(tmp_path, monkeypatch):
    """A pinned-environment mismatch keeps the tools entry pending with an
    actionable note; a gated tokenizer names the human-only prerequisite; the
    launch refuses and no work is started."""
    monkeypatch.setattr(subprocess, "run", run_fake(probe="0.11.0\n5.17.0\n", fetch_exit=3))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    result = run_setup_cli(setup_dir, setup_answers())
    assert result.exit_code == 0, result.output  # a preparation problem is not a session failure
    summary = summary_of(result)
    entries = {entry["name"]: entry for entry in summary["preparations"]}
    assert entries["performance_tools"]["status"] == "pending"
    assert "pinned performance tools do not match" in entries["performance_tools"]["note"]
    assert entries["tokenizer"]["status"] == "pending"
    assert "human-only prerequisite" in entries["tokenizer"]["note"]
    assert "HF_TOKEN" in entries["tokenizer"]["note"]
    assert summary["speed_ready"] is False
    # The quality and long-context preparations are independent of the speed
    # tools: the coding and long-context submissions are prepared while the
    # speed submissions stay absent.
    submissions = summary["prepared_submissions"]
    assert set(submissions) == {"humaneval-plus", "sampled-ruler"}

    # The launch refuses naming both gaps and starts no work.
    client = api(results)
    refused = client.post("/api/runs", json=fresh_payload())
    assert refused.status_code == 422, refused.text
    assert "performance tools" in refused.text
    assert "tokenizer" in refused.text
    assert {p.name for p in results.iterdir()} == {"setup"}  # no run work on a refused launch


def test_gated_tokenizer_is_ready_after_human_only_step(tmp_path, monkeypatch):
    """A gated tokenizer names the token step; once the operator has it, the
    resume completes the preparation and the launch passes the gate."""
    monkeypatch.setattr(subprocess, "run", run_fake(fetch_exit=3))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    result = run_setup_cli(setup_dir, setup_answers())
    assert result.exit_code == 0
    summary = summary_of(result)
    entries = {entry["name"]: entry for entry in summary["preparations"]}
    assert entries["performance_tools"]["status"] == "complete"
    assert entries["tokenizer"]["status"] == "pending"
    assert "HF_TOKEN" in entries["tokenizer"]["note"]
    assert summary["speed_ready"] is False
    # The human-only step happens: the tokenizer is now fetchable.
    monkeypatch.setattr(subprocess, "run", run_fake())
    resumed = run_setup_cli(setup_dir, setup_answers())
    assert resumed.exit_code == 0
    assert state(setup_dir).preparations["tokenizer"].status == "complete"
    client = api(results)
    launched = client.post("/api/runs", json=fresh_payload())
    assert launched.status_code == 201, launched.text
    assert launched.json()["status"] == "completed"


def test_stale_prepared_inputs_cannot_pass_as_ready(tmp_path, monkeypatch):
    """After the tokenizer changes and the fetch fails, the old prepared
    submission is refused at launch (hash mismatch) and no work starts; a
    submission matching the current configuration passes once re-prepared."""
    monkeypatch.setattr(subprocess, "run", run_fake())
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    assert run_setup_cli(setup_dir, setup_answers()).exit_code == 0
    assert (setup_dir / "submissions" / "fresh-context.json").is_file()
    # Change the tokenizer; the fetch for the new source fails.
    fail_b = {"calls": 0}

    def fetch_b(command, **kwargs):
        source = option_value(command, "--source")
        if source == TOKENIZER_B:
            fail_b["calls"] += 1
            return subprocess.CompletedProcess(command, 1, "", f"fetch failed for {source}")
        return fake_preparation(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_fake(fetch=fetch_b))
    changed = run_setup_cli(setup_dir, {"model.tokenizer": TOKENIZER_B})
    assert changed.exit_code == 0, changed.output
    summary = summary_of(changed)
    entries = {entry["name"]: entry for entry in summary["preparations"]}
    assert entries["tokenizer"]["status"] == "pending"
    assert entries["performance_tools"]["status"] == "complete"
    assert summary["speed_ready"] is False
    # The old prepared submission no longer matches the prepared state: the
    # launch refuses and starts no work.
    client = api(results)
    refused = client.post("/api/runs", json=fresh_payload(TOKENIZER))
    assert refused.status_code == 422, refused.text
    assert "tokenizer" in refused.text
    assert {p.name for p in results.iterdir()} == {"setup"}  # no run work on a refused launch
    # A launch matching the current (still-failing) configuration is refused
    # too, because the preparation is pending.
    refused_b = client.post("/api/runs", json=fresh_payload(TOKENIZER_B))
    assert refused_b.status_code == 422, refused_b.text
    assert fail_b["calls"] == 1, "a refused launch must not fetch the tokenizer"
    # The fetch recovers: the re-prepared submission launches.
    monkeypatch.setattr(subprocess, "run", run_fake())
    repaired = run_setup_cli(setup_dir, {"model.tokenizer": TOKENIZER_B})
    assert repaired.exit_code == 0
    assert state(setup_dir).preparations["tokenizer"].status == "complete"
    launched = client.post("/api/runs", json=fresh_payload(TOKENIZER_B))
    assert launched.status_code == 201, launched.text
    assert launched.json()["status"] == "completed"
    gate = json.loads((results / launched.json()["run_id"] / "speed-preparation.json").read_text())
    assert gate["tokenizer"]["source"] == TOKENIZER_B
    assert gate["tokenizer"]["path"] == str(setup_dir / "speed" / "tokenizer")


# --- 6. Incomplete preparation is clearly identified ---------------------------


def test_incomplete_setup_is_not_advertised_ready(tmp_path, monkeypatch):
    """An incomplete setup is named and never advertised ready or launched;
    once complete, the speed and quality workloads are prepared while the full
    default suite stays not-ready while long context is unprepared."""
    monkeypatch.setattr(subprocess, "run", run_fake())
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    partial = dict(setup_answers())
    del partial["requests.base_url"]
    result = run_setup_cli(setup_dir, partial)
    assert result.exit_code == 2, result.output
    assert "requests.base_url is execution-critical and required" in result.output
    # An incomplete record is never prepared or advertised, and no
    # submissions exist to launch from.
    assert not (setup_dir / "submissions").exists()
    assert not (setup_dir / "speed").exists()
    # Completing the setup prepares the speed, quality, and long-context
    # workloads; the full default suite is ready once every workload is
    # prepared.
    complete = run_setup_cli(setup_dir, setup_answers())
    assert complete.exit_code == 0
    final = summary_of(complete)
    assert final["speed_ready"] is True
    assert final["default_suite_ready"] is True
    entries = {entry["name"]: entry for entry in final["preparations"]}
    assert entries["datasets"]["status"] == "complete"
    assert entries["grading_container"]["status"] == "complete"
    assert entries["long_context"]["status"] == "complete"
    assert entries["compatibility"]["status"] == "unimplemented"
    assert "fresh-context" in final["prepared_submissions"]
    assert "humaneval-plus" in final["prepared_submissions"]
    assert "sampled-ruler" in final["prepared_submissions"]


def test_growing_session_launches_prepared_and_keeps_independent_histories(
    tmp_path,
    monkeypatch,
):
    """The prepared growing-session submission launches through the same gate;
    each session sizes turns locally and builds its own independent history."""
    monkeypatch.setattr(subprocess, "run", run_fake())
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    result = run_setup_cli(setup_dir, setup_answers())
    assert result.exit_code == 0, result.output
    payload = {
        "setup": {"declared": declared(), "observed": {}},
        "run_config": {
            "reasoning_effort": "high",
            "temperature": 0.5,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "presence_penalty": None,
            "repetition_penalty": None,
        },
        "workload": speed_workload("growing-session-speed", "growing-session"),
    }
    client = api(results)
    submitted = client.post("/api/runs", json=payload)
    assert submitted.status_code == 201, submitted.text
    run = submitted.json()
    assert run["status"] == "completed"
    assert run["coverage"]["growing_session"] == {"expected": 60, "accounted": 60, "successful": 60}
    assert len(run["growing_session_curve"]) == 12
    gate = json.loads((results / run["run_id"] / "speed-preparation.json").read_text())
    assert gate["prepared"] is True
    # Independent histories: each turn is sized from its own session's
    # accumulated turns only, so sessions never see each other's history.
    accumulated: dict[str, list] = {}
    for outcome in run["outcomes"]:
        if outcome["outcome"] != "success":
            continue
        session = outcome["session"]
        turn_dir = results / run["run_id"] / session / f"turn-{outcome['input_tokens']}"
        spec = json.loads((turn_dir / "sizing-spec.json").read_text())
        assert spec["history"] == accumulated.get(session, []), (
            "a session must not see another session's turns"
        )
        row_messages = json.loads((turn_dir / "input.jsonl").read_text())["extra"]["messages"]
        # A prompt must lead with its own accumulated history.
        assert row_messages[: len(spec["history"])] == spec["history"]
        assistant = {
            "role": "assistant",
            "content": f"assistant-response-turn-{outcome['input_tokens']}",
            "reasoning_content": "thought",
        }
        accumulated[session] = row_messages + [assistant]
    # Each (output target, repetition) pair is one independent session.
    assert len(accumulated) == 10


def test_template_change_revalidates_prepared_tokenizer(tmp_path, monkeypatch):
    """A chat-template change revalidates the prepared tokenizer: the old
    submission is refused until setup re-prepares against the new template."""
    monkeypatch.setattr(subprocess, "run", run_fake())
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    assert run_setup_cli(setup_dir, setup_answers()).exit_code == 0
    # The prepared submission matches the previous (empty) template; changing
    # the template overrides must invalidate it and re-prep the tokenizer.
    changed = run_setup_cli(setup_dir, {"model.template_overrides": {"chat_template": "x"}})
    assert changed.exit_code == 0, changed.output
    summary = summary_of(changed)
    entries = {entry["name"]: entry for entry in summary["preparations"]}
    assert entries["tokenizer"]["status"] == "complete"
    assert summary["speed_ready"] is True
    # The readiness the API advertises reads the saved state against the disk:
    # a deleted tokenizer must be visible as not-ready, and repair restores it.
    client = api(results)
    first = client.get("/api/setup", params=[("directory", str(setup_dir))])
    assert first.json()["speed_ready"] is True
    (setup_dir / "speed" / "tokenizer" / "tokenizer.json").unlink()
    second = client.get("/api/setup", params=[("directory", str(setup_dir))])
    # A deleted tokenizer must not stay advertised ready.
    assert second.json()["speed_ready"] is False
    repaired = run_setup_cli(setup_dir, {"model.template_overrides": {"chat_template": "x"}})
    assert repaired.exit_code == 0, repaired.output
    assert summary_of(repaired)["speed_ready"] is True
    # The old submission still matches the previous template: the launch
    # refuses it and starts no work.
    refused = client.post("/api/runs", json=fresh_payload())
    assert refused.status_code == 422, refused.text
    assert "tokenizer" in refused.text
    assert {p.name for p in results.iterdir()} == {"setup"}  # no run work on a refused launch
    # A launch matching the new template passes the gate.
    payload = fresh_payload()
    payload["setup"]["declared"]["model"]["template_overrides"] = {"chat_template": "x"}
    launched = client.post("/api/runs", json=payload)
    assert launched.status_code == 201, launched.text
    assert launched.json()["status"] == "completed"
    gate = json.loads((results / launched.json()["run_id"] / "speed-preparation.json").read_text())
    assert gate["prepared"] is True
    assert gate["tokenizer"]["path"] == str(setup_dir / "speed" / "tokenizer")
