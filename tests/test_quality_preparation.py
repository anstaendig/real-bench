"""Setup-to-coding launch: quality preparation, reuse, interruption, failure.

Every case is driven through the public command boundary (the ``bench setup``
and ``bench run`` CLIs and the ``/api/setup`` / ``/api/runs`` endpoints) with
the preparation subprocesses (pinned-tools probe, tokenizer fetch, the Docker
CLI) and the scored run subprocesses (aiperf, EvalPlus grading) served by a
controlled ``subprocess.run``: no network, no native tool, no model weights,
no Docker daemon. The API uses its own ``results/setup`` directory as the
setup state, exactly as ``bench run`` serves a ``--setup-directory``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import DOCKER_VERSION_OUTPUT, fake_preparation
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from llm_bench import setup_state
from llm_bench.humaneval import IMAGE, SOURCE_SHA256
from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_cli import app
from llm_bench.ordinary_runs import CODING_OUTPUT_BUDGET

pytestmark = [
    pytest.mark.xdist_group("slow-setup"),
    pytest.mark.usefixtures("preparation_processes", "external_tool_process_adapter"),
]

SOURCE = Path("tests/fixtures/pilots/humaneval-122")
PROBLEMS = Path("src/llm_bench/datasets/HumanEvalPlus-v0.1.10.jsonl.gz")
PILOT_EXPORT = SOURCE / "remote/HumanEval_0/native/profile_export_raw.jsonl.gz"


def setup_answers() -> dict[str, Any]:
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
        "model.tokenizer": "org/model",
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


def declared() -> dict[str, Any]:
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


def coding_workload(problems_path: str) -> dict[str, Any]:
    return {
        "profile": "humaneval-plus",
        "problems_path": problems_path,
        "solutions_per_task": 1,
        "request_timeout_seconds": 900,
        "grading_timeout_seconds": 3600,
    }


def coding_payload() -> dict[str, Any]:
    return {
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
        "workload": coding_workload(str(PROBLEMS)),
    }


def run_fake(
    counts: dict[str, Any] | None = None,
    *,
    daemon: bool = True,
    machine: str = "x86_64",
    profile_fail: tuple[int, int] | None = None,
) -> Any:
    """Controlled ``subprocess.run`` for the quality preparation and a coding run.

    The Docker CLI is stateful per test: a pulled image stays present for
    every later call, so the reuse (no re-pull) assertions are observable.
    ``daemon`` emulates an unreachable daemon; ``machine`` is what the grading
    image reports for ``uname -m``; ``profile_fail`` makes aiperf fail with an
    HTTP 500 on the named tasks of the coding run.
    """
    pulled: set[str] = set()

    def external(command, **kwargs):
        args = [str(part) for part in command]
        if any(part.endswith("ruler_source_prepare.py") for part in args):
            return fake_preparation(command, **kwargs)
        if any(part.endswith("speed_prepare.py") for part in args):
            return fake_preparation(command, **kwargs)
        if "-c" in args:
            return subprocess.CompletedProcess(command, 0, "0.12.0\n5.17.0\n", "")
        if args and args[0] == "git":
            return fake_preparation(command, **kwargs)
        if args and args[0] == "docker":
            if counts is not None:
                counts.setdefault("docker", []).append(args[1:])
            if args[1] == "version":
                if not daemon:
                    return subprocess.CompletedProcess(
                        command, 1, "", "Cannot connect to the Docker daemon"
                    )
                return subprocess.CompletedProcess(command, 0, DOCKER_VERSION_OUTPUT, "")
            if args[1] == "pull":
                if not daemon:
                    return subprocess.CompletedProcess(
                        command, 1, "", "Cannot connect to the Docker daemon"
                    )
                if counts is not None:
                    counts["pulls"] = counts.get("pulls", 0) + 1
                pulled.add(args[-1])
                return subprocess.CompletedProcess(command, 0, "", f"pulled {args[-1]}")
            if args[1] == "image":
                # inspect: present only for images this fake has pulled.
                present = args[-1] in pulled
                payload = json.dumps([{}]) if present else "[]"
                return subprocess.CompletedProcess(command, 0, payload, "")
            if args[1] == "run":
                if "uname" in args:
                    return subprocess.CompletedProcess(command, 0, machine + "\n", "")
                directory = Path(args[args.index("-v") + 1].removesuffix(":/app"))
                if "evalplus.sanitize" in args:
                    (directory / "solutions-sanitized.jsonl").write_text(
                        (directory / "solutions.jsonl").read_text()
                    )
                    solutions = [
                        json.loads(line)
                        for line in (
                            (directory / "solutions-sanitized.jsonl").read_text().splitlines()
                        )
                    ]
                    native = {
                        "hash": hashlib.md5(
                            (directory / "grading-problems.jsonl").read_bytes()
                        ).hexdigest(),
                        "eval": {
                            row["task_id"]: [
                                {
                                    **row,
                                    "base_status": "pass",
                                    "plus_status": "pass"
                                    if int(row["task_id"].split("/")[1]) < 82
                                    else "fail",
                                }
                            ]
                            for row in solutions
                        },
                    }
                    (directory / "solutions-sanitized_eval_results.json").write_text(
                        json.dumps(native)
                    )
                return subprocess.CompletedProcess(command, 0, "", "")
            raise LookupError(f"unhandled docker command: {command}")
        if "profile" in args:
            native = Path(args[args.index("--output-artifact-dir") + 1])
            native.mkdir(parents=True, exist_ok=True)
            task = native.parent.name
            if profile_fail is not None and task.replace("HumanEval_", "").isdigit():
                index = int(task.replace("HumanEval_", ""))
                if profile_fail[0] <= index <= profile_fail[1]:
                    (native / "profile_export_raw.jsonl").write_text(
                        json.dumps(
                            {
                                "metadata": {"x_request_id": "failed"},
                                "status": 500,
                                "responses": [],
                            }
                        )
                        + "\n"
                    )
                    return subprocess.CompletedProcess(command, 1, "", "HTTP 500")
            raw = json.loads(gzip.decompress(PILOT_EXPORT.read_bytes()))
            (native / "profile_export_raw.jsonl").write_text(json.dumps(raw) + "\n")
            return subprocess.CompletedProcess(command, 0, "", "")
        raise LookupError(f"unhandled command: {command}")

    return external


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


def run_result(results: Path, run_id: str) -> dict[str, Any]:
    return json.loads((results / run_id / "quality-preparation.json").read_text())


# --- 1. Setup-to-coding execution and independent result reads -----------------


def test_setup_preparation_launches_coding_and_reads_independently(tmp_path, monkeypatch):
    """A prepared setup launches the coding profile against the prepared
    dataset; the result is read back independently by the API and the CLI."""
    counts: dict[str, Any] = {}
    monkeypatch.setattr(subprocess, "run", run_fake(counts))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    result = run_setup_cli(setup_dir, setup_answers())
    assert result.exit_code == 0, result.output
    summary = summary_of(result)
    entries = {entry["name"]: entry for entry in summary["preparations"]}
    assert entries["datasets"]["status"] == "complete"
    assert entries["grading_container"]["status"] == "complete"
    assert entries["long_context"]["status"] == "complete"
    assert summary["speed_ready"] is True
    assert summary["default_suite_ready"] is True
    # The preparation evidence is retained in the state directory.
    problems = json.loads((setup_dir / "quality" / "problems.json").read_text())
    assert problems["source_sha256"] == SOURCE_SHA256
    assert problems["tasks"] == 164
    container = json.loads((setup_dir / "quality" / "grading-container.json").read_text())
    assert container["machine"] == "x86_64"
    assert container["pulled"] is True
    # The prepared dataset copy is byte-identical to the pinned source.
    prepared = setup_dir / "quality" / "HumanEvalPlus-v0.1.10.jsonl.gz"
    assert hashlib.sha256(prepared.read_bytes()).hexdigest() == (
        hashlib.sha256(PROBLEMS.read_bytes()).hexdigest()
    )
    submissions = summary["prepared_submissions"]
    assert set(submissions) == {
        "fresh-context",
        "growing-session",
        "humaneval-plus",
        "sampled-ruler",
        "default-suite",
    }
    coding_text = Path(submissions["humaneval-plus"]).read_text()
    payload = json.loads(coding_text)
    assert payload["workload"]["profile"] == "humaneval-plus"
    assert payload["workload"]["problems_path"] == str(prepared)
    assert "max_tokens" not in payload["workload"]
    assert payload["run_config"] is None
    assert "secret-124" not in coding_text
    assert "hf_sensitive_124" not in coding_text
    assert "secret-124" not in json.dumps(summary)

    client = api(results)
    launched = client.post("/api/runs", json=coding_payload())
    assert launched.status_code == 201, launched.text
    run = launched.json()
    assert run["status"] == "completed"
    assert run["coverage"]["coding"] == {
        "expected": 164,
        "attempted": 164,
        "generated": 164,
        "graded": 164,
    }
    assert run["coding"]["correct"] == 82
    assert run["coding"]["accuracy"] == 0.5
    assert run["coding"]["identity"]["status"] == "verified"
    # The launch graded against the prepared copy, not the submitted path.
    assert run["workload"]["problems_path"] == str(prepared)
    gate = run_result(results, run["run_id"])
    assert gate["prepared"] is True
    assert gate["datasets"]["source_sha256"] == SOURCE_SHA256
    assert gate["grading_container"]["source"] == str(IMAGE)
    assert gate["grading_container"]["machine"] == "x86_64"
    html = client.get("/runs/" + run["run_id"]).text
    assert "HumanEval+ coding results" in html
    assert client.get("/api/runs/" + run["run_id"]).json() == run
    cli = CliRunner().invoke(app, ["result", run["run_id"], "--results", str(results)])
    assert cli.exit_code == 0
    assert json.loads(cli.stdout) == run
    assert counts["pulls"] == 1
    inputs = list((results / run["run_id"]).glob("HumanEval_*/input.jsonl"))
    assert all(
        json.loads(path.read_text())["output_length"] == CODING_OUTPUT_BUDGET for path in inputs
    )


def test_setup_preparation_launches_coding_through_cli(tmp_path, monkeypatch):
    """The prepared coding submission launches through the public CLI gate."""
    monkeypatch.setattr(subprocess, "run", run_fake())
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    result = run_setup_cli(setup_dir, setup_answers())
    assert result.exit_code == 0, result.output
    summary = summary_of(result)
    submitted = Path(summary["prepared_submissions"]["humaneval-plus"])
    run_config = json.dumps(
        {
            "reasoning_effort": "high",
            "temperature": 0.5,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "presence_penalty": None,
            "repetition_penalty": None,
        }
    )
    cli = CliRunner().invoke(
        app,
        [
            "run",
            str(submitted),
            "--results",
            str(results),
            "--setup-directory",
            str(setup_dir),
            "--no-interactive",
            "--run-config",
            run_config,
        ],
    )
    assert cli.exit_code == 0, cli.output
    run = json.loads(cli.stdout[cli.stdout.index("{") :])
    assert run["status"] == "completed"
    gate = run_result(results, run["run_id"])
    assert gate["prepared"] is True
    expected = setup_dir / "quality" / "HumanEvalPlus-v0.1.10.jsonl.gz"
    assert run["workload"]["problems_path"] == str(expected)


# --- 2. Reuse and repair -------------------------------------------------------


def test_prepared_quality_resources_are_reused_and_repaired(tmp_path, monkeypatch):
    """A second setup re-verifies the pinned dataset and grading container
    instead of re-pulling; a tampered or truncated dataset copy is refused at
    launch and a re-run of the setup repairs it."""
    counts: dict[str, Any] = {}
    monkeypatch.setattr(subprocess, "run", run_fake(counts))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    assert run_setup_cli(setup_dir, setup_answers()).exit_code == 0
    assert counts["pulls"] == 1
    prepared = setup_dir / "quality" / "HumanEvalPlus-v0.1.10.jsonl.gz"
    problems_path = setup_dir / "quality" / "problems.json"
    first_stat = problems_path.stat()
    second = run_setup_cli(setup_dir, setup_answers())
    assert second.exit_code == 0, second.output
    assert counts["pulls"] == 1, "the grading container must not be re-pulled"
    assert problems_path.stat().st_mtime_ns == first_stat.st_mtime_ns
    # The prepared copy is re-verified, not re-copied.
    assert hashlib.sha256(prepared.read_bytes()).hexdigest() == (
        hashlib.sha256(PROBLEMS.read_bytes()).hexdigest()
    )

    client = api(results)
    # Tamper the prepared dataset: the launch refuses, naming the dataset.
    prepared.write_bytes(b"tampered")
    refused = client.post("/api/runs", json=coding_payload())
    assert refused.status_code == 422, refused.text
    assert "dataset" in refused.text
    assert {p.name for p in results.iterdir()} == {"setup"}

    # Re-running the setup re-prepares the dataset; the launch then passes.
    repaired = run_setup_cli(setup_dir, setup_answers())
    assert repaired.exit_code == 0, repaired.output
    assert counts["pulls"] == 1
    assert hashlib.sha256(prepared.read_bytes()).hexdigest() == (
        hashlib.sha256(PROBLEMS.read_bytes()).hexdigest()
    )
    launched = client.post("/api/runs", json=coding_payload())
    assert launched.status_code == 201, launched.text
    assert launched.json()["status"] == "completed"

    # A truncated archive (a valid gzip prefix cut short) is refused the same
    # way and the setup repairs the copy without re-pulling.
    truncated = prepared.read_bytes()[:100]
    prepared.write_bytes(truncated)
    refused = client.post("/api/runs", json=coding_payload())
    assert refused.status_code == 422, refused.text
    assert "dataset" in refused.text
    assert {p.name for p in results.iterdir()} == {
        "setup",
        launched.json()["run_id"],
    }
    repaired = run_setup_cli(setup_dir, setup_answers())
    assert repaired.exit_code == 0, repaired.output
    assert counts["pulls"] == 1
    assert hashlib.sha256(prepared.read_bytes()).hexdigest() == (
        hashlib.sha256(PROBLEMS.read_bytes()).hexdigest()
    )


# --- 3. Interrupted preparation -----------------------------------------------


def test_interrupted_preparation_resumes_after_save(tmp_path, monkeypatch):
    """A Ctrl-C during the grading-container pull saves the interrupted
    preparation; the session resumes and completes the same preparation."""
    interrupted = {"n": 0}

    base = run_fake()

    def interrupt_once(command, **kwargs):
        args = [str(part) for part in command]
        if args[:2] == ["docker", "pull"]:
            interrupted["n"] += 1
            if interrupted["n"] == 1:
                raise KeyboardInterrupt
        return base(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", interrupt_once)
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    first = run_setup_cli(setup_dir, setup_answers())
    assert first.exit_code == 130, first.output
    assert "Setup interrupted; saved progress remains" in first.output
    saved = state(setup_dir)
    assert saved.preparations["datasets"].status == "complete"
    assert saved.preparations["grading_container"].status == "in-progress"
    # The coding submission exists only while both quality preparations pass.
    submissions = setup_dir / "submissions"
    assert (submissions / "fresh-context.json").is_file()
    assert (submissions / "humaneval-plus.json").is_file() is False
    client = api(results)
    view = client.get("/api/setup", params=[("directory", str(setup_dir))]).json()
    assert view["speed_ready"] is True
    assert "humaneval-plus" not in view.get("prepared_submissions", {})

    # Resume: the dataset re-verifies, the pull completes, the submission
    # appears.
    monkeypatch.setattr(subprocess, "run", run_fake())
    resumed = run_setup_cli(setup_dir, setup_answers())
    assert resumed.exit_code == 0, resumed.output
    final = state(setup_dir)
    assert final.preparations["grading_container"].status == "complete"
    assert (submissions / "humaneval-plus.json").is_file()


# --- 4. Dependency failures ----------------------------------------------------


def test_docker_unavailability_is_honest_and_refuses_launch(tmp_path, monkeypatch):
    """Without a Docker daemon the grading-container entry stays pending with
    an actionable note; the coding launch refuses and no work is started."""
    monkeypatch.setattr(subprocess, "run", run_fake(daemon=False))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    result = run_setup_cli(setup_dir, setup_answers())
    assert result.exit_code == 0, result.output  # a preparation problem is not a session failure
    summary = summary_of(result)
    entries = {entry["name"]: entry for entry in summary["preparations"]}
    assert entries["datasets"]["status"] == "complete"
    assert entries["grading_container"]["status"] == "pending"
    assert "Docker is not available" in entries["grading_container"]["note"]
    assert "re-run bench setup" in entries["grading_container"]["note"]
    assert summary["speed_ready"] is True
    assert summary["default_suite_ready"] is False
    assert set(summary["prepared_submissions"]) == {
        "fresh-context",
        "growing-session",
        "sampled-ruler",
    }

    client = api(results)
    refused = client.post("/api/runs", json=coding_payload())
    assert refused.status_code == 422, refused.text
    assert "grading_container" in refused.text
    assert {p.name for p in results.iterdir()} == {"setup"}  # no run work on a refused launch


def test_wrong_grading_platform_is_honest_and_refuses_launch(tmp_path, monkeypatch):
    """An image that does not run as x86_64 under linux/amd64 keeps the
    grading-container entry pending; the coding launch refuses it."""
    monkeypatch.setattr(subprocess, "run", run_fake(machine="aarch64"))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    result = run_setup_cli(setup_dir, setup_answers())
    assert result.exit_code == 0, result.output
    summary = summary_of(result)
    entries = {entry["name"]: entry for entry in summary["preparations"]}
    assert entries["datasets"]["status"] == "complete"
    assert entries["grading_container"]["status"] == "pending"
    assert "x86_64" in entries["grading_container"]["note"]
    assert "aarch64" in entries["grading_container"]["note"]
    assert summary["default_suite_ready"] is False

    client = api(results)
    refused = client.post("/api/runs", json=coding_payload())
    assert refused.status_code == 422, refused.text
    assert {p.name for p in results.iterdir()} == {"setup"}  # no run work on a refused launch


def test_incomplete_generation_is_partial_and_keeps_no_runtime_secret(tmp_path, monkeypatch):
    """A prepared coding launch whose generation partially fails reports
    partial coverage, retains distinct failure outcomes, and keeps the
    runtime bearer key out of every retained artifact."""
    monkeypatch.setattr(subprocess, "run", run_fake(profile_fail=(1, 3)))
    monkeypatch.setenv("REAL_BENCH_API_KEY", "rt-secret-999")
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    assert run_setup_cli(setup_dir, setup_answers()).exit_code == 0

    client = api(results)
    launched = client.post("/api/runs", json=coding_payload())
    assert launched.status_code == 201, launched.text
    run = launched.json()
    assert run["status"] == "partial"
    coverage = run["coverage"]["coding"]
    # The HTTP-500 generations fail per task; the loop continues, so the run
    # still generates and grades the remaining 161 tasks.
    assert coverage == {"expected": 164, "attempted": 164, "generated": 161, "graded": 161}
    counts_outcome = run["coding"]["outcome_counts"]
    assert counts_outcome["generation_failed"] == 3
    # The fake passes index < 82; the three failed tasks were 1, 2, 3.
    assert counts_outcome["correct"] == 79
    assert counts_outcome["not_attempted"] == 0
    assert run["coding"]["correct"] == 79
    assert run["coding"]["full_suite_accuracy"] is None
    assert run["coding"]["identity"]["status"] == "verified"
    assert [r["outcome"] for r in run["outcomes"][:4]] == [
        "correct",
        "generation_failed",
        "generation_failed",
        "generation_failed",
    ]
    # Independent reads agree.
    assert client.get("/api/runs/" + run["run_id"]).json() == run

    # The runtime bearer key is never retained in any run artifact.
    run_dir = results / run["run_id"]
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, ValueError):
            continue
        assert "rt-secret-999" not in text, f"runtime key retained in {path.name}"
    # The retained setup record still redacts the declared serving key.
    setup_text = (run_dir / "setup.json").read_text()
    assert "secret-124" not in setup_text
    assert "[REDACTED]" in setup_text


def test_missing_docker_cli_is_honest_and_refuses_launch(tmp_path, monkeypatch):
    """A host without a Docker CLI keeps the grading-container entry pending
    with installation guidance (not a raw errno); the coding launch refuses."""
    base = run_fake()

    def docker_absent(command, **kwargs):
        if [str(part) for part in command][:1] == ["docker"]:
            raise FileNotFoundError(2, "No such file or directory", "docker")
        return base(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", docker_absent)
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    result = run_setup_cli(setup_dir, setup_answers())
    assert result.exit_code == 0, result.output
    summary = summary_of(result)
    entries = {entry["name"]: entry for entry in summary["preparations"]}
    assert entries["datasets"]["status"] == "complete"
    assert entries["grading_container"]["status"] == "pending"
    note = entries["grading_container"]["note"]
    assert "Docker CLI was not found" in note
    assert "install Docker" in note
    assert "re-run bench setup" in note
    assert summary["default_suite_ready"] is False

    client = api(results)
    refused = client.post("/api/runs", json=coding_payload())
    assert refused.status_code == 422, refused.text
    assert "grading_container" in refused.text
    assert {p.name for p in results.iterdir()} == {"setup"}  # no run work on a refused launch
