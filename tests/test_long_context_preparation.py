"""Setup-to-long-context launch: RULER preparation, reuse, interruption, failure.

Every case is driven through the public command boundary (the ``bench setup``
and ``bench run`` CLIs and the ``/api/setup`` / ``/api/runs`` endpoints) with
the preparation subprocesses (pinned-tools probe, tokenizer fetch, the Docker
CLI, the RULER source clone) and the scored run subprocesses (the RULER
preparation and aiperf) served by a controlled ``subprocess.run``: no network,
no native tool, no model weights, no git remote. The API uses its own
``results/setup`` directory as the setup state, exactly as ``bench run`` serves
a ``--setup-directory``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import fake_preparation
from fastapi.testclient import TestClient
from test_ruler_runs import CHECKPOINTS, TASKS
from typer.testing import CliRunner

from llm_bench import setup_state
from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_cli import app
from llm_bench.ordinary_runs import RULER_OUTPUT_BUDGET
from llm_bench.ruler_preparation import DATA_FILES, REVISION, SOURCE_URL

pytestmark = [
    pytest.mark.xdist_group("slow-setup"),
    pytest.mark.usefixtures("preparation_processes", "external_tool_process_adapter"),
]


def _load_source_prepare() -> Any:
    """Load ``scripts/ruler_source_prepare.py`` (a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "ruler_source_prepare",
        Path(__file__).parent.parent / "scripts" / "ruler_source_prepare.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        "serving.environment": {"API_KEY": "secret-124"},
        "requests.base_url": "http://localhost:8080",
        "requests.model_id": "test-model",
        "requests.context_window": 262144,
        "requests.generation": {"temperature": 0.5},
        "requests.reasoning": {"chat_template_kwargs": {"enable_thinking": True}},
    }


def run_config_payload(effort: str = "high") -> dict[str, Any]:
    return {
        "reasoning_effort": effort,
        "temperature": 0.5,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "presence_penalty": None,
        "repetition_penalty": None,
    }


def run_fake(
    counts: dict[str, Any] | None = None,
    *,
    helper: str | None = None,
    profile_fail: tuple[int, int] | None = None,
    budget_exhaust: bool = False,
) -> Any:
    """Controlled ``subprocess.run`` for the long-context preparation and a run.

    ``helper`` emulates the RULER source helper failing: ``"network"`` exits 3
    (a human-only prerequisite), ``"clone"`` exits 1. ``counts`` tallies helper
    fetches so reuse (no re-fetch) is observable. ``budget_exhaust`` makes the
    first sampled generation hit the fixed output budget (a ``length`` finish
    at ``RULER_OUTPUT_BUDGET``) to exercise budget exhaustion.
    """
    seen_prep_commands: list[dict[str, Any]] = []

    def external(command, **kwargs):
        args = [str(part) for part in command]
        if any(part.endswith("ruler_source_prepare.py") for part in args):
            if counts is not None:
                counts["fetches"] = counts.get("fetches", 0) + 1
            if helper is not None:
                code = 3 if helper == "network" else 1
                message = (
                    "The pinned RULER source could not be reached. This host needs "
                    "outbound network access. Restore network access and re-run "
                    "bench setup."
                    if helper == "network"
                    else "RULER source clone failed: boom"
                )
                return subprocess.CompletedProcess(command, code, "", message)
            return fake_preparation(command, **kwargs)
        if "-c" in args:
            return fake_preparation(command, **kwargs)
        if args and args[0] == "git" and "-C" in args:
            return fake_preparation(command, **kwargs)
        if any(part.endswith("speed_prepare.py") for part in args) or args[:1] == ["docker"]:
            return fake_preparation(command, **kwargs)
        if any(part.endswith("ruler_pilot.py") for part in args):
            directory = Path(args[args.index("--directory") + 1])
            directory.mkdir()
            max_tokens = int(args[args.index("--quality-output-limit") + 1])
            window = int(args[args.index("--context-window") + 1])
            template = json.loads(args[args.index("--chat-template-kwargs") + 1])
            template_path = Path(args[args.index("--chat-template-path") + 1])
            tokenizer = Path(args[args.index("--tokenizer") + 1])
            seed = int(args[args.index("--seed") + 1])
            if counts is not None:
                seen_prep_commands.append(
                    {
                        "chat_template_kwargs": template,
                        "chat_template_text": template_path.read_text(),
                        "chat_template_sha256": hashlib.sha256(
                            template_path.read_text().encode()
                        ).hexdigest(),
                        "seed": seed,
                        "context_window": window,
                        "quality_output_limit": max_tokens,
                    }
                )
                counts["preparations"] = seen_prep_commands
            directory.joinpath("workload.json").write_text(
                json.dumps(
                    {
                        "source_revision": REVISION,
                        "planned_requests": 117,
                        "tasks": list(TASKS),
                        "checkpoints": list(CHECKPOINTS),
                        "samples_per_task_checkpoint": 3,
                        "quality_output_limit": max_tokens,
                        "context_window": window,
                        "chat_template_kwargs": template,
                        "chat_template_sha256": hashlib.sha256(
                            template_path.read_text().encode()
                        ).hexdigest(),
                        "seed": seed,
                        "tokenizer_sha256": {
                            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in tokenizer.iterdir()
                            if path.is_file()
                        },
                    }
                )
            )
            preparation = []
            for task in TASKS:
                for checkpoint in CHECKPOINTS:
                    condition = directory / f"{task}-{checkpoint}" / task
                    condition.mkdir(parents=True)
                    condition.joinpath("validation.jsonl").write_text(
                        "".join(
                            json.dumps(
                                {"input": f"{task}-{checkpoint}-{index}", "outputs": ["expected"]}
                            )
                            + "\n"
                            for index in range(3)
                        )
                    )
                    preparation.append(
                        {
                            "task": task,
                            "checkpoint": checkpoint,
                            "exit_code": 0,
                            "samples": [
                                {"index": index, "input_tokens": checkpoint - 1}
                                for index in range(3)
                            ],
                        }
                    )
            directory.joinpath("preparation.json").write_text(json.dumps(preparation))
            return subprocess.CompletedProcess(command, 0, "", "")
        if "profile" in args:
            native = Path(args[args.index("--output-artifact-dir") + 1])
            native.mkdir(parents=True, exist_ok=True)
            name = native.parent.name.removeprefix("ruler-")
            task, checkpoint, index = name.rsplit("-", 2)
            if (
                profile_fail is not None
                and task == TASKS[0]
                and int(index) == 0
                and int(checkpoint) <= profile_fail[1]
            ):
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
            content = "expected"
            checkpoint_value = int(checkpoint)
            # Budget exhaustion: the first sampled generation runs the full
            # fixed output budget and stops with a "length" finish reason.
            exhausted = budget_exhaust and task == TASKS[0] and int(index) == 0
            finish_reason = "length" if exhausted else "stop"
            completion_tokens = RULER_OUTPUT_BUDGET if exhausted else 1
            packet = json.dumps(
                {
                    "usage": {
                        "prompt_tokens": checkpoint_value - 1,
                        "completion_tokens": completion_tokens,
                    },
                    "choices": [{"delta": {"content": content}, "finish_reason": finish_reason}],
                }
            )
            (native / "profile_export_raw.jsonl").write_text(
                json.dumps(
                    {
                        "metadata": {"x_request_id": "ruler-request"},
                        "status": 200,
                        "responses": [
                            {
                                "packets": [
                                    {"name": "data", "value": packet},
                                    {"name": "data", "value": "[DONE]"},
                                ]
                            }
                        ],
                    }
                )
                + "\n"
            )
            return subprocess.CompletedProcess(command, 0, "", "")
        return fake_preparation(command, **kwargs)

    return external


def run_setup_cli(directory: Path, answers: dict[str, Any]) -> Any:
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


def state(directory: Path) -> setup_state.SetupState:
    loaded = setup_state.load_state(directory)
    assert loaded is not None
    return loaded


def api(results: Path) -> TestClient:
    return TestClient(create_app(results))


def prepared_submission(results: Path, setup_dir: Path) -> dict[str, Any]:
    path = Path(setup_dir) / "submissions" / "sampled-ruler.json"
    payload = json.loads(path.read_text())
    payload["run_config"] = run_config_payload()
    return payload


# --- 1. Setup-to-long-context execution and independent result reads -----------


def test_setup_preparation_launches_long_context_and_reads_independently(
    tmp_path, monkeypatch
) -> None:
    """A prepared setup launches the sampled-RULER profile against the prepared
    source; the result is read back independently by the API and the CLI."""
    counts: dict[str, Any] = {}
    monkeypatch.setattr(subprocess, "run", run_fake(counts))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    result = run_setup_cli(setup_dir, setup_answers())
    assert result.exit_code == 0, result.output
    summary = summary_of(result)
    entries = {entry["name"]: entry for entry in summary["preparations"]}
    assert entries["long_context"]["status"] == "complete"
    assert summary["speed_ready"] is True
    assert summary["default_suite_ready"] is True
    # The source evidence is retained in the state directory.
    evidence = json.loads((setup_dir / "long-context" / "source.json").read_text())
    assert evidence["source"] == SOURCE_URL
    assert evidence["revision"] == REVISION
    assert set(evidence["data"]) == set(DATA_FILES)
    # The prepared submission exists with the prepared checkout as source and
    # carries no secret.
    submission_path = Path(summary["prepared_submissions"]["sampled-ruler"])
    payload = json.loads(submission_path.read_text())
    assert payload["workload"]["profile"] == "sampled-ruler"
    assert payload["workload"]["source_path"] == str(setup_dir / "long-context" / "source")
    assert "max_tokens" not in payload["workload"]
    assert payload["run_config"] is None
    assert "secret-124" not in submission_path.read_text()
    assert "secret-124" not in json.dumps(summary)

    client = api(results)
    launched = client.post("/api/runs", json=prepared_submission(results, setup_dir))
    assert launched.status_code == 201, launched.text
    run = launched.json()
    assert run["status"] == "completed"
    assert run["coverage"]["long_context"] == {"expected": 117, "accounted": 117, "scored": 117}
    assert run["ruler"]["sampled_diagnostic"] is True
    assert all(row["output_reservation_tokens"] == RULER_OUTPUT_BUDGET for row in run["outcomes"])
    # The launch generated from the prepared checkout, not the submitted path.
    assert run["workload"]["source_path"] == str(setup_dir / "long-context" / "source")
    gate = json.loads((results / run["run_id"] / "long-context-preparation.json").read_text())
    assert gate["prepared"] is True
    assert gate["source"]["revision"] == REVISION
    assert gate["path"] == str(setup_dir / "long-context" / "source")
    # The 117 prompts were prepared under the final run configuration.
    assert counts["preparations"][0]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "high",
    }
    assert len(run["outcomes"]) == 117
    html = client.get("/runs/" + run["run_id"]).text
    assert "Sampled RULER long-context results" in html
    assert client.get("/api/runs/" + run["run_id"]).json() == run
    cli = CliRunner().invoke(app, ["result", run["run_id"], "--results", str(results)])
    assert cli.exit_code == 0
    assert json.loads(cli.stdout) == run
    assert counts["fetches"] == 1


def test_prepared_long_context_source_is_reused_without_refetch(tmp_path, monkeypatch) -> None:
    """A second setup re-verifies the prepared source instead of re-cloning it."""
    counts: dict[str, Any] = {}
    monkeypatch.setattr(subprocess, "run", run_fake(counts))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    assert run_setup_cli(setup_dir, setup_answers()).exit_code == 0
    assert counts["fetches"] == 1
    evidence_path = setup_dir / "long-context" / "source.json"
    first_stat = evidence_path.stat()
    second = run_setup_cli(setup_dir, setup_answers())
    assert second.exit_code == 0, second.output
    assert counts["fetches"] == 1, "the RULER source must not be re-fetched"
    assert evidence_path.stat().st_mtime_ns == first_stat.st_mtime_ns
    assert summary_of(second)["default_suite_ready"] is True


# --- 2. Interrupted preparation -----------------------------------------------


def test_interrupted_preparation_resumes_after_save(tmp_path, monkeypatch) -> None:
    """A Ctrl-C during the RULER source fetch saves the interrupted
    preparation; the session resumes and completes the same preparation."""
    interrupted = {"n": 0}
    base = run_fake()

    def interrupt_once(command, **kwargs):
        args = [str(part) for part in command]
        if any(part.endswith("ruler_source_prepare.py") for part in args):
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
    assert saved.preparations["long_context"].status == "in-progress"
    # The long-context submission exists only while the preparation passes.
    submissions = setup_dir / "submissions"
    assert (submissions / "fresh-context.json").is_file()
    assert (submissions / "sampled-ruler.json").is_file() is False
    client = api(results)
    view = client.get("/api/setup", params=[("directory", str(setup_dir))]).json()
    assert view["speed_ready"] is True
    assert view["default_suite_ready"] is False
    assert "sampled-ruler" not in view.get("prepared_submissions", {})

    # Resume: the source re-verifies, the submission appears.
    monkeypatch.setattr(subprocess, "run", run_fake())
    resumed = run_setup_cli(setup_dir, setup_answers())
    assert resumed.exit_code == 0, resumed.output
    final = state(setup_dir)
    assert final.preparations["long_context"].status == "complete"
    assert (submissions / "sampled-ruler.json").is_file()


# --- 3. Prerequisite failures --------------------------------------------------


def test_source_download_failure_is_honest_and_refuses_launch(tmp_path, monkeypatch) -> None:
    """An unreachable RULER source keeps the entry pending with actionable
    guidance; the long-context launch refuses and no work is started."""
    monkeypatch.setattr(subprocess, "run", run_fake(helper="network"))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    result = run_setup_cli(setup_dir, setup_answers())
    assert result.exit_code == 0, result.output  # a preparation problem is not a session failure
    summary = summary_of(result)
    entries = {entry["name"]: entry for entry in summary["preparations"]}
    assert entries["datasets"]["status"] == "complete"
    assert entries["long_context"]["status"] == "pending"
    note = entries["long_context"]["note"]
    assert "re-run bench setup" in note
    assert "network" in note.lower()
    assert summary["speed_ready"] is True
    assert summary["default_suite_ready"] is False
    assert "sampled-ruler" not in summary.get("prepared_submissions", {})

    client = api(results)
    refused = client.post(
        "/api/runs",
        json={
            "setup": {"declared": declared(), "observed": {}},
            "run_config": run_config_payload(),
            "workload": {"profile": "sampled-ruler", "source_path": "/pinned/ruler"},
        },
    )
    assert refused.status_code == 422, refused.text
    assert "long-context" in refused.text
    assert {p.name for p in results.iterdir()} == {"setup"}  # no run work on a refused launch


def test_tampered_source_is_refused_and_repaired(tmp_path, monkeypatch) -> None:
    """A tampered or deleted prepared source is refused at launch; a re-run of
    the setup repairs it without re-fetching."""
    counts: dict[str, Any] = {}
    monkeypatch.setattr(subprocess, "run", run_fake(counts))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    assert run_setup_cli(setup_dir, setup_answers()).exit_code == 0
    client = api(results)
    payload = prepared_submission(results, setup_dir)
    assert client.post("/api/runs", json=payload).status_code == 201

    # Tamper a prepared data file: the launch refuses, naming the source.
    json_dir = setup_dir / "long-context" / "source" / "scripts" / "data" / "synthetic" / "json"
    data_file = json_dir / "squad.json"
    data_file.write_bytes(b"tampered")
    refused = client.post("/api/runs", json=payload)
    assert refused.status_code == 422, refused.text
    assert "source" in refused.text

    # Re-running the setup re-fetches the source; the launch then passes.
    repaired = run_setup_cli(setup_dir, setup_answers())
    assert repaired.exit_code == 0, repaired.output
    assert counts["fetches"] == 2
    launched = client.post("/api/runs", json=payload)
    assert launched.status_code == 201, launched.text
    assert launched.json()["status"] == "completed"

    # A deleted checkout is not advertised ready and the launch refuses.
    shutil.rmtree(setup_dir / "long-context" / "source" / "scripts")
    view = client.get("/api/setup", params=[("directory", str(setup_dir))]).json()
    assert view["default_suite_ready"] is False
    refused = client.post("/api/runs", json=payload)
    assert refused.status_code == 422, refused.text


# --- 4. Repreparation under a changed run configuration -------------------------


def test_prepared_inputs_reprepare_under_the_final_run_configuration(tmp_path, monkeypatch) -> None:
    """A reasoning change re-prepares the 117 prompts against the final
    configuration before scored execution; the scored run never uses inputs
    prepared under a different reasoning choice."""
    counts: dict[str, Any] = {}
    monkeypatch.setattr(subprocess, "run", run_fake(counts))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    assert run_setup_cli(setup_dir, setup_answers()).exit_code == 0
    client = api(results)
    payload = prepared_submission(results, setup_dir)

    first = client.post("/api/runs", json=payload | {"run_config": run_config_payload("high")})
    assert first.status_code == 201, first.text
    second = client.post("/api/runs", json=payload | {"run_config": run_config_payload("medium")})
    assert second.status_code == 201, second.text
    # Each scored run prepared its own inputs against the final configuration.
    efforts = [
        entry["chat_template_kwargs"]["reasoning_effort"] for entry in counts["preparations"]
    ]
    assert efforts == ["high", "medium"]
    assert all(
        entry["chat_template_kwargs"]["enable_thinking"] is True for entry in counts["preparations"]
    )
    assert first.json()["coverage"]["long_context"]["scored"] == 117
    assert second.json()["coverage"]["long_context"]["scored"] == 117


def test_template_change_reprepares_prompts_under_the_new_template(tmp_path, monkeypatch) -> None:
    """A declared template change re-prepares the 117 prompts against the new
    effective template on the next launch, without invalidating the pinned source."""
    counts: dict[str, Any] = {}
    monkeypatch.setattr(subprocess, "run", run_fake(counts))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    assert run_setup_cli(setup_dir, setup_answers()).exit_code == 0
    client = api(results)
    payload = prepared_submission(results, setup_dir)
    first = client.post("/api/runs", json=payload)
    assert first.status_code == 201, first.text
    # The first launch sized the prompts with the tokenizer's own template.
    original_template = (setup_dir / "speed" / "tokenizer" / "chat_template.jinja").read_text()
    assert counts["preparations"][0]["chat_template_text"] == original_template
    assert first.json()["coverage"]["long_context"]["scored"] == 117

    changed = run_setup_cli(setup_dir, {"model.template_overrides": {"chat_template": "x"}})
    assert changed.exit_code == 0, changed.output
    summary = summary_of(changed)
    entries = {entry["name"]: entry for entry in summary["preparations"]}
    # The template change does not revalidate the pinned source entry: only the
    # configuration-dependent 117 prompts are re-prepared, at launch.
    assert entries["long_context"]["status"] == "complete"
    assert entries["long_context"]["needs_revalidation"] is False
    assert summary["default_suite_ready"] is True
    payload = prepared_submission(results, setup_dir)
    second = client.post("/api/runs", json=payload)
    assert second.status_code == 201, second.text
    # The second run generated its prompts with the declared override in place
    # of the tokenizer's own template, under the unchanged reasoning settings.
    assert counts["preparations"][-1]["chat_template_text"] == "x"
    assert counts["preparations"][-1]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "high",
    }
    assert second.json()["coverage"]["long_context"]["scored"] == 117


# --- 5. Credential nonretention -------------------------------------------------


def test_prepared_long_context_launch_keeps_no_runtime_secret(tmp_path, monkeypatch) -> None:
    """A prepared long-context launch keeps the runtime bearer key out of every
    retained artifact, and the retained setup record redacts the declared key."""
    monkeypatch.setattr(subprocess, "run", run_fake())
    monkeypatch.setenv("REAL_BENCH_API_KEY", "rt-secret-999")
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    assert run_setup_cli(setup_dir, setup_answers()).exit_code == 0
    client = api(results)
    launched = client.post("/api/runs", json=prepared_submission(results, setup_dir))
    assert launched.status_code == 201, launched.text
    run = launched.json()
    assert run["status"] == "completed"
    run_dir = results / run["run_id"]
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, ValueError):
            continue
        assert "rt-secret-999" not in text, f"runtime key retained in {path.name}"
    setup_text = (run_dir / "setup.json").read_text()
    assert "secret-124" not in setup_text
    assert "[REDACTED]" in setup_text


# --- 6. Incomplete generation stays partial --------------------------------------


def test_incomplete_generation_is_partial_with_distinct_outcomes(tmp_path, monkeypatch) -> None:
    """A prepared launch whose first generation fails reports partial coverage
    with distinct failure outcomes; the remaining requests are scored."""
    monkeypatch.setattr(subprocess, "run", run_fake(profile_fail=(0, 32768)))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    assert run_setup_cli(setup_dir, setup_answers()).exit_code == 0
    client = api(results)
    launched = client.post("/api/runs", json=prepared_submission(results, setup_dir))
    assert launched.status_code == 201, launched.text
    run = launched.json()
    assert run["status"] == "partial"
    assert run["coverage"]["long_context"] == {"expected": 117, "accounted": 117, "scored": 116}
    assert run["outcomes"][0]["outcome"] == "failed"
    assert run["outcomes"][0]["score"] is None


def test_generation_that_hits_the_output_budget_is_still_scored(tmp_path, monkeypatch) -> None:
    """A generation that runs the full fixed RULER output budget stops with a
    ``length`` finish reason; the request is still scored and the finish
    reason stays visible in the outcome."""
    monkeypatch.setattr(subprocess, "run", run_fake(budget_exhaust=True))
    results = tmp_path / "results"
    results.mkdir()
    setup_dir = results / "setup"
    assert run_setup_cli(setup_dir, setup_answers()).exit_code == 0
    client = api(results)
    launched = client.post("/api/runs", json=prepared_submission(results, setup_dir))
    assert launched.status_code == 201, launched.text
    run = launched.json()
    assert run["status"] == "completed"
    assert run["coverage"]["long_context"] == {"expected": 117, "accounted": 117, "scored": 117}
    exhausted = run["outcomes"][0]
    assert exhausted["outcome"] == "success"
    assert exhausted["finish_reason"] == "length"
    assert exhausted["score"] == 100.0
    # The budget exhaustion is visible in the retained outcome and in the
    # native evidence, and the fixed budget is recorded per request.
    outcomes = json.loads((results / run["run_id"] / "outcomes.json").read_text())
    assert outcomes[0]["finish_reason"] == "length"
    assert outcomes[0]["output_reservation_tokens"] == RULER_OUTPUT_BUDGET
    native_raw = (
        results
        / run["run_id"]
        / f"ruler-{TASKS[0]}-{CHECKPOINTS[0]}-0"
        / "native"
        / "profile_export_raw.jsonl"
    ).read_text()
    # The native export retains the streamed packet as a JSON string, so parse
    # it out to confirm the budget exhaustion the endpoint reported.
    outer = json.loads(native_raw)
    packet = json.loads(outer["responses"][0]["packets"][0]["value"])
    assert packet["choices"][0]["finish_reason"] == "length"
    assert packet["usage"]["completion_tokens"] == RULER_OUTPUT_BUDGET


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
            "environment": {"API_KEY": "secret-124"},
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


def test_hydration_validation_catches_helpers_that_exit_clean_with_bad_data(tmp_path) -> None:
    """The pinned helpers can exit 0 with an empty or incomplete download:
    the QA script's status is its final command and the essay helper catches
    per-URL failures. The preparation must therefore check the data itself
    before trusting it, or a bad file would be hashed into the manifest a
    later launch verifies."""
    prepare = _load_source_prepare()
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    # A valid hydration: the structures the upstream generators read.
    (json_dir / "squad.json").write_text('{"data": [{"id": 1}]}')
    (json_dir / "hotpotqa.json").write_text('[{"id": 2}]')
    (json_dir / "PaulGrahamEssays.json").write_text('{"text": "' + "x" * 5000 + '"}')
    (json_dir / "english_words.json").write_bytes(b"word " * 100)
    assert prepare.hydration_problems(json_dir) == []

    # An empty SQuAD file (a failed wget masked by a later successful
    # download) is caught.
    (json_dir / "squad.json").write_text("")
    problems = prepare.hydration_problems(json_dir)
    assert any("squad.json" in problem for problem in problems)

    # An unfetched git-lfs pointer for the word list is caught.
    (json_dir / "squad.json").write_text('{"data": [{"id": 1}]}')
    (json_dir / "english_words.json").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:affcd6d4\nsize 8564991\n"
    )
    problems = prepare.hydration_problems(json_dir)
    assert any("english_words.json" in problem for problem in problems)
