import gzip
import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from typer.testing import CliRunner

from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_cli import app

pytestmark = pytest.mark.usefixtures("external_tool_process_adapter")


def submission():
    return {
        "setup": {
            "declared": {
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
            },
            "observed": {},
        },
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


def fresh_submission(context_window=262144):
    payload = submission()
    payload["setup"]["declared"]["requests"]["context_window"] = context_window
    payload["workload"] = {
        "profile": "fresh-context-speed",
        "scenario": "fresh-context",
        "checkpoints": [4096, 16384, 32768, 65536, 131072, 196608],
        "output_targets": [256, 1024],
        "repetitions": 5,
        "warmup_repetitions": 1,
        "request_timeout_seconds": 900,
        "cache_configuration": {"prefix_caching": True},
        "cache_observations": [],
    }
    return payload


def growing_submission(context_window=196608):
    payload = submission()
    payload["setup"]["declared"]["requests"]["context_window"] = context_window
    payload["workload"] = {
        "profile": "growing-session-speed",
        "scenario": "growing-session",
        "checkpoints": [4096, 16384, 32768, 65536, 131072, 196608],
        "output_targets": [256, 1024],
        "repetitions": 5,
        "warmup_repetitions": 1,
        "request_timeout_seconds": 900,
        "cache_configuration": {"prefix_caching": True},
        "cache_observations": [],
    }
    return payload


def _growing_fake(calls, expected_history):
    def external(command, **kwargs):
        calls.append(list(command))
        if any(str(part).endswith("performance_inputs.py") for part in command):
            specification = json.loads(Path(command[command.index("--spec") + 1]).read_text())
            directory = Path(command[command.index("--directory") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            nonce = specification["nonce"]
            is_warmup = nonce.endswith("-warmup")
            assert specification["chat_template_kwargs"] == {
                "enable_thinking": True,
                "reasoning_effort": "xhigh",
            }
            assert specification["request_settings"] == {
                "temperature": 0.5,
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
            }
            if is_warmup:
                base, expected = nonce, []
            else:
                base = nonce[: nonce.rindex("-turn")]
                expected = expected_history.setdefault(base, [])
            assert specification["history"] == expected
            history = list(specification["history"])
            structural = list(specification["structural"])
            assert len(structural) == 4
            assert structural[0]["role"] == "user"
            assert structural[1]["tool_calls"][0]["function"]["name"] == "read_log"
            assert structural[1]["tool_calls"][0]["function"]["arguments"] == "{}"
            assert structural[2]["role"] == "tool"
            assert structural[3]["role"] == "user"
            messages = history + structural
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
            (directory / "sizing.json").write_text(
                json.dumps(
                    {
                        "checkpoint": specification["checkpoint"],
                        "local_input_tokens": specification["checkpoint"],
                        "history_messages": len(history),
                        "bare_content_tokens": 0,
                        "template_and_structured_history_tokens": 0,
                    }
                )
                + "\n"
            )
            if not is_warmup:
                assistant = {
                    "role": "assistant",
                    "content": f"assistant-response-{directory.name}",
                    "reasoning_content": "thought",
                }
                expected_history[base] = [*history, *structural, assistant]
            return subprocess.CompletedProcess(command, 0, "", "")

        dataset = Path(command[command.index("--input-file") + 1])
        native = Path(command[command.index("--output-artifact-dir") + 1])
        native.mkdir(parents=True, exist_ok=True)
        row = json.loads(dataset.read_text().splitlines()[0])
        output_target = row["output_length"]
        name = dataset.parent.name
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
                "output_token_throughput_per_user": {
                    "value": 80.0,
                    "unit": "tokens/sec/user",
                },
                "prefill_throughput_per_user": {
                    "value": 900.0,
                    "unit": "tokens/sec/user",
                },
                "usage_prompt_tokens": {"value": checkpoint, "unit": "tokens"},
                "usage_completion_tokens": {"value": output_target, "unit": "tokens"},
            },
        }
        (native / "profile_export_raw.jsonl").write_text(json.dumps(raw) + "\n")
        (native / "profile_export.jsonl").write_text(json.dumps(measured) + "\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    return external


def test_growing_session_curve_accumulates_history_and_separates_sessions(tmp_path, monkeypatch):
    payload = growing_submission(context_window=131072)
    calls = []
    expected_history = {}
    monkeypatch.setattr(subprocess, "run", _growing_fake(calls, expected_history))
    client = TestClient(create_app(tmp_path))
    submitted = client.post("/api/runs", json=payload)
    assert submitted.status_code == 201, submitted.text
    result = TestClient(create_app(tmp_path)).get("/api/runs/" + submitted.json()["run_id"]).json()

    assert result["evidence"] == "run"
    assert result["status"] == "completed"
    assert result["coverage"] == {
        "growing_session": {"expected": 60, "accounted": 60, "successful": 40}
    }
    outcomes = result["outcomes"]
    assert len(outcomes) == 60
    assert len({row["session"] for row in outcomes}) == 10
    assert len({(row["session"], row["input_tokens"]) for row in outcomes}) == 60
    assert {(row["repetition"], row["output_target"]) for row in outcomes} == {
        (repetition, target) for repetition in range(5) for target in (256, 1024)
    }
    unsupported = [row for row in outcomes if row["outcome"] == "unsupported"]
    assert {row["input_tokens"] for row in unsupported} == {131072, 196608}
    assert len(unsupported) == 20
    assert all(
        row["endpoint_input_matches"] is True for row in outcomes if row["outcome"] == "success"
    )
    for session, accumulated in expected_history.items():
        assert accumulated
        assert accumulated[0] == {
            "role": "user",
            "content": f"{session} checkpoint 4096. Read the log.",
        }
    curve = result["growing_session_curve"]
    assert len(curve) == 12
    tail = next(p for p in curve if p["input_tokens"] == 196608 and p["output_target"] == 256)
    assert tail["outcomes"]["unsupported"] == 5
    mid = next(p for p in curve if p["input_tokens"] == 131072 and p["output_target"] == 256)
    assert mid["outcomes"]["unsupported"] == 5
    head = next(p for p in curve if p["input_tokens"] == 4096 and p["output_target"] == 256)
    assert head["outcomes"]["success"] == 5
    assert head["metrics"]["ttft"]["values"] == [1000.0] * 5
    construction = result["construction"]
    assert construction["sizing_requests"] > 0
    assert construction["construction_seconds"] >= 0
    assert len(result["warmups"]) == 10
    assert all(
        w["excluded_from_measurements"] is True and w["exit_code"] == 0 for w in result["warmups"]
    )
    assert "Growing-session speed curve" in client.get("/runs/" + result["run_id"]).text


def test_growing_session_accounts_unsupported_without_invoking_tools(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported matrix invoked an external tool")

    monkeypatch.setattr(subprocess, "run", forbidden)
    payload = growing_submission(context_window=4096)
    client = TestClient(create_app(tmp_path))
    submitted = client.post("/api/runs", json=payload)
    assert submitted.status_code == 201, submitted.text
    result = TestClient(create_app(tmp_path)).get("/api/runs/" + submitted.json()["run_id"]).json()
    assert len(result["outcomes"]) == 60
    assert all(row["outcome"] == "unsupported" for row in result["outcomes"])
    assert result["warmups"] == []
    assert result["growing_session_curve"][0]["outcomes"]["unsupported"] == 5


def test_cli_growing_session_run_readable_by_api_view(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported growing matrix invoked an external tool")

    monkeypatch.setattr(subprocess, "run", forbidden)
    payload = growing_submission(context_window=4096)
    source = tmp_path / "submission.json"
    source.write_text(json.dumps(payload))
    runs = tmp_path / "runs"
    execution = CliRunner().invoke(
        app,
        ["run", str(source), "--results", str(runs), "--setup-directory", str(tmp_path / "setup")],
    )
    assert execution.exit_code == 0, execution.output
    run_id = json.loads(execution.output)["run_id"]
    result = TestClient(create_app(runs)).get("/api/runs/" + run_id).json()
    assert result["evidence"] == "run"
    assert len(result["outcomes"]) == 60
    assert all(row["outcome"] == "unsupported" for row in result["outcomes"])
    assert result["growing_session_curve"][0]["outcomes"]["unsupported"] == 5
    view = TestClient(create_app(runs)).get("/runs/" + run_id)
    assert "Growing-session speed curve" in view.text
    read = CliRunner().invoke(app, ["result", run_id, "--results", str(runs)])
    assert read.exit_code == 0, read.output
    assert json.loads(read.output) == result


def test_submit_retains_native_results_readable_by_api_cli_and_view(tmp_path, monkeypatch):
    def tool(command, **kwargs):
        dataset = Path(command[command.index("--input-file") + 1])
        sent = json.loads(dataset.read_text())
        assert sent["extra"]["chat_template_kwargs"] == {
            "enable_thinking": True,
            "reasoning_effort": "xhigh",
        }
        assert sent["extra"]["temperature"] == 0.5
        assert sent["output_length"] == 256
        native = Path(command[command.index("--output-artifact-dir") + 1])
        native.mkdir()
        source = Path("tests/fixtures/pilots/performance-121/native")
        for name in ["profile_export.jsonl", "profile_export_raw.jsonl"]:
            (native / name).write_bytes(gzip.decompress((source / (name + ".gz")).read_bytes()))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", tool)
    client = TestClient(create_app(tmp_path))
    response = client.post("/api/runs", json=submission())
    assert response.status_code == 201, response.text
    run = response.json()
    result = TestClient(create_app(tmp_path)).get("/api/runs/" + run["run_id"]).json()
    assert result["evidence"] == "validation"
    assert result["status"] == "completed"
    outcome = result["outcomes"][0]
    assert outcome["outcome"] == "success"
    assert outcome["metrics"]["ttft"]["value"] == 954.6156249999999
    assert outcome["metrics"]["ttft"]["unit"] == "ms"
    assert outcome["metrics"]["ttft"]["source"] == "native/profile_export.jsonl:time_to_first_token"
    assert outcome["actual_output_tokens"] == 256
    assert result["families"]["coding"] == "missing"
    directory = tmp_path / run["run_id"]
    assert (directory / "native/profile_export_raw.jsonl").exists()
    assert "secret-124" not in "".join(p.read_text() for p in directory.rglob("*") if p.is_file())
    read = CliRunner().invoke(app, ["result", run["run_id"], "--results", str(tmp_path)])
    assert read.exit_code == 0, read.output
    assert json.loads(read.output) == result
    view = client.get("/runs/" + run["run_id"])
    assert view.status_code == 200
    assert "954.615" in view.text and "validation" in view.text


def test_rejects_malformed_metadata_before_invoking_tool(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid submission invoked a tool")

    monkeypatch.setattr(subprocess, "run", forbidden)
    client = TestClient(create_app(tmp_path))
    for category in ("hardware", "model", "serving", "requests", "benchmark"):
        payload = submission()
        del payload["setup"]["declared"][category]
        assert client.post("/api/runs", json=payload).status_code == 422
    for key, value in [
        ("generation", {"max_tokens": 9}),
        ("base_url", "http://user:password@host"),
        ("model_id", 42),
    ]:
        payload = submission()
        payload["setup"]["declared"]["requests"][key] = value
        assert client.post("/api/runs", json=payload).status_code == 422
    assert list(tmp_path.iterdir()) == []


def test_missing_and_failed_outputs_are_partial_without_timings(tmp_path, monkeypatch):
    for code, expected in [(0, "missing"), (1, "failed")]:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda command, code=code, **kwargs: subprocess.CompletedProcess(command, code, "", ""),
        )
        client = TestClient(create_app(tmp_path))
        response = client.post("/api/runs", json=submission())
        assert response.status_code == 201
        result = client.get("/api/runs/" + response.json()["run_id"]).json()
        assert result["status"] == "partial"
        assert result["outcomes"][0]["outcome"] == expected
        assert result["outcomes"][0]["metrics"] == {}


def test_cli_submission_retains_unsupported_condition_without_tool(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported condition invoked tool")

    monkeypatch.setattr(subprocess, "run", forbidden)
    payload = submission()
    payload["setup"]["declared"]["requests"]["context_window"] = 4096
    source = tmp_path / "submission.json"
    source.write_text(json.dumps(payload))
    runs = tmp_path / "runs"
    execution = CliRunner().invoke(app, ["run", str(source), "--results", str(runs)])
    assert execution.exit_code == 0, execution.output
    result = (
        TestClient(create_app(runs))
        .get("/api/runs/" + json.loads(execution.output)["run_id"])
        .json()
    )
    assert result["outcomes"][0]["outcome"] == "unsupported"
    assert result["outcomes"][0]["metrics"] == {}
    assert result["default_workload_complete"] is False
    assert result["workload"]["request_timeout_seconds"] == 900
    assert (runs / result["run_id"] / "input.jsonl").exists()


def test_credentials_are_redacted_in_native_json_logs_and_declared_argv(tmp_path, monkeypatch):
    payload = submission()
    payload["setup"]["declared"]["serving"]["launch_command"] = ["serve", "--api-key", "secret-124"]

    def tool(command, **kwargs):
        native = Path(command[command.index("--output-artifact-dir") + 1])
        native.mkdir()
        (native / "profile_export_raw.jsonl").write_text(
            json.dumps(
                {
                    "metadata": {"x_request_id": "failed"},
                    "status": 401,
                    "headers": {"Authorization": "Bearer secret-124"},
                    "responses": [],
                }
            )
            + "\n"
        )
        return subprocess.CompletedProcess(command, 1, "server echoed secret-124", "")

    monkeypatch.setattr(subprocess, "run", tool)
    response = TestClient(create_app(tmp_path)).post("/api/runs", json=payload)
    assert response.status_code == 201
    directory = tmp_path / response.json()["run_id"]
    for path in directory.rglob("*"):
        if path.is_file():
            assert "secret-124" not in path.read_text(), path
            assert "hf_sensitive_124" not in path.read_text(), path
    raw = json.loads((directory / "native/profile_export_raw.jsonl").read_text())
    assert raw["headers"]["Authorization"] == "[REDACTED]"


def test_native_measurement_mismatch_is_retained_as_failed(tmp_path, monkeypatch):
    def tool(command, **kwargs):
        native = Path(command[command.index("--output-artifact-dir") + 1])
        native.mkdir()
        source = Path("tests/fixtures/pilots/performance-121/native")
        for name in ["profile_export.jsonl", "profile_export_raw.jsonl"]:
            row = json.loads(gzip.decompress((source / (name + ".gz")).read_bytes()))
            if name == "profile_export.jsonl":
                row["metadata"]["x_request_id"] = "wrong-request"
            (native / name).write_text(json.dumps(row) + "\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", tool)
    response = TestClient(create_app(tmp_path)).post("/api/runs", json=submission())
    result = response.json()
    assert result["status"] == "partial"
    assert result["outcomes"][0]["outcome"] == "failed"
    assert result["outcomes"][0]["metrics"] == {}
    assert "identities disagree" in result["outcomes"][0]["diagnostic"]


def test_malformed_native_export_does_not_lose_public_result(tmp_path, monkeypatch):
    def tool(command, **kwargs):
        native = Path(command[command.index("--output-artifact-dir") + 1])
        native.mkdir()
        (native / "profile_export_raw.jsonl").write_text("{broken")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", tool)
    response = TestClient(create_app(tmp_path)).post("/api/runs", json=submission())
    assert response.status_code == 201
    assert response.json()["outcomes"][0]["outcome"] == "failed"


def test_validation_errors_do_not_echo_credentials(tmp_path):
    payload = submission()
    del payload["setup"]["declared"]["hardware"]
    response = TestClient(create_app(tmp_path)).post("/api/runs", json=payload)
    assert response.status_code == 422
    assert "secret-124" not in response.text


def test_conflicting_request_settings_are_rejected(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("conflicting settings invoked tool")

    monkeypatch.setattr(subprocess, "run", forbidden)
    payload = submission()
    payload["setup"]["declared"]["requests"]["reasoning"]["temperature"] = 1.0
    response = TestClient(create_app(tmp_path)).post("/api/runs", json=payload)
    assert response.status_code == 422


def test_complete_fresh_context_curve_uses_sized_distinct_inputs_and_native_metrics(
    tmp_path, monkeypatch
):
    payload = fresh_submission()
    calls = []

    def external(command, **kwargs):
        calls.append(command)
        if any(str(part).endswith("performance_inputs.py") for part in command):
            specification = json.loads(Path(command[command.index("--spec") + 1]).read_text())
            directory = Path(command[command.index("--directory") + 1])
            assert specification["tokenizer"] == "org/model"
            assert specification["chat_template_kwargs"] == {
                "enable_thinking": True,
                "reasoning_effort": "xhigh",
            }
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
                        messages = [
                            {"role": "system", "content": "unique-prefix-" + nonce},
                            {"role": "user", "content": "sized payload"},
                        ]
                        rows.append(
                            {
                                "text": "Messages supplied in extra.messages",
                                "output_length": condition["output_target"],
                                "extra": {
                                    "temperature": 0.5,
                                    "chat_template_kwargs": {
                                        "enable_thinking": True,
                                        "reasoning_effort": "xhigh",
                                    },
                                    "messages": messages,
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
            (directory / "preparation.json").write_text(
                json.dumps({"observations": prepared}) + "\n"
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        dataset = Path(command[command.index("--input-file") + 1])
        rows = [json.loads(line) for line in dataset.read_text().splitlines()]
        request_count = int(command[command.index("--request-count") + 1])
        assert len(rows) == request_count
        native = Path(command[command.index("--output-artifact-dir") + 1])
        native.mkdir()
        condition_parent = native.parent
        condition = (
            condition_parent.parent.name
            if condition_parent.name.startswith("repetition-")
            else (
                condition_parent.name
                if condition_parent.name != "warmup"
                else condition_parent.parent.name
            )
        )
        input_tokens, output_target = (int(value) for value in condition.split("-")[-2:])
        raw_rows = []
        metric_rows = []
        request_repetition = (
            int(condition_parent.name.removeprefix("repetition-"))
            if condition_parent.name.startswith("repetition-")
            else 0
        )
        for repetition, _row in enumerate(rows):
            repetition += request_repetition
            request_id = f"{condition}-{native.parent.name}-{repetition}"
            actual_output = output_target if repetition != 4 else output_target - 1
            raw_rows.append(
                {
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
                                                    "delta": {"content": "result"},
                                                    "finish_reason": (
                                                        "length"
                                                        if actual_output == output_target
                                                        else "stop"
                                                    ),
                                                }
                                            ]
                                        }
                                    ),
                                },
                                {
                                    "name": "data",
                                    "value": json.dumps(
                                        {
                                            "choices": [],
                                            "usage": {
                                                **(
                                                    {}
                                                    if condition == "fresh-context-4096-256"
                                                    and repetition == 0
                                                    and native.parent.name != "warmup"
                                                    else {
                                                        "prompt_tokens": input_tokens,
                                                        "completion_tokens": actual_output,
                                                    }
                                                ),
                                            },
                                        }
                                    ),
                                },
                                {"name": "data", "value": "[DONE]"},
                            ]
                        }
                    ],
                }
            )
            metric_rows.append(
                {
                    "metadata": {"x_request_id": request_id},
                    "metrics": {
                        "time_to_first_token": {"value": 1000 + repetition, "unit": "ms"},
                        "time_to_second_token": {"value": 20 + repetition, "unit": "ms"},
                        "time_to_first_output_token": {
                            "value": 1500 + repetition,
                            "unit": "ms",
                        },
                        "request_latency": {"value": 2000 + repetition, "unit": "ms"},
                        "output_token_throughput_per_user": {
                            "value": 80 + repetition,
                            "unit": "tokens/sec/user",
                        },
                        "usage_prompt_tokens": {"value": input_tokens, "unit": "tokens"},
                        "usage_completion_tokens": {
                            "value": actual_output,
                            "unit": "tokens",
                        },
                    },
                }
            )
        raw_contents = "".join(json.dumps(row) + "\n" for row in raw_rows)
        metric_contents = "".join(json.dumps(row) + "\n" for row in metric_rows)
        if condition == "fresh-context-196608-1024" and condition_parent.name != "warmup":
            (native / "profile_export_raw.jsonl.gz").write_bytes(
                gzip.compress(raw_contents.encode())
            )
            (native / "profile_export.jsonl.gz").write_bytes(
                gzip.compress(metric_contents.encode())
            )
        else:
            (native / "profile_export_raw.jsonl").write_text(raw_contents)
            (native / "profile_export.jsonl").write_text(metric_contents)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", external)
    client = TestClient(create_app(tmp_path))
    submitted = client.post("/api/runs", json=payload)
    assert submitted.status_code == 201, submitted.text
    result = TestClient(create_app(tmp_path)).get("/api/runs/" + submitted.json()["run_id"]).json()

    assert result["evidence"] == "run"
    assert result["status"] == "completed"
    assert result["default_workload_complete"] is False
    assert result["coverage"] == {
        "fresh_context": {"expected": 60, "accounted": 60, "successful": 60}
    }
    assert len(result["outcomes"]) == 60
    assert {(row["input_tokens"], row["output_target"]) for row in result["outcomes"]} == {
        (checkpoint, target)
        for checkpoint in [4096, 16384, 32768, 65536, 131072, 196608]
        for target in [256, 1024]
    }
    assert {row["repetition"] for row in result["outcomes"]} == set(range(5))
    assert sum(row["output_target_attained"] is False for row in result["outcomes"]) == 12
    native_metric_token_count = next(
        row
        for row in result["outcomes"]
        if row["input_tokens"] == 4096 and row["output_target"] == 256 and row["repetition"] == 0
    )
    assert native_metric_token_count["outcome"] == "success"
    assert native_metric_token_count["local_input_tokens"] == 4096
    assert native_metric_token_count["actual_input_tokens"] == 4096
    assert native_metric_token_count["actual_output_tokens"] == 256
    assert native_metric_token_count["endpoint_input_matches"] is True
    assert native_metric_token_count["actual_input_tokens_source"].endswith(":usage_prompt_tokens")
    compressed = next(
        row
        for row in result["outcomes"]
        if row["input_tokens"] == 196608 and row["output_target"] == 1024
    )
    assert compressed["metrics"]["ttft"]["source"].startswith("native/profile_export.jsonl.gz:")
    assert len(result["fresh_context_curve"]) == 12
    curve = result["fresh_context_curve"][0]
    assert curve["metrics"]["ttft"] == {
        "unit": "ms",
        "values": [1000, 1001, 1002, 1003, 1004],
        "minimum": 1000,
        "maximum": 1004,
        "mean": 1002,
        "unavailable": 0,
    }
    retained_directory = tmp_path / result["run_id"]
    retained_metrics = json.loads(
        (retained_directory / "fresh-context-4096-256/repetition-000/native/profile_export.jsonl")
        .read_text()
        .splitlines()[0]
    )["metrics"]
    assert {
        key: retained_metrics[key]
        for key in (
            "time_to_first_token",
            "time_to_second_token",
            "time_to_first_output_token",
        )
    } == {
        "time_to_first_token": {"value": 1000, "unit": "ms"},
        "time_to_second_token": {"value": 20, "unit": "ms"},
        "time_to_first_output_token": {"value": 1500, "unit": "ms"},
    }
    compressed_metrics = json.loads(
        gzip.decompress(
            (
                retained_directory
                / "fresh-context-196608-1024/repetition-000/native/profile_export.jsonl.gz"
            ).read_bytes()
        )
        .decode()
        .splitlines()[0]
    )["metrics"]
    assert {
        key: compressed_metrics[key]
        for key in (
            "time_to_first_token",
            "time_to_second_token",
            "time_to_first_output_token",
        )
    } == {
        "time_to_first_token": {"value": 1000, "unit": "ms"},
        "time_to_second_token": {"value": 20, "unit": "ms"},
        "time_to_first_output_token": {"value": 1500, "unit": "ms"},
    }
    assert result["cache"] == {
        "configuration": {"prefix_caching": True},
        "observations": [],
        "request_design": "distinct prefixes discourage reuse; cache state is not controlled",
    }

    directory = tmp_path / result["run_id"]
    inputs = list(directory.glob("fresh-context-*/input.jsonl"))
    warmups = list(directory.glob("fresh-context-*/warmup/input.jsonl"))
    assert len(inputs) == len(warmups) == 12
    prefixes = {
        json.loads(line)["extra"]["messages"][0]["content"]
        for path in inputs + warmups
        for line in path.read_text().splitlines()
    }
    assert len(prefixes) == 72
    assert len([call for call in calls if "aiperf" in call and "profile" in call]) == 72
    assert all(
        "--use-server-token-count" in call
        for call in calls
        if "aiperf" in call and "profile" in call
    )
    assert "Fresh-context speed curve" in client.get("/runs/" + result["run_id"]).text


def test_input_preparer_uses_submitted_tokenizer_template_and_distinct_prefixes(tmp_path):
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "system": 1,
                "user": 2,
                "assistant": 3,
                "entry": 4,
                "think": 5,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tokenizer_directory / "tokenizer.json"))
    (tokenizer_directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "[UNK]",
                "chat_template": (
                    "{% for message in messages %}{{ message['role'] }} "
                    "{{ message['content'] }} {% endfor %}"
                    "{% if enable_thinking %}think {% endif %}assistant"
                ),
            }
        )
    )
    specification = {
        "tokenizer": str(tokenizer_directory),
        "chat_template_kwargs": {"enable_thinking": True},
        "request_settings": {"chat_template_kwargs": {"enable_thinking": True}},
        "seed": 125,
        "repetitions": 2,
        "warmup_repetitions": 1,
        "conditions": [
            {"name": "fresh-context-4-8", "input_tokens": 4, "output_target": 8},
            {"name": "fresh-context-64-8", "input_tokens": 64, "output_target": 8},
        ],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(specification))
    helper = Path(__file__).parents[1] / "src/llm_bench/performance_inputs.py"

    completed = subprocess.run(
        [
            "uv",
            "run",
            "--with",
            "aiperf==0.12.0",
            "--with",
            "transformers==5.17.0",
            "python",
            str(helper),
            "--spec",
            str(spec_path),
            "--directory",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    prepared = json.loads((tmp_path / "preparation.json").read_text())
    assert [row["local_input_tokens"] for row in prepared["observations"]] == [64, 64, 64]
    assert prepared["failures"] == [
        {
            "condition": "fresh-context-4-8",
            "diagnostic": (
                "Cannot size fresh-context input exactly to 4 tokens with the submitted "
                "tokenizer and chat template"
            ),
        }
    ]
    assert len({row["nonce"] for row in prepared["observations"]}) == 3
    assert all(
        row["template_and_structured_overhead_tokens"] > 0 for row in prepared["observations"]
    )
    measured = [
        json.loads(line)
        for line in (tmp_path / "fresh-context-64-8/input.jsonl").read_text().splitlines()
    ]
    assert len(measured) == 2
    assert measured[0]["extra"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert measured[0]["extra"]["messages"] != measured[1]["extra"]["messages"]


def test_grow_preparer_sizes_turn_exactly_and_preserves_history(tmp_path):
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "system": 1,
                "user": 2,
                "assistant": 3,
                "entry": 4,
                "think": 5,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tokenizer_directory / "tokenizer.json"))
    (tokenizer_directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "[UNK]",
                "chat_template": (
                    "{% for message in messages %}{{ message['role'] }} "
                    "{{ message['content'] }} {% endfor %}"
                    "{% if enable_thinking %}think {% endif %}assistant"
                ),
            }
        )
    )
    history = [
        {"role": "user", "content": "entry entry"},
        {"role": "assistant", "content": "entry"},
    ]
    structural = [
        {"role": "user", "content": "entry"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_log", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": ""},
        {"role": "user", "content": "entry entry"},
    ]
    specification = {
        "tokenizer": str(tokenizer_directory),
        "chat_template_kwargs": {"enable_thinking": True},
        "request_settings": {"chat_template_kwargs": {"enable_thinking": True}},
        "history": history,
        "structural": structural,
        "checkpoint": 20,
        "output_target": 8,
        "nonce": "grow-20",
    }
    spec_path = tmp_path / "grow-spec.json"
    spec_path.write_text(json.dumps(specification))
    helper = Path(__file__).parents[1] / "src/llm_bench/performance_inputs.py"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--with",
            "aiperf==0.12.0",
            "--with",
            "transformers==5.17.0",
            "python",
            str(helper),
            "--mode",
            "grow",
            "--spec",
            str(spec_path),
            "--directory",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    messages = json.loads((tmp_path / "input.jsonl").read_text())["extra"]["messages"]
    assert len(messages) == 6
    assert messages[:2] == history
    assert messages[3]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert messages[4]["role"] == "tool"
    assert messages[4]["content"] == "log entry entry entry entry entry"
    sizing = json.loads((tmp_path / "sizing.json").read_text())
    assert sizing["checkpoint"] == 20
    assert sizing["local_input_tokens"] == 20
    assert sizing["history_messages"] == 2
    assert sizing["bare_content_tokens"] + sizing["template_and_structured_history_tokens"] == 20


def test_fresh_context_curve_accounts_for_unsupported_sizes_without_invoking_tools(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported matrix invoked an external tool")

    monkeypatch.setattr(subprocess, "run", forbidden)
    payload = fresh_submission(2048)
    payload["workload"]["cache_configuration"] = {}

    response = TestClient(create_app(tmp_path)).post("/api/runs", json=payload)

    assert response.status_code == 201, response.text
    result = response.json()
    assert result["status"] == "completed"
    assert result["coverage"]["fresh_context"] == {
        "expected": 60,
        "accounted": 60,
        "successful": 0,
    }
    assert {row["outcome"] for row in result["outcomes"]} == {"unsupported"}
    assert all(
        condition["outcomes"]["unsupported"] == 5 for condition in result["fresh_context_curve"]
    )
    assert result["warmups"] == []


def test_fresh_context_profile_rejects_reduced_matrix_and_invalid_template_kwargs(tmp_path):
    payload = fresh_submission()
    payload["workload"]["checkpoints"] = [4096]
    client = TestClient(create_app(tmp_path))

    assert client.post("/api/runs", json=payload).status_code == 422
    payload = submission()
    payload["setup"]["declared"]["requests"]["reasoning"]["chat_template_kwargs"] = True
    assert client.post("/api/runs", json=payload).status_code == 422
    assert list(tmp_path.iterdir()) == []


def test_nonzero_matrix_tool_exit_retains_native_rows_as_failed_observations(tmp_path, monkeypatch):
    def external(command, **kwargs):
        if any(str(part).endswith("performance_inputs.py") for part in command):
            specification = json.loads(Path(command[command.index("--spec") + 1]).read_text())
            directory = Path(command[command.index("--directory") + 1])
            condition = specification["conditions"][0]
            condition_directory = directory / condition["name"]
            condition_directory.mkdir()
            observations = []
            for phase, count in (("warmup", 1), ("measured", 5)):
                path = (
                    condition_directory / "warmup/input.jsonl"
                    if phase == "warmup"
                    else condition_directory / "input.jsonl"
                )
                path.parent.mkdir(exist_ok=True)
                row = {
                    "text": "Messages supplied in extra.messages",
                    "output_length": 256,
                    "extra": {"messages": [{"role": "user", "content": "sized"}]},
                }
                path.write_text((json.dumps(row) + "\n") * count)
                observations.extend(
                    {
                        "condition": condition["name"],
                        "phase": phase,
                        "repetition": repetition,
                        "nonce": f"{phase}-{repetition}",
                        "local_input_tokens": 4096,
                    }
                    for repetition in range(count)
                )
            (directory / "preparation.json").write_text(
                json.dumps({"observations": observations, "failures": []}) + "\n"
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        count = int(command[command.index("--request-count") + 1])
        native = Path(command[command.index("--output-artifact-dir") + 1])
        native.mkdir()
        source = Path("tests/fixtures/pilots/performance-121/native")
        raw = json.loads(gzip.decompress((source / "profile_export_raw.jsonl.gz").read_bytes()))
        metric = json.loads(gzip.decompress((source / "profile_export.jsonl.gz").read_bytes()))
        raw_rows = []
        metric_rows = []
        for repetition in range(count):
            request_id = f"request-{native.parent.name}-{repetition}"
            raw_row = json.loads(json.dumps(raw))
            metric_row = json.loads(json.dumps(metric))
            raw_row["metadata"]["x_request_id"] = request_id
            metric_row["metadata"]["x_request_id"] = request_id
            raw_rows.append(raw_row)
            metric_rows.append(metric_row)
        (native / "profile_export_raw.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in raw_rows)
        )
        (native / "profile_export.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in metric_rows)
        )
        exit_code = 0 if native.parent.name == "warmup" else 7
        return subprocess.CompletedProcess(command, exit_code, "", "tool failure")

    monkeypatch.setattr(subprocess, "run", external)
    response = TestClient(create_app(tmp_path)).post("/api/runs", json=fresh_submission(5000))

    assert response.status_code == 201, response.text
    result = response.json()
    assert result["status"] == "partial"
    completed = [row for row in result["outcomes"] if row["outcome"] == "success"]
    assert len(completed) == 5
    assert all(row["metrics"] for row in completed)
    assert result["tool_failures"] == [
        {"condition": "fresh-context-4096-256", "repetition": index, "exit_code": 7}
        for index in range(5)
    ]
    assert sum(row["outcome"] == "unsupported" for row in result["outcomes"]) == 55


def test_one_unsizeable_condition_does_not_prevent_later_conditions(tmp_path, monkeypatch):
    invoked_conditions = []

    def external(command, **kwargs):
        if any(str(part).endswith("performance_inputs.py") for part in command):
            specification = json.loads(Path(command[command.index("--spec") + 1]).read_text())
            directory = Path(command[command.index("--directory") + 1])
            failed_name = specification["conditions"][0]["name"]
            observations = []
            for condition in specification["conditions"]:
                condition_directory = directory / condition["name"]
                condition_directory.mkdir()
                if condition["name"] == failed_name:
                    continue
                for phase, count in (("warmup", 1), ("measured", 5)):
                    path = (
                        condition_directory / "warmup/input.jsonl"
                        if phase == "warmup"
                        else condition_directory / "input.jsonl"
                    )
                    path.parent.mkdir(exist_ok=True)
                    row = {
                        "text": "Messages supplied in extra.messages",
                        "output_length": condition["output_target"],
                        "extra": {"messages": [{"role": "user", "content": "sized"}]},
                    }
                    path.write_text((json.dumps(row) + "\n") * count)
                    observations.extend(
                        {
                            "condition": condition["name"],
                            "phase": phase,
                            "repetition": repetition,
                            "nonce": f"{condition['name']}-{phase}-{repetition}",
                            "local_input_tokens": condition["input_tokens"],
                        }
                        for repetition in range(count)
                    )
            (directory / "preparation.json").write_text(
                json.dumps(
                    {
                        "observations": observations,
                        "failures": [
                            {"condition": failed_name, "diagnostic": "cannot size condition"}
                        ],
                    }
                )
                + "\n"
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        native = Path(command[command.index("--output-artifact-dir") + 1])
        condition_directory = native.parent
        if condition_directory.name == "warmup":
            condition_directory = condition_directory.parent
        if condition_directory.name.startswith("repetition-"):
            condition_directory = condition_directory.parent
        invoked_conditions.append(condition_directory.name)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", external)
    response = TestClient(create_app(tmp_path)).post("/api/runs", json=fresh_submission(20000))

    assert response.status_code == 201, response.text
    result = response.json()
    first_condition = "fresh-context-4096-256"
    first = [row for row in result["outcomes"] if row["condition"] == first_condition]
    assert len(first) == 5
    assert {row["diagnostic"] for row in first} == {"cannot size condition"}
    assert first_condition not in invoked_conditions
    assert set(invoked_conditions) == {
        "fresh-context-4096-1024",
        "fresh-context-16384-256",
        "fresh-context-16384-1024",
    }
