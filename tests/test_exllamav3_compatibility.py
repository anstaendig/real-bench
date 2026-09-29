"""ExLlamaV3 setup and launch compatibility: settings, reasoning, and the gate.

Every case is driven through the public command boundary: the ``bench check``
CLI and the ``bench run`` CLI submission contract. A local HTTP server plays
the externally operated ExLlamaV3 endpoint with controlled responses. The
shared setup stores reasoning_effort in ``chat_template_kwargs`` for engines
such as vLLM; ExLlamaV3 moves it to the top-level field its server reads.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llm_bench import setup_state
from llm_bench.ordinary_cli import app

pytestmark = pytest.mark.usefixtures("external_tool_process_adapter")


class Endpoint:
    """A controlled ExLlamaV3 (TabbyAPI) endpoint: records every request, can
    enforce a bearer key, and answers with controlled status bodies. It
    exposes only reads and inference."""

    def __init__(
        self, handler: Callable[[dict[str, Any]], tuple[int, Any]], *, auth: str | None = None
    ):
        self.handler = handler
        self.auth = auth
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, status: int, body: Any) -> None:
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _authorized(self) -> bool:
                if outer.auth is None:
                    return True
                return self.headers.get("authorization") == f"Bearer {outer.auth}"

            def do_GET(self) -> None:
                outer.requests.append(
                    {
                        "method": "GET",
                        "path": self.path,
                        "headers": {"authorization": self.headers.get("authorization")},
                    }
                )
                if not self._authorized():
                    self._send(401, {"error": {"message": "Invalid API key", "type": "auth"}})
                    return
                if self.path.rstrip("/") == "/v1/models":
                    self._send(200, {"object": "list", "data": [{"id": "test-model"}]})
                else:
                    self._send(404, {"error": {"message": "unknown route", "type": "NotFound"}})

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "headers": {"authorization": self.headers.get("authorization")},
                        "body": body,
                    }
                )
                if not self._authorized():
                    self._send(401, {"error": {"message": "Invalid API key", "type": "auth"}})
                    return
                status, response = outer.handler(body)
                self._send(status, response)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def paths(self) -> list[str]:
        return [request["path"] for request in self.requests]

    def completions(self) -> list[dict[str, Any]]:
        return [
            request for request in self.requests if request["path"].endswith("/v1/chat/completions")
        ]


def completion_ok(_body: dict[str, Any]) -> tuple[int, Any]:
    return 200, {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def ok_handler(_body: dict[str, Any]) -> tuple[int, Any]:
    return completion_ok(_body)


def repetition_penalty_rejecting(body: dict[str, Any]) -> tuple[int, Any]:
    """An ExLlamaV3 (TabbyAPI) build whose OpenAI-compatible layer rejects the
    shared repeated-penalty setting with a named 400; every other request is
    accepted."""
    if "repetition_penalty" in body:
        return (
            400,
            {
                "error": {
                    "message": "Unsupported sampling parameter: repetition_penalty",
                    "type": "BadRequestError",
                    "code": 400,
                }
            },
        )
    return completion_ok(body)


def save_setup(directory: Path, setup: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    state = setup_state.SetupState(
        schema_version=setup_state.SCHEMA_VERSION,
        updated=datetime.now(UTC).isoformat(timespec="seconds"),
        setup={"declared": {c: dict(f) for c, f in setup["declared"].items()}, "observed": {}},
        provided={},
        preparations={},
    )
    setup_state.save_state(state, directory)


def setup_record(**overrides: Any) -> dict[str, Any]:
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
            "quantized_artifact": "model-exl3",
            "tokenizer": "org/model",
            "template_overrides": {},
        },
        "serving": {
            "engine": "exl3",
            "version": "1.5.0",
            "launch_command": "python -m tabbyAPI --api_key secret-124",
            "environment": {"API_KEY": "secret-124"},
        },
        "requests": {
            "base_url": "http://127.0.0.1:0",
            "model_id": "test-model",
            "generation": {"temperature": 0.5, "repetition_penalty": 1.1},
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
        category, field = path.split("__")
        declared[category][field] = value
    return {"declared": declared, "observed": {}}


def submission(setup: dict[str, Any], base_url: str) -> dict[str, Any]:
    setup["declared"]["requests"]["base_url"] = base_url
    return {
        "setup": setup,
        "run_config": {
            "reasoning_effort": "high",
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


def pilot_tool(command, **kwargs):
    """Serve the retained export so a validation run can complete."""
    native = Path(command[command.index("--output-artifact-dir") + 1])
    native.mkdir(parents=True, exist_ok=True)
    source = Path("tests/fixtures/pilots/performance-121/native")
    for name in ("profile_export.jsonl", "profile_export_raw.jsonl"):
        (native / name).write_bytes(gzip.decompress((source / (name + ".gz")).read_bytes()))
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def unboxed(output: str) -> str:
    """Collapse a Rich error box into plain text so phrase checks survive
    line wrapping at the box border."""
    lines = [line.strip().strip("│").strip() for line in output.splitlines()]
    return " ".join(lines)


def result_json(output: str) -> dict:
    return json.loads(output[output.index("{") :])


def retained_blob(directory: Path) -> str:
    """Every retained file under a directory, as one searchable blob."""
    parts = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            try:
                parts.append(path.read_bytes().decode(errors="replace"))
            except OSError:
                pass
    return "\n".join(parts)


# --- 1. Setup compatibility check --------------------------------------------


def test_engine_identification_accepts_the_common_names():
    """The declared engine identifies an ExLlamaV3 route by its common names,
    case-insensitively, so the gate and the probe apply to it."""
    from llm_bench import compatibility

    for engine in ("exllamav3", "ExLlamaV3", "exl3", "EXLLAMA V3", "  exl3  "):
        setup = {"serving": {"engine": engine}}
        assert compatibility.exllamav3_engine(setup), engine
        assert compatibility.gated_engine(setup), engine
    # Unmapped engines are not gated.
    assert not compatibility.gated_engine({"serving": {"engine": "tensorrt-llm"}})
    # ExLlamaV3 reads reasoning_effort at the top level.
    assert compatibility.setting_mappings({"serving": {"engine": "exl3"}}) == {
        "chat_template_kwargs.reasoning_effort": "reasoning_effort"
    }


def test_check_unauthenticated_success_probes_shared_settings(tmp_path, monkeypatch):
    """Without any key the check authenticates nothing, uses only the model
    list and a minimal completion, and the repeated-penalty setting rides the
    wire under its shared (OpenAI) name."""
    monkeypatch.delenv("REAL_BENCH_API_KEY", raising=False)
    endpoint = Endpoint(ok_handler)
    from llm_bench import compatibility

    setup = setup_record()
    setup["declared"]["requests"]["base_url"] = endpoint.base_url
    evidence = compatibility.check(setup["declared"], include_settings=True, key=None)
    assert endpoint.completions(), "no completion request was sent"
    assert all(request["headers"]["authorization"] is None for request in endpoint.requests)
    assert set(endpoint.paths()) == {"/v1/models", "/v1/chat/completions"}
    assert evidence["authenticated"] is False
    assert evidence["engine"] == "exl3"
    assert evidence["checks"]["models"]["ok"] is True
    assert evidence["checks"]["completion"]["ok"] is True
    assert evidence["unsupported"] == []
    # No rename is delivered: the probe carries the shared per-run name.
    assert "repetition_penalty" in evidence["checks"]
    probe_bodies = [request["body"] for request in endpoint.completions()]
    assert any(body.get("repetition_penalty") == 1.1 for body in probe_bodies)


def test_check_records_exllama_reasoning_mapping(tmp_path, monkeypatch):
    """ExLlamaV3 moves reasoning_effort to the top level while retaining
    enable_thinking inside chat_template_kwargs."""
    monkeypatch.delenv("REAL_BENCH_API_KEY", raising=False)
    endpoint = Endpoint(ok_handler)
    from llm_bench import compatibility

    setup = setup_record()
    setup["declared"]["requests"]["reasoning"]["chat_template_kwargs"]["reasoning_effort"] = "high"
    setup["declared"]["requests"]["base_url"] = endpoint.base_url
    evidence = compatibility.check(setup["declared"], include_settings=True, key=None)
    assert evidence["setting_mappings"] == {
        "chat_template_kwargs.reasoning_effort": "reasoning_effort"
    }
    assert "reasoning_effort" in evidence["checks"]
    assert "chat_template_kwargs.reasoning_effort" not in evidence["checks"]
    for body in [request["body"] for request in endpoint.completions()]:
        assert "repeat_penalty" not in body


def test_check_authenticated_success_keeps_key_out_of_evidence(tmp_path, monkeypatch):
    """A key supplied through the environment authenticates every probe
    request — the endpoint enforces it — only reads and inference are used,
    and the key never appears in the evidence."""
    key = "live-secret-key-999"
    monkeypatch.setenv("REAL_BENCH_API_KEY", key)
    endpoint = Endpoint(ok_handler, auth=key)
    from llm_bench import compatibility

    setup = setup_record()
    setup["declared"]["requests"]["base_url"] = endpoint.base_url
    evidence = compatibility.check(setup["declared"], include_settings=True, key=key)
    assert endpoint.completions(), "no completion request was sent"
    assert all(
        request["headers"]["authorization"] == f"Bearer {key}" for request in endpoint.requests
    )
    assert set(endpoint.paths()) == {"/v1/models", "/v1/chat/completions"}
    assert evidence["authenticated"] is True
    assert evidence["checks"]["completion"]["ok"] is True
    blob = json.dumps(evidence)
    assert key not in blob
    assert "secret-124" not in blob


def test_check_reports_unsupported_setting_from_a_named_400(tmp_path, monkeypatch):
    """A 400 that names a shared parameter is evidence the setting was not
    applied; the check reports it as unsupported with the server's own
    message and records the (empty) mapping."""
    monkeypatch.delenv("REAL_BENCH_API_KEY", raising=False)
    endpoint = Endpoint(repetition_penalty_rejecting)
    from llm_bench import compatibility

    setup = setup_record()
    setup["declared"]["requests"]["base_url"] = endpoint.base_url
    evidence = compatibility.check(setup["declared"], include_settings=True, key=None)
    assert evidence["unsupported"] == ["repetition_penalty"]
    assert evidence["checks"]["temperature"]["ok"] is True
    assert evidence["checks"]["repetition_penalty"]["ok"] is False
    assert "repetition_penalty" in evidence["checks"]["repetition_penalty"]["error"]
    assert evidence["setting_mappings"] == {
        "chat_template_kwargs.reasoning_effort": "reasoning_effort"
    }


def test_check_records_advisory_against_the_pinned_revisions(tmp_path, monkeypatch):
    """The retained evidence records the declared version against the pinned
    TabbyAPI/exllamav3 revisions the mapping was established for, with the
    drop-unregistered-fields limitation, advisory only."""
    monkeypatch.delenv("REAL_BENCH_API_KEY", raising=False)
    endpoint = Endpoint(ok_handler)
    from llm_bench import compatibility

    setup = setup_record()
    setup["declared"]["requests"]["base_url"] = endpoint.base_url
    evidence = compatibility.check(setup["declared"], include_settings=True, key=None)
    assert evidence["engine"] == "exl3"
    assert evidence["version"] == "1.5.0"
    advisory = [a for a in evidence["advisory"] if a.startswith("ExLlamaV3:")]
    assert advisory, "no ExLlamaV3 advisory was recorded"
    assert "53da7919d4e4" in advisory[0]
    assert "v1.5.0" in advisory[0]
    assert "declared version: 1.5.0" in advisory[0]


def test_check_treats_a_server_error_as_not_ok(tmp_path, monkeypatch):
    """A 500 completion is not proof that anything is accepted: the check
    records the failure and the launch gate refuses to proceed on it."""
    monkeypatch.delenv("REAL_BENCH_API_KEY", raising=False)
    endpoint = Endpoint(lambda body: (500, {"error": {"message": "upstream exploded"}}))
    from llm_bench import compatibility

    setup = setup_record()
    setup["declared"]["requests"]["base_url"] = endpoint.base_url
    evidence = compatibility.check(setup["declared"])
    assert evidence["checks"]["completion"]["ok"] is False
    assert evidence["checks"]["completion"]["http"] == 500
    with pytest.raises(ValueError, match="completion check failed"):
        compatibility.launch_gate(setup["declared"])


def test_check_scrubs_an_externally_echoed_key(tmp_path, monkeypatch):
    """An unlabelled echo of the bearer key in the endpoint's own error text
    is scrubbed from the retained evidence."""
    key = "echoed-secret-555"
    monkeypatch.setenv("REAL_BENCH_API_KEY", key)
    endpoint = Endpoint(lambda body: (400, {"error": {"message": f"rejected: {key}"}}))
    from llm_bench import compatibility

    setup = setup_record(requests__generation={"min_p": 0.02})
    setup["declared"]["requests"]["base_url"] = endpoint.base_url
    evidence = compatibility.check(setup["declared"], include_settings=True, key=key)
    assert evidence["checks"]["min_p"]["ok"] is False
    blob = json.dumps(evidence)
    assert key not in blob
    assert "[REDACTED]" in evidence["checks"]["min_p"]["error"]


# --- 2. bench check public command -------------------------------------------


def test_bench_check_command_probes_and_retains_separately(tmp_path, monkeypatch):
    """bench check reads the saved setup, probes the endpoint, and retains the
    result in the setup state directory, separate from any scored run."""
    monkeypatch.setenv("REAL_BENCH_API_KEY", "live-key")
    endpoint = Endpoint(ok_handler)
    directory = tmp_path / "setup"
    setup = setup_record()
    setup["declared"]["requests"]["base_url"] = endpoint.base_url
    save_setup(directory, setup)

    result = CliRunner().invoke(app, ["check", "--directory", str(directory), "--settings"])
    assert result.exit_code == 0, result.output
    evidence = result_json(result.output)
    assert evidence["checks"]["completion"]["ok"] is True
    assert "repetition_penalty" in evidence["checks"]
    retained = directory / "compatibility" / "check.json"
    assert retained.is_file()
    stored = json.loads(retained.read_text())
    assert stored["checks"] == evidence["checks"]
    assert stored["setting_mappings"] == {
        "chat_template_kwargs.reasoning_effort": "reasoning_effort"
    }
    assert set(endpoint.paths()) == {"/v1/models", "/v1/chat/completions"}
    assert "live-key" not in retained.read_text()


def test_bench_check_authentication_failure_is_actionable(tmp_path, monkeypatch):
    """A 401 from an enforcing endpoint is reported with the endpoint's own
    message, and the launch gate turns it into an actionable error that names
    the environment variable."""
    monkeypatch.delenv("REAL_BENCH_API_KEY", raising=False)
    endpoint = Endpoint(ok_handler, auth="required-key")
    from llm_bench import compatibility

    setup = setup_record()
    setup["declared"]["requests"]["base_url"] = endpoint.base_url
    evidence = compatibility.check(setup["declared"])
    assert evidence["checks"]["models"]["ok"] is False
    assert evidence["checks"].get("completion", {}).get("ok") is not True
    with pytest.raises(ValueError, match="REAL_BENCH_API_KEY"):
        compatibility.launch_gate(setup["declared"])


# --- 3. Launch gate ------------------------------------------------------------


def test_launch_gate_stops_on_unsupported_setting_without_running(tmp_path, monkeypatch):
    """A 400 that names a final setting stops the launch before any tool is
    invoked: no run directory and the error names the setting."""
    monkeypatch.setenv("REAL_BENCH_API_KEY", "live-key")
    results = tmp_path / "results"

    endpoint = Endpoint(repetition_penalty_rejecting)
    monkeypatch.setattr(subprocess, "run", pilot_tool)
    setup = setup_record()
    payload = submission(setup, endpoint.base_url)
    payload["run_config"]["repetition_penalty"] = 1.1
    source = tmp_path / "submission.json"
    source.write_text(json.dumps(payload))

    result = CliRunner().invoke(
        app, ["run", str(source), "--results", str(results), "--no-interactive"]
    )
    assert result.exit_code != 0, result.output
    output = unboxed(result.output)
    assert "repetition_penalty" in output
    assert "unsupported" in output.lower()
    assert "no work was started" in output
    # No scored run was created.
    assert not results.exists() or list(results.iterdir()) == []


def test_launch_gate_stops_on_a_server_error_without_running(tmp_path, monkeypatch):
    """A 500 from the endpoint is a failure, not an acceptance: the launch is
    stopped before any scored work with the endpoint's own message."""
    monkeypatch.delenv("REAL_BENCH_API_KEY", raising=False)
    results = tmp_path / "results"
    endpoint = Endpoint(lambda body: (500, {"error": {"message": "engine wedged"}}))
    monkeypatch.setattr(subprocess, "run", pilot_tool)
    setup = setup_record()
    payload = submission(setup, endpoint.base_url)
    source = tmp_path / "submission.json"
    source.write_text(json.dumps(payload))

    result = CliRunner().invoke(
        app, ["run", str(source), "--results", str(results), "--no-interactive"]
    )
    assert result.exit_code != 0, result.output
    output = unboxed(result.output)
    assert "engine wedged" in output
    assert "no work was started" in output
    assert not results.exists() or list(results.iterdir()) == []


def test_launch_gate_rejects_a_wrong_bearer_key_before_running(tmp_path, monkeypatch):
    """A key the endpoint rejects stops the launch with the key contract
    named, before any scored work."""
    key = "wrong-key-123"
    monkeypatch.setenv("REAL_BENCH_API_KEY", key)
    results = tmp_path / "results"
    endpoint = Endpoint(ok_handler, auth="right-key-456")
    monkeypatch.setattr(subprocess, "run", pilot_tool)
    setup = setup_record()
    payload = submission(setup, endpoint.base_url)
    source = tmp_path / "submission.json"
    source.write_text(json.dumps(payload))

    result = CliRunner().invoke(
        app, ["run", str(source), "--results", str(results), "--no-interactive"]
    )
    output = unboxed(result.output)
    assert "rejected the bearer key" in output
    assert "REAL_BENCH_API_KEY" in output
    assert "no work was started" in output
    assert not results.exists() or list(results.iterdir()) == []


def test_launch_gate_passes_and_retains_evidence(tmp_path, monkeypatch):
    """When the endpoint accepts every final setting, the launch proceeds and
    the retained run directory carries the compatibility evidence with the
    shared setting names and the advisory."""
    monkeypatch.setenv("REAL_BENCH_API_KEY", "live-key")
    results = tmp_path / "results"
    endpoint = Endpoint(ok_handler)
    monkeypatch.setattr(subprocess, "run", pilot_tool)
    setup = setup_record()
    payload = submission(setup, endpoint.base_url)
    payload["run_config"]["repetition_penalty"] = 1.1
    source = tmp_path / "submission.json"
    source.write_text(json.dumps(payload))

    result = CliRunner().invoke(
        app, ["run", str(source), "--results", str(results), "--no-interactive"]
    )
    assert result.exit_code == 0, result.output
    run_id = result_json(result.output)["run_id"]
    compatibility_file = results / run_id / "compatibility.json"
    assert compatibility_file.is_file()
    evidence = json.loads(compatibility_file.read_text())
    assert evidence["engine"] == "exl3"
    assert evidence["checks"]["completion"]["ok"] is True
    assert evidence["unsupported"] == []
    assert evidence["checks"]["temperature"]["ok"] is True
    assert evidence["checks"]["repetition_penalty"]["ok"] is True
    assert evidence["checks"]["reasoning_effort"]["ok"] is True
    assert any(a.startswith("ExLlamaV3:") for a in evidence["advisory"])
    assert "live-key" not in compatibility_file.read_text()


# --- 4. Outgoing settings ride to the endpoint --------------------------------


def test_outgoing_settings_carry_shared_names_end_to_end(tmp_path, monkeypatch):
    """The final settings ride to the endpoint exactly as resolved: the launch
    gate probes each one under its shared name, and the prepared scored input
    carries the same merged settings with the reasoning template kwargs."""
    monkeypatch.setenv("REAL_BENCH_API_KEY", "live-key")
    results = tmp_path / "results"
    seen: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> tuple[int, Any]:
        seen.append(dict(body))
        return completion_ok(body)

    original_loads = json.loads

    def record_json(raw, *args, **kwargs):
        parsed = original_loads(raw, *args, **kwargs)
        if isinstance(parsed, dict) and isinstance(parsed.get("extra"), dict):
            input_records.append(parsed)
        return parsed

    monkeypatch.setattr(json, "loads", record_json)
    endpoint = Endpoint(handler)
    monkeypatch.setattr(subprocess, "run", pilot_tool)
    setup = setup_record(requests__generation={"temperature": 0.7})
    payload = submission(setup, endpoint.base_url)
    payload["run_config"].update(
        {"temperature": 0.3, "top_p": 0.9, "repetition_penalty": 1.2, "reasoning_effort": "high"}
    )
    source = tmp_path / "submission.json"
    source.write_text(json.dumps(payload))

    result = CliRunner().invoke(
        app, ["run", str(source), "--results", str(results), "--no-interactive"]
    )
    assert result.exit_code == 0, result.output
    # Each final setting rode its own probe with the resolved value, under its
    # shared (OpenAI) name: TabbyAPI registers the shared names unchanged.
    temp_probe = next(body for body in seen if "temperature" in body)
    assert temp_probe["temperature"] == 0.3
    top_p_probe = next(body for body in seen if "top_p" in body)
    assert top_p_probe["top_p"] == 0.9
    rep_probe = next(body for body in seen if "repetition_penalty" in body)
    assert rep_probe["repetition_penalty"] == 1.2
    # No probe carried a llama.cpp rename.
    assert not any("repeat_penalty" in body for body in seen)
    # The prepared scored inputs are the outgoing settings aiperf would send:
    # the merged generation settings plus the resolved template kwargs.
    assert input_records, "no prepared scored input was observed"
    for record in input_records:
        extra = record["extra"]
        assert extra["temperature"] == 0.3
        assert extra["top_p"] == 0.9
        assert extra["repetition_penalty"] == 1.2
        assert "repeat_penalty" not in extra
        assert extra["chat_template_kwargs"] == {"enable_thinking": True}
        assert extra["reasoning_effort"] == "high"
    # And the retained record is the resolved setup as executed.
    run_id = result_json(result.output)["run_id"]
    retained = json.loads((results / run_id / "setup.json").read_text())
    assert retained["declared"]["requests"]["generation"] == {
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
    }
    assert retained["declared"]["requests"]["reasoning"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "high",
    }
    compatibility = json.loads((results / run_id / "compatibility.json").read_text())
    assert compatibility["setting_mappings"] == {
        "chat_template_kwargs.reasoning_effort": "reasoning_effort"
    }
    assert compatibility["checks"]["reasoning_effort"]["ok"] is True
    assert "chat_template_kwargs.reasoning_effort" not in compatibility["checks"]
    assert compatibility["checks"]["repetition_penalty"]["ok"] is True


def test_off_run_disables_thinking_at_tabbyapi_top_level_and_probes_it(monkeypatch):
    monkeypatch.delenv("REAL_BENCH_API_KEY", raising=False)
    from llm_bench import compatibility
    from llm_bench.ordinary_runs import Submission, mapped_request_settings, resolve_run_config

    endpoint = Endpoint(ok_handler)
    payload = submission(
        setup_record(
            requests__reasoning={
                "chat_template_kwargs": {"enable_thinking": True},
                "enable_thinking": True,
            }
        ),
        endpoint.base_url,
    )
    payload["run_config"]["reasoning_effort"] = "off"
    payload["run_config"]["repetition_penalty"] = 1.1
    resolved = resolve_run_config(Submission.model_validate(payload)).declared
    assert mapped_request_settings(resolved) == {
        "temperature": 0.5,
        "repetition_penalty": 1.1,
        "enable_thinking": False,
    }

    evidence = compatibility.check(resolved, include_settings=True, key=None)
    assert evidence["engine"] == "exl3"
    assert evidence["checks"]["enable_thinking"]["ok"] is True
    assert "reasoning_effort" not in evidence["checks"]
    assert any(
        request["body"].get("enable_thinking") is False for request in endpoint.completions()
    )


def test_off_probe_rejects_named_tabbyapi_enable_thinking_400(monkeypatch):
    monkeypatch.delenv("REAL_BENCH_API_KEY", raising=False)
    from llm_bench import compatibility
    from llm_bench.ordinary_runs import Submission, resolve_run_config

    def rejects_off(body):
        if body.get("enable_thinking") is False:
            return 400, {"error": {"message": "enable_thinking is unsupported"}}
        return completion_ok(body)

    endpoint = Endpoint(rejects_off)
    payload = submission(setup_record(), endpoint.base_url)
    payload["run_config"]["reasoning_effort"] = "off"
    resolved = resolve_run_config(Submission.model_validate(payload)).declared
    evidence = compatibility.check(resolved, include_settings=True, key=None)
    assert evidence["unsupported"] == ["enable_thinking"]
    assert evidence["checks"]["enable_thinking"]["http"] == 400


def test_mapped_request_settings_moves_reasoning_effort_for_exllamav3():
    exl3 = setup_record()
    exl3["declared"]["requests"]["reasoning"]["chat_template_kwargs"]["reasoning_effort"] = "high"
    from llm_bench.ordinary_runs import mapped_request_settings

    mapped = mapped_request_settings(exl3["declared"])
    assert mapped["reasoning_effort"] == "high"
    assert mapped["chat_template_kwargs"] == {"enable_thinking": True}
    assert mapped["repetition_penalty"] == 1.1
    assert "chat_template_kwargs.reasoning_effort" not in mapped
    assert "repeat_penalty" not in mapped
    assert mapped["temperature"] == 0.5


# --- 5. Unmapped engines still pass through ------------------------------------


def test_unmapped_engine_skips_the_gate_and_runs(tmp_path, monkeypatch):
    """An engine without a delivered mapping (e.g. TensorRT-LLM) performs no
    compatibility probe and runs as before; no compatibility evidence is
    retained for it."""
    monkeypatch.setenv("REAL_BENCH_API_KEY", "live-key")
    results = tmp_path / "results"
    endpoint = Endpoint(ok_handler)
    monkeypatch.setattr(subprocess, "run", pilot_tool)
    setup = setup_record(serving__engine="tensorrt-llm")
    payload = submission(setup, endpoint.base_url)
    source = tmp_path / "submission.json"
    source.write_text(json.dumps(payload))

    result = CliRunner().invoke(
        app, ["run", str(source), "--results", str(results), "--no-interactive"]
    )
    assert result.exit_code == 0, result.output
    # No endpoint probe happened for an unmapped engine.
    assert endpoint.requests == []
    run_id = result_json(result.output)["run_id"]
    assert not (results / run_id / "compatibility.json").exists()


# --- 6. Credentials never reach retained evidence ------------------------------


def test_key_rides_aiperf_command_and_is_scrubbed_from_retained(tmp_path, monkeypatch):
    """The bearer key is passed to the native tool so the endpoint receives
    it, but never appears in the retained command record, the export, or any
    run file; the declared setup secret is scrubbed as before."""
    key = "native-secret-777"
    monkeypatch.setenv("REAL_BENCH_API_KEY", key)
    results = tmp_path / "results"
    endpoint = Endpoint(ok_handler)
    observed_commands: list[list[str]] = []

    def tool(command, **kwargs):
        observed_commands.append(list(command))
        return pilot_tool(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", tool)
    setup = setup_record()
    payload = submission(setup, endpoint.base_url)
    source = tmp_path / "submission.json"
    source.write_text(json.dumps(payload))

    result = CliRunner().invoke(
        app, ["run", str(source), "--results", str(results), "--no-interactive"]
    )
    assert result.exit_code == 0, result.output
    assert observed_commands, "no native tool command was observed"
    for command in observed_commands:
        assert "--api-key" in command
        assert key in command
    run_id = result_json(result.output)["run_id"]
    blob = retained_blob(results / run_id)
    assert key not in blob
    assert "secret-124" not in blob
