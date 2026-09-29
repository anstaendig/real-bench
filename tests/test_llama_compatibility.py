"""llama.cpp setup and launch compatibility: parameter mapping, settings, and
the gate.

Every case is driven through the public command boundary: the ``bench check``
CLI and the ``bench run`` CLI submission contract. A local HTTP server plays
the externally operated llama.cpp OpenAI-compatible endpoint with controlled
responses (and, where the case requires, enforceable authorization), so no
real service is launched, stopped, or reconfigured. The engine-appropriate
mapping (``repetition_penalty`` -> ``repeat_penalty``) is verified both on the
wire and in the prepared scored inputs, so the probe and the scored request
carry the same settings.
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
    """A controlled llama.cpp endpoint: records every request, can enforce a
    bearer key, and answers with controlled status bodies. It exposes only
    reads and inference."""

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


def repeat_penalty_rejecting(body: dict[str, Any]) -> tuple[int, Any]:
    """A llama.cpp build whose OpenAI-compatible layer rejects the mapped
    repeat penalty with a named 400; every other request is accepted."""
    if "repeat_penalty" in body:
        return (
            400,
            {
                "error": {
                    "message": "Unknown or invalid field: repeat_penalty",
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
            "accelerator": "CPU",
            "count": 1,
            "memory": "32 GiB",
            "cpu": "CPU",
            "ram": "64 GiB",
            "os": "macOS",
            "driver": "n/a",
        },
        "model": {
            "source": "org/model",
            "revision": "abc",
            "quantized_artifact": "model-q4.gguf",
            "tokenizer": "org/model",
            "template_overrides": {},
        },
        "serving": {
            "engine": "llama.cpp",
            "version": "b6150",
            "launch_command": "llama-server --api-key secret-124",
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


def test_check_unauthenticated_success_probes_mapped_settings(tmp_path, monkeypatch):
    """Without any key the check authenticates nothing, uses only the model
    list and a minimal completion, and the repeated-penalty setting rides the
    wire under its llama.cpp name."""
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
    assert evidence["engine"] == "llama.cpp"
    assert evidence["checks"]["models"]["ok"] is True
    assert evidence["checks"]["completion"]["ok"] is True
    assert evidence["unsupported"] == []
    # The engine-appropriate mapping is applied: no request carries the
    # shared per-run name.
    assert "repeat_penalty" in evidence["checks"]
    assert "repetition_penalty" not in evidence["checks"]
    probe_bodies = [request["body"] for request in endpoint.completions()]
    assert not any("repetition_penalty" in body for body in probe_bodies)
    assert any(body.get("repeat_penalty") == 1.1 for body in probe_bodies)


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


def test_check_reports_unsupported_mapped_setting_from_a_named_400(tmp_path):
    """A 400 that names the mapped parameter is evidence the setting was not
    applied; the check reports it as unsupported with the server's own
    message and the mapping is retained."""
    endpoint = Endpoint(repeat_penalty_rejecting)
    from llm_bench import compatibility

    setup = setup_record()
    setup["declared"]["requests"]["base_url"] = endpoint.base_url
    evidence = compatibility.check(setup["declared"], include_settings=True, key=None)
    assert evidence["unsupported"] == ["repeat_penalty"]
    assert evidence["checks"]["temperature"]["ok"] is True
    assert evidence["checks"]["repeat_penalty"]["ok"] is False
    assert "repeat_penalty" in evidence["checks"]["repeat_penalty"]["error"]
    assert evidence["setting_mappings"] == {"repetition_penalty": "repeat_penalty"}


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
    assert "repeat_penalty" in evidence["checks"]
    retained = directory / "compatibility" / "check.json"
    assert retained.is_file()
    stored = json.loads(retained.read_text())
    assert stored["checks"] == evidence["checks"]
    assert stored["setting_mappings"] == {"repetition_penalty": "repeat_penalty"}
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

    endpoint = Endpoint(repeat_penalty_rejecting)
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
    assert "repeat_penalty" in output
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
    applied mapping."""
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
    assert evidence["engine"] == "llama.cpp"
    assert evidence["checks"]["completion"]["ok"] is True
    assert evidence["unsupported"] == []
    assert evidence["checks"]["temperature"]["ok"] is True
    assert evidence["checks"]["repeat_penalty"]["ok"] is True
    assert evidence["checks"]["chat_template_kwargs.reasoning_effort"]["ok"] is True
    assert "live-key" not in compatibility_file.read_text()


# --- 4. Outgoing parameter mapping --------------------------------------------


def test_outgoing_parameter_mapping_matches_final_settings(tmp_path, monkeypatch):
    """The final settings ride to the endpoint exactly as resolved: the launch
    gate probes each one under its llama.cpp name, and the prepared scored
    input carries the same merged, mapped settings."""
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
    # Each final setting rode its own probe with the resolved value; the
    # repeated-penalty setting under its llama.cpp name.
    temp_probe = next(body for body in seen if "temperature" in body)
    assert temp_probe["temperature"] == 0.3
    top_p_probe = next(body for body in seen if "top_p" in body)
    assert top_p_probe["top_p"] == 0.9
    repeat_probe = next(body for body in seen if "repeat_penalty" in body)
    assert repeat_probe["repeat_penalty"] == 1.2
    # No probe carried the shared per-run name: the mapping reached the wire.
    assert not any("repetition_penalty" in body for body in seen)
    # The prepared scored inputs are the outgoing mapping aiperf would send:
    # the merged generation settings plus the resolved template kwargs, with
    # the mapped penalty name.
    assert input_records, "no prepared scored input was observed"
    for record in input_records:
        extra = record["extra"]
        assert extra["temperature"] == 0.3
        assert extra["top_p"] == 0.9
        assert extra["repeat_penalty"] == 1.2
        assert "repetition_penalty" not in extra
        assert extra["chat_template_kwargs"] == {
            "enable_thinking": True,
            "reasoning_effort": "high",
        }
    # And the retained record is the resolved setup as executed: the declared
    # record keeps the shared per-run name, the compatibility evidence keeps
    # the applied mapping.
    run_id = result_json(result.output)["run_id"]
    retained = json.loads((results / run_id / "setup.json").read_text())
    assert retained["declared"]["requests"]["generation"] == {
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
    }
    compatibility = json.loads((results / run_id / "compatibility.json").read_text())
    assert compatibility["setting_mappings"] == {"repetition_penalty": "repeat_penalty"}
    assert compatibility["checks"]["repeat_penalty"]["ok"] is True


def test_off_run_disables_thinking_with_llama_penalty_mapping(monkeypatch):
    monkeypatch.delenv("REAL_BENCH_API_KEY", raising=False)
    from llm_bench import compatibility
    from llm_bench.ordinary_runs import Submission, mapped_request_settings, resolve_run_config

    endpoint = Endpoint(ok_handler)
    payload = submission(setup_record(), endpoint.base_url)
    payload["run_config"]["reasoning_effort"] = "off"
    payload["run_config"]["repetition_penalty"] = 1.1
    resolved = resolve_run_config(Submission.model_validate(payload)).declared
    assert mapped_request_settings(resolved) == {
        "temperature": 0.5,
        "repeat_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    evidence = compatibility.check(resolved, include_settings=True, key=None)
    assert evidence["engine"] == "llama.cpp"
    assert evidence["checks"]["chat_template_kwargs.enable_thinking"]["ok"] is True
    assert "chat_template_kwargs.reasoning_effort" not in evidence["checks"]
    assert {"enable_thinking": False} in [
        request["body"]["chat_template_kwargs"]
        for request in endpoint.completions()
        if "chat_template_kwargs" in request["body"]
    ]


def test_mapped_request_settings_renames_for_llama_and_keeps_vllm_unchanged():
    """Every outgoing request-settings merge site (fresh-context, growing
    session, coding, RULER) resolves through one shared helper. llama.cpp
    renames the shared repeated-penalty setting to its own name and passes
    the reasoning template kwargs through; vLLM keeps every shared name so a
    delivered vLLM request is byte-identical to before."""
    from llm_bench.ordinary_runs import mapped_request_settings

    llama = setup_record()
    mapped = mapped_request_settings(llama["declared"])
    assert mapped["repeat_penalty"] == 1.1
    assert "repetition_penalty" not in mapped
    assert mapped["temperature"] == 0.5
    assert mapped["chat_template_kwargs"] == {"enable_thinking": True}

    vllm = setup_record(serving__engine="vllm")
    assert mapped_request_settings(vllm["declared"]) == {
        **vllm["declared"]["requests"]["generation"],
        **vllm["declared"]["requests"]["reasoning"],
    }


# --- 5. Other engines pass through unchanged -----------------------------------


def test_other_engine_skips_the_gate_and_runs(tmp_path, monkeypatch):
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
