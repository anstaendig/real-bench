"""Automatic discovery of instance details inside the guided setup flow.

Discovery is a choice within the same ``bench setup`` flow, not a requirement
for successful setup. It uses only read routes on the OpenAI-compatible
serving API, presents discovered values with their source for review, keeps
declarations distinct from sourced observations, and falls back to guided
manual entry when discovery is unavailable, partial, or contradicts a
declaration. Every case is driven through the public command boundary: the
``bench setup`` CLI and the ``/api/setup`` endpoints.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from llm_bench import setup_state, speed_preparation
from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_cli import app

pytestmark = [
    pytest.mark.usefixtures("external_tool_process_adapter"),
    pytest.mark.usefixtures("preparation_processes"),
]

MARKER = '{\n  "state_file"'


def parse(result_output: str) -> dict[str, Any]:
    """Extract the JSON result block from setup command output."""
    obj, _ = json.JSONDecoder().raw_decode(result_output[result_output.index(MARKER) :])
    return obj


class DiscoveryEndpoint:
    """A controlled OpenAI-compatible endpoint: serves the model list and the
    vLLM loads route, records every request, can enforce a bearer key, and
    rejects POST so a completion or mutation attempt is observable."""

    def __init__(
        self,
        models: list[str] | dict[str, Any] | None = None,
        loads: dict[str, Any] | None = None,
        *,
        auth: str | None = None,
        loads_status: int = 200,
        models_status: int = 200,
    ):
        self.models = models if models is not None else ["test-model"]
        self.loads = (
            loads
            if loads is not None
            else {
                "vllm_version": "0.8.4",
                "model": {"max_model_len": 32768},
                "gpu_memory": [{"device_id": 0, "memory_total": 81920.0, "memory_used": 1024.0}],
            }
        )
        self.loads_status = loads_status
        self.models_status = models_status
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

            def _record(self, body: Any = None) -> None:
                outer.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": {"authorization": self.headers.get("authorization")},
                        **({"body": body} if body is not None else {}),
                    }
                )

            def do_GET(self) -> None:
                self._record()
                if not self._authorized():
                    self._send(401, {"error": {"message": "Invalid API key", "type": "auth"}})
                    return
                if self.path.rstrip("/") == "/v1/models":
                    if outer.models_status == 200:
                        body = outer.models
                        if not isinstance(body, dict):
                            body = {"object": "list", "data": [{"id": i} for i in body]}
                        self._send(200, body)
                    else:
                        self._send(
                            outer.models_status,
                            {"error": {"message": "unknown route", "type": "NotFound"}},
                        )
                elif self.path.rstrip("/") == "/v1/loads":
                    if outer.loads_status == 200:
                        self._send(200, outer.loads)
                    else:
                        self._send(
                            outer.loads_status,
                            {"error": {"message": "unknown route", "type": "NotFound"}},
                        )
                else:
                    self._send(404, {"error": {"message": "unknown route", "type": "NotFound"}})

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                self._record(body)
                if not self._authorized():
                    self._send(401, {"error": {"message": "Invalid API key", "type": "auth"}})
                    return
                self._send(
                    405, {"error": {"message": "method not allowed", "type": "MethodNotAllowed"}}
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def paths(self) -> list[str]:
        return [request["path"] for request in self.requests]

    def methods(self) -> list[str]:
        return [request["method"] for request in self.requests]


def pre_discovery_answers(base_url: str) -> dict[str, Any]:
    """A complete record except the fields discovery is expected to find:
    model ID, context window, and serving version."""
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
        "serving.engine": "vllm",
        "serving.launch_command": "vllm serve org/model",
        "serving.environment": {},
        "requests.base_url": base_url,
        "requests.generation": {"temperature": 0.5},
        "requests.reasoning": {"chat_template_kwargs": {"enable_thinking": True}},
    }


def invoke_setup(
    directory: Path,
    *,
    answers: dict[str, Any] | None = None,
    discover=True,
    extra: list[str] | None = None,
    **input_args: Any,
) -> Any:
    args = ["setup", "--directory", str(directory)]
    if discover:
        args.append("--discover")
    if answers is not None:
        args += ["--answers", json.dumps(answers)]
    if extra:
        args += extra
    return CliRunner().invoke(app, args, **input_args)


# --- 1. Discovery success and the manual path ---------------------------------


def test_discovery_success_offers_values_with_provenance_and_does_not_mutate(tmp_path):
    """Discovery inside bench setup finds endpoint facts, presents them for
    review, saves accepted values with their source, uses only reads, and the
    record completes without re-asking answered fields."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    try:
        result = invoke_setup(
            directory,
            answers=pre_discovery_answers(endpoint.base_url),
            input="a\na\na\n",  # accept all three discovered values
        )
        assert result.exit_code == 0, result.output
        summary = parse(result.output)
        assert summary["critical_complete"] is True
        assert summary["missing_critical"] == []
        # Only the two read routes were used; nothing was POSTed, so the
        # service received no completion and no mutation.
        assert set(endpoint.paths()) == {"/v1/models", "/v1/loads"}
        assert set(endpoint.methods()) == {"GET"}
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.setup["declared"]["requests"]["model_id"] == "test-model"
        assert state.setup["declared"]["requests"]["context_window"] == 32768
        assert state.setup["declared"]["serving"]["version"] == "0.8.4"
        # Observations carry provenance and stay distinct from declarations.
        observed = state.setup["observed"]
        assert observed["requests"]["model_id"] == {
            "value": "test-model",
            "source": "GET /v1/models",
            "date": observed["requests"]["model_id"]["date"],
        }
        assert observed["requests"]["context_window"]["source"] == "GET /v1/loads"
        assert observed["serving"]["version"]["source"] == "GET /v1/loads"
        assert all("date" in entry for category in observed.values() for entry in category.values())
        # The review showed the source for each value.
        assert "GET /v1/models" in result.output
        assert "GET /v1/loads" in result.output
        # Accepted discovered values are saved answers: a resume does not
        # re-ask them.
        assert summary["discovered"] == [
            "requests.context_window",
            "requests.model_id",
            "serving.version",
        ]
    finally:
        endpoint.stop()


def test_setup_without_discovery_sends_no_requests(tmp_path):
    """Discovery is a choice within the setup flow: without it, the complete
    manual path works and the endpoint is never contacted."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    answers = pre_discovery_answers(endpoint.base_url)
    answers.update(
        {
            "requests.model_id": "test-model",
            "requests.context_window": 32768,
            "serving.version": "0.8.4",
        }
    )
    try:
        result = invoke_setup(directory, answers=answers, discover=False)
        assert result.exit_code == 0, result.output
        assert endpoint.requests == []
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.setup["observed"] == {}
    finally:
        endpoint.stop()


def test_partial_discovery_reports_unavailable_fields_and_guides_entry(tmp_path):
    """A 404 loads route means context window and version cannot be observed:
    they are reported unavailable and entered manually, never fabricated, and
    only the model list observation is saved."""
    endpoint = DiscoveryEndpoint(loads_status=404)
    directory = tmp_path / "setup"
    try:
        result = invoke_setup(
            directory,
            answers=pre_discovery_answers(endpoint.base_url),
            input="a\n1\n8192\n",  # accept model_id; manual version, context window
        )
        assert result.exit_code == 0, result.output
        assert "unavailable" in result.output.lower()
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.setup["declared"]["requests"]["model_id"] == "test-model"
        # Manually entered values, not discovered ones.
        assert state.setup["declared"]["requests"]["context_window"] == 8192
        assert state.setup["declared"]["serving"]["version"] == "1"
        # Only what the endpoint actually supplied was observed.
        assert list(state.setup["observed"]) == ["requests"]
        assert list(state.setup["observed"]["requests"]) == ["model_id"]
    finally:
        endpoint.stop()


def test_unreachable_endpoint_falls_back_to_manual_entry(tmp_path):
    """A refused connection is a failed discovery, not a failed setup: the
    session reports it and continues through guided manual entry to completion."""
    # A closed port: the discovery routes never answer.
    endpoint = DiscoveryEndpoint()
    base_url = f"http://127.0.0.1:{endpoint.server.server_address[1]}"
    endpoint.stop()
    directory = tmp_path / "setup"
    answers = pre_discovery_answers(base_url)
    result = invoke_setup(
        directory,
        answers=answers,
        input="1\ntest-model\n8192\n",  # manual version, model_id, context window
    )
    assert result.exit_code == 0, result.output
    assert "unavailable" in result.output.lower()
    state = setup_state.load_state(directory)
    assert state is not None
    assert state.setup["observed"] == {}
    assert state.setup["declared"]["requests"]["model_id"] == "test-model"
    assert state.setup["declared"]["requests"]["context_window"] == 8192
    # The failed discovery is recorded honestly, not as a usable finding.
    assert state.discovery is not None
    assert state.discovery.status == "unavailable"


# --- 2. Contradictions, corrections, and the review boundary -----------------


def test_contradictory_value_is_reviewed_with_both_values_shown(tmp_path):
    """A declared value the endpoint contradicts is a contradiction to review,
    not a silent overwrite: both values and their sources are shown, the
    operator's choice is stored, and the observation retains its provenance."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    answers = pre_discovery_answers(endpoint.base_url)
    answers["requests.model_id"] = "declared-model"
    try:
        result = invoke_setup(
            directory,
            answers=answers,
            input="k\na\na\n",  # keep, accept, accept
        )
        assert result.exit_code == 0, result.output
        assert "conflict" in result.output.lower()
        assert "declared-model" in result.output
        assert "test-model" in result.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.setup["declared"]["requests"]["model_id"] == "declared-model"
        assert state.setup["declared"]["requests"]["context_window"] == 32768
        # The observation is retained even though the declaration was kept:
        # it records what the endpoint actually reported, with its source.
        observed = state.setup["observed"]["requests"]["model_id"]
        assert observed["value"] == "test-model"
        assert observed["source"] == "GET /v1/models"
        assert state.discovery is not None
        assert state.discovery.status == "review"
        assert state.discovery.rejected == ["requests.model_id"]
        assert sorted(state.discovery.accepted) == [
            "requests.context_window",
            "serving.version",
        ]
    finally:
        endpoint.stop()


def test_reviewed_value_can_be_corrected_to_an_operator_value(tmp_path):
    """At review the operator can reject both the discovery and the existing
    declaration and enter the value the runner should actually use."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    answers = pre_discovery_answers(endpoint.base_url)
    answers["requests.model_id"] = "declared-model"
    try:
        result = invoke_setup(directory, answers=answers, input="r\nreal-model-id\na\na\n")
        assert result.exit_code == 0, result.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.setup["declared"]["requests"]["model_id"] == "real-model-id"
        # The correction is a saved answer like any other.
        assert state.provided["requests"]["model_id"] == "real-model-id"
        # The observation still records what the endpoint said.
        assert state.setup["observed"]["requests"]["model_id"]["value"] == "test-model"
    finally:
        endpoint.stop()


def test_repeated_discovery_does_not_reask_unchanged_findings(tmp_path):
    """Running setup again with discovery re-observes the endpoint; a finding
    that is already accepted and still matches is reported without re-asking,
    and the session completes without prompts."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    try:
        first = invoke_setup(
            directory,
            answers=pre_discovery_answers(endpoint.base_url),
            input="a\na\na\n",
        )
        assert first.exit_code == 0, first.output
        again = invoke_setup(directory)
        assert again.exit_code == 0, again.output
        summary = parse(again.output)
        assert summary["critical_complete"] is True
        # No new prompt was needed for the unchanged finding.
        assert "requests.model_id" not in again.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.setup["declared"]["requests"]["model_id"] == "test-model"
    finally:
        endpoint.stop()


def test_accepted_value_change_invalidates_dependent_preparation(tmp_path):
    """Discovery fills context window; changing the endpoint afterwards marks
    the dependent compatibility preparation for revalidation and removes the
    composed suite prepared for the old endpoint."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    try:
        result = invoke_setup(
            directory,
            answers=pre_discovery_answers(endpoint.base_url),
            input="a\na\na\n",
        )
        assert result.exit_code == 0, result.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.preparations["compatibility"].needs_revalidation is False
        suite = speed_preparation.suite_submission_path(directory)
        assert suite.is_file()
        # The endpoint moves: a new base URL changes the configuration that
        # the compatibility preparation depends on.
        changed = invoke_setup(
            directory,
            answers={"requests.base_url": "http://127.0.0.1:9999"},
            discover=False,
            extra=["--no-interactive"],
        )
        assert changed.exit_code != 0  # not critical-complete for the new endpoint
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.preparations["compatibility"].needs_revalidation is True
        assert not suite.is_file()
    finally:
        endpoint.stop()


def test_setup_api_offers_discovery_with_the_same_state(tmp_path, monkeypatch):
    """POST /api/setup accepts a discovery choice and returns the same
    resumable state with provenance as the CLI; GET /api/setup reads it back."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    client = TestClient(create_app(results=tmp_path / "results"))
    try:
        response = client.post(
            "/api/setup",
            json={
                "directory": str(directory),
                "discover": True,
                "answers": pre_discovery_answers(endpoint.base_url),
            },
        )
        assert response.status_code == 200, response.text
        state = response.json()
        assert state["critical_complete"] is True
        # Unreviewed findings are presented for review, not silently stored.
        assert "requests.model_id" in state["discovery"]["findings"]
        assert state["discovery"]["findings"]["requests.model_id"]["source"] == "GET /v1/models"
        # The discovered value is not yet a declaration.
        assert state["setup"]["declared"]["requests"].get("model_id") is None
        assert set(endpoint.methods()) == {"GET"}
        # Reviewing through the same API stores the accepted value.
        reviewed = client.post(
            "/api/setup",
            json={
                "directory": str(directory),
                "discover": True,
                "answers": {
                    "requests.model_id": "test-model",
                    "requests.context_window": 32768,
                    "serving.version": "0.8.4",
                },
            },
        )
        assert reviewed.status_code == 200, reviewed.text
        saved = client.get("/api/setup", params={"directory": str(directory)})
        assert saved.status_code == 200
        body = saved.json()
        assert body["setup"]["declared"]["requests"]["model_id"] == "test-model"
        assert body["setup"]["observed"]["requests"]["model_id"]["source"] == "GET /v1/models"
        assert body["discovery"]["status"] == "review"
    finally:
        endpoint.stop()


# Guided-entry answers for every field discovery does not cover, in prompt
# order, following the blank-records-unknown convention: blanks record
# descriptive unknowns, and object fields end with an empty key line.
MANUAL_TAIL = [
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
    "serve --api-key secret-124",
    "",  # environment: no keys
    "temperature",
    "0.5",
    "",  # generation done
    "chat_template_kwargs",
    '{"enable_thinking": true}',
    "",  # reasoning done
]


# --- 4. Hardening regressions ------------------------------------------------


def test_repeated_discovery_keeps_ownership_of_its_values(tmp_path):
    """A re-run that confirms every finding silently still owns the values it
    accepted: a later endpoint change must clear them even though the second
    review presented nothing new."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    try:
        first = invoke_setup(
            directory,
            answers=pre_discovery_answers(endpoint.base_url),
            input="a\na\na\n",
        )
        assert first.exit_code == 0, first.output
        again = invoke_setup(directory)
        assert again.exit_code == 0, again.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.discovery is not None
        assert state.discovery.status == "review"
        assert sorted(state.discovery.accepted) == [
            "requests.context_window",
            "requests.model_id",
            "serving.version",
        ]
        changed = invoke_setup(
            directory,
            answers={"requests.base_url": "http://127.0.0.1:9999"},
            discover=False,
            extra=["--no-interactive"],
        )
        assert changed.exit_code != 0, changed.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.setup["declared"]["requests"]["model_id"] is None
        assert state.setup["declared"]["requests"]["context_window"] is None
        assert state.setup["declared"]["serving"]["version"] == "unknown"
        assert state.discovery is not None
        assert state.discovery.status == "stale"
    finally:
        endpoint.stop()


def test_repeated_discovery_preserves_rejected_operator_values(tmp_path):
    """Values the operator rejected at review (kept declared) and corrected are
    not owned by discovery: an endpoint change leaves them in place and clears
    only what discovery actually accepted."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    answers = pre_discovery_answers(endpoint.base_url)
    answers["requests.model_id"] = "declared-model"
    try:
        result = invoke_setup(directory, answers=answers, input="r\nreal-model-id\na\na\n")
        assert result.exit_code == 0, result.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.setup["declared"]["requests"]["model_id"] == "real-model-id"
        assert state.discovery is not None
        assert state.discovery.accepted == ["requests.context_window", "serving.version"]
        changed = invoke_setup(
            directory,
            answers={"requests.base_url": "http://127.0.0.1:9999"},
            discover=False,
            extra=["--no-interactive"],
        )
        assert changed.exit_code != 0, changed.output
        state = setup_state.load_state(directory)
        assert state is not None
        # Operator-owned: kept. Discovery-owned: cleared.
        assert state.setup["declared"]["requests"]["model_id"] == "real-model-id"
        assert state.setup["declared"]["requests"]["context_window"] is None
        assert state.setup["declared"]["serving"]["version"] == "unknown"
    finally:
        endpoint.stop()


def test_endpoint_change_survives_incomplete_preparation_inputs(tmp_path):
    """Changing the endpoint drops the compatibility preparation's inputs to a
    gap, but the entry with its revalidation flag survives the gap instead of
    being dropped; the review it came from is stale."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    try:
        result = invoke_setup(
            directory,
            answers=pre_discovery_answers(endpoint.base_url),
            input="a\na\na\n",
        )
        assert result.exit_code == 0, result.output
        changed = invoke_setup(
            directory,
            answers={"requests.base_url": "http://127.0.0.1:9999"},
            discover=False,
            extra=["--no-interactive"],
        )
        assert changed.exit_code != 0, changed.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert "compatibility" in state.preparations
        assert state.preparations["compatibility"].needs_revalidation is True
    finally:
        endpoint.stop()


def test_stale_review_cannot_complete_the_new_endpoint(tmp_path):
    """A review with nothing left to offer is still history once the endpoint
    moved: it is marked stale, so its old findings cannot report the new
    endpoint's critical values as reviewable."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    answers = pre_discovery_answers(endpoint.base_url)
    # Every declared value already matches the endpoint, so the review
    # confirms silently and has nothing left to offer.
    answers["requests.model_id"] = "test-model"
    answers["serving.version"] = "0.8.4"
    answers["requests.context_window"] = 32768
    try:
        result = invoke_setup(directory, answers=answers)
        assert result.exit_code == 0, result.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.discovery is not None
        assert state.discovery.status == "review"
        assert state.discovery.findings == {}
        changed = invoke_setup(
            directory,
            answers={"requests.base_url": "http://127.0.0.1:9999"},
            discover=False,
            extra=["--no-interactive"],
        )
        assert changed.exit_code != 0, changed.output
        # A stale review offers nothing, so the new endpoint's critical values
        # are cleared and the session reports them missing.
        assert "execution-critical and required" in changed.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.discovery is not None
        assert state.discovery.status == "stale"
        assert state.setup["declared"]["requests"]["model_id"] is None
        assert state.setup["declared"]["requests"]["context_window"] is None
        assert state.setup["declared"]["serving"]["version"] == "unknown"
    finally:
        endpoint.stop()


def test_operator_supplied_replacement_survives_endpoint_change(tmp_path):
    """An operator-supplied value that differs from what discovery reported is
    the operator's own answer, not discovery's: an endpoint change keeps it
    and clears only the discovery-accepted fields."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    answers = pre_discovery_answers(endpoint.base_url)
    answers["requests.model_id"] = "declared-model"
    try:
        result = invoke_setup(directory, answers=answers, input="r\nreal-model-id\na\na\n")
        assert result.exit_code == 0, result.output
        # The endpoint moves and the operator supplies the new model at once.
        changed = invoke_setup(
            directory,
            answers={"requests.base_url": "http://127.0.0.1:9999", "requests.model_id": "other"},
            discover=False,
            extra=["--no-interactive"],
        )
        assert changed.exit_code != 0, changed.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.setup["declared"]["requests"]["model_id"] == "other"
        assert state.setup["declared"]["requests"]["context_window"] is None
        assert state.setup["declared"]["serving"]["version"] == "unknown"
    finally:
        endpoint.stop()


def test_fresh_answer_survives_endpoint_change_after_silent_confirm(tmp_path):
    """A value the operator supplies after an unchanged re-run confirmed the
    findings silently is the operator's own answer: the endpoint change keeps
    it and records the endpoint it replaced, instead of erasing it."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    try:
        first = invoke_setup(
            directory,
            answers=pre_discovery_answers(endpoint.base_url),
            input="a\na\na\n",
        )
        assert first.exit_code == 0, first.output
        # An unchanged re-run re-confirms every finding silently: the review
        # presents nothing new, but ownership of the values is kept.
        again = invoke_setup(directory)
        assert again.exit_code == 0, again.output
        # The endpoint moves and the operator supplies the new model id at
        # once: that answer is operator-owned, not discovery's, so it must
        changed = invoke_setup(
            directory,
            answers={
                "requests.base_url": "http://127.0.0.1:9999",
                "requests.model_id": "fresh-model",
            },
            discover=False,
            extra=["--no-interactive"],
        )
        assert changed.exit_code != 0, changed.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.setup["declared"]["requests"]["model_id"] == "fresh-model"
        assert state.setup["declared"]["requests"]["context_window"] is None
        assert state.setup["declared"]["serving"]["version"] == "unknown"
        # The stale review still names the endpoint its values came from.
        assert state.discovery is not None
        assert state.discovery.status == "stale"
        assert state.discovery.endpoint == endpoint.base_url
    finally:
        endpoint.stop()


def test_interrupted_discovery_review_saves_progress(tmp_path, monkeypatch):
    """An interruption during the discovery review still saves the session:
    the supplied answers persist and the review record is resumable."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    try:
        calls = {"n": 0}

        def fake_input(prompt: str = "") -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                return "a"  # accept the model id finding
            raise KeyboardInterrupt  # interrupt at the version prompt

        with monkeypatch.context() as input_patch:
            input_patch.setattr("builtins.input", fake_input)
            interrupted = invoke_setup(
                directory,
                answers={"requests.base_url": endpoint.base_url},
            )
        assert interrupted.exit_code == 130
        state = setup_state.load_state(directory)
        assert state is not None
        # The supplied answer and the reviewed finding both persist.
        assert state.provided["requests"]["base_url"] == endpoint.base_url
        assert state.setup["declared"]["requests"]["model_id"] == "test-model"
        assert state.discovery is not None
        assert state.discovery.status == "review"
        # Only the finding resolved before the interrupt is in the record; the
        # review is resumable rather than blank.
        assert set(state.discovery.findings) == {"requests.model_id"}
        assert state.discovery.accepted == ["requests.model_id"]
        # Only input is restored; preparation and process fixtures stay installed.
        # Resume: the endpoint is known (saved), the model id is re-observed
        # silently, the two left-pending findings are confirmed, and guided
        # entry completes the record without re-asking answered fields.
        completed = invoke_setup(
            directory,
            input="a\na\n" + "\n".join(MANUAL_TAIL) + "\n",
        )
        assert completed.exit_code == 0, completed.output
        summary = parse(completed.output)
        assert summary["critical_complete"] is True
    finally:
        endpoint.stop()


def test_discovery_prompts_for_the_endpoint_first(tmp_path):
    """Choosing discovery without a known endpoint asks for the endpoint
    first, then discovers; the session completes and the value is saved."""
    endpoint = DiscoveryEndpoint()
    directory = tmp_path / "setup"
    try:
        result = invoke_setup(
            directory,
            input=endpoint.base_url + "\na\na\na\n" + "\n".join(MANUAL_TAIL) + "\n",
        )
        assert result.exit_code == 0, result.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.setup["declared"]["requests"]["base_url"] == endpoint.base_url
        assert state.setup["declared"]["requests"]["model_id"] == "test-model"
        assert state.setup["declared"]["requests"]["context_window"] == 32768
        assert state.setup["declared"]["serving"]["version"] == "0.8.4"
        assert set(endpoint.methods()) == {"GET"}
    finally:
        endpoint.stop()


def test_unreadable_model_list_still_reports_loads_findings(tmp_path):
    """A 404 model list must not throw away a readable loads route: the
    version and context window are still reported with their source, and the
    model id is guided into manual entry rather than fabricated."""
    endpoint = DiscoveryEndpoint(models_status=404)
    directory = tmp_path / "setup"
    try:
        result = invoke_setup(
            directory,
            answers=pre_discovery_answers(endpoint.base_url),
            input="a\na\nmanual-model\n",  # version, context window, manual model id
        )
        assert result.exit_code == 0, result.output
        assert "did not return a readable model list" in result.output
        state = setup_state.load_state(directory)
        assert state is not None
        # The loads route was read despite the broken model list.
        assert set(endpoint.paths()) == {"/v1/models", "/v1/loads"}
        assert state.setup["declared"]["requests"]["model_id"] == "manual-model"
        assert state.setup["declared"]["requests"]["context_window"] == 32768
        assert state.setup["declared"]["serving"]["version"] == "0.8.4"
        assert state.discovery is not None
        assert state.discovery.status == "review"
        assert sorted(state.discovery.findings) == ["requests.context_window", "serving.version"]
        assert sorted(state.discovery.accepted) == ["requests.context_window", "serving.version"]
        # No observation was fabricated for the unreadable route.
        assert "model_id" not in state.setup["observed"].get("requests", {})
    finally:
        endpoint.stop()


def test_malformed_model_list_is_not_fabricated(tmp_path):
    """A model list whose entries lack readable ids (or that is not a list at
    all) yields no model finding: the value is entered manually and the rest
    of discovery still works."""
    endpoint = DiscoveryEndpoint(models={"object": "list", "data": [{"id": None}, {}]})
    directory = tmp_path / "setup"
    try:
        result = invoke_setup(
            directory,
            answers=pre_discovery_answers(endpoint.base_url),
            input="a\na\nmanual-model\n",
        )
        assert result.exit_code == 0, result.output
        assert "did not return a readable model list" in result.output
        state = setup_state.load_state(directory)
        assert state is not None
        assert state.setup["declared"]["requests"]["model_id"] == "manual-model"
        assert state.setup["declared"]["requests"]["context_window"] == 32768
        assert state.discovery is not None
        assert state.discovery.status == "review"
    finally:
        endpoint.stop()
