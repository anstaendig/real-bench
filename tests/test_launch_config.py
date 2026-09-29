"""Explicit per-launch configuration: the CLI/API submission contract.

Driven through the public boundary — the ``bench run`` CLI and ``POST /api/runs``
— with assertions against independently read retained results. Covers missing
choices, historical review, explicit inputs, outbound settings, and retained
setup.
"""

import gzip
import json
import os
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_ordinary_runs import submission
from typer.testing import CliRunner

from llm_bench.launch_config import REVIEW_ORDER, collect_run_config, historical_sampling
from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_cli import app
from llm_bench.ordinary_runs import RunConfig

pytestmark = pytest.mark.usefixtures("external_tool_process_adapter")


def serve_retained_export(command, **kwargs):
    """Serve the retained export without asserting the outbound settings."""
    native = Path(command[command.index("--output-artifact-dir") + 1])
    native.mkdir(parents=True, exist_ok=True)
    source = Path("tests/fixtures/pilots/performance-121/native")
    for name in ("profile_export.jsonl", "profile_export_raw.jsonl"):
        (native / name).write_bytes(gzip.decompress((source / (name + ".gz")).read_bytes()))
    return subprocess.CompletedProcess(command, 0, "", "")


def native_tool(command, **kwargs):
    """Assert the outbound single-condition settings, then serve the retained export."""
    dataset = Path(command[command.index("--input-file") + 1])
    sent = json.loads(dataset.read_text())
    assert sent["extra"]["temperature"] == 0.5
    assert sent["extra"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    }
    return serve_retained_export(command, **kwargs)


def write_submission(tmp_path: Path, payload: dict, name: str = "submission.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def retained_setup(results: Path, run_id: str) -> dict:
    return json.loads((results / run_id / "setup.json").read_text())


def result_json(output: str) -> dict:
    """The run result JSON; a prompt banner may precede it on stdout."""
    return json.loads(output[output.index("{") :])


def test_collect_run_config_presents_history_and_keeps_reasoning_unselected():
    """Historical sampling values are offered, but off is a fresh, valid choice."""
    prompts: list[str] = []
    errors: list[str] = []
    answers = iter(["1.2", "", "", "", "", "", "off"])

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    config = collect_run_config(
        {"temperature": 1.0, "top_p": 0.95}, input_fn=input_fn, output_fn=errors.append
    )
    # Historical values are shown as recommendations (blank accepts them); a
    # typed answer edits, a blank records no override once history exists.
    assert config.reasoning_effort == "off"
    assert config.temperature == 1.2
    assert config.top_p == 0.95
    assert (config.top_k, config.min_p, config.presence_penalty, config.repetition_penalty) == (
        None,
        None,
        None,
        None,
    )
    # The historical keys are presented for review; the reasoning key is not.
    assert any(p.startswith("temperature (previous 1.0") for p in prompts)
    assert any(p.startswith("top_p (previous 0.95") for p in prompts)
    reasoning_prompt = next(p for p in prompts if p.startswith("reasoning_effort"))
    assert reasoning_prompt == "reasoning_effort (required; 'off' disables thinking): "
    assert len([p for p in prompts if p.startswith("reasoning_effort")]) == 1
    assert errors == []


def test_collect_run_config_can_clear_a_historical_override():
    """Once a value exists in history, the interactive review can still record
    the API's explicit no-override via 'none' — it is not locked to accepting."""
    prompts: list[str] = []
    answers = iter(["none", "", "", "", "", "", "low"])

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    config = collect_run_config({"temperature": 1.0}, input_fn=input_fn)
    # The historical temperature is cleared, not silently accepted.
    assert config.temperature is None
    assert (
        config.top_p,
        config.top_k,
        config.min_p,
        config.presence_penalty,
        config.repetition_penalty,
    ) == (
        None,
        None,
        None,
        None,
        None,
    )
    assert config.reasoning_effort == "low"
    assert any(p.startswith("temperature (previous 1.0") for p in prompts)


def test_missing_run_config_rejected_by_api_and_unattended_cli(tmp_path, monkeypatch):
    """A submission without per-run choices is 422 via API and an actionable
    error via unattended CLI; no work occurs."""
    payload = submission()
    payload.pop("run_config")
    tool_calls = []

    def external(command, **kwargs):
        tool_calls.append(command)
        return native_tool(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", external)
    client = TestClient(create_app(tmp_path))
    assert client.post("/api/runs", json=payload).status_code == 422
    assert list(tmp_path.iterdir()) == []

    source = write_submission(tmp_path, payload)
    # --no-interactive and piped/no input are both unattended; neither may
    # bypass the required choices.
    no_interactive = CliRunner().invoke(
        app, ["run", str(source), "--results", str(tmp_path / "results"), "--no-interactive"]
    )
    assert no_interactive.exit_code != 0
    assert "run_config is required at launch" in no_interactive.output
    unattended = CliRunner().invoke(
        app, ["run", str(source), "--results", str(tmp_path / "results")]
    )
    assert unattended.exit_code != 0
    assert "run_config is required at launch" in unattended.output
    assert tool_calls == []
    assert list(tmp_path.iterdir()) == [source]


def test_invalid_run_config_rejected_before_execution(tmp_path, monkeypatch):
    """Empty and alias efforts, frequency penalty, and seed remain invalid."""
    tool_calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: tool_calls.append(command) or native_tool(command, **kwargs),
    )
    for effort in ("", "   ", "none", "null"):
        payload = submission()
        payload["run_config"]["reasoning_effort"] = effort
        assert TestClient(create_app(tmp_path)).post("/api/runs", json=payload).status_code == 422
    for key in ("frequency_penalty", "seed"):
        payload = submission()
        payload["run_config"][key] = 1
        assert TestClient(create_app(tmp_path)).post("/api/runs", json=payload).status_code == 422
    assert tool_calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("effort", ["off", " Off "])
def test_run_config_canonicalizes_explicit_off(effort):
    config = RunConfig.model_validate({**submission()["run_config"], "reasoning_effort": effort})
    assert config.reasoning_effort == "off"


def test_cli_interactive_launch_completes_missing_choices_into_retained_result(
    tmp_path, monkeypatch
):
    """An interactive launch prompts for the missing per-run choices, resolves
    them into the final setup record, and the retained result is independently
    readable. A seeded prior run's sampling is accepted from the prompt."""
    results = tmp_path / "results"
    prior = results / "11111111111111111111111111111111"
    prior.mkdir(parents=True)
    (prior / "setup.json").write_text(
        json.dumps(
            {
                "declared": {
                    "requests": {
                        "generation": {"temperature": 1.0, "top_p": 0.95},
                        "reasoning": {
                            "chat_template_kwargs": {
                                "enable_thinking": True,
                                "reasoning_effort": "low",
                            }
                        },
                    }
                }
            }
        )
    )

    monkeypatch.setattr(subprocess, "run", serve_retained_export)
    payload = submission()
    payload.pop("run_config")
    source = write_submission(tmp_path, payload)
    # temperature is edited to 1.2, top_p is accepted from history (0.95), the
    # rest record an explicit no-override, and reasoning is chosen fresh.
    answers = ["1.2", "", "", "", "", "", "medium"]
    launched = CliRunner().invoke(
        app, ["run", str(source), "--results", str(results)], input="\n".join(answers) + "\n"
    )
    assert launched.exit_code == 0, launched.output
    run_id = result_json(launched.output)["run_id"]

    retained = retained_setup(results, run_id)
    assert retained["declared"]["requests"]["generation"] == {"temperature": 1.2, "top_p": 0.95}
    assert retained["declared"]["requests"]["reasoning"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    result = TestClient(create_app(results)).get("/api/runs/" + run_id).json()
    assert result["status"] == "completed"
    assert result["setup"]["declared"]["requests"]["generation"] == {
        "temperature": 1.2,
        "top_p": 0.95,
    }


def test_explicit_inputs_run_as_supplied_via_cli_and_api(tmp_path, monkeypatch):
    """Explicit per-run choices (CLI file and API body) run as supplied without
    prompting; the retained record and outbound settings reflect them."""
    monkeypatch.setattr(subprocess, "run", serve_retained_export)
    payload = submission()
    # Choices deliberately differ from the builder defaults to show the
    # explicit configuration is what runs, not an inherited fallback.
    payload["run_config"] = {
        "reasoning_effort": "high",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": None,
        "min_p": None,
        "presence_penalty": None,
        "repetition_penalty": None,
    }

    source = write_submission(tmp_path, payload)
    cli = CliRunner().invoke(app, ["run", str(source), "--results", str(tmp_path / "results")])
    assert cli.exit_code == 0, cli.output
    cli_id = result_json(cli.output)["run_id"]
    # No prompt occurs when the choices are explicit.
    assert "Review the per-run configuration" not in cli.output

    api = TestClient(create_app(tmp_path / "api")).post("/api/runs", json=payload).json()
    assert api["status"] == "completed"

    for root, run_id in ((tmp_path / "results", cli_id), (tmp_path / "api", api["run_id"])):
        retained = retained_setup(root, run_id)
        assert retained["declared"]["requests"]["generation"] == {"temperature": 0.7, "top_p": 0.9}
        assert retained["declared"]["requests"]["reasoning"]["chat_template_kwargs"] == {
            "enable_thinking": True,
            "reasoning_effort": "high",
        }


def test_null_sampling_records_no_override_and_preserves_declared_keys(tmp_path, monkeypatch):
    """A null sampling choice records no override; declared non-sampling keys
    such as seed are preserved, and a value overrides the declared setting."""
    monkeypatch.setattr(subprocess, "run", native_tool)
    payload = submission()
    # Declared generation also carries a non-sampling key the resolver keeps.
    payload["setup"]["declared"]["requests"]["generation"] = {"temperature": 0.9, "seed": 121}
    payload["run_config"] = {
        "reasoning_effort": "xhigh",
        "temperature": 0.5,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "presence_penalty": None,
        "repetition_penalty": None,
    }
    result = TestClient(create_app(tmp_path)).post("/api/runs", json=payload).json()
    generation = retained_setup(tmp_path, result["run_id"])["declared"]["requests"]["generation"]
    # temperature is overridden to the per-run choice; seed survives; the null
    # choices add nothing.
    assert generation == {"temperature": 0.5, "seed": 121}


def test_historical_sampling_reads_retained_runs_most_recent_first(tmp_path):
    """The historical reviewer prefers the most recent retained run and never
    re-recommends a key a later run explicitly dropped."""
    results = tmp_path / "results"
    older = results / "11111111111111111111111111111111"
    newer = results / "22222222222222222222222222222222"
    for directory, generation, reasoning in (
        (older, {"temperature": 0.9, "top_p": 0.95}, {"reasoning_effort": "high"}),
        (newer, {"temperature": 0.5}, {"reasoning_effort": "off"}),
    ):
        directory.mkdir(parents=True)
        (directory / "setup.json").write_text(
            json.dumps(
                {"declared": {"requests": {"generation": generation, "reasoning": reasoning}}}
            )
        )
    # Force an mtime ordering that does not rely on wall-clock spacing.
    os.utime(older / "setup.json", (100, 100))
    os.utime(newer / "setup.json", (200, 200))

    # Only the newest run's sent keys are recommended: its absent top_p is not
    # resurrected from the older run's 0.95.
    assert historical_sampling(results) == {"temperature": 0.5}
    # The reviewer presents only the sampling parameters, never reasoning.
    assert "reasoning_effort" not in REVIEW_ORDER
