"""Public sampled-RULER submission coverage."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_ordinary_runs import submission
from typer.testing import CliRunner

from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_cli import app
from llm_bench.ordinary_runs import RULER_OUTPUT_BUDGET

pytestmark = pytest.mark.usefixtures("external_tool_process_adapter")

TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
CHECKPOINTS = (32768, 131072, 196608)


def ruler_submission(source: Path):
    payload = submission()
    payload["setup"]["declared"]["requests"]["context_window"] = 262144
    payload["setup"]["declared"]["model"]["tokenizer"] = str(source / "tokenizer")
    payload["workload"] = {
        "profile": "sampled-ruler",
        "source_path": str(source),
    }
    return payload


def source(results: Path) -> Path:
    """A controlled RULER checkout inside results/setup, where API runs may read it."""
    root = results / "setup" / "ruler"
    (root / "scripts/eval/synthetic").mkdir(parents=True)
    (root / "tokenizer").mkdir()
    (root / "tokenizer/tokenizer.json").write_text("controlled tokenizer")
    (root / "tokenizer/chat_template.jinja").write_text("controlled template")
    fixture = Path("tests/fixtures/ruler-upstream")
    shutil.copyfile(fixture / "synthetic.yaml", root / "scripts/synthetic.yaml")
    shutil.copyfile(fixture / "constants.py.txt", root / "scripts/eval/synthetic/constants.py")
    return root


def controlled_tools(
    malformed=False,
    invalid_preparation: str | None = None,
    cancel=False,
    endpoint_prompt_delta: int = 0,
    omit_usage: bool = False,
    reasoning_only: bool = False,
):
    def external(command, **kwargs):
        if any(str(part).endswith("ruler_pilot.py") for part in command):
            directory = Path(command[command.index("--directory") + 1])
            directory.mkdir()
            max_tokens = int(command[command.index("--quality-output-limit") + 1])
            window = int(command[command.index("--context-window") + 1])
            template = json.loads(command[command.index("--chat-template-kwargs") + 1])
            seed = int(command[command.index("--seed") + 1])
            tokenizer = Path(command[command.index("--tokenizer") + 1])
            template_path = Path(command[command.index("--chat-template-path") + 1])
            directory.joinpath("workload.json").write_text(
                json.dumps(
                    {
                        "source_revision": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
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
            if invalid_preparation == "overshoot":
                preparation[0]["samples"][0]["input_tokens"] = 32769
            elif invalid_preparation == "duplicate":
                preparation[0]["samples"][1]["index"] = 0
            directory.joinpath("preparation.json").write_text(json.dumps(preparation))
        elif "profile" in command:
            if cancel:
                raise KeyboardInterrupt()
            native = Path(command[command.index("--output-artifact-dir") + 1])
            native.mkdir()
            if not malformed or "niah_single_1-32768-0" not in str(native):
                content = "expected"
                checkpoint = int(native.parent.name.rsplit("-", 2)[1])
                usage = {"completion_tokens": 1}
                if not omit_usage:
                    usage["prompt_tokens"] = checkpoint - 1 + endpoint_prompt_delta
                delta = {"reasoning_content": content} if reasoning_only else {"content": content}
                packet = json.dumps(
                    {
                        "usage": usage,
                        "choices": [{"delta": delta, "finish_reason": "stop"}],
                    }
                )
                native.joinpath("profile_export_raw.jsonl").write_text(
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
        return subprocess.CompletedProcess(command, 0, "secret-124", "")

    return external


def test_sampled_ruler_public_run_retains_inputs_scores_and_result_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", controlled_tools())
    client = TestClient(create_app(tmp_path / "results"))
    payload = ruler_submission(source(tmp_path / "results"))
    assert "max_tokens" not in payload["workload"]
    result = client.post("/api/runs", json=payload).json()
    assert result["status"] == "completed"
    assert result["families"] == {
        "performance": "missing",
        "coding": "missing",
        "long_context": "completed",
    }
    assert result["coverage"]["long_context"] == {"expected": 117, "accounted": 117, "scored": 117}
    assert result["ruler"]["sampled_diagnostic"] is True
    assert result["ruler"]["quality_by_context"]["196608"]["mean_score"] == 100.0
    assert result["ruler"]["task_coverage"]["qa_2"]["scored"] == 9
    assert len(result["outcomes"]) == 117
    assert all(row["scorer"] == "pinned upstream RULER metric_fn" for row in result["outcomes"])
    directory = tmp_path / "results" / result["run_id"]
    assert len(list(directory.glob("ruler-*/input.jsonl"))) == 117
    assert all(
        json.loads(path.read_text())["output_length"] == RULER_OUTPUT_BUDGET
        for path in directory.glob("ruler-*/input.jsonl")
    )
    assert all(
        row["output_reservation_tokens"] == RULER_OUTPUT_BUDGET for row in result["outcomes"]
    )
    assert result["ruler"]["input_accounting"]["output_reservation_tokens"] == [RULER_OUTPUT_BUDGET]
    retained_workload = json.loads((directory / "workload.json").read_text())
    assert retained_workload["max_tokens"] == RULER_OUTPUT_BUDGET
    assert (directory / "prepared/preparation.json").exists()
    assert "secret-124" not in "".join(
        path.read_text() for path in directory.rglob("*") if path.is_file()
    )
    assert "Sampled RULER long-context results" in client.get("/runs/" + result["run_id"]).text
    cli = CliRunner().invoke(
        app, ["result", result["run_id"], "--results", str(tmp_path / "results")]
    )
    assert cli.exit_code == 0
    assert json.loads(cli.stdout) == result


def test_sampled_ruler_accepts_bounded_endpoint_count_semantics_and_exposes_provenance(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(subprocess, "run", controlled_tools(endpoint_prompt_delta=1))
    client = TestClient(create_app(tmp_path / "results"))
    result = client.post("/api/runs", json=ruler_submission(source(tmp_path / "results"))).json()

    assert result["status"] == "completed"
    assert result["coverage"]["long_context"] == {
        "expected": 117,
        "accounted": 117,
        "scored": 117,
    }
    row = result["outcomes"][0]
    assert row["nominal_input_ceiling"] == 32768
    assert row["local_input_tokens"] == 32767
    assert row["local_input_tokens_source"].startswith(
        "prepared/preparation.json:niah_single_1-32768.samples[0].input_tokens"
    )
    assert row["endpoint_input_tokens"] == 32768
    assert row["endpoint_input_tokens_source"].endswith(
        "native/profile_export_raw.jsonl:usage.prompt_tokens"
    )
    assert row["input_count_delta_tokens"] == 1
    assert row["input_count_compatibility"] == "endpoint-authoritative"
    assert result["ruler"]["input_accounting"]["compatibility_counts"] == {
        "exact": 0,
        "endpoint-authoritative": 117,
        "incompatible": 0,
        "unavailable": 0,
    }
    assert result["ruler"]["input_accounting"]["endpoint"]["samples"] == 117

    api_result = client.get("/api/runs/" + result["run_id"]).json()
    assert api_result == result
    comparison = client.get("/api/reports").json()
    indexed = next(row for row in comparison["runs"] if row["run_id"] == result["run_id"])
    assert indexed["ruler"]["input_accounting"] == result["ruler"]["input_accounting"]
    cli = CliRunner().invoke(
        app, ["result", result["run_id"], "--results", str(tmp_path / "results")]
    )
    assert cli.exit_code == 0
    assert json.loads(cli.stdout) == result


def test_sampled_ruler_missing_endpoint_usage_stays_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", controlled_tools(omit_usage=True))
    result = (
        TestClient(create_app(tmp_path))
        .post("/api/runs", json=ruler_submission(source(tmp_path)))
        .json()
    )

    assert result["status"] == "completed"
    assert result["coverage"]["long_context"]["scored"] == 117
    row = result["outcomes"][0]
    assert row["local_input_tokens"] == 32767
    assert row["endpoint_input_tokens"] is None
    assert row["endpoint_input_tokens_source"] is None
    assert row["input_count_delta_tokens"] is None
    assert row["input_count_compatibility"] == "unavailable"
    assert result["ruler"]["input_accounting"]["endpoint"]["samples"] == 0
    assert result["ruler"]["input_accounting"]["compatibility_counts"] == {
        "exact": 0,
        "endpoint-authoritative": 0,
        "incompatible": 0,
        "unavailable": 117,
    }


def test_sampled_ruler_null_usage_stays_unavailable(tmp_path, monkeypatch):
    baseline = controlled_tools()

    def external(command, **kwargs):
        result = baseline(command, **kwargs)
        if "profile" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            if native.parent.name == "ruler-niah_single_1-32768-0":
                path = native / "profile_export_raw.jsonl"
                row = json.loads(path.read_text())
                packet = row["responses"][0]["packets"][0]
                chunk = json.loads(packet["value"])
                chunk["usage"] = None
                packet["value"] = json.dumps(chunk)
                path.write_text(json.dumps(row) + "\n")
        return result

    monkeypatch.setattr(subprocess, "run", external)
    result = (
        TestClient(create_app(tmp_path))
        .post("/api/runs", json=ruler_submission(source(tmp_path)))
        .json()
    )

    assert result["status"] == "completed"
    assert result["coverage"]["long_context"]["scored"] == 117
    row = result["outcomes"][0]
    assert row["endpoint_input_tokens"] is None
    assert row["input_count_compatibility"] == "unavailable"


def test_sampled_ruler_reasoning_only_response_is_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", controlled_tools(reasoning_only=True))
    result = (
        TestClient(create_app(tmp_path))
        .post("/api/runs", json=ruler_submission(source(tmp_path)))
        .json()
    )

    assert result["status"] == "completed"
    assert result["coverage"]["long_context"]["scored"] == 117
    assert all(row["outcome"] == "success" for row in result["outcomes"])


def test_sampled_ruler_malformed_native_output_remains_partial_and_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", controlled_tools(malformed=True))
    result = (
        TestClient(create_app(tmp_path))
        .post("/api/runs", json=ruler_submission(source(tmp_path)))
        .json()
    )
    assert result["status"] == "partial"
    assert result["outcomes"][0]["outcome"] == "failed"
    assert result["outcomes"][0]["score"] is None
    assert result["coverage"]["long_context"]["scored"] == 116


def test_sampled_ruler_declared_context_overflow_is_retained_as_unsupported(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", controlled_tools())
    payload = ruler_submission(source(tmp_path))
    payload["setup"]["declared"]["requests"]["context_window"] = 196608
    result = TestClient(create_app(tmp_path)).post("/api/runs", json=payload).json()
    assert result["status"] == "completed"
    assert result["ruler"]["outcome_counts"]["unsupported"] == 39
    assert result["coverage"]["long_context"]["scored"] == 78


@pytest.mark.parametrize("invalid_preparation", ["overshoot", "duplicate"])
def test_invalid_preparation_is_partial_and_never_invokes_scoring(
    tmp_path, monkeypatch, invalid_preparation
):
    monkeypatch.setattr(
        subprocess, "run", controlled_tools(invalid_preparation=invalid_preparation)
    )
    result = (
        TestClient(create_app(tmp_path))
        .post("/api/runs", json=ruler_submission(source(tmp_path)))
        .json()
    )
    assert result["status"] == "partial"
    assert result["coverage"]["long_context"]["scored"] == 0
    assert result["ruler"]["outcome_counts"]["missing"] == 117


def test_cancellation_retains_accounted_partial_result(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", controlled_tools(cancel=True))
    result = (
        TestClient(create_app(tmp_path))
        .post("/api/runs", json=ruler_submission(source(tmp_path)))
        .json()
    )
    assert result["status"] == "partial"
    assert result["outcomes"][0]["outcome"] == "cancelled"
    assert result["ruler"]["outcome_counts"]["missing"] == 116


def test_native_transport_outage_stops_without_retry_and_retains_missing(tmp_path, monkeypatch):
    baseline = controlled_tools()
    calls = []

    def external(command, **kwargs):
        result = baseline(command, **kwargs)
        if "profile" in command:
            calls.append(command)
            native = Path(command[command.index("--output-artifact-dir") + 1])
            (native / "profile_export_raw.jsonl").write_text(
                json.dumps(
                    {"status": None, "responses": [], "metadata": {"x_request_id": "outage"}}
                )
                + "\n"
            )
        return result

    monkeypatch.setattr(subprocess, "run", external)
    result = (
        TestClient(create_app(tmp_path))
        .post("/api/runs", json=ruler_submission(source(tmp_path)))
        .json()
    )
    assert len(calls) == 1
    assert result["ruler"]["outcome_counts"] == {
        "success": 0,
        "failed": 1,
        "missing": 116,
        "unsupported": 0,
        "cancelled": 0,
    }


def test_preparation_cancellation_retains_public_partial_result(tmp_path, monkeypatch):
    def external(command, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(subprocess, "run", external)
    result = (
        TestClient(create_app(tmp_path))
        .post("/api/runs", json=ruler_submission(source(tmp_path)))
        .json()
    )
    assert result["status"] == "partial"
    assert result["coverage"]["long_context"] == {"expected": 117, "accounted": 117, "scored": 0}


def test_retained_live_pilot_exports_score_through_public_run(tmp_path, monkeypatch):
    import gzip

    pilot = Path("tests/fixtures/pilots/ruler-123/run")
    selected = {
        ("niah_single_2", 32768, 2),
        ("niah_single_2", 131072, 0),
        ("niah_single_3", 196608, 2),
    }
    baseline = controlled_tools()

    def external(command, **kwargs):
        result = baseline(command, **kwargs)
        if any(str(part).endswith("ruler_pilot.py") for part in command):
            directory = Path(command[command.index("--directory") + 1])
            preparations = json.loads((directory / "preparation.json").read_text())
            for task, checkpoint, index in selected:
                relative = f"{task}-{checkpoint}/{task}/validation.jsonl"
                original = [
                    json.loads(line)
                    for line in gzip.decompress((pilot / (relative + ".gz")).read_bytes())
                    .decode()
                    .splitlines()
                ]
                destination = directory / relative
                rows = [json.loads(line) for line in destination.read_text().splitlines()]
                rows[index] = original[index]
                destination.write_text("".join(json.dumps(row) + "\n" for row in rows))
                next(
                    row
                    for row in preparations
                    if row["task"] == task and row["checkpoint"] == checkpoint
                )["samples"][index]["input_tokens"] = checkpoint
            (directory / "preparation.json").write_text(json.dumps(preparations))
        elif "profile" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            name = native.parent.name.removeprefix("ruler-")
            task, checkpoint, index = name.rsplit("-", 2)
            if (task, int(checkpoint), int(index)) in selected:
                raw = gzip.decompress(
                    (pilot / "live" / name / "native/profile_export_raw.jsonl.gz").read_bytes()
                )
                (native / "profile_export_raw.jsonl").write_bytes(raw)
        return result

    monkeypatch.setattr(subprocess, "run", external)
    client = TestClient(create_app(tmp_path))
    result = client.post("/api/runs", json=ruler_submission(source(tmp_path))).json()
    observed = [
        row
        for row in result["outcomes"]
        if (row["task"], row["checkpoint"], row["index"]) in selected
    ]
    assert len(observed) == 3
    assert [row["score"] for row in observed] == [100.0, 100.0, 100.0]
    assert [row["native"]["usage"]["completion_tokens"] for row in observed] == [178, 112, 155]
    assert client.get("/api/runs/" + result["run_id"]).json() == result


@pytest.mark.parametrize(
    "defect", ["failed_condition", "tokenizer", "template", "seed", "missing_condition"]
)
def test_preparation_failures_and_binding_are_visible(tmp_path, monkeypatch, defect):
    baseline = controlled_tools()

    def external(command, **kwargs):
        result = baseline(command, **kwargs)
        if any(str(part).endswith("ruler_pilot.py") for part in command):
            directory = Path(command[command.index("--directory") + 1])
            path = directory / "preparation.json"
            conditions = json.loads(path.read_text())
            if defect == "failed_condition":
                conditions[0]["exit_code"] = 1
            elif defect == "missing_condition":
                conditions.pop()
            else:
                metadata_path = directory / "workload.json"
                metadata = json.loads(metadata_path.read_text())
                if defect == "tokenizer":
                    metadata["tokenizer_sha256"] = {}
                elif defect == "template":
                    metadata["chat_template_kwargs"] = {"enable_thinking": "different"}
                else:
                    metadata["seed"] = -1
                metadata_path.write_text(json.dumps(metadata))
            path.write_text(json.dumps(conditions))
        return result

    monkeypatch.setattr(subprocess, "run", external)
    result = (
        TestClient(create_app(tmp_path))
        .post("/api/runs", json=ruler_submission(source(tmp_path)))
        .json()
    )
    assert result["status"] == "partial"
    assert result["coverage"]["long_context"]["scored"] == (
        114 if defect == "failed_condition" else 0
    )
    assert result["coverage"]["long_context"]["accounted"] == 117
    if defect == "failed_condition":
        assert result["outcomes"][0]["stage"] == "preparation"
        assert result["outcomes"][0]["outcome"] == "failed"
        assert result["outcomes"][0]["attempted"] is False


@pytest.mark.parametrize(
    "defect", ["choices", "choice", "message", "usage", "usage_mismatch", "prompt_tokens_type"]
)
def test_malformed_native_structures_remain_failed_and_continue(tmp_path, monkeypatch, defect):
    baseline = controlled_tools()

    def external(command, **kwargs):
        result = baseline(command, **kwargs)
        if "profile" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            if native.parent.name == "ruler-niah_single_1-32768-0":
                path = native / "profile_export_raw.jsonl"
                row = json.loads(path.read_text())
                packet = row["responses"][0]["packets"][0]
                chunk = json.loads(packet["value"])
                if defect == "choices":
                    chunk["choices"] = None
                elif defect == "choice":
                    chunk["choices"] = [None]
                elif defect == "message":
                    chunk["choices"][0]["delta"] = ["bad"]
                elif defect == "usage":
                    chunk["usage"] = []
                elif defect == "prompt_tokens_type":
                    chunk["usage"]["prompt_tokens"] = "bad"
                else:
                    chunk["usage"]["prompt_tokens"] = 32769
                packet["value"] = json.dumps(chunk)
                path.write_text(json.dumps(row) + "\n")
        return result

    monkeypatch.setattr(subprocess, "run", external)
    result = (
        TestClient(create_app(tmp_path))
        .post("/api/runs", json=ruler_submission(source(tmp_path)))
        .json()
    )
    assert result["outcomes"][0]["outcome"] == "failed"
    assert result["coverage"]["long_context"]["scored"] == 116
    if defect in ("usage_mismatch", "prompt_tokens_type"):
        row = result["outcomes"][0]
        assert row["endpoint_input_tokens"] == (32769 if defect == "usage_mismatch" else "bad")
        assert row["input_count_compatibility"] == "incompatible"
        assert row["diagnostic"] == (
            "Endpoint input count exceeds nominal input ceiling"
            if defect == "usage_mismatch"
            else "Malformed endpoint prompt token count"
        )


def test_sampled_ruler_rejects_endpoint_undercount(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", controlled_tools(endpoint_prompt_delta=-2))
    result = (
        TestClient(create_app(tmp_path))
        .post("/api/runs", json=ruler_submission(source(tmp_path)))
        .json()
    )

    assert result["status"] == "partial"
    assert result["coverage"]["long_context"] == {
        "expected": 117,
        "accounted": 117,
        "scored": 0,
    }
    row = result["outcomes"][0]
    assert row["outcome"] == "failed"
    assert row["score"] is None
    assert row["local_input_tokens"] == 32767
    assert row["endpoint_input_tokens"] == 32765
    assert row["input_count_compatibility"] == "incompatible"
    assert row["diagnostic"] == "Endpoint input count is below retained local sizing"
    assert result["ruler"]["input_accounting"]["compatibility_counts"] == {
        "exact": 0,
        "endpoint-authoritative": 0,
        "incompatible": 117,
        "unavailable": 0,
    }
