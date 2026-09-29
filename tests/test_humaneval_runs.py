"""Public submission → existing tools → ordinary coding results."""

import gzip
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
from llm_bench.ordinary_runs import CODING_OUTPUT_BUDGET

pytestmark = pytest.mark.usefixtures("external_tool_process_adapter")

SOURCE = Path("tests/fixtures/pilots/humaneval-122")
PROBLEMS = Path("src/llm_bench/datasets/HumanEvalPlus-v0.1.10.jsonl.gz")


def stage_problems(results: Path) -> Path:
    """Place the pinned dataset where API submissions may read it: results/setup."""
    staged = Path(results) / "setup" / PROBLEMS.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROBLEMS, staged)
    return staged


def coding_submission(results: Path):
    payload = submission()
    payload["setup"]["declared"]["requests"]["context_window"] = 131072
    payload["workload"] = {
        "profile": "humaneval-plus",
        "problems_path": str(stage_problems(results)),
    }
    return payload


def controlled_tools(calls):
    def external(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "version"]:
            return subprocess.CompletedProcess(command, 0, "controlled Docker version", "")
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, "[{}]", "")
        if "--output-artifact-dir" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            native.mkdir()
            raw = json.loads(
                gzip.decompress(
                    (SOURCE / "remote/HumanEval_0/native/profile_export_raw.jsonl.gz").read_bytes()
                )
            )
            (native / "profile_export_raw.jsonl").write_text(json.dumps(raw) + "\n")
        elif "docker" == command[0] and "-v" not in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        elif "docker" == command[0]:
            directory = Path(command[command.index("-v") + 1].removesuffix(":/app"))
            if "evalplus.sanitize" in command:
                (directory / "solutions-sanitized.jsonl").write_text(
                    (directory / "solutions.jsonl").read_text()
                )
            else:
                solutions = [
                    json.loads(line)
                    for line in (directory / "solutions-sanitized.jsonl").read_text().splitlines()
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
                (directory / "solutions-sanitized_eval_results.json").write_text(json.dumps(native))
        return subprocess.CompletedProcess(command, 0, "", "")

    return external


def test_complete_coding_submission_retains_quality_and_exposes_cli_api_view(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", controlled_tools(calls))
    client = TestClient(create_app(tmp_path))
    payload = coding_submission(tmp_path)
    assert "max_tokens" not in payload["workload"]
    response = client.post("/api/runs", json=payload)
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["status"] == "completed"
    assert result["schema_version"] == 1
    assert result["evidence"] == "run"
    assert result["default_workload_complete"] is False
    assert result["families"] == {
        "coding": "completed",
        "performance": "missing",
        "long_context": "missing",
    }
    assert result["coverage"]["coding"] == {
        "expected": 164,
        "attempted": 164,
        "generated": 164,
        "graded": 164,
    }
    assert result["coding"]["correct"] == 82
    assert result["coding"]["accuracy"] == 0.5
    assert result["coding"]["identity"]["status"] == "verified"
    assert result["coding"]["identity"]["counts"] == {
        "original": 164,
        "generated": 164,
        "sanitized": 164,
        "native": 164,
        "verified": 164,
    }
    assert "diagnostic" not in result
    assert result["coding"]["uncertainty"]["method"] == "Wilson 95% interval over graded tasks"
    assert 0.42 < result["coding"]["uncertainty"]["lower"] < 0.43
    assert 0.57 < result["coding"]["uncertainty"]["upper"] < 0.58
    assert len(result["outcomes"]) == 164
    assert result["outcomes"][82]["outcome"] == "incorrect"
    assert result["outcomes"][0]["grading_available"] is True
    assert result["outcomes"][0]["generation"]["usage"]["completion_tokens"] == 751
    directory = tmp_path / result["run_id"]
    inputs = list(directory.glob("HumanEval_*/input.jsonl"))
    assert all(
        json.loads(path.read_text())["output_length"] == CODING_OUTPUT_BUDGET for path in inputs
    )
    retained_workload = json.loads((directory / "workload.json").read_text())
    assert retained_workload["max_tokens"] == CODING_OUTPUT_BUDGET
    assert json.loads(inputs[0].read_text())["extra"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    }
    assert len([c for c in calls if "profile" in c]) == 164
    assert all(c[c.index("--request-count") + 1] == "1" for c in calls if "profile" in c)
    assert "HumanEval+ coding results" in client.get("/runs/" + result["run_id"]).text
    assert client.get("/api/runs/" + result["run_id"]).json() == result
    cli = CliRunner().invoke(app, ["result", result["run_id"], "--results", str(tmp_path)])
    assert cli.exit_code == 0
    assert json.loads(cli.stdout) == result


def test_outage_preserves_earlier_grades_and_distinct_failure_coverage(tmp_path, monkeypatch):
    calls = []
    normal = controlled_tools(calls)

    def external(command, **kwargs):
        if "profile" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            if native.parent.name == "HumanEval_1":
                native.mkdir()
                (native / "profile_export_raw.jsonl").write_text(
                    json.dumps(
                        {"metadata": {"x_request_id": "failed"}, "status": 500, "responses": []}
                    )
                    + "\n"
                )
                return subprocess.CompletedProcess(command, 1, "", "HTTP 500")
            if native.parent.name == "HumanEval_2":
                native.mkdir()
                (native / "profile_export_raw.jsonl").write_text(
                    json.dumps(
                        {"metadata": {"x_request_id": "outage"}, "status": 0, "responses": []}
                    )
                    + "\n"
                )
                return subprocess.CompletedProcess(command, 1, "", "Connection refused")
        return normal(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", external)
    response = TestClient(create_app(tmp_path)).post("/api/runs", json=coding_submission(tmp_path))
    assert response.status_code == 201
    result = response.json()
    assert result["status"] == "partial"
    assert result["coverage"]["coding"] == {
        "expected": 164,
        "attempted": 3,
        "generated": 1,
        "graded": 1,
    }
    assert [r["outcome"] for r in result["outcomes"][:4]] == [
        "correct",
        "generation_failed",
        "generation_failed",
        "not_attempted",
    ]
    assert result["coding"]["accuracy"] == 1.0
    assert result["coding"]["full_suite_accuracy"] is None


def test_pinned_pilot_native_outputs_normalize_through_public_submission(tmp_path, monkeypatch):
    import shutil

    def external(command, **kwargs):
        if "profile" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            source = SOURCE / "remote" / native.parent.name / "native"
            if source.exists():
                shutil.copytree(source, native)
                return subprocess.CompletedProcess(command, 0, "", "")
            native.mkdir()
            native.joinpath("profile_export_raw.jsonl").write_text(
                json.dumps({"metadata": {"x_request_id": "outage"}, "status": 0, "responses": []})
                + "\n"
            )
            return subprocess.CompletedProcess(
                command, 1, "", "controlled outage after pilot tasks"
            )
        if "evalplus.sanitize" in command or "evalplus.evaluate" in command:
            directory = Path(command[command.index("-v") + 1].removesuffix(":/app"))
            name = (
                "solutions-sanitized.jsonl"
                if "evalplus.sanitize" in command
                else "solutions-sanitized_eval_results.json"
            )
            shutil.copyfile(SOURCE / "remote" / name, directory / name)
        return subprocess.CompletedProcess(command, 0, "controlled runtime", "")

    monkeypatch.setattr(subprocess, "run", external)
    payload = coding_submission(tmp_path)
    result = TestClient(create_app(tmp_path)).post("/api/runs", json=payload).json()
    assert [r["outcome"] for r in result["outcomes"][:5]] == [
        "correct",
        "incorrect",
        "correct",
        "generation_failed",
        "not_attempted",
    ]
    assert result["outcomes"][1]["generation"]["assistant"]["content"] == ""
    assert result["outcomes"][1]["generation"]["finish_reason"] == "length"
    assert result["coding"]["correct"] == 2
    assert result["coding"]["full_suite_accuracy"] is None
    assert result["coverage"]["coding"]["graded"] == 3


def test_malformed_native_grading_is_retained_as_infrastructure_failure(tmp_path, monkeypatch):
    normal = controlled_tools([])

    def external(command, **kwargs):
        if "profile" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            if native.parent.name != "HumanEval_0":
                return subprocess.CompletedProcess(command, 1, "", "controlled outage")
        result = normal(command, **kwargs)
        if "evalplus.evaluate" in command:
            directory = Path(command[command.index("-v") + 1].removesuffix(":/app"))
            path = directory / "solutions-sanitized_eval_results.json"
            data = json.loads(path.read_text())
            data["eval"]["HumanEval/0"] = [None]
            path.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(subprocess, "run", external)
    response = TestClient(create_app(tmp_path)).post("/api/runs", json=coding_submission(tmp_path))
    assert response.status_code == 201
    result = response.json()
    assert result["outcomes"][0]["outcome"] == "grading_infrastructure"
    assert result["outcomes"][0]["grading_available"] is False
    assert result["coding"]["accuracy"] is None
    assert result["coding"]["uncertainty"] is None


@pytest.mark.parametrize(
    "failure",
    [
        "sanitize_exit",
        "grading_exit",
        "hash",
        "duplicate",
        "solution",
        "missing",
        "sanitized_duplicate",
        "sanitized_missing",
        "sanitized_unexpected",
    ],
)
def test_grading_failures_never_become_incorrect_answers(tmp_path, monkeypatch, failure):
    normal = controlled_tools([])

    def external(command, **kwargs):
        if "profile" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            if native.parent.name != "HumanEval_0":
                return subprocess.CompletedProcess(command, 1, "", "controlled outage")
        if (failure == "sanitize_exit" and "evalplus.sanitize" in command) or (
            failure == "grading_exit" and "evalplus.evaluate" in command
        ):
            return subprocess.CompletedProcess(command, 125, "", "controlled Docker failure")
        result = normal(command, **kwargs)
        if "evalplus.sanitize" in command:
            directory = Path(command[command.index("-v") + 1].removesuffix(":/app"))
            path = directory / "solutions-sanitized.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if failure == "sanitized_duplicate":
                rows.append(rows[0])
            elif failure == "sanitized_missing":
                rows = []
            elif failure == "sanitized_unexpected":
                rows[0]["task_id"] = "HumanEval/unexpected"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if "evalplus.evaluate" in command:
            directory = Path(command[command.index("-v") + 1].removesuffix(":/app"))
            path = directory / "solutions-sanitized_eval_results.json"
            data = json.loads(path.read_text())
            if failure == "hash":
                data["hash"] = "wrong-dataset"
            elif failure == "duplicate":
                data["eval"]["HumanEval/0"] *= 2
            elif failure == "solution":
                data["eval"]["HumanEval/0"][0]["solution"] = "different solution"
            elif failure == "missing":
                data["eval"] = {}
            path.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(subprocess, "run", external)
    result = (
        TestClient(create_app(tmp_path)).post("/api/runs", json=coding_submission(tmp_path)).json()
    )
    assert result["outcomes"][0]["outcome"] == "grading_infrastructure"
    assert result["coding"]["outcome_counts"]["incorrect"] == 0
    assert result["coding"]["accuracy"] is None
    assert result["coverage"]["coding"]["generated"] == 1
    assert result["coverage"]["coding"]["graded"] == 0
    assert (tmp_path / result["run_id"] / "solutions.jsonl").exists()


def test_invalid_native_identity_is_visible_through_cli_api_and_reports(tmp_path, monkeypatch):
    normal = controlled_tools([])

    def external(command, **kwargs):
        if "profile" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            if native.parent.name != "HumanEval_0":
                return subprocess.CompletedProcess(command, 1, "", "controlled outage")
        result = normal(command, **kwargs)
        if "evalplus.evaluate" in command:
            directory = Path(command[command.index("-v") + 1].removesuffix(":/app"))
            path = directory / "solutions-sanitized_eval_results.json"
            data = json.loads(path.read_text())
            duplicate = json.dumps(data["eval"]["HumanEval/0"])
            path.write_text(
                '{"hash": '
                + json.dumps(data["hash"])
                + ', "eval": {"HumanEval/0": '
                + duplicate
                + ', "HumanEval/0": '
                + duplicate
                + "}}"
            )
        return result

    monkeypatch.setattr(subprocess, "run", external)
    client = TestClient(create_app(tmp_path))
    result = client.post("/api/runs", json=coding_submission(tmp_path)).json()
    assert result["coverage"]["coding"] == {
        "expected": 164,
        "attempted": 164,
        "generated": 1,
        "graded": 0,
    }
    assert result["coding"]["accuracy"] is None
    assert result["coding"]["uncertainty"] is None
    assert result["coding"]["identity"]["status"] == "invalid"
    assert "duplicate identities" in result["diagnostic"]
    run_id = result["run_id"]
    cli = CliRunner().invoke(app, ["result", run_id, "--results", str(tmp_path)])
    assert json.loads(cli.stdout) == result
    assert client.get("/api/runs/" + run_id).json() == result
    report = client.get("/api/reports").json()
    assert report["runs"][0]["coding"]["identity"]["status"] == "invalid"
    assert report["runs"][0]["coverage"]["coding"]["graded"] == 0
    assert "duplicate identities" in client.get("/runs/" + run_id).text


def test_cli_cancellation_retains_solutions_and_redacts_tool_logs(tmp_path, monkeypatch):
    normal = controlled_tools([])

    def external(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, '[{"Architecture":"amd64"}]', "")
        if command and command[0] == "docker" and "-v" not in command:
            return subprocess.CompletedProcess(
                command, 0, "x86_64" if "uname" in command else "", ""
            )
        if "--output-artifact-dir" in command:
            native = Path(command[command.index("--output-artifact-dir") + 1])
            if native.parent.name == "HumanEval_1":
                raise KeyboardInterrupt()
        normal(command, **kwargs)
        return subprocess.CompletedProcess(command, 0, "secret-124 hf_sensitive_124", "")

    monkeypatch.setattr(subprocess, "run", external)
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(coding_submission(tmp_path / "results")))
    result = CliRunner().invoke(app, ["run", str(path), "--results", str(tmp_path / "results")])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["diagnostic"] == "Cancelled"
    assert data["coverage"]["coding"]["generated"] == 1
    directory = tmp_path / "results" / data["run_id"]
    assert (directory / "solutions.jsonl").exists()
    assert "secret-124" not in (directory / "HumanEval_0/execution.json").read_text()
    assert "hf_sensitive_124" not in result.stdout


def test_source_pin_and_fixed_budget_are_enforced_before_generation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input must not invoke a tool")

    monkeypatch.setattr(subprocess, "run", forbidden)
    client = TestClient(create_app(tmp_path / "results"))
    payload = coding_submission(tmp_path / "results")
    payload["workload"]["max_tokens"] = 4096
    assert client.post("/api/runs", json=payload).status_code == 422
    payload = coding_submission(tmp_path / "results")
    source = tmp_path / "results" / "setup" / "wrong.jsonl.gz"
    source.write_bytes(gzip.compress(b"{}\n"))
    payload["workload"]["problems_path"] = str(source)
    result = client.post("/api/runs", json=payload).json()
    assert result["status"] == "partial"
    assert result["coverage"]["coding"]["attempted"] == 0
    assert "source differs" in result["diagnostic"]
