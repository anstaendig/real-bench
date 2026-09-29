"""Comparison through public submission, retained directories and independent reads."""

import gzip
import json
import subprocess

import pytest
from fastapi.testclient import TestClient
from test_ordinary_runs import submission
from typer.testing import CliRunner

from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_cli import app

pytestmark = pytest.mark.usefixtures("external_tool_process_adapter")


def test_compare_submitted_runs_and_rebuild_without_index(tmp_path, monkeypatch):
    from pathlib import Path

    def tool(command, **kwargs):
        native = Path(command[command.index("--output-artifact-dir") + 1])
        native.mkdir()
        source = Path("tests/fixtures/pilots/performance-121/native")
        for name in ("profile_export.jsonl", "profile_export_raw.jsonl"):
            (native / name).write_bytes(gzip.decompress((source / (name + ".gz")).read_bytes()))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", tool)
    client = TestClient(create_app(tmp_path))
    first = client.post("/api/runs", json=submission()).json()
    other = submission()
    declared = other["setup"]["declared"]
    declared["hardware"]["count"] = 10
    declared["model"]["source"] = "other/model"
    declared["model"]["quantized_artifact"] = "q8"
    declared["serving"]["engine"] = "other-engine"
    declared["requests"]["reasoning"] = {"enable_thinking": False}
    declared["requests"]["context_window"] = 4096
    second = client.post("/api/runs", json=other).json()
    response = client.get("/api/reports")
    assert response.status_code == 200
    report = response.json()
    assert len(report["runs"]) == 2
    by_id = {run["run_id"]: run for run in report["runs"]}
    assert by_id[first["run_id"]]["outcomes"][0]["metrics"]["ttft"]["value"] == 954.6156249999999
    assert by_id[second["run_id"]]["outcomes"][0]["outcome"] == "unsupported"
    assert by_id[second["run_id"]]["families"]["coding"] == "missing"
    field = "setup.declared.hardware.count"
    ordered = client.get("/api/reports", params={"sort": field, "descending": True}).json()
    assert [row["run_id"] for row in ordered["runs"]] == [second["run_id"], first["run_id"]]
    filtered = client.get(
        "/api/reports", params={"filter": 'setup.declared.model.source="other/model"'}
    ).json()
    assert [row["run_id"] for row in filtered["runs"]] == [second["run_id"]]
    assert client.get("/api/reports", params={"filter": "invalid"}).status_code == 422
    assert "secret-124" not in response.text
    assert "hf_sensitive_124" not in response.text
    index = tmp_path / "report-index.json"
    assert index.exists()
    index.unlink()
    rebuilt = CliRunner().invoke(app, ["rebuild-index", "--results", str(tmp_path)])
    assert rebuilt.exit_code == 0, rebuilt.output
    assert json.loads(rebuilt.output) == report
    assert TestClient(create_app(tmp_path)).get("/api/reports").json() == report


def test_compare_quality_profiles_keeps_independent_coverage(tmp_path, monkeypatch):
    from test_humaneval_runs import coding_submission
    from test_humaneval_runs import controlled_tools as coding_tools
    from test_ruler_runs import controlled_tools as ruler_tools
    from test_ruler_runs import ruler_submission, source

    client = TestClient(create_app(tmp_path / "results"))
    monkeypatch.setattr(subprocess, "run", coding_tools([]))
    coding = client.post("/api/runs", json=coding_submission(tmp_path / "results")).json()
    monkeypatch.setattr(subprocess, "run", ruler_tools())
    ruler = client.post("/api/runs", json=ruler_submission(source(tmp_path / "results"))).json()
    report = client.get("/api/reports").json()
    rows = {row["run_id"]: row for row in report["runs"]}
    assert rows[coding["run_id"]]["coding"]["accuracy"] == 0.5
    assert rows[coding["run_id"]]["coverage"]["coding"]["graded"] == 164
    assert rows[coding["run_id"]]["coding"]["identity"]["status"] == "verified"
    assert rows[coding["run_id"]]["coding"]["identity"]["counts"]["verified"] == 164
    assert rows[ruler["run_id"]]["ruler"]["quality_by_context"]["196608"]["mean_score"] == 100
    assert (
        rows[ruler["run_id"]]["ruler"]["actual_input_ranges"]
        == ruler["ruler"]["actual_input_ranges"]
    )
    assert "coding" not in rows[ruler["run_id"]]
    assert "ruler" not in rows[coding["run_id"]]
    assert client.post("/api/reports/rebuild").json() == report
    assert "sampled-ruler" in report["parameters"]["workload.profile"]
    assert "humaneval-plus" in report["parameters"]["workload.profile"]


def test_rebuild_refreshes_partial_results_and_exposes_unreadable_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "")
    )
    client = TestClient(create_app(tmp_path))
    first = client.post("/api/runs", json=submission()).json()
    report = client.get("/api/reports").json()
    assert report["runs"][0]["status"] == "partial"
    assert report["runs"][0]["outcomes"][0]["metrics"] == {}
    (tmp_path / "report-index.json").write_text("corrupt disposable cache")
    assert client.get("/api/reports").json() == report
    (tmp_path / first["run_id"] / "summary.json").write_text("broken retained summary")
    refreshed = client.get("/api/reports").json()
    assert refreshed["runs"] == []
    assert refreshed["errors"] == [
        {"run_id": first["run_id"], "error": "Unreadable ordinary run directory"}
    ]
    assert client.get("/reports").status_code == 200


def test_exact_filters_distinguish_boolean_and_numeric_settings(tmp_path):
    client = TestClient(create_app(tmp_path))
    ids = []
    for value in (True, 1):
        payload = submission()
        payload["setup"]["declared"]["requests"]["context_window"] = 1
        payload["setup"]["declared"]["serving"]["environment"]["flag"] = value
        response = client.post("/api/runs", json=payload)
        assert response.status_code == 201
        ids.append(response.json()["run_id"])
    field = "setup.declared.serving.environment.flag"
    report = client.get("/api/reports").json()
    assert sorted(json.dumps(v) for v in report["parameters"][field]) == ["1", "true"]
    for value, run_id in zip(("true", "1"), ids, strict=True):
        filtered = client.get("/api/reports", params={"filter": f"{field}={value}"}).json()
        assert [row["run_id"] for row in filtered["runs"]] == [run_id]
