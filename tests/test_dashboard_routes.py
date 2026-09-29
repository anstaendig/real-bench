"""The report application and its assets are available through the public server."""

from fastapi.testclient import TestClient

from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_runs import save


def test_report_shell_and_local_assets(tmp_path):
    client = TestClient(create_app(tmp_path))
    for route in ("/", "/reports"):
        response = client.get(route)
        assert response.status_code == 200
        assert "/assets/report.js" in response.text
        assert "/assets/report.css" in response.text
    for name, mime in (("report.js", "javascript"), ("report.css", "text/css")):
        response = client.get("/assets/" + name)
        assert response.status_code == 200
        assert mime in response.headers["content-type"]
    assert client.get("/assets/ordinary_api.py").status_code == 404
    assert client.get("/runs/" + "a" * 32).status_code == 404


def test_compact_reports_keep_metrics_without_transcript_or_index_writes(tmp_path):
    run_id = "b" * 32
    directory = tmp_path / run_id
    directory.mkdir()
    outcome = {
        "outcome": "success",
        "prompt": "large prompt " * 1000,
        "prediction": "answer",
        "metrics": {"ttft": {"value": 12.5, "unit": "ms"}},
    }
    save(
        directory / "summary.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": "completed",
            "outcomes": [outcome],
        },
    )
    save(directory / "setup.json", {"declared": {"prompt": "exact setup parameter"}})
    save(directory / "workload.json", {"profile": "validation"})
    client = TestClient(create_app(tmp_path))
    compact = client.get("/api/reports?compact=true").json()["runs"][0]
    assert compact["outcomes"][0]["metrics"] == outcome["metrics"]
    assert compact["outcomes"][0]["prompt"] == "[Transcript retained in run artifacts]"
    assert compact["setup"]["declared"]["prompt"] == "exact setup parameter"
    assert not (tmp_path / "report-index.json").exists()
    detail = client.get(f"/api/runs/{run_id}?compact=true").json()
    assert detail["outcomes"] == compact["outcomes"]
    assert client.get(f"/api/runs/{run_id}").json()["outcomes"] == [outcome]
