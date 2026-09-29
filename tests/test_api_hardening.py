"""Request guards, fixed setup directory, and error mapping for the local API."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from llm_bench import paths, setup_state, speed_preparation
from llm_bench.dashboard import dashboard_snapshot
from llm_bench.ordinary_api import create_app
from llm_bench.ordinary_cli import app as cli


def guarded(tmp_path: Path, host: str = "localhost:8000") -> TestClient:
    return TestClient(create_app(tmp_path, allow_remote=False), headers={"Host": host})


@pytest.mark.parametrize("host", ["localhost:8000", "127.0.0.1", "[::1]:9"])
def test_loopback_hosts_are_served(tmp_path: Path, host: str) -> None:
    assert guarded(tmp_path, host).get("/api/reports").status_code == 200


def test_foreign_host_rejected_unless_remote_allowed(tmp_path: Path) -> None:
    assert guarded(tmp_path, "evil.example:8000").get("/api/reports").status_code == 400
    remote = TestClient(create_app(tmp_path, allow_remote=True), headers={"Host": "box.lan:1"})
    assert remote.get("/api/reports").status_code == 200


def test_cross_origin_post_rejected(tmp_path: Path) -> None:
    client = guarded(tmp_path)
    blocked = client.post("/api/reports/rebuild", headers={"Origin": "http://evil.example"})
    assert blocked.status_code == 403
    allowed = client.post("/api/reports/rebuild", headers={"Origin": "http://localhost:8000"})
    assert allowed.status_code == 200
    assert client.get("/api/reports", headers={"Origin": "http://evil.example"}).status_code == 200


def test_setup_ignores_client_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    used: list[Path] = []

    def fake(directory: Path, *_: object, **__: object) -> dict[str, str]:
        used.append(directory)
        return {}

    monkeypatch.setattr(setup_state, "run_setup", fake)
    client = guarded(tmp_path)
    client.post("/api/setup", json={"directory": str(tmp_path / "elsewhere"), "answers": {}})
    assert used == [tmp_path / "setup"]
    assert client.get("/api/setup", params={"directory": "/tmp/x"}).status_code == 404


def test_corrupt_setup_and_run_are_client_errors(tmp_path: Path) -> None:
    client = guarded(tmp_path)
    (tmp_path / "setup").mkdir()
    (tmp_path / "setup" / "setup-state.json").write_text("{not json")
    assert client.get("/api/setup").status_code in {404, 422}
    run_id = "a" * 32
    (tmp_path / run_id).mkdir()
    (tmp_path / run_id / "summary.json").write_text("{not json")
    for url in (f"/api/runs/{run_id}", f"/runs/{run_id}"):
        assert client.get(url).status_code in {404, 422}


def test_dashboard_errors_hide_local_paths(tmp_path: Path) -> None:
    run_id = "b" * 32
    (tmp_path / run_id).mkdir()
    (tmp_path / run_id / "summary.json").write_text("{not json")
    errors = dashboard_snapshot(tmp_path)["errors"]
    assert errors and str(tmp_path) not in json.dumps(errors)


def test_api_rejects_problem_paths_outside_setup(tmp_path: Path) -> None:
    payload = {
        "setup": {"declared": {}, "observed": {}},
        "workload": {"profile": "humaneval-plus", "problems_path": "/etc/passwd"},
    }
    response = guarded(tmp_path).post("/api/runs", json=payload)
    assert response.status_code == 422


def test_serve_requires_allow_remote_for_public_host() -> None:
    result = CliRunner().invoke(cli, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code != 0
    assert "--allow-remote" in result.output


def test_cli_hides_locals() -> None:
    assert cli.pretty_exceptions_show_locals is False


def test_check_reports_unreadable_state_cleanly(tmp_path: Path) -> None:
    (tmp_path / "setup-state.json").write_text("{not json")
    result = CliRunner().invoke(cli, ["check", "--directory", str(tmp_path)])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_source_checkout_root_requires_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (paths.source_checkout_root() / "scripts").is_dir()
    monkeypatch.setattr(paths, "__file__", str(tmp_path / "a" / "b" / "paths.py"))
    with pytest.raises(RuntimeError, match="source checkout"):
        paths.source_checkout_root()


def test_ruler_commands_do_not_depend_on_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_bench import ruler_preparation

    monkeypatch.chdir(tmp_path)
    helper = ruler_preparation._helper_path()
    assert helper.is_absolute() and helper.exists()


def test_speed_versions_validated_and_source_is_attached() -> None:
    record = {
        "declared": {"benchmark": {"version": "0.1 --index-url x", "transformers_version": "4"}}
    }
    with pytest.raises(ValueError, match="invalid benchmark version"):
        speed_preparation._benchmark_pins(record)
    record["declared"]["benchmark"]["version"] = "0.12.1"
    assert speed_preparation._benchmark_pins(record) == ("0.12.1", "4")


def test_tokenizer_source_uses_attached_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(speed_preparation.subprocess, "run", fake_run)
    monkeypatch.setattr(speed_preparation, "read_verified_tokenizer", lambda *_: None)
    with pytest.raises(ValueError):
        speed_preparation.prepare_tokenizer(
            {"declared": {"model": {"tokenizer": "--help"}}}, tmp_path
        )
    assert "--source=--help" in seen[0]
