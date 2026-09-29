"""Public pilot report boundary, using native AIPerf 0.12 output."""

import gzip
import json
import subprocess
import sys
from pathlib import Path


def test_report_keeps_native_counts_and_reasoning_attainment() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/performance_pilot.py",
            "report",
            "tests/fixtures/aiperf_012",
            "--output-target",
            "256",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["outcome"] == "success"
    assert report["usage"]["prompt_tokens"] == 180
    assert report["usage"]["completion_tokens"] == 256
    assert report["output_target_attained"] is True
    assert report["assistant"]["content"] == ""
    assert report["assistant"]["reasoning_content"]
    assert report["finish_reason"] == "length"
    assert report["metrics"]["time_to_first_token"] == {"value": 448.982625, "unit": "ms"}
    assert "time_to_first_output_token" not in report["metrics"]


def test_pilot_retains_unsupported_conditions_without_invoking_tool(tmp_path: Path) -> None:
    setup = tmp_path / "setup.json"
    setup.write_text(json.dumps({"context_window": 128}))
    run = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/performance_pilot.py",
            "run",
            str(setup),
            "--tokenizer",
            str(tmp_path / "absent"),
            "--directory",
            str(run),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((run / "summary.json").read_text())
    assert summary["evidence"] == "pilot"
    assert len(summary["observations"]) == 8
    assert all(row["outcome"] == "unsupported" for row in summary["observations"])
    assert all("metrics" not in row for row in summary["observations"])


def test_report_retains_interrupted_stream_without_success_timings(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/aiperf_012")
    raw = json.loads(gzip.decompress((fixture / "profile_export_raw.jsonl.gz").read_bytes()))
    raw["responses"] = raw["responses"][:-1]  # Lost connection before terminal DONE.
    (tmp_path / "profile_export_raw.jsonl").write_text(json.dumps(raw) + "\n")
    (tmp_path / "profile_export.jsonl").write_bytes((fixture / "profile_export.jsonl").read_bytes())
    result = subprocess.run(
        [
            sys.executable,
            "scripts/performance_pilot.py",
            "report",
            str(tmp_path),
            "--output-target",
            "256",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["outcome"] == "failed"
    assert report["metrics"] == {}
    assert report["output_target_attained"] is None
    assert report["assistant"]["reasoning_content"]
