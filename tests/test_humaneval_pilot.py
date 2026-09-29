"""Public retained-directory report seam for the EvalPlus pilot."""

import json
import subprocess
import sys
from pathlib import Path


def test_report_distinguishes_upstream_correctness_from_infrastructure(tmp_path: Path) -> None:
    (tmp_path / "workload.json").write_text(
        json.dumps({"task_ids": ["HumanEval/0", "HumanEval/1", "HumanEval/2"]})
    )
    (tmp_path / "grading.json").write_text(json.dumps({"exit_code": 0}))
    (tmp_path / "grading-problems.jsonl").write_text("{}\n")
    import hashlib

    dataset_hash = hashlib.md5(b"{}\n").hexdigest()
    (tmp_path / "solutions-sanitized.jsonl").write_text(
        "".join(
            json.dumps({"task_id": f"HumanEval/{i}", "solution": "code"}) + "\n" for i in range(3)
        )
    )
    (tmp_path / "solutions-sanitized_eval_results.json").write_text(
        json.dumps(
            {
                "hash": dataset_hash,
                "eval": {
                    "HumanEval/0": [
                        {
                            "task_id": "HumanEval/0",
                            "solution": "code",
                            "base_status": "pass",
                            "plus_status": "pass",
                        }
                    ],
                    "HumanEval/1": [
                        {
                            "task_id": "HumanEval/1",
                            "solution": "code",
                            "base_status": "pass",
                            "plus_status": "fail",
                        }
                    ],
                },
            }
        )
    )
    result = subprocess.run(
        [sys.executable, "scripts/humaneval_pilot.py", "report", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert [r["outcome"] for r in report["tasks"]] == [
        "correct",
        "incorrect",
        "grading_infrastructure",
    ]
    assert report["graded"] == 2
    assert report["complete"] is False
    assert report["full_suite_accuracy"] is None


def test_submission_retains_grading_start_failure(tmp_path: Path) -> None:
    import os

    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text('#!/bin/sh\necho "controlled daemon unavailable" >&2\nexit 125\n')
    docker.chmod(0o755)
    directory = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/humaneval_pilot.py",
            "controls",
            "--problems",
            "src/llm_bench/datasets/HumanEvalPlus-v0.1.10.jsonl.gz",
            "--directory",
            str(directory),
            "--kind",
            "correct",
            "--tasks",
            "0",
            "1",
        ],
        env={**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["graded"] == 0
    assert summary["complete"] is False
    assert len(summary["tasks"]) == 2
    assert all(t["outcome"] == "grading_infrastructure" for t in summary["tasks"])
    assert (directory / "solutions.jsonl").exists()
    assert "controlled daemon unavailable" in (directory / "sanitize.log").read_text()


def test_remote_submission_retains_failure_and_redacts_setup(tmp_path: Path) -> None:
    import os

    binaries = tmp_path / "bin"
    binaries.mkdir()
    tool = binaries / "uv"
    tool.write_text('#!/bin/sh\necho "controlled transport failure" >&2\nexit 1\n')
    tool.chmod(0o755)
    setup = tmp_path / "setup.json"
    setup.write_text(
        json.dumps(
            {
                "base_url": "http://127.0.0.1:12345",
                "model_id": "test",
                "hardware": {"gpu": "fixture"},
                "model": {},
                "serving": {
                    "api_key": "secret-value",
                    "environment": {"HF_TOKEN": "hf-fixture-credential"},
                },
                "request_settings": {"temperature": 0, "seed": 122},
                "max_tokens": 4096,
            }
        )
    )
    directory = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/humaneval_pilot.py",
            "run",
            str(setup),
            "--problems",
            "src/llm_bench/datasets/HumanEvalPlus-v0.1.10.jsonl.gz",
            "--directory",
            str(directory),
            "--tokenizer",
            str(tmp_path),
            "--tasks",
            "0",
            "1",
        ],
        env={**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((directory / "summary.json").read_text())
    assert [t["outcome"] for t in summary["tasks"]] == ["generation_failed", "not_attempted"]
    assert summary["graded"] == 0
    assert "secret-value" not in (directory / "setup.json").read_text()
    assert "hf-fixture-credential" not in (directory / "setup.json").read_text()
    request = json.loads((directory / "HumanEval_0" / "input.jsonl").read_text())
    assert request["output_length"] == 4096
    assert request["extra"]["seed"] == 122


def test_retained_native_report_rejects_mismatched_dataset(tmp_path: Path) -> None:
    import shutil

    source = Path("tests/fixtures/pilots/humaneval-122/correct")
    for name in (
        "workload.json",
        "grading.json",
        "grading-problems.jsonl",
        "solutions-sanitized.jsonl",
        "solutions-sanitized_eval_results.json",
    ):
        shutil.copyfile(source / name, tmp_path / name)
    result_path = tmp_path / "solutions-sanitized_eval_results.json"
    native = json.loads(result_path.read_text())
    native["hash"] = "wrong-dataset"
    result_path.write_text(json.dumps(native))
    result = subprocess.run(
        [sys.executable, "scripts/humaneval_pilot.py", "report", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["graded"] == 0
    assert summary["correct"] == 0
    assert "dataset hash" in summary["diagnostic"]
