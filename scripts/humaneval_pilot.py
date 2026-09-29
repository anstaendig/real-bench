"""Tool validation, not a full benchmark or a production execution mode."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from performance_pilot import TOOL
from performance_pilot import report as generation_report


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report(directory: Path) -> dict[str, Any]:
    workload = json.loads((directory / "workload.json").read_text())
    tasks = []
    native: dict[str, Any] = {}
    solutions: dict[str, Any] = {}
    diagnostic = None
    try:
        execution = json.loads((directory / "grading.json").read_text())
        if execution["exit_code"] != 0:
            raise ValueError("Grading process did not complete successfully")
        native = json.loads((directory / "solutions-sanitized_eval_results.json").read_text())
        if (
            native["hash"]
            != hashlib.md5((directory / "grading-problems.jsonl").read_bytes()).hexdigest()
        ):
            raise ValueError("Native dataset hash disagrees with grading input")
        solutions = {
            r["task_id"]: r["solution"] for r in rows(directory / "solutions-sanitized.jsonl")
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        diagnostic = str(exc)
        native = {}
    generation_path = directory / "generation.json"
    generation = json.loads(generation_path.read_text()) if generation_path.exists() else None
    for task_id in workload["task_ids"]:
        outcome = "grading_infrastructure"
        results = native.get("eval", {}).get(task_id, [])
        row: dict[str, Any] = {"task_id": task_id}
        if len(results) == 1:
            result = results[0]
            base, plus = result.get("base_status"), result.get("plus_status")
            if (
                result.get("task_id") == task_id
                and task_id in solutions
                and result.get("solution") == solutions[task_id]
                and base in ("pass", "fail", "timeout")
                and plus in ("pass", "fail", "timeout")
            ):
                outcome = "correct" if base == plus == "pass" else "incorrect"
                row.update(base_status=base, plus_status=plus)
        if generation is not None and generation.get(task_id, {}).get("outcome") != "success":
            outcome = generation.get(task_id, {}).get("outcome", "not_attempted")
        if generation is not None:
            observed = generation.get(task_id, {})
            row["generation"] = {
                key: observed[key]
                for key in ("finish_reason", "usage", "output_target")
                if key in observed
            }
        row["outcome"] = outcome
        tasks.append(row)
    graded = sum(t["outcome"] in ("correct", "incorrect") for t in tasks)
    return {
        "evidence": "tool-validation",
        "tasks": tasks,
        "selected": len(tasks),
        "graded": graded,
        "correct": sum(t["outcome"] == "correct" for t in tasks),
        "complete": graded == len(tasks),
        "full_suite_accuracy": None,
        "diagnostic": diagnostic,
        "limitations": "Small selected sample; no full-suite accuracy or runtime claim.",
    }


IMAGE = "ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740"
SOURCE_SHA256 = "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def prepare(problems_path: Path, task_numbers: list[int], directory: Path) -> list[dict[str, Any]]:
    data = gzip.decompress(problems_path.read_bytes())
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("HumanEval+ source differs from pinned v0.1.10 bytes")
    problems = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len(problems) != 164 or len(set(task_numbers)) != len(task_numbers):
        raise ValueError("Expected 164 source problems and distinct selected tasks")
    by_id = {p["task_id"]: p for p in problems}
    selected = [by_id[f"HumanEval/{i}"] for i in task_numbers]
    directory.mkdir(parents=True, exist_ok=False)
    write_rows(directory / "problems.jsonl", selected)
    save(
        directory / "workload.json",
        {
            "evidence": "tool-validation",
            "task_ids": [p["task_id"] for p in selected],
            "source_sha256": SOURCE_SHA256,
            "source_version": "HumanEval+ v0.1.10",
            "evalplus_version": "0.4.0.dev2",
            "evalplus_revision": "f11cfb92c1d52896a87f988cbebbd74727d56c7e",
            "image": IMAGE,
            "solutions_per_task": 1,
            "full_suite_size": 164,
            "grading": {"parallel": 1, "min_time_limit": 1, "gt_time_limit_factor": 4},
        },
    )
    return selected


def execute(command: list[str], directory: Path, name: str) -> int:
    save(directory / f"{name}-command.json", command)
    start = time.monotonic()
    code = -1
    try:
        with (directory / f"{name}.log").open("w") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    except OSError as exc:
        (directory / f"{name}.log").write_text(str(exc))
    finally:
        save(
            directory / f"{name}.json",
            {"exit_code": code, "wall_seconds": time.monotonic() - start},
        )
    return code


def grade(directory: Path, infrastructure_control: bool = False) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        "4g",
        "--pids-limit",
        "128",
        "--cpus",
        "2",
        "-v",
        f"{directory.resolve()}:/app",
        "-e",
        "HUMANEVAL_OVERRIDE_PATH=/app/grading-problems.jsonl",
        IMAGE,
    ]
    try:
        if infrastructure_control:
            execute(command + ["/deliberately-missing-evalplus-command"], directory, "grading")
        elif (
            execute(
                command + ["evalplus.sanitize", "--samples", "/app/solutions.jsonl"],
                directory,
                "sanitize",
            )
            == 0
        ):
            execute(
                command
                + [
                    "evalplus.evaluate",
                    "--dataset",
                    "humaneval",
                    "--samples",
                    "/app/solutions-sanitized.jsonl",
                    "--parallel",
                    "1",
                    "--min-time-limit",
                    "1",
                    "--gt-time-limit-factor",
                    "4",
                ],
                directory,
                "grading",
            )
    finally:
        save(directory / "summary.json", report(directory))


def controls(args: argparse.Namespace) -> None:
    problems = prepare(args.problems, args.tasks, args.directory)
    save(args.directory / "setup.json", {"kind": "offline-grading-control", "control": args.kind})
    write_rows(args.directory / "grading-problems.jsonl", problems)
    write_rows(
        args.directory / "solutions.jsonl",
        [
            {
                "task_id": p["task_id"],
                "solution": p["prompt"]
                + (
                    p["canonical_solution"]
                    if args.kind != "incorrect"
                    else "    raise AssertionError('negative control')\n"
                ),
            }
            for p in problems
        ],
    )
    grade(args.directory, args.kind == "infrastructure")


SECRET_KEY = re.compile(
    r"api.?key|authorization|password|secret|access.?token|(?:^|[_-])token(?:$|[_-])", re.I
)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: "[REDACTED]" if SECRET_KEY.search(k) else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        result = []
        hide_next = False
        for item in value:
            result.append("[REDACTED]" if hide_next else redact(item))
            hide_next = (
                isinstance(item, str)
                and item.startswith("-")
                and bool(SECRET_KEY.search(item))
                and "=" not in item
            )
        return result
    if isinstance(value, str):
        return re.sub(
            r"(?i)(bearer\s+|(?:api[_-]?key|token|password|secret)[= :]+)\S+",
            r"\1[REDACTED]",
            value,
        )
    return value


def run(args: argparse.Namespace) -> None:
    setup = json.loads(args.setup.read_text())
    url = urlsplit(setup["base_url"])
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "Pilot requires an HTTP(S) endpoint URL without credentials or query parameters"
        )
    allowed = {"temperature", "top_p", "top_k", "seed", "chat_template_kwargs"}
    settings = setup["request_settings"]
    if set(settings) - allowed or set(settings.get("chat_template_kwargs", {})) - {
        "enable_thinking",
        "reasoning_effort",
    }:
        raise ValueError("Unsupported pilot request settings; credentials are not accepted")
    if not isinstance(setup["max_tokens"], int) or setup["max_tokens"] <= 0:
        raise ValueError("A positive quality-specific max_tokens is required")
    for category in ("hardware", "model", "serving"):
        if category not in setup:
            raise ValueError(f"Missing setup category: {category}")
    problems = prepare(args.problems, args.tasks, args.directory)
    directory = args.directory
    save(directory / "setup.json", redact(setup))
    workload = json.loads((directory / "workload.json").read_text())
    workload.update(
        generation_tool="aiperf==0.12.0",
        request_settings=settings,
        max_tokens=setup["max_tokens"],
        request_timeout_seconds=setup.get("request_timeout_seconds", 900),
        tokenizer=str(args.tokenizer),
        concurrency=1,
        retries=0,
    )
    save(directory / "workload.json", workload)
    generated: dict[str, Any] = {}
    solutions = []
    grading_problems = []
    start = time.monotonic()
    try:
        for problem in problems:
            task_id = problem["task_id"]
            condition = directory / task_id.replace("/", "_")
            condition.mkdir()
            generated[task_id] = {"outcome": "generation_failed"}
            save(directory / "generation.json", generated)
            write_rows(
                condition / "input.jsonl",
                [
                    {
                        "text": "Message array supplied in extra.messages",
                        "output_length": setup["max_tokens"],
                        "extra": {
                            **settings,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": (
                                        "Complete the following Python function. "
                                        "Return the complete implementation including imports "
                                        "and signature.\n```python\n"
                                    )
                                    + problem["prompt"]
                                    + "\n```",
                                }
                            ],
                        },
                    }
                ],
            )
            command = TOOL + [
                "profile",
                "--model",
                setup["model_id"],
                "--url",
                setup["base_url"],
                "--endpoint",
                "/v1/chat/completions",
                "--endpoint-type",
                "chat",
                "--streaming",
                "--tokenizer",
                str(args.tokenizer.resolve()),
                "--use-server-token-count",
                "--request-count",
                "1",
                "--concurrency",
                "1",
                "--request-timeout-seconds",
                str(workload["request_timeout_seconds"]),
                "--random-seed",
                str(settings.get("seed", 122)),
                "--input-file",
                str((condition / "input.jsonl").resolve()),
                "--custom-dataset-type",
                "single_turn",
                "--export-level",
                "raw",
                "--no-server-metrics",
                "--no-gpu-telemetry",
                "--output-artifact-dir",
                str((condition / "native").resolve()),
            ]
            print(f"Generating {task_id}", flush=True)
            exit_code = execute(command, condition, "generation")
            try:
                observed = generation_report(condition / "native", setup["max_tokens"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                generated[task_id]["diagnostic"] = str(exc)
                break  # No native response: availability is uncertain, preserve partial work.
            generated[task_id] = observed
            if exit_code != 0 or observed["outcome"] != "success":
                generated[task_id]["outcome"] = "generation_failed"
                if observed.get("status") is None:
                    break
                continue
            solutions.append({"task_id": task_id, "solution": observed["assistant"]["content"]})
            grading_problems.append(problem)
            write_rows(directory / "solutions.jsonl", solutions)
            write_rows(directory / "grading-problems.jsonl", grading_problems)
            save(directory / "generation.json", generated)
    finally:
        save(directory / "generation.json", generated)
        save(directory / "runtime.json", {"generation_wall_seconds": time.monotonic() - start})
        save(directory / "summary.json", report(directory))
    if solutions:
        grade(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("directory", type=Path)
    control_parser = subparsers.add_parser("controls")
    control_parser.add_argument("--problems", type=Path, required=True)
    control_parser.add_argument("--directory", type=Path, required=True)
    control_parser.add_argument("--tasks", nargs="+", type=int, default=[0, 1, 2])
    control_parser.add_argument(
        "--kind", choices=["correct", "incorrect", "infrastructure"], required=True
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("setup", type=Path)
    run_parser.add_argument("--problems", type=Path, required=True)
    run_parser.add_argument("--directory", type=Path, required=True)
    run_parser.add_argument("--tokenizer", type=Path, required=True)
    run_parser.add_argument("--tasks", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if args.command == "report":
        print(json.dumps(report(args.directory), indent=2))
    elif args.command == "controls":
        controls(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
