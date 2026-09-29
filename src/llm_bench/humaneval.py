"""Pinned HumanEval+ generation and EvalPlus Docker grading for ordinary runs."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from .ordinary_runs import (
    TOOL,
    EndpointUnreachable,
    HumanEvalWorkload,
    RunCancelled,
    Submission,
    aiperf_request_settings,
    api_key_args,
    assistant_from_raw,
    create_run_directory,
    execute_tool,
    finish_run,
    finished_on_exit,
    mapped_request_settings,
    read_jsonl,
    read_result,
    run_secrets,
    sanitize_files,
    save,
)

IMAGE = "ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740"
SOURCE_SHA256 = "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"
TASK_IDS = [f"HumanEval/{i}" for i in range(164)]
# Actionable Docker guidance for grading-container failures: the pinned
# image runs as linux/amd64, so a host that cannot run that platform (or
# has no daemon) must say what to change, not just that grading failed.
DOCKER_PREREQUISITE_INSTRUCTIONS = (
    "The grading container requires a running Docker daemon that can run "
    "linux/amd64 images. On macOS, use Docker Desktop or OrbStack with amd64 "
    "emulation enabled; on Linux, start the daemon (systemctl start docker) "
    "and install qemu user-static emulation if the host is not x86_64. "
    "Then re-run bench setup to prepare."
)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def generation(directory: Path) -> dict[str, Any]:
    exported = read_jsonl(directory / "native/profile_export_raw.jsonl")
    if exported is None or len(exported[0]) != 1:
        raise ValueError("Expected exactly one native generation request")
    raw = exported[0][0]
    assistant = assistant_from_raw(directory)
    usage: dict[str, Any] = {}
    finish = None
    for response in raw.get("responses", []):
        for packet in response.get("packets", []):
            if packet.get("name") != "data" or packet["value"] == "[DONE]":
                continue
            chunk = json.loads(packet["value"])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                finish = choice.get("finish_reason") or finish
    return {
        "outcome": "generated" if assistant is not None else "generation_failed",
        "http_status": raw.get("status"),
        "request_id": raw["metadata"]["x_request_id"],
        "assistant": assistant,
        "usage": usage,
        "finish_reason": finish,
    }


def grade(directory: Path, timeout: int, cancel_marker: Path) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"real-bench-{uuid.uuid4().hex}",
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
    for name, arguments in (
        ("sanitize", ["evalplus.sanitize", "--samples", "/app/solutions.jsonl"]),
        (
            "grading",
            [
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
        ),
    ):
        log_directory = directory / name
        log_directory.mkdir()
        result = execute_tool(
            command + arguments,
            log_directory,
            1,
            timeout,
            cancel_marker=cancel_marker,
        )
        if result.returncode:
            raise ValueError(f"EvalPlus {name} exited {result.returncode}")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def rows_by_task(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id in indexed:
            raise ValueError(f"Missing or duplicate task identities in {name.lower()}")
        indexed[task_id] = row
    return indexed


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Native grading output contains duplicate identities")
        value[key] = item
    return value


def normalize_grades(directory: Path, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    native = json.loads(
        (directory / "solutions-sanitized_eval_results.json").read_text(),
        object_pairs_hook=unique_json_object,
    )
    if not isinstance(native, dict) or not isinstance(native.get("eval"), dict):
        raise ValueError("Malformed native grading output")
    if (
        native["hash"]
        != hashlib.md5((directory / "grading-problems.jsonl").read_bytes()).hexdigest()
    ):
        raise ValueError("Native dataset hash disagrees with grading input")
    original = read_jsonl(directory / "problems.jsonl")
    grading = read_jsonl(directory / "grading-problems.jsonl")
    generated = read_jsonl(directory / "solutions.jsonl")
    sanitized = read_jsonl(directory / "solutions-sanitized.jsonl")
    if original is None or grading is None or generated is None or sanitized is None:
        raise ValueError("Missing sanitized solutions")
    originals = rows_by_task(original[0], "Original problems")
    grading_problems = rows_by_task(grading[0], "Grading problems")
    generated_solutions = rows_by_task(generated[0], "Generated solutions")
    sanitized_solutions = rows_by_task(sanitized[0], "Sanitized solutions")
    expected = set(generated_solutions)
    outcome_by_task = rows_by_task(outcomes, "Outcomes")
    if (
        not expected <= set(originals)
        or not expected <= set(grading_problems)
        or not expected <= set(outcome_by_task)
    ):
        raise ValueError("Generated solution identities disagree with original tasks")
    if any(originals[task_id] != grading_problems[task_id] for task_id in expected):
        raise ValueError("Grading problem identities disagree with original tasks")
    if set(sanitized_solutions) != expected:
        raise ValueError("Sanitized solution identities disagree with generated tasks")
    if set(native["eval"]) != expected:
        raise ValueError("Native grading identities disagree with sanitized solutions")

    mapping: list[dict[str, Any]] = []
    verified = 0
    for task_id in sorted(expected):
        row = outcome_by_task[task_id]
        row.update(outcome="grading_infrastructure", grading_available=False)
        results = native["eval"][task_id]
        entry = {
            "task_id": task_id,
            "original_task_sha256": content_sha256(originals[task_id]),
            "generated_solution_sha256": content_sha256(generated_solutions[task_id]["solution"]),
            "sanitized_solution_sha256": content_sha256(sanitized_solutions[task_id]["solution"]),
        }
        if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
            row["diagnostic"] = "Expected one native grading result"
            entry["status"] = "invalid"
            entry["diagnostic"] = row["diagnostic"]
            mapping.append(entry)
            continue
        result = results[0]
        entry["native_result_sha256"] = content_sha256(result)
        base, plus = result.get("base_status"), result.get("plus_status")
        if (
            result.get("task_id") != task_id
            or result.get("solution") != sanitized_solutions[task_id]["solution"]
            or base not in ("pass", "fail", "timeout")
            or plus not in ("pass", "fail", "timeout")
        ):
            row["diagnostic"] = "Native grading identity, solution or status mismatch"
            entry["status"] = "invalid"
            entry["diagnostic"] = row["diagnostic"]
            mapping.append(entry)
            continue
        row.update(
            outcome="correct" if base == plus == "pass" else "incorrect",
            grading_available=True,
            base_status=base,
            plus_status=plus,
        )
        row.pop("diagnostic", None)
        entry["status"] = "verified"
        mapping.append(entry)
        verified += 1
    identity = {
        "schema_version": 1,
        "status": "verified" if verified == len(expected) else "invalid",
        "source": {
            "human_eval_plus_sha256": SOURCE_SHA256,
            "evalplus_version": "0.4.0.dev2",
            "evalplus_revision": "f11cfb92c1d52896a87f988cbebbd74727d56c7e",
            "image": IMAGE,
            "native_dataset_md5": native["hash"],
        },
        "counts": {
            "original": len(originals),
            "generated": len(generated_solutions),
            "sanitized": len(sanitized_solutions),
            "native": len(native["eval"]),
            "verified": verified,
        },
        "mappings": mapping,
    }
    save(directory / "identity.json", identity)
    return identity


def coding_summary(
    outcomes: list[dict[str, Any]], identity: dict[str, Any] | None
) -> dict[str, Any]:
    graded = sum(row["grading_available"] for row in outcomes)
    correct = sum(row["outcome"] == "correct" for row in outcomes)
    interval = None
    if graded:
        proportion = correct / graded
        z = 1.959963984540054
        denominator = 1 + z * z / graded
        center = (proportion + z * z / (2 * graded)) / denominator
        radius = (
            z
            * math.sqrt(proportion * (1 - proportion) / graded + z * z / (4 * graded**2))
            / denominator
        )
        interval = {
            "method": "Wilson 95% interval over graded tasks",
            "lower": max(0, center - radius),
            "upper": min(1, center + radius),
        }
    return {
        "correct": correct,
        "accuracy": correct / graded if graded else None,
        "accuracy_denominator": "graded tasks only",
        "full_suite_accuracy": correct / 164 if graded == 164 else None,
        "uncertainty": interval,
        "outcome_counts": {
            name: sum(row["outcome"] == name for row in outcomes)
            for name in (
                "correct",
                "incorrect",
                "generation_failed",
                "not_attempted",
                "grading_infrastructure",
                "unsupported",
                "cancelled",
            )
        },
        "limitations": "Standalone code quality, not agent success. "
        "Incomplete coverage may be biased; "
        "the interval is descriptive and does not establish a model ranking. "
        "Scores are comparable only when identity is verified and coverage is shown.",
        "identity": identity,
    }


@finished_on_exit
def execute_coding(
    root: Path,
    submission: Submission,
    workload: HumanEvalWorkload,
    run_id: str | None = None,
    cancel_marker: Path | None = None,
) -> dict[str, Any]:
    run_id, directory = create_run_directory(root, run_id)
    marker = cancel_marker or directory / "cancel.json"
    setup = submission.setup.declared
    settings = mapped_request_settings(setup)
    settings.setdefault("seed", setup["benchmark"]["seed"])
    save(directory / "setup.json", submission.setup.model_dump())
    save(
        directory / "workload.json",
        {
            **workload.model_dump(),
            "evidence": "run",
            "task_ids": TASK_IDS,
            "source_sha256": SOURCE_SHA256,
            "source_version": "HumanEval+ v0.1.10",
            "evalplus_version": "0.4.0.dev2",
            "evalplus_revision": "f11cfb92c1d52896a87f988cbebbd74727d56c7e",
            "image": IMAGE,
            "generation_tool": "aiperf==0.12.0",
            "transformers_version": "5.17.0",
            "request_settings": settings,
            "concurrency": 1,
            "retries": 0,
            "grading": {"parallel": 1, "min_time_limit": 1, "gt_time_limit_factor": 4},
        },
    )
    outcomes: list[dict[str, Any]] = [
        {"task_id": task_id, "outcome": "not_attempted", "grading_available": False}
        for task_id in TASK_IDS
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "evidence": "run",
        "default_workload_complete": False,
        "status": "partial",
        "outcomes": outcomes,
        "families": {"coding": "partial", "performance": "missing", "long_context": "missing"},
    }
    started = time.monotonic()
    solutions = []
    grading_problems = []
    identity: dict[str, Any] | None = None
    active_row: dict[str, Any] | None = None
    active_problem: dict[str, Any] | None = None
    active_task_directory: Path | None = None
    try:
        data = gzip.decompress(Path(workload.problems_path).read_bytes())
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
            raise ValueError("HumanEval+ source differs from pinned v0.1.10 bytes")
        by_id = {
            row["task_id"]: row
            for row in [json.loads(line) for line in data.splitlines() if line.strip()]
        }
        problems = [by_id[task_id] for task_id in TASK_IDS]
        write_rows(directory / "problems.jsonl", problems)
        for problem, row in zip(problems, outcomes, strict=True):
            active_row = row
            active_problem = problem
            task_directory = directory / row["task_id"].replace("/", "_")
            active_task_directory = task_directory
            task_directory.mkdir()
            if workload.max_tokens >= setup["requests"]["context_window"]:
                row.update(
                    outcome="unsupported",
                    diagnostic=(
                        "Requested output target leaves no room in the declared context window"
                    ),
                )
                save(directory / "outcomes.json", outcomes)
                continue
            row["outcome"] = "generation_failed"
            save(directory / "outcomes.json", outcomes)
            prompt = (
                "Complete the following Python function. Return the complete implementation "
                "including imports and signature.\n```python\n" + problem["prompt"] + "\n```"
            )
            write_rows(
                task_directory / "input.jsonl",
                [
                    {
                        "text": "Message array supplied in extra.messages",
                        "output_length": workload.max_tokens,
                        "extra": aiperf_request_settings(
                            setup,
                            {**settings, "messages": [{"role": "user", "content": prompt}]},
                        ),
                    }
                ],
            )
            command = TOOL + [
                "profile",
                "--model",
                setup["requests"]["model_id"],
                "--url",
                setup["requests"]["base_url"],
                "--endpoint",
                "/v1/chat/completions",
                "--endpoint-type",
                "chat",
                "--streaming",
                "--tokenizer",
                setup["model"]["tokenizer"],
                "--use-server-token-count",
                "--request-count",
                "1",
                "--concurrency",
                "1",
                "--request-timeout-seconds",
                str(workload.request_timeout_seconds),
                "--random-seed",
                str(setup["benchmark"]["seed"]),
                "--input-file",
                str((task_directory / "input.jsonl").resolve()),
                "--custom-dataset-type",
                "single_turn",
                "--export-level",
                "raw",
                "--no-server-metrics",
                "--no-gpu-telemetry",
                "--output-artifact-dir",
                str((task_directory / "native").resolve()),
            ]
            command = api_key_args(command)
            try:
                result = execute_tool(
                    command,
                    task_directory,
                    1,
                    workload.request_timeout_seconds,
                    cancel_marker=marker,
                )
                observed = generation(task_directory)
                row["generation"] = observed
                if result.returncode or observed["outcome"] != "generated":
                    row["diagnostic"] = f"Generation failed (tool exit {result.returncode})"
                    if observed["http_status"] in (None, 0):
                        break
                    continue
            except subprocess.TimeoutExpired as exc:
                row["diagnostic"] = str(exc)
                continue
            except EndpointUnreachable as exc:
                row["diagnostic"] = str(exc)
                summary["diagnostic"] = "Endpoint became unreachable"
                break
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row["diagnostic"] = str(exc)
                continue
            row["outcome"] = "grading_infrastructure"
            solutions.append(
                {"task_id": row["task_id"], "solution": observed["assistant"]["content"]}
            )
            grading_problems.append(problem)
            write_rows(directory / "solutions.jsonl", solutions)
            write_rows(directory / "grading-problems.jsonl", grading_problems)
            save(directory / "outcomes.json", outcomes)
        if solutions:
            active_row = None
            version_directory = directory / "docker-version"
            version_directory.mkdir()
            execute_tool(
                ["docker", "version"],
                version_directory,
                1,
                30,
                cancel_marker=marker,
            )
            grade(directory, workload.grading_timeout_seconds, marker)
            identity = normalize_grades(directory, outcomes)
    except RunCancelled:
        if active_row is not None and active_row["outcome"] == "generation_failed":
            try:
                retained = (
                    generation(active_task_directory) if active_task_directory is not None else None
                )
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                retained = None
            if retained is not None and retained["outcome"] == "generated":
                active_row.update(outcome="grading_infrastructure", generation=retained)
                solutions.append(
                    {"task_id": active_row["task_id"], "solution": retained["assistant"]["content"]}
                )
                if active_problem is not None:
                    grading_problems.append(active_problem)
                write_rows(directory / "solutions.jsonl", solutions)
                write_rows(directory / "grading-problems.jsonl", grading_problems)
            else:
                active_row.update(outcome="cancelled", diagnostic="Cancelled during generation")
        summary["diagnostic"] = "Cancelled"
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.TimeoutExpired,
        KeyboardInterrupt,
    ) as exc:
        summary["diagnostic"] = str(exc) or "Cancelled"
        if solutions:
            identity = {"schema_version": 1, "status": "invalid", "diagnostic": str(exc)}
    finally:
        # A cancelled or timed-out Docker client can still have retained a complete
        # EvalPlus export. Normalize that finished evidence without launching more work.
        if solutions and (directory / "solutions-sanitized_eval_results.json").is_file():
            try:
                identity = normalize_grades(directory, outcomes)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                summary.setdefault("diagnostic", str(exc))
                identity = {"schema_version": 1, "status": "invalid", "diagnostic": str(exc)}
        graded = sum(row["grading_available"] for row in outcomes)
        summary["status"] = (
            "completed"
            if all(row["outcome"] in {"correct", "incorrect", "unsupported"} for row in outcomes)
            else "partial"
        )
        summary["families"]["coding"] = summary["status"]
        summary["coverage"] = {
            "coding": {
                "expected": 164,
                "attempted": sum(
                    row["outcome"] not in {"not_attempted", "unsupported"} for row in outcomes
                ),
                "generated": len(solutions),
                "graded": graded,
            }
        }
        summary["coding"] = coding_summary(outcomes, identity)
        summary["wall_seconds"] = time.monotonic() - started
        save(directory / "outcomes.json", outcomes)
        save(directory / "summary.json", summary)
        sanitize_files(directory, run_secrets(submission))
        finish_run(directory)
    return read_result(root, run_id)
