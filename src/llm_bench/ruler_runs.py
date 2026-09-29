"""Sampled RULER ordinary-run execution using the validated upstream route."""

from __future__ import annotations

import hashlib
import json
import math
import runpy
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from .ordinary_runs import (
    TOOL,
    RunCancelled,
    SampledRulerWorkload,
    Submission,
    aiperf_request_settings,
    api_key_args,
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
from .paths import source_checkout_root

REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
CHECKPOINTS = (32768, 131072, 196608)
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
CASE_COUNT = len(TASKS) * len(CHECKPOINTS) * 3
ENDPOINT_INPUT_SOURCE = "native/profile_export_raw.jsonl:usage.prompt_tokens"


def _local_input_source(task: str, checkpoint: int, index: int) -> str:
    return f"prepared/preparation.json:{task}-{checkpoint}.samples[{index}].input_tokens"


def _template_kwargs(submission: Submission) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for group in (
        submission.setup.declared["requests"]["generation"],
        submission.setup.declared["requests"]["reasoning"],
    ):
        candidate = group.get("chat_template_kwargs")
        if candidate is not None:
            values = candidate
    return values


def _effective_template(setup: dict[str, Any]) -> str:
    """The template that sizes the 117 prompts: a declared override, else the
    submitted tokenizer's own template."""
    overrides = setup["model"]["template_overrides"]
    template = overrides.get("chat_template")
    if isinstance(template, str) and template:
        return template
    return (Path(setup["model"]["tokenizer"]) / "chat_template.jinja").read_text()


def _effective_template_sha256(setup: dict[str, Any]) -> str:
    return hashlib.sha256(_effective_template(setup).encode()).hexdigest()


def _prepared_rows(
    directory: Path,
    workload: SampledRulerWorkload,
    setup: dict[str, Any],
    template_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    pilot_workload = json.loads((directory / "workload.json").read_text())
    preparation = json.loads((directory / "preparation.json").read_text())
    if (
        pilot_workload.get("source_revision") != REVISION
        or pilot_workload.get("planned_requests") != CASE_COUNT
        or pilot_workload.get("tasks") != list(TASKS)
        or pilot_workload.get("checkpoints") != list(CHECKPOINTS)
        or pilot_workload.get("samples_per_task_checkpoint") != 3
        or pilot_workload.get("quality_output_limit") != workload.max_tokens
        or pilot_workload.get("context_window") != setup["requests"]["context_window"]
        or pilot_workload.get("chat_template_kwargs") != template_kwargs
        or pilot_workload.get("seed") != setup["benchmark"]["seed"]
    ):
        raise ValueError("RULER preparation metadata does not bind the submitted workload")
    tokenizer = Path(setup["model"]["tokenizer"])
    actual_hashes = (
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in tokenizer.iterdir()
            if path.is_file()
        }
        if tokenizer.is_dir()
        else {}
    )
    if not actual_hashes or pilot_workload.get("tokenizer_sha256") != actual_hashes:
        raise ValueError("RULER preparation tokenizer hashes differ from submitted tokenizer")
    if pilot_workload.get("chat_template_sha256") != _effective_template_sha256(setup):
        raise ValueError("RULER preparation template differs from the declared effective template")
    expected = [(task, checkpoint) for task in TASKS for checkpoint in CHECKPOINTS]
    if [(row.get("task"), row.get("checkpoint")) for row in preparation] != expected:
        raise ValueError("RULER preparation conditions do not match the sampled profile")
    rows: list[dict[str, Any]] = []
    for condition in preparation:
        if condition.get("exit_code") != 0 or len(condition.get("samples", [])) != 3:
            rows.extend(
                {
                    "task": condition["task"],
                    "checkpoint": condition["checkpoint"],
                    "index": index,
                    "outcome": "failed",
                    "score": None,
                    "attempted": False,
                    "stage": "preparation",
                    "nominal_input_ceiling": condition["checkpoint"],
                    "local_input_tokens": None,
                    "local_input_tokens_source": None,
                    "endpoint_input_tokens": None,
                    "endpoint_input_tokens_source": None,
                    "input_count_delta_tokens": None,
                    "input_count_compatibility": None,
                    "output_reservation_tokens": workload.max_tokens,
                    "diagnostic": condition.get("diagnostic")
                    or "Upstream preparation failed or returned incomplete samples",
                    "preparation_exit_code": condition.get("exit_code"),
                    "prepared_source": "preparation.json",
                }
                for index in range(3)
            )
            continue
        source = (
            directory
            / f"{condition['task']}-{condition['checkpoint']}"
            / condition["task"]
            / "validation.jsonl"
        )
        exported = read_jsonl(source)
        if exported is None or len(exported[0]) != 3:
            raise ValueError("RULER prepared input is missing or malformed")
        for index, (metadata, sample) in enumerate(
            zip(condition["samples"], exported[0], strict=True)
        ):
            prompt = sample.get("input", "") + sample.get("answer_prefix", "")
            references = sample.get("outputs")
            if (
                metadata.get("index") != index
                or not isinstance(metadata.get("input_tokens"), int)
                or metadata["input_tokens"] <= 0
                or metadata["input_tokens"] > condition["checkpoint"]
                or not isinstance(prompt, str)
                or not isinstance(references, list)
                or not references
                or not all(isinstance(value, str) for value in references)
            ):
                raise ValueError("RULER prepared input violates sizing or scoring contract")
            rows.append(
                {
                    "task": condition["task"],
                    "checkpoint": condition["checkpoint"],
                    "index": index,
                    "input_tokens": metadata["input_tokens"],
                    "nominal_input_ceiling": condition["checkpoint"],
                    "local_input_tokens": metadata["input_tokens"],
                    "local_input_tokens_source": _local_input_source(
                        condition["task"], condition["checkpoint"], index
                    ),
                    "output_reservation_tokens": workload.max_tokens,
                    "prompt": prompt,
                    "references": references,
                    "prepared_source": str(source.relative_to(directory)),
                    "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
    return rows


def _native_generation(directory: Path) -> tuple[str, dict[str, Any]]:
    exported = read_jsonl(directory / "native/profile_export_raw.jsonl")
    if exported is None or len(exported[0]) != 1:
        raise ValueError("Expected exactly one native RULER generation request")
    raw = exported[0][0]
    if isinstance(raw, dict) and "status" in raw and raw["status"] in (None, 0):
        raise ConnectionError("Native request reports unavailable endpoint transport")
    content = ""
    reasoning = ""
    usage: dict[str, Any] = {}
    finish = None
    done = False
    if not isinstance(raw, dict) or not isinstance(raw.get("responses"), list):
        raise ValueError("Malformed native RULER response list")
    for response in raw["responses"]:
        if not isinstance(response, dict) or not isinstance(response.get("packets"), list):
            raise ValueError("Malformed native RULER response packets")
        for packet in response.get("packets", []):
            if not isinstance(packet, dict):
                raise ValueError("Malformed native RULER packet")
            if packet.get("name") != "data":
                continue
            if packet.get("value") == "[DONE]":
                done = True
                continue
            chunk = json.loads(packet["value"])
            if not isinstance(chunk, dict):
                raise ValueError("Malformed native RULER chunk")
            if "usage" in chunk:
                candidate_usage = chunk["usage"]
                if candidate_usage is not None:
                    if not isinstance(candidate_usage, dict):
                        raise ValueError("Malformed native RULER usage")
                    usage = candidate_usage
            choices = chunk.get("choices", [])
            if not isinstance(choices, list):
                raise ValueError("Malformed native RULER choices")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise ValueError("Malformed native RULER choice")
                message = choice.get("delta") or choice.get("message") or {}
                if not isinstance(message, dict):
                    raise ValueError("Malformed native RULER message")
                if isinstance(message.get("content"), str):
                    content += message["content"]
                if isinstance(message.get("reasoning_content"), str):
                    reasoning += message["reasoning_content"]
                finish = choice.get("finish_reason") or finish
    if not content:
        content = reasoning
    if raw.get("status") != 200 or not done or finish is None or not content:
        raise ValueError("No usable assistant response in native RULER export")
    return content, {
        "request_id": raw["metadata"]["x_request_id"],
        "http_status": raw.get("status"),
        "usage": usage,
        "finish_reason": finish,
    }


def ruler_summary(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    def quality_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
        scores = [row["score"] for row in rows if isinstance(row.get("score"), (int, float))]
        return {
            "attempted": sum(row.get("attempted", False) for row in rows),
            "scored": len(scores),
            "mean_score": sum(scores) / len(scores) if scores else None,
            "minimum_score": min(scores) if scores else None,
            "maximum_score": max(scores) if scores else None,
            "uncertainty": {
                "available": False,
                "reason": (
                    "Three samples per task/context with fractional task metrics; "
                    "no calibrated interval. Do not infer a ranking or significance "
                    "from differences."
                ),
            },
        }

    contexts = {
        str(checkpoint): quality_coverage(
            [row for row in outcomes if row["checkpoint"] == checkpoint]
        )
        for checkpoint in CHECKPOINTS
    }

    tasks = {
        task: quality_coverage([row for row in outcomes if row["task"] == task]) for task in TASKS
    }
    local_input_ranges = {}
    endpoint_input_ranges = {}
    for checkpoint in CHECKPOINTS:
        rows = [row for row in outcomes if row["checkpoint"] == checkpoint]
        local_counts = [
            row["local_input_tokens"] for row in rows if type(row.get("local_input_tokens")) is int
        ]
        endpoint_counts = [
            row["endpoint_input_tokens"]
            for row in rows
            if type(row.get("endpoint_input_tokens")) is int
        ]
        local_input_ranges[str(checkpoint)] = {
            "samples": len(local_counts),
            "minimum": min(local_counts) if local_counts else None,
            "maximum": max(local_counts) if local_counts else None,
        }
        endpoint_input_ranges[str(checkpoint)] = {
            "samples": len(endpoint_counts),
            "minimum": min(endpoint_counts) if endpoint_counts else None,
            "maximum": max(endpoint_counts) if endpoint_counts else None,
        }

    compatibility_counts = {
        name: sum(row.get("input_count_compatibility") == name for row in outcomes)
        for name in ("exact", "endpoint-authoritative", "incompatible", "unavailable")
    }
    local_sample_count = len(
        [row for row in outcomes if type(row.get("local_input_tokens")) is int]
    )
    endpoint_sample_count = len(
        [row for row in outcomes if type(row.get("endpoint_input_tokens")) is int]
    )
    input_accounting = {
        "policy": (
            "The submitted tokenizer and explicit chat template provide local preparation "
            "counts. A positive integer endpoint usage.prompt_tokens count is authoritative "
            "for the executed request; it may differ because the endpoint owns its tokenizer "
            "and template serialization. A differing endpoint count is compatible only when "
            "it is at least the retained local count and remains at or below the nominal input "
            "ceiling. Undercounts and ceiling overshoots are incompatible; missing usage "
            "remains unavailable."
        ),
        "local": {
            "source": "prepared/preparation.json:*.samples[].input_tokens",
            "ranges": local_input_ranges,
            "samples": local_sample_count,
        },
        "endpoint": {
            "source": ENDPOINT_INPUT_SOURCE,
            "ranges": endpoint_input_ranges,
            "samples": endpoint_sample_count,
        },
        "compatibility_counts": compatibility_counts,
        "output_reservation_tokens": sorted(
            {
                row["output_reservation_tokens"]
                for row in outcomes
                if type(row.get("output_reservation_tokens")) is int
            }
        ),
    }
    return {
        "sampled_diagnostic": True,
        "limited_precision": (
            "Three samples per task and nominal context; descriptive means only, "
            "not a full official RULER score or ranking."
        ),
        "sizing_contract": "upstream-input-ceilings with retained actual local counts",
        "quality_by_context": contexts,
        "actual_input_ranges": local_input_ranges,
        "local_input_ranges": local_input_ranges,
        "endpoint_input_ranges": endpoint_input_ranges,
        "input_accounting": input_accounting,
        "task_coverage": tasks,
        "outcome_counts": {
            name: sum(row["outcome"] == name for row in outcomes)
            for name in ("success", "failed", "missing", "unsupported", "cancelled")
        },
    }


def score_native_generation(
    row: dict[str, Any],
    condition: Path,
    sample: dict[str, Any],
    configs: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Validate and score a completed retained native response once."""
    prediction, native = _native_generation(condition)
    row["native"] = native
    endpoint_input = native["usage"].get("prompt_tokens")
    row.update(
        nominal_input_ceiling=sample["checkpoint"],
        local_input_tokens=sample["input_tokens"],
        local_input_tokens_source=_local_input_source(
            sample["task"], sample["checkpoint"], sample["index"]
        ),
        output_reservation_tokens=row.get("output_reservation_tokens"),
        endpoint_input_tokens=None,
        endpoint_input_tokens_source=None,
        input_count_delta_tokens=None,
        input_count_compatibility="unavailable",
    )
    if endpoint_input is not None:
        row.update(
            endpoint_input_tokens=endpoint_input,
            endpoint_input_tokens_source=ENDPOINT_INPUT_SOURCE,
            input_count_compatibility="incompatible",
        )
        if type(endpoint_input) is not int or endpoint_input <= 0:
            raise ValueError("Malformed endpoint prompt token count")
        row.update(
            input_count_delta_tokens=endpoint_input - sample["input_tokens"],
        )
        if endpoint_input > sample["checkpoint"]:
            row["input_count_compatibility"] = "incompatible"
            raise ValueError("Endpoint input count exceeds nominal input ceiling")
        if endpoint_input < sample["input_tokens"]:
            row["input_count_compatibility"] = "incompatible"
            raise ValueError("Endpoint input count is below retained local sizing")
        row["input_count_compatibility"] = (
            "exact" if endpoint_input == sample["input_tokens"] else "endpoint-authoritative"
        )
    metric = metrics[configs[sample["task"]]["task"]]["metric_fn"]
    score = metric([prediction], [sample["references"]])
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("Upstream RULER scorer returned a non-numeric score")
    # The finish reason is retained so a generation that hit the fixed
    # RULER_OUTPUT_BUDGET ("length") stays visible in the outcome; the budget
    # is part of the profile, so the request is still scored.
    row.update(
        outcome="success",
        score=float(score),
        prediction=prediction,
        finish_reason=native["finish_reason"],
        scorer="pinned upstream RULER metric_fn",
        scorer_source=f"{REVISION}:scripts/eval/synthetic/constants.py",
    )


@finished_on_exit
def execute_ruler(
    root: Path,
    submission: Submission,
    workload: SampledRulerWorkload,
    run_id: str | None = None,
    cancel_marker: Path | None = None,
) -> dict[str, Any]:
    run_id, directory = create_run_directory(root, run_id)
    marker = cancel_marker or directory / "cancel.json"
    setup = submission.setup.declared
    template_kwargs = _template_kwargs(submission)
    prepared = directory / "prepared"
    settings = mapped_request_settings(setup)
    settings.setdefault("seed", setup["benchmark"]["seed"])
    save(directory / "setup.json", submission.setup.model_dump())
    save(
        directory / "workload.json",
        {
            **workload.model_dump(),
            "evidence": "run",
            "source_revision": REVISION,
            "tasks": list(TASKS),
            "checkpoints": list(CHECKPOINTS),
            "samples_per_task_checkpoint": 3,
            "planned_requests": CASE_COUNT,
            "request_settings": settings,
            "chat_template_kwargs": template_kwargs,
            "concurrency": 1,
            "retries": 0,
            "sizing_contract": "upstream-input-ceilings",
        },
    )
    outcomes: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "evidence": "run",
        "default_workload_complete": False,
        "status": "partial",
        "families": {"performance": "missing", "coding": "missing", "long_context": "partial"},
        "outcomes": outcomes,
    }
    started = time.monotonic()
    try:
        # The effective template (a declared override or the tokenizer's own)
        # sizes the 117 prompts. It is written beside the prepared directory
        # and removed after preparation so the run retains no template copy.
        template_file = directory / "chat-template.effective.jinja"
        template_file.write_text(_effective_template(setup))
        checkout = source_checkout_root()
        preparation_command = [
            "uv",
            "run",
            "--project",
            str(checkout / "third_party" / "ruler"),
            "--with",
            "jinja2==3.1.6",
            "python",
            str(checkout / "scripts" / "ruler_pilot.py"),
            "--source",
            str(Path(workload.source_path).resolve()),
            "--tokenizer",
            str(Path(setup["model"]["tokenizer"]).resolve()),
            "--directory",
            str(prepared.resolve()),
            "--quality-output-limit",
            str(workload.max_tokens),
            "--context-window",
            str(setup["requests"]["context_window"]),
            "--chat-template-kwargs",
            json.dumps(template_kwargs, sort_keys=True),
            "--chat-template-path",
            str(template_file),
            "--seed",
            str(setup["benchmark"]["seed"]),
        ]
        execute_tool(
            preparation_command,
            prepared.parent,
            1,
            workload.request_timeout_seconds,
            cancel_marker=marker,
        )
        template_file.unlink()
        if not prepared.is_dir():
            raise ValueError("RULER preparation did not retain a run directory")
        rows = _prepared_rows(prepared, workload, setup, template_kwargs)
        source_path = Path(workload.source_path)
        configs = yaml.safe_load((source_path / "scripts/synthetic.yaml").read_text())
        metrics = runpy.run_path(str(source_path / "scripts/eval/synthetic/constants.py"))["TASKS"]
        for sample in rows:
            condition = (
                directory / f"ruler-{sample['task']}-{sample['checkpoint']}-{sample['index']}"
            )
            condition.mkdir()
            row = {**sample, "outcome": "failed", "score": None, "attempted": False}
            if sample.get("stage") == "preparation":
                if sample["checkpoint"] + workload.max_tokens > setup["requests"]["context_window"]:
                    row.update(
                        outcome="unsupported",
                        diagnostic=(
                            "Nominal input ceiling plus output exceeds declared context window"
                        ),
                    )
                outcomes.append(row)
                continue
            input_record = {
                "text": "RULER prompt supplied in extra.messages",
                "output_length": workload.max_tokens,
                "extra": aiperf_request_settings(
                    setup,
                    {**settings, "messages": [{"role": "user", "content": sample["prompt"]}]},
                ),
            }
            (condition / "input.jsonl").write_text(json.dumps(input_record) + "\n")
            if sample["checkpoint"] + workload.max_tokens > setup["requests"]["context_window"]:
                row.update(
                    outcome="unsupported",
                    diagnostic=(
                        "Nominal input ceiling plus output exceeds declared context window"
                    ),
                )
                outcomes.append(row)
                continue
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
            command = api_key_args(command)
            try:
                row["attempted"] = True
                result = execute_tool(
                    command,
                    condition,
                    1,
                    workload.request_timeout_seconds,
                    cancel_marker=marker,
                )
                if result.returncode:
                    raise ValueError(f"Tool exited {result.returncode}")
                score_native_generation(row, condition, sample, configs, metrics)
            except ConnectionError as exc:
                row.update(outcome="failed", diagnostic=str(exc))
                summary["diagnostic"] = "Endpoint became unreachable"
                outcomes.append(row)
                break
            except (KeyboardInterrupt, RunCancelled):
                summary["diagnostic"] = "Cancelled"
                try:
                    score_native_generation(row, condition, sample, configs, metrics)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    row.update(outcome="cancelled", diagnostic="Cancelled")
                outcomes.append(row)
                break
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                subprocess.TimeoutExpired,
                json.JSONDecodeError,
            ) as exc:
                row.update(outcome="failed", diagnostic=str(exc))
            outcomes.append(row)
            save(directory / "outcomes.json", outcomes)
    except (KeyboardInterrupt, RunCancelled):
        summary["diagnostic"] = "Cancelled during preparation"
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as exc:
        summary["diagnostic"] = str(exc)
    finally:
        observed = {(row["task"], row["checkpoint"], row["index"]) for row in outcomes}
        outcomes.extend(
            {
                "task": task,
                "checkpoint": checkpoint,
                "index": index,
                "outcome": "missing",
                "score": None,
                "diagnostic": "Not attempted after partial preparation or execution",
            }
            for task in TASKS
            for checkpoint in CHECKPOINTS
            for index in range(3)
            if (task, checkpoint, index) not in observed
        )
        summary["coverage"] = {
            "long_context": {
                "expected": CASE_COUNT,
                "accounted": len(outcomes),
                "scored": sum(isinstance(row.get("score"), (int, float)) for row in outcomes),
            }
        }
        summary["ruler"] = ruler_summary(outcomes)
        summary["status"] = (
            "completed"
            if len(outcomes) == CASE_COUNT
            and all(row["outcome"] in {"success", "unsupported"} for row in outcomes)
            else "partial"
        )
        summary["families"]["long_context"] = summary["status"]
        summary["wall_seconds"] = time.monotonic() - started
        save(directory / "outcomes.json", outcomes)
        save(directory / "summary.json", summary)
        sanitize_files(directory, run_secrets(submission))
        finish_run(directory)
    return read_result(root, run_id)
