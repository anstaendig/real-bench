"""Pilot orchestration; AIPerf owns requests and all measurement.

This is a validation script, not the default benchmark or a second product mode.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import time
from pathlib import Path
from typing import Any

TOOL = [
    "uv",
    "tool",
    "run",
    "--from",
    "aiperf==0.12.0",
    "--with",
    "transformers==5.17.0",
    "aiperf",
]


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.exists():
        contents = path.read_text()
    else:
        contents = gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes()).decode()
    return [json.loads(line) for line in contents.splitlines() if line.strip()]


def report(directory: Path, output_target: int) -> dict[str, Any]:
    raw_rows = read_rows(directory / "profile_export_raw.jsonl")
    metric_rows = read_rows(directory / "profile_export.jsonl")
    if len(raw_rows) != 1 or len(metric_rows) != 1:
        raise ValueError("Pilot invocation must contain exactly one retained request")
    raw, measured = raw_rows[0], metric_rows[0]
    if raw["metadata"]["x_request_id"] != measured["metadata"]["x_request_id"]:
        raise ValueError("Native request and metric identities disagree")
    assistant = {"role": "assistant", "content": "", "reasoning_content": ""}
    usage: dict[str, Any] = {}
    finish_reason = None
    done = False
    for response in raw["responses"]:
        for packet in response["packets"]:
            if packet["name"] != "data":
                continue
            if packet["value"] == "[DONE]":
                done = True
                continue
            chunk = json.loads(packet["value"])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                assistant["content"] += delta.get("content") or ""
                assistant["reasoning_content"] += (
                    delta.get("reasoning_content") or delta.get("reasoning") or ""
                )
                finish_reason = choice.get("finish_reason") or finish_reason
    success = raw.get("status") == 200 and finish_reason is not None and done
    return {
        "evidence": "pilot",
        "outcome": "success" if success else "failed",
        "status": raw.get("status"),
        "request_id": raw["metadata"]["x_request_id"],
        "usage": usage,
        "finish_reason": finish_reason,
        "output_target": output_target,
        "output_target_attained": (
            usage["completion_tokens"] >= output_target
            if success and "completion_tokens" in usage
            else None
        ),
        "assistant": assistant,
        "metrics": measured.get("metrics", {}) if success else {},
    }


def run(setup_path: Path, tokenizer_path: Path, directory: Path) -> None:
    setup = json.loads(setup_path.read_text())
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / "setup.json", setup)
    workload: dict[str, Any] = {
        "evidence": "pilot",
        "checkpoints": [4096, 196608],
        "output_targets": [256, 1024],
        "repetitions": 1,
        "seed": 121,
        "tool": "aiperf==0.12.0",
        "concurrency": 1,
        "request_timeout_seconds": 900,
        "history": "live assistant responses plus authored synthetic tool results",
    }
    save(directory / "workload.json", workload)
    summary: dict[str, Any] = {"evidence": "pilot", "observations": [], "complete": False}
    started = time.monotonic()
    tokenizer = None
    try:
        for scenario in ("fresh-context", "growing-session"):
            for target in workload["output_targets"]:
                history: list[dict[str, Any]] = []
                for checkpoint in workload["checkpoints"]:
                    name = f"{scenario}-{target}-{checkpoint}"
                    row: dict[str, Any] = {
                        "condition": name,
                        "scenario": scenario,
                        "input_checkpoint": checkpoint,
                        "output_target": target,
                    }
                    summary["observations"].append(row)
                    if checkpoint + target > setup["context_window"]:
                        row["outcome"] = "unsupported"
                        continue
                    if tokenizer is None:
                        from transformers import AutoTokenizer

                        tokenizer = AutoTokenizer.from_pretrained(
                            str(tokenizer_path),
                            local_files_only=True,
                        )
                    preparation_start = time.monotonic()
                    messages = list(history) if scenario == "growing-session" else []
                    messages.append({"role": "user", "content": f"Pilot 121 {name}. Read the log."})
                    call_id = f"log-{name}"
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": "read_log", "arguments": "{}"},
                                }
                            ],
                        }
                    )
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": ""})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Summarize the log and explain a method for analyzing it in detail."
                            ),
                        }
                    )
                    kwargs = setup["request_settings"]["chat_template_kwargs"]

                    # Workload sizing only: no requests or performance measurements here.
                    low, high = 0, checkpoint
                    while low <= high:
                        middle = (low + high) // 2
                        messages[-2]["content"] = "log" + " entry" * middle
                        template_messages = json.loads(json.dumps(messages))
                        for message in template_messages:
                            for call in message.get("tool_calls", []):
                                call["function"]["arguments"] = json.loads(
                                    call["function"]["arguments"]
                                )
                        actual = len(
                            tokenizer.apply_chat_template(
                                template_messages,
                                tokenize=True,
                                return_dict=False,
                                add_generation_prompt=True,
                                **kwargs,
                            )
                        )
                        if actual == checkpoint:
                            break
                        if actual < checkpoint:
                            low = middle + 1
                        else:
                            high = middle - 1
                    else:
                        raise ValueError(f"Cannot size {name} exactly with supplied tokenizer")
                    row["local_input_tokens"] = actual
                    row["bare_content_tokens"] = sum(
                        len(tokenizer.encode(m.get("content", ""), add_special_tokens=False))
                        for m in messages
                    )
                    row["template_and_structured_history_tokens"] = (
                        actual - row["bare_content_tokens"]
                    )
                    row["preparation_seconds"] = time.monotonic() - preparation_start
                    condition = directory / name
                    condition.mkdir()
                    dataset = condition / "input.jsonl"
                    dataset.write_text(
                        json.dumps(
                            {
                                "text": "Message array supplied in extra.messages",
                                "output_length": target,
                                "extra": {**setup["request_settings"], "messages": messages},
                            }
                        )
                        + "\n"
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
                        str(tokenizer_path.resolve()),
                        "--use-server-token-count",
                        "--request-count",
                        "1",
                        "--concurrency",
                        "1",
                        "--input-file",
                        str(dataset.resolve()),
                        "--custom-dataset-type",
                        "single_turn",
                        "--random-seed",
                        "121",
                        "--request-timeout-seconds",
                        "900",
                        "--export-level",
                        "raw",
                        "--no-server-metrics",
                        "--no-gpu-telemetry",
                        "--output-artifact-dir",
                        str((condition / "native").resolve()),
                    ]
                    save(condition / "command.json", command)
                    execution_start = time.monotonic()
                    print(f"Executing {name}", flush=True)
                    with (condition / "execution.log").open("w") as log:
                        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                    row["tool_wall_seconds"] = time.monotonic() - execution_start
                    row["exit_code"] = completed.returncode
                    try:
                        normalized = report(condition / "native", target)
                    except (OSError, ValueError, KeyError) as exc:
                        row.update(outcome="failed", diagnostic=str(exc))
                        save(directory / "summary.json", summary)
                        # Without a native response, endpoint availability is uncertain.
                        return
                    row.update(normalized)
                    save(condition / "report.json", normalized)
                    if row["outcome"] == "success":
                        history = messages + [normalized["assistant"]]
                        row["endpoint_input_matches"] = row["usage"].get("prompt_tokens") == actual
                    save(directory / "summary.json", summary)
        summary["complete"] = True
    finally:
        summary["wall_seconds"] = time.monotonic() - started
        save(directory / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("directory", type=Path)
    report_parser.add_argument("--output-target", type=int, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("setup", type=Path)
    run_parser.add_argument("--tokenizer", type=Path, required=True)
    run_parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.setup, args.tokenizer, args.directory)
    else:
        print(json.dumps(report(args.directory, args.output_target), indent=2))


if __name__ == "__main__":
    main()
