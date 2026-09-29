"""Inspect official RULER preparation without changing its generators."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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


def check_source(source: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION or subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--", "*.py", "*.yaml"]
    ):
        raise ValueError("Expected unmodified pinned upstream RULER scripts")


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_input(path: Path) -> str:
    if path.exists():
        return path.read_text()
    return gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes()).decode()


def score_controls(directory: Path, source: Path) -> None:
    """Exercise the pinned metric for every prepared reference, without model claims."""
    import yaml

    check_source(source)
    tasks = yaml.safe_load((source / "scripts/synthetic.yaml").read_text())
    metrics = runpy.run_path(str(source / "scripts/eval/synthetic/constants.py"))["TASKS"]
    outcomes = []
    for condition in json.loads((directory / "preparation.json").read_text()):
        task, checkpoint = condition["task"], condition["checkpoint"]
        if condition["exit_code"] != 0:
            continue
        path = directory / f"{task}-{checkpoint}" / task / "validation.jsonl"
        metric = metrics[tasks[task]["task"]]["metric_fn"]
        for index, line in enumerate(read_input(path).splitlines()):
            references = json.loads(line)["outputs"]
            prediction = " ".join(references)
            outcomes.append(
                {
                    "task": task,
                    "checkpoint": checkpoint,
                    "index": index,
                    "reference_prediction": prediction,
                    "reference_score": metric([prediction], [references]),
                    "empty_prediction": "",
                    "empty_score": metric([""], [references]),
                }
            )
    save(directory / "scoring-controls.json", outcomes)


def report(directory: Path) -> dict[str, Any]:
    workload = json.loads((directory / "workload.json").read_text())
    preparations = json.loads((directory / "preparation.json").read_text())
    samples = [
        sample
        for condition in preparations
        if condition["exit_code"] == 0
        for sample in condition["samples"]
    ]
    differences = [
        sample["input_tokens"] - condition["checkpoint"]
        for condition in preparations
        if condition["exit_code"] == 0
        for sample in condition["samples"]
    ]
    expected_conditions = {(task, checkpoint) for task in TASKS for checkpoint in CHECKPOINTS}
    actual_conditions = [(row["task"], row["checkpoint"]) for row in preparations]
    preparation_validated = (
        workload.get("source_revision") == REVISION
        and workload.get("planned_requests") == 117
        and set(workload.get("tasks", [])) == set(TASKS)
        and workload.get("checkpoints") == list(CHECKPOINTS)
        and workload.get("samples_per_task_checkpoint") == 3
        and len(actual_conditions) == len(expected_conditions)
        and set(actual_conditions) == expected_conditions
        and isinstance(workload.get("quality_output_limit"), int)
        and workload["quality_output_limit"] > 0
        and isinstance(workload.get("context_window"), int)
        and all(
            row["exit_code"] == 0
            and len(row["samples"]) == 3
            and {sample["index"] for sample in row["samples"]} == {0, 1, 2}
            and all(
                type(sample["input_tokens"]) is int
                and 0 < sample["input_tokens"] <= row["checkpoint"]
                and sample["input_tokens"] + workload["quality_output_limit"]
                <= workload["context_window"]
                for sample in row["samples"]
            )
            for row in preparations
        )
    )
    input_ranges = {}
    for checkpoint in CHECKPOINTS:
        counts = [
            sample["input_tokens"]
            for row in preparations
            if row["checkpoint"] == checkpoint and row["exit_code"] == 0
            for sample in row["samples"]
        ]
        input_ranges[str(checkpoint)] = {
            "samples": len(counts),
            "minimum": min(counts) if counts else None,
            "maximum": max(counts) if counts else None,
        }
    live_path = directory / "live" / "outcomes.json"
    live_outcomes = json.loads(live_path.read_text()) if live_path.exists() else []
    return {
        "evidence": "sampled diagnostic preparation probe",
        "planned_requests": workload["planned_requests"],
        "prepared_requests": len(samples),
        "exact_checkpoint_requests": sum(value == 0 for value in differences),
        "largest_shortfall_tokens": max([0, *(-value for value in differences)]),
        "largest_overshoot_tokens": max([0, *differences]),
        "preparation_seconds": sum(row["elapsed_seconds"] for row in preparations),
        "sizing_contract": "upstream-input-ceilings",
        "preparation_validated": preparation_validated,
        "input_ranges": input_ranges,
        "full_profile_executed": False,
        "full_official_score": None,
        "live_outcomes": live_outcomes,
        "limitations": "Preparation alone does not validate live execution or scoring.",
    }


def live(directory: Path, source: Path, tokenizer: Path, setup_path: Path) -> None:
    """Exercise three exactly sized, already generated samples with AIPerf."""
    import httpx
    from humaneval_pilot import execute, redact, write_rows
    from performance_pilot import TOOL
    from performance_pilot import report as generation_report

    setup = json.loads(setup_path.read_text())
    # This is a bounded diagnostic against the endpoint recorded in the submitted setup.
    if not setup.get("base_url") or not setup.get("model_id"):
        raise ValueError("The setup must record the endpoint base_url and model_id")
    check_source(source)
    workload = json.loads((directory / "workload.json").read_text())
    for name, expected in workload["tokenizer_sha256"].items():
        if hashlib.sha256((tokenizer / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Tokenizer differs from preparation")
    if setup["request_settings"]["chat_template_kwargs"] != workload["chat_template_kwargs"]:
        raise ValueError("Chat settings differ from preparation")
    metrics = runpy.run_path(str(source / "scripts/eval/synthetic/constants.py"))["TASKS"]
    root = directory / "live"
    root.mkdir(exist_ok=False)
    save(root / "setup.json", redact(setup))
    selected = [
        ("niah_single_2", 32768, 2),
        ("niah_single_2", 131072, 0),
        ("niah_single_3", 196608, 2),
    ]
    save(
        root / "workload.json",
        {
            "selected": selected,
            "tool": TOOL,
            "max_tokens": 4096,
            "scorer": "upstream niah metric_fn on final content",
            "source_revision": REVISION,
            "retries": 0,
        },
    )
    outcomes = []

    def reachable() -> bool:
        try:
            httpx.get(setup["base_url"] + "/v1/models", timeout=10)
        except httpx.RequestError:
            return False
        return True

    for task, checkpoint, index in selected:
        condition = root / f"{task}-{checkpoint}-{index}"
        condition.mkdir()
        native_path = directory / f"{task}-{checkpoint}" / task / "validation.jsonl"
        row = json.loads(read_input(native_path).splitlines()[index])
        prompt = row["input"] + row.get("answer_prefix", "")
        write_rows(
            condition / "input.jsonl",
            [
                {
                    "text": "Messages in extra",
                    "output_length": 4096,
                    "extra": {
                        **setup["request_settings"],
                        "messages": [{"role": "user", "content": prompt}],
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
            str(tokenizer.resolve()),
            "--use-server-token-count",
            "--request-count",
            "1",
            "--concurrency",
            "1",
            "--request-timeout-seconds",
            "900",
            "--random-seed",
            "123",
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
        code = execute(command, condition, "generation")
        try:
            observed = generation_report(condition / "native", 4096)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            outcomes.append(
                {
                    "task": task,
                    "checkpoint": checkpoint,
                    "index": index,
                    "exit_code": code,
                    "score": None,
                    "diagnostic": str(exc),
                    "outcome": "native_output_unavailable",
                }
            )
            save(root / "outcomes.json", outcomes)
            if not reachable():
                break
            continue
        outcome = {
            "task": task,
            "checkpoint": checkpoint,
            "index": index,
            "generation": observed,
            "exit_code": code,
            "score": None,
            "sizing_matches": observed.get("usage", {}).get("prompt_tokens") == checkpoint,
        }
        if code == 0 and observed["outcome"] == "success":
            outcome["score"] = metrics["niah"]["metric_fn"](
                [observed["assistant"]["content"]], [row["outputs"]]
            )
        outcomes.append(outcome)
        save(root / "outcomes.json", outcomes)
        print(task, checkpoint, outcome["score"], flush=True)
        if observed.get("status") is None and not reachable():
            break


def prepare(
    source: Path,
    tokenizer_path: Path,
    directory: Path,
    *,
    quality_output_limit: int = 4096,
    context_window: int = 262144,
    chat_template_kwargs: dict[str, Any] | None = None,
    chat_template_path: Path | None = None,
    seed: int = 123,
) -> None:
    import yaml
    from transformers import AutoTokenizer

    source, tokenizer_path = source.resolve(), tokenizer_path.resolve()
    check_source(source)
    tasks = yaml.safe_load((source / "scripts/synthetic.yaml").read_text())
    constants = runpy.run_path(str(source / "scripts/data/synthetic/constants.py"))["TASKS"]
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=False)
    if chat_template_path is not None:
        # The effective template (a declared override or the tokenizer's own)
        # is written beside the run directory by the runner, so the 117
        # inputs hold for the template the endpoint actually applies.
        template = chat_template_path.read_text()
    else:
        template = (tokenizer_path / "chat_template.jinja").read_text()
    settings = (
        {"enable_thinking": True, "reasoning_effort": "xhigh"}
        if chat_template_kwargs is None
        else chat_template_kwargs
    )

    def count(prompt: str) -> int:
        return len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                chat_template=template,
                tokenize=True,
                add_generation_prompt=True,
                **settings,
            )
        )

    overhead = count("")
    directory.mkdir(parents=True, exist_ok=False)
    save(
        directory / "workload.json",
        {
            "evidence": "sampled diagnostic preparation probe",
            "source_revision": REVISION,
            "tasks": list(tasks),
            "checkpoints": CHECKPOINTS,
            "samples_per_task_checkpoint": 3,
            "planned_requests": 117,
            "seed": seed,
            "sizing_contract": "upstream-input-ceilings",
            "chat_template_kwargs": settings,
            "empty_chat_overhead": overhead,
            "quality_output_limit": quality_output_limit,
            "context_window": context_window,
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "tokenizer_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in tokenizer_path.iterdir()
                if p.is_file()
            },
            "source_data_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (source / "scripts/data/synthetic/json").glob("*.json")
            },
            "sizing": (
                "Upstream total limit = checkpoint + native output reserve - chat overhead; "
                "measure actual inputs independently"
            ),
        },
    )
    observations = []
    for task, config in tasks.items():
        constant = constants[config["task"]]
        for checkpoint in CHECKPOINTS:
            condition = directory / f"{task}-{checkpoint}"
            condition.mkdir()
            command = [
                sys.executable,
                str(source / "scripts/data/synthetic" / f"{config['task']}.py"),
                "--save_dir",
                str(condition.resolve()),
                "--save_name",
                task,
                "--tokenizer_path",
                str(tokenizer_path),
                "--tokenizer_type",
                "hf",
                "--max_seq_length",
                str(checkpoint + constant["tokens_to_generate"] - overhead),
                "--tokens_to_generate",
                str(constant["tokens_to_generate"]),
                "--num_samples",
                "3",
                "--random_seed",
                str(seed),
                "--template",
                constant["template"] + constant.get("answer_prefix", ""),
            ]
            for key, value in config.get("args", {}).items():
                command.extend([f"--{key}", str(value)])
            save(condition / "command.json", command)
            start = time.monotonic()
            diagnostic = None
            code = -1
            try:
                with (condition / "preparation.log").open("w") as log:
                    code = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, timeout=900
                    ).returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                diagnostic = str(exc)
            observation: dict[str, Any] = {
                "task": task,
                "checkpoint": checkpoint,
                "exit_code": code,
                "diagnostic": diagnostic,
                "elapsed_seconds": time.monotonic() - start,
                "samples": [],
            }
            path = condition / task / "validation.jsonl"
            if code == 0 and path.exists():
                for index, line in enumerate(path.read_text().splitlines()):
                    row = json.loads(line)
                    prompt = row["input"] + row.get("answer_prefix", "")
                    actual = count(prompt)
                    observation["samples"].append(
                        {
                            "index": index,
                            "input_tokens": actual,
                            "difference_from_checkpoint": actual - checkpoint,
                        }
                    )
            observations.append(observation)
            save(directory / "preparation.json", observations)
            print(task, checkpoint, observation, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--score-controls", action="store_true")
    parser.add_argument("--setup", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--quality-output-limit", type=int, default=4096)
    parser.add_argument("--context-window", type=int, default=262144)
    parser.add_argument("--chat-template-kwargs", default="{}")
    parser.add_argument("--chat-template-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.report:
        print(json.dumps(report(args.report), indent=2))
    elif args.score_controls and args.source and args.directory:
        score_controls(args.directory, args.source)
    elif args.live and args.source and args.tokenizer and args.directory and args.setup:
        live(args.directory, args.source, args.tokenizer, args.setup)
    elif args.source and args.tokenizer and args.directory:
        try:
            settings = json.loads(args.chat_template_kwargs)
        except json.JSONDecodeError as exc:
            parser.error(f"--chat-template-kwargs must be JSON: {exc.msg}")
        if not isinstance(settings, dict):
            parser.error("--chat-template-kwargs must be a JSON object")
        prepare(
            args.source,
            args.tokenizer,
            args.directory,
            quality_output_limit=args.quality_output_limit,
            context_window=args.context_window,
            chat_template_kwargs=settings,
            chat_template_path=args.chat_template_path,
            seed=args.seed,
        )
    else:
        parser.error("Preparation requires --source, --tokenizer, and --directory")
