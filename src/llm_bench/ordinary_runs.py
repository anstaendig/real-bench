"""Ordinary run records. External tools own requests, token accounting and timing."""

from __future__ import annotations

import fcntl
import functools
import gzip
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TextIO
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

TOOL = ["uv", "tool", "run", "--from", "aiperf==0.12.0", "--with", "transformers==5.17.0", "aiperf"]
TOKENIZER_TOOL = [
    "uv",
    "run",
    "--with",
    "aiperf==0.12.0",
    "--with",
    "transformers==5.17.0",
    "python",
]
FRESH_CHECKPOINTS = [4096, 16384, 32768, 65536, 131072, 196608]
OUTPUT_TARGETS = [256, 1024]
# Fixed quality output budgets (tokens). These bound reasoning plus final-answer
# generation where the endpoint counts reasoning within the total allowance.
# Selected from representative bounded development evidence on the benchmark
# model itself not from the provisional 8192/4096 values that
# truncated earlier runs. Every coding task the 27B run truncated at 8192
# was re-probed at the benchmark settings: 11 stop within 32768 (max 28858) and
# the 12th (HumanEval/32) stops at 34673, so coding gets 65536 (8x the truncating
# value, 1.9x the observed 27B maximum). The 14 RULER cwe/fwe rows that run
# truncated at 4096 were re-probed too: 2 stop at 43665/46847 and the rest keep
# generating past 65536 (one cwe row stops at 69322 at 131072), the 27B
# reasoning_effort xhigh reasoning traces. RULER gets 65536, the maximum budget
# whose reservation (196608 + 65536 = 262144) keeps the largest sampled
# checkpoint supported in a 262144-token window; longer 27B traces retain a
# length finish reason, which the issue requires to stay visible.
CODING_OUTPUT_BUDGET = 65536
RULER_OUTPUT_BUDGET = 65536
REQUIRED = {
    "hardware": {"accelerator", "count", "memory", "cpu", "ram", "os", "driver"},
    "model": {"source", "revision", "quantized_artifact", "tokenizer", "template_overrides"},
    "serving": {"engine", "version", "launch_command", "environment"},
    "requests": {"base_url", "model_id", "generation", "reasoning", "context_window"},
    "benchmark": {"tool", "version", "transformers_version", "seed", "settings"},
}
# Routing, model identity, context window, tokenizer, and template information
# gate request routing and input preparation. Missing or invalid values make a
# run impossible, so guided setup must reject them before saving.
CRITICAL_SETUP = (
    ("requests", "base_url"),
    ("requests", "model_id"),
    ("requests", "context_window"),
    ("model", "tokenizer"),
    ("model", "template_overrides"),
    ("requests", "reasoning"),
    ("requests", "generation"),
)
DEFAULT_BENCHMARK = {
    "tool": "aiperf",
    "version": "0.12.0",
    "transformers_version": "5.17.0",
    "seed": 121,
    "settings": {"concurrency": 1},
}
SECRET = re.compile(
    r"api[-_]?key|authorization|password|secret|(?:^|[-_])token$|access[-_]?token|cookie", re.I
)
NATIVE_NON_SECRET_KEYS = {
    "time_to_first_token",
    "time_to_second_token",
    "time_to_first_output_token",
}


class RunCancelled(Exception):
    """A durable cancellation request stopped owned local work."""


class EndpointUnreachable(ConnectionError):
    """Native tool evidence reports that the remote endpoint transport vanished."""


def cancellation_path(directory: Path) -> Path:
    return directory / "cancel.json"


def request_cancellation(root: Path, run_id: str) -> dict[str, Any]:
    """Persist a cancellation request for an existing ordinary run."""
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise FileNotFoundError("unknown run")
    directory = root / run_id
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError("unknown run")
    save(cancellation_path(directory), {"requested": True, "requested_at": time.time()})
    return {"run_id": run_id, "cancellation_requested": True}


def cancellation_requested(path: Path | None) -> bool:
    return path is not None and path.is_file() and not path.is_symlink()


def native_endpoint_unreachable(directory: Path) -> bool:
    """Detect a transport failure while a streaming tool is still writing raw rows."""
    export = directory / "native" / "profile_export_raw.jsonl"
    try:
        for line in export.read_text().splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("status") not in (None, 0):
                continue
            details = json.dumps(
                raw.get("error") or raw.get("error_details") or raw.get("exception") or {}
            ).lower()
            if "timeout" in details:
                continue
            return True
    except (OSError, ValueError, AttributeError):
        pass
    return False


def _terminate_owned_process(process: subprocess.Popen[str]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGINT)
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            except ProcessLookupError:
                pass
    # The group leader can exit after SIGINT while descendants remain alive.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass


def _remove_owned_container(command: list[str]) -> None:
    if not command or command[0] != "docker" or "--name" not in command:
        return
    name = command[command.index("--name") + 1]
    subprocess.Popen(
        ["docker", "rm", "--force", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name == "posix"),
    ).wait(timeout=15)


def secret_key(name: str) -> bool:
    return name not in NATIVE_NON_SECRET_KEYS and SECRET.search(name) is not None


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: "[REDACTED]" if secret_key(k) else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [
            "[REDACTED]"
            if i
            and isinstance(value[i - 1], str)
            and SECRET.search(value[i - 1])
            and value[i - 1].startswith("--")
            else redact(v)
            for i, v in enumerate(value)
        ]
    if isinstance(value, str):
        if value.lstrip().startswith(("{", "[")):
            try:
                return json.dumps(redact(json.loads(value)))
            except ValueError:
                pass
        value = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", value)
        return re.sub(
            r"(?i)((?:api[-_]?key|authorization|password|secret|access[-_]?token)[\s=:]+)(?:Bearer\s+)?[^\s,;]+",
            r"\1[REDACTED]",
            value,
        )
    return value


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Setup(Record):
    declared: dict[str, dict[str, Any]]
    observed: dict[str, Any]

    @model_validator(mode="after")
    def validate_categories(self) -> Setup:
        for category, field in CRITICAL_SETUP:
            value = self.declared.get(category, {}).get(field)
            if value in (None, "", "unknown"):
                raise ValueError(f"{category}.{field} is execution-critical and cannot be unknown")
        for category, fields in REQUIRED.items():
            values = self.declared.get(category, {})
            missing = fields - values.keys()
            if missing or any(values.get(field) in (None, "") for field in fields):
                raise ValueError(f"{category}: required reproduction fields: {sorted(fields)}")
        for category, fields in REQUIRED.items():
            for field in fields - {
                "count",
                "context_window",
                "seed",
                "generation",
                "reasoning",
                "settings",
                "environment",
                "template_overrides",
                "launch_command",
            }:
                value = self.declared[category][field]
                if not isinstance(value, str) or value == "":
                    raise ValueError(f"{category}.{field} must be a readable string or unknown")
        for category, field in (("serving", "environment"), ("model", "template_overrides")):
            if not isinstance(self.declared[category][field], dict):
                raise ValueError(f"{category}.{field} must be an object")
        requests = self.declared["requests"]
        url = urlsplit(requests["base_url"])
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("base_url must be HTTP(S), without credentials, query or fragment")
        for key in ("generation", "reasoning"):
            if not isinstance(requests[key], dict):
                raise ValueError(f"requests.{key} must be an object (empty is explicit)")
            if {
                "model",
                "messages",
                "stream",
                "max_tokens",
                "max_completion_tokens",
                "n",
            } & requests[key].keys():
                raise ValueError("request settings cannot override workload or routing")
        if requests["generation"].keys() & requests["reasoning"].keys():
            raise ValueError("generation and reasoning settings must not overlap")
        for request_group in (requests["generation"], requests["reasoning"]):
            if "chat_template_kwargs" in request_group and not isinstance(
                request_group["chat_template_kwargs"], dict
            ):
                raise ValueError("chat_template_kwargs must be an object")
        if not isinstance(requests["context_window"], int) or requests["context_window"] <= 0:
            raise ValueError("context_window must be a positive integer")
        benchmark = self.declared["benchmark"]
        if (benchmark["tool"], benchmark["version"], benchmark["transformers_version"]) != (
            "aiperf",
            "0.12.0",
            "5.17.0",
        ):
            raise ValueError("this slice requires aiperf 0.12.0 and transformers 5.17.0")
        if benchmark["settings"] != {"concurrency": 1} or not isinstance(benchmark["seed"], int):
            raise ValueError("benchmark requires integer seed and settings: {concurrency: 1}")
        return self


class SpeedCondition(Record):
    scenario: Literal["fresh-context", "growing-session"]
    input_tokens: int = Field(gt=0)
    output_target: int = Field(gt=0)
    messages: list[dict[str, Any]] = Field(min_length=1)
    request_timeout_seconds: int = Field(default=900, gt=0)
    repetitions: Literal[1] = 1


class FreshContextWorkload(Record):
    profile: Literal["fresh-context-speed"]
    scenario: Literal["fresh-context"]
    checkpoints: list[int]
    output_targets: list[int]
    repetitions: Literal[5] = 5
    warmup_repetitions: Literal[1] = 1
    request_timeout_seconds: int = Field(default=900, gt=0)
    cache_configuration: dict[str, Any]
    cache_observations: list[dict[str, Any]]

    @model_validator(mode="after")
    def validate_fixed_matrix(self) -> FreshContextWorkload:
        if self.checkpoints != FRESH_CHECKPOINTS:
            raise ValueError(f"fresh-context checkpoints must be {FRESH_CHECKPOINTS}")
        if self.output_targets != OUTPUT_TARGETS:
            raise ValueError(f"fresh-context output targets must be {OUTPUT_TARGETS}")
        return self


class FreshContextSpeedCondition(Record):
    name: str
    input_tokens: int
    output_target: int


class GrowingSessionWorkload(Record):
    profile: Literal["growing-session-speed"]
    scenario: Literal["growing-session"]
    checkpoints: list[int]
    output_targets: list[int]
    repetitions: Literal[5] = 5
    warmup_repetitions: Literal[1] = 1
    request_timeout_seconds: int = Field(default=900, gt=0)
    cache_configuration: dict[str, Any]
    cache_observations: list[dict[str, Any]]

    @model_validator(mode="after")
    def validate_fixed_matrix(self) -> GrowingSessionWorkload:
        if self.checkpoints != FRESH_CHECKPOINTS:
            raise ValueError(f"growing-session checkpoints must be {FRESH_CHECKPOINTS}")
        if self.output_targets != OUTPUT_TARGETS:
            raise ValueError(f"growing-session output targets must be {OUTPUT_TARGETS}")
        return self


class HumanEvalWorkload(Record):
    profile: Literal["humaneval-plus"]
    problems_path: str
    max_tokens: int = Field(default=CODING_OUTPUT_BUDGET, gt=0)
    solutions_per_task: Literal[1] = 1
    request_timeout_seconds: int = Field(default=900, gt=0)
    grading_timeout_seconds: int = Field(default=3600, gt=0)

    @model_validator(mode="after")
    def validate_fixed_budget(self) -> HumanEvalWorkload:
        if self.max_tokens != CODING_OUTPUT_BUDGET:
            raise ValueError(f"coding output budget must be the fixed {CODING_OUTPUT_BUDGET}")
        return self


class SampledRulerWorkload(Record):
    profile: Literal["sampled-ruler"]
    source_path: str = Field(min_length=1)
    max_tokens: int = Field(default=RULER_OUTPUT_BUDGET, gt=0)
    request_timeout_seconds: int = Field(default=900, gt=0)

    @model_validator(mode="after")
    def validate_fixed_budget(self) -> SampledRulerWorkload:
        if self.max_tokens != RULER_OUTPUT_BUDGET:
            raise ValueError(f"sampled RULER output budget must be the fixed {RULER_OUTPUT_BUDGET}")
        return self


DEFAULT_SUITE_UNITS = {
    "fresh_context": 60,
    "growing_session": 60,
    "coding": 164,
    "long_context": 117,
}


class DefaultSuiteWorkload(Record):
    profile: Literal["default-suite"]
    fresh_context: FreshContextWorkload
    growing_session: GrowingSessionWorkload
    coding: HumanEvalWorkload
    long_context: SampledRulerWorkload


class RunConfig(Record):
    """The per-launch configuration choices.

    ``reasoning_effort`` is required; ``off`` (case-insensitive input, stored
    lowercase) disables thinking for the run. The six sampling parameters are
    all present but individually nullable, where null records an explicit "no
    override". ``extra="forbid"`` keeps frequency penalty and seed out of the
    per-run configuration surface.
    """

    reasoning_effort: str
    temperature: float | int | None
    top_p: float | int | None
    top_k: float | int | None
    min_p: float | int | None
    presence_penalty: float | int | None
    repetition_penalty: float | int | None

    @model_validator(mode="after")
    def _effort(self) -> RunConfig:
        effort = self.reasoning_effort.strip()
        if not effort or effort.lower() in {"none", "null"}:
            raise ValueError("reasoning_effort is required and cannot be none or null")
        self.reasoning_effort = "off" if effort.lower() == "off" else effort
        return self


class Submission(Record):
    setup: Setup
    workload: (
        SpeedCondition
        | FreshContextWorkload
        | GrowingSessionWorkload
        | HumanEvalWorkload
        | SampledRulerWorkload
        | DefaultSuiteWorkload
    )
    # Present and complete for an executable submission; setup-produced
    # submissions carry it as None and complete it at the launch boundary.
    run_config: RunConfig | None = None


def credential_values(value: Any) -> set[str]:
    """Collect declared credentials to scrub unlabelled echoes in tool logs."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if SECRET.search(key) and isinstance(item, str) and item:
                found.add(item)
            found.update(credential_values(item))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            if (
                i
                and isinstance(value[i - 1], str)
                and value[i - 1].startswith("--")
                and SECRET.search(value[i - 1])
                and isinstance(item, str)
            ):
                found.add(item)
            found.update(credential_values(item))
    elif isinstance(value, str):
        found.update(
            re.findall(
                r"(?i)(?:api[-_]?key|password|secret|access[-_]?token)[\s=:]+([^\s,;]+)", value
            )
        )
    return found


API_KEY_ENV = "REAL_BENCH_API_KEY"


def runtime_api_key() -> str | None:
    """The bearer key supplied through the environment, never persisted."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def api_key_args(command: list[str], key: str | None = None) -> list[str]:
    """Return the command with the endpoint bearer key appended, when set.

    Every AIPerf invocation routes the key through this single point so the
    authentication policy cannot drift between workloads. The key rides the
    command so the externally operated endpoint receives it; the retained
    ``command.json`` copy is redacted by position, and the live key joins the
    run's scrub set so any unlabelled echo is removed by ``sanitize_files``.
    """
    key = key if key is not None else runtime_api_key()
    return [*command, "--api-key", key] if key else command


def run_secrets(submission: Submission) -> set[str]:
    """Declared credentials plus the live bearer key.

    The key is not part of the setup record, so ``credential_values`` alone
    would miss an unlabelled echo of it; adding it here lets ``sanitize_files``
    scrub the key from tool logs and diagnostics while it still drives the
    authenticated requests that produced them.
    """
    secrets = credential_values(submission.model_dump())
    key = runtime_api_key()
    if key:
        secrets.add(key)
    return secrets


def sanitize_files(directory: Path, secrets: set[str]) -> None:
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        compressed = path.suffix == ".gz"
        try:
            contents = (
                gzip.decompress(path.read_bytes()).decode() if compressed else path.read_text()
            )
        except (UnicodeError, OSError):
            continue
        try:
            if path.name.endswith((".jsonl", ".jsonl.gz")):
                contents = (
                    "\n".join(
                        json.dumps(redact(json.loads(line)))
                        for line in contents.splitlines()
                        if line.strip()
                    )
                    + "\n"
                )
            elif path.suffix == ".json":
                contents = json.dumps(redact(json.loads(contents)), indent=2) + "\n"
            else:
                contents = redact(contents)
        except ValueError:
            contents = redact(contents)
        for secret in sorted(secrets, key=len, reverse=True):
            contents = contents.replace(json.dumps(secret)[1:-1], "[REDACTED]")
            contents = contents.replace(secret, "[REDACTED]")
        if compressed:
            path.write_bytes(gzip.compress(contents.encode()))
        else:
            path.write_text(contents)


def save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(redact(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_result(root: Path, run_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise FileNotFoundError("unknown run")
    directory = root / run_id
    if directory.is_symlink():
        raise FileNotFoundError("unknown run")
    result: dict[str, Any] = json.loads((directory / "summary.json").read_text())
    result["setup"] = json.loads((directory / "setup.json").read_text())
    result["workload"] = json.loads((directory / "workload.json").read_text())
    if result["workload"].get("profile") == "default-suite":
        from .default_suite import refresh_progress

        refresh_progress(directory, result)
    return result


_RUN_LOCKS: dict[Path, TextIO] = {}
_RUN_OWNERS: dict[Path, int] = {}
_RUN_LOCKS_GUARD = threading.Lock()


def finished_on_exit[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Release this invocation's run locks even if execution raises."""

    @functools.wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        owner = threading.get_ident()
        with _RUN_LOCKS_GUARD:
            before = {path for path, thread in _RUN_OWNERS.items() if thread == owner}
        try:
            return fn(*args, **kwargs)
        finally:
            with _RUN_LOCKS_GUARD:
                owned = [
                    path
                    for path, thread in _RUN_OWNERS.items()
                    if thread == owner and path not in before
                ]
            for directory in owned:
                finish_run(directory)

    return wrapped


def create_run_directory(root: Path, requested_run_id: str | None = None) -> tuple[str, Path]:
    run_id = requested_run_id or uuid.uuid4().hex
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("run_id must be 32 lowercase hexadecimal characters")
    directory = root / run_id
    try:
        directory.mkdir(parents=True)
    except FileExistsError as exc:
        raise ValueError("run_id already exists") from exc
    lock = (directory / "execution.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with _RUN_LOCKS_GUARD:
        _RUN_LOCKS[directory] = lock
        _RUN_OWNERS[directory] = threading.get_ident()
    save(
        directory / "execution-state.json",
        {"state": "running", "hostname": socket.gethostname(), "pid": os.getpid()},
    )
    return run_id, directory


def run_lock_held(directory: Path) -> bool:
    """An OS lock proves the original runner still owns the run."""
    try:
        with (directory / "execution.lock").open("r+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(lock, fcntl.LOCK_UN)
    except OSError:
        pass
    return False


def finish_run(directory: Path) -> None:
    """Persist terminal lifecycle state after the final summary is durable."""
    path = directory / "execution-state.json"
    try:
        state = json.loads(path.read_text())
        if isinstance(state, dict):
            state["state"] = "finished"
            save(path, state)
    except (OSError, ValueError):
        pass
    finally:
        with _RUN_LOCKS_GUARD:
            _RUN_OWNERS.pop(directory, None)
            lock = _RUN_LOCKS.pop(directory, None)
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str] | None:
    if path.exists():
        contents = path.read_text()
        source = path.name
    elif path.with_suffix(path.suffix + ".gz").exists():
        compressed = path.with_suffix(path.suffix + ".gz")
        contents = gzip.decompress(compressed.read_bytes()).decode()
        source = compressed.name
    else:
        return None
    return ([json.loads(line) for line in contents.splitlines() if line.strip()], source)


def normalize_native_row(
    raw: dict[str, Any],
    measured: dict[str, Any] | None,
    input_tokens: int,
    output_target: int,
    raw_source: str,
    metric_source: str,
) -> dict[str, Any]:
    request_id = raw["metadata"]["x_request_id"]
    usage: dict[str, Any] = {}
    finish = None
    done = False
    for response in raw.get("responses", []):
        for packet in response.get("packets", []):
            if packet.get("name") != "data":
                continue
            if packet["value"] == "[DONE]":
                done = True
                continue
            chunk = json.loads(packet["value"])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                finish = choice.get("finish_reason") or finish
    success = raw.get("status") == 200 and done and finish is not None
    native_metrics = measured.get("metrics", {}) if measured is not None else {}

    def native_token(*names: str) -> tuple[int | float | None, str | None]:
        for name in names:
            metric = native_metrics.get(name)
            if metric is None:
                continue
            value = metric.get("value")
            if (
                metric.get("unit") != "tokens"
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid native token metric: {name}")
            return value, f"native/{metric_source}:{name}"
        return None, None

    actual_input = usage.get("prompt_tokens")
    actual_input_source = (
        f"native/{raw_source}:usage.prompt_tokens" if actual_input is not None else None
    )
    if actual_input is None:
        actual_input, actual_input_source = native_token(
            "usage_prompt_tokens", "input_sequence_length"
        )
    actual_output = usage.get("completion_tokens")
    actual_output_source = (
        f"native/{raw_source}:usage.completion_tokens" if actual_output is not None else None
    )
    if actual_output is None:
        actual_output, actual_output_source = native_token(
            "usage_completion_tokens", "output_token_count", "output_sequence_length"
        )
    result: dict[str, Any] = {
        "outcome": "success" if success else "failed",
        "request_id": request_id,
        "http_status": raw.get("status"),
        "finish_reason": finish,
        "metrics": {},
        "local_input_tokens": input_tokens,
        "actual_input_tokens": actual_input,
        "actual_input_tokens_source": actual_input_source,
        "actual_output_tokens": actual_output,
        "actual_output_tokens_source": actual_output_source,
        "output_target": output_target,
        "output_target_attained": None,
        "endpoint_input_matches": None,
        "token_accounting": (
            "endpoint usage when available; includes endpoint-defined reasoning accounting"
        ),
    }
    if not success:
        return result
    if actual_input is not None:
        result["endpoint_input_matches"] = actual_input == input_tokens
    if result["endpoint_input_matches"] is False:
        result.update(
            outcome="failed",
            diagnostic="Endpoint input count differs from exact context checkpoint",
        )
        return result
    if actual_output is not None:
        result["output_target_attained"] = actual_output >= output_target
    if measured is None:
        result.update(outcome="missing", diagnostic="No native measurement output")
        return result
    metrics = measured.get("metrics", {})
    for name, upstream, unit in (
        ("ttft", "time_to_first_token", "ms"),
        ("completion_latency", "request_latency", "ms"),
        ("decode_rate", "output_token_throughput_per_user", "tokens/sec/user"),
        (
            "effective_endpoint_observed_prefill_throughput",
            "prefill_throughput_per_user",
            "tokens/sec/user",
        ),
    ):
        metric = metrics.get(upstream)
        if metric is None:
            result["metrics"][name] = {
                "value": None,
                "unit": unit,
                "unavailable_reason": "not exported by tool",
            }
        elif (
            metric["unit"] != unit
            or not isinstance(metric["value"], (int, float))
            or not math.isfinite(metric["value"])
            or metric["value"] < 0
        ):
            raise ValueError(f"invalid native metric: {upstream}")
        else:
            result["metrics"][name] = {
                **metric,
                "source": f"native/{metric_source}:{upstream}",
                "tool": "aiperf==0.12.0",
            }
    return result


def normalize_many(
    directory: Path, input_tokens: int, output_target: int, expected: int
) -> list[dict[str, Any]]:
    raw_export = read_jsonl(directory / "native/profile_export_raw.jsonl")
    if raw_export is None:
        return [
            {"outcome": "missing", "diagnostic": "No native request output", "metrics": {}}
            for _ in range(expected)
        ]
    raw_rows, raw_source = raw_export
    if len(raw_rows) > expected:
        raise ValueError(f"expected at most {expected} native requests")
    metric_export = read_jsonl(directory / "native/profile_export.jsonl")
    metric_rows, metric_source = metric_export or ([], "profile_export.jsonl")
    measured_by_id: dict[str, dict[str, Any]] = {}
    for metric_row in metric_rows:
        request_id = metric_row["metadata"]["x_request_id"]
        if request_id in measured_by_id:
            raise ValueError("duplicate native measurement identity")
        measured_by_id[request_id] = metric_row
    outcomes = []
    raw_request_ids: set[str] = set()
    for raw in raw_rows:
        request_id = raw["metadata"]["x_request_id"]
        if request_id in raw_request_ids:
            raise ValueError("duplicate native request identity")
        raw_request_ids.add(request_id)
        measured_row = measured_by_id.get(request_id)
        if measured_row is not None:
            del measured_by_id[request_id]
        outcomes.append(
            normalize_native_row(
                raw,
                measured_row,
                input_tokens,
                output_target,
                raw_source,
                metric_source,
            )
        )
    if measured_by_id:
        raise ValueError("native measurement/request identities disagree")
    outcomes.extend(
        {"outcome": "missing", "diagnostic": "No native request output", "metrics": {}}
        for _ in range(expected - len(outcomes))
    )
    return outcomes


def normalize(directory: Path, condition: SpeedCondition) -> dict[str, Any]:
    outcomes = normalize_many(directory, condition.input_tokens, condition.output_target, 1)
    return outcomes[0]


@finished_on_exit
def execute_single(
    root: Path,
    submission: Submission,
    condition: SpeedCondition,
    run_id: str | None = None,
) -> dict[str, Any]:
    run_id, directory = create_run_directory(root, run_id)
    setup = submission.setup.declared
    save(directory / "setup.json", submission.setup.model_dump())
    save(directory / "workload.json", {"evidence": "validation", **condition.model_dump()})
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "evidence": "validation",
        "status": "partial",
        "default_workload_complete": False,
        "families": {"performance": "partial", "coding": "missing", "long_context": "missing"},
        "outcomes": [{"outcome": "missing", "metrics": {}}],
    }
    save(directory / "summary.json", summary)
    requests = setup["requests"]
    dataset = directory / "input.jsonl"
    input_record = {
        "text": "Messages supplied in extra.messages",
        "output_length": condition.output_target,
        "extra": {
            **aiperf_request_settings(setup, mapped_request_settings(setup)),
            "messages": condition.messages,
        },
    }
    # Preserve execution inputs exactly; scrub the retained copy after invocation.
    dataset.write_text(json.dumps(input_record) + "\n")
    cancelled = False
    if condition.input_tokens + condition.output_target > requests["context_window"]:
        outcome = {
            "outcome": "unsupported",
            "diagnostic": "Input plus output exceeds declared context window",
            "metrics": {},
        }
    else:
        command = api_key_args(
            aiperf_command(
                setup,
                dataset,
                directory / "native",
                condition.repetitions,
                condition.request_timeout_seconds,
            )
        )
        try:
            completed = execute_tool(
                command,
                directory,
                condition.repetitions,
                condition.request_timeout_seconds,
                cancel_marker=cancellation_path(directory),
            )
            outcome = normalize(directory, condition)
            if completed.returncode != 0:
                outcome = {
                    "outcome": "failed",
                    "diagnostic": f"Tool exited {completed.returncode}",
                    "metrics": {},
                    "native_outcome": outcome,
                }
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            outcome = {"outcome": "failed", "diagnostic": str(exc), "metrics": {}}
        except (KeyboardInterrupt, RunCancelled):
            cancelled = True
            retained = normalize(directory, condition)
            outcome = (
                retained
                if retained["outcome"] == "success"
                else {"outcome": "cancelled", "metrics": {}, "native_outcome": retained}
            )
    summary["outcomes"] = [outcome]
    summary["status"] = (
        "completed"
        if not cancelled and outcome["outcome"] in {"success", "unsupported"}
        else "partial"
    )
    summary["families"]["performance"] = summary["status"]
    save(directory / "outcomes.json", summary["outcomes"])
    save(directory / "summary.json", summary)
    sanitize_files(directory, run_secrets(submission))
    finish_run(directory)
    return read_result(root, run_id)


def aiperf_command(
    setup: dict[str, dict[str, Any]],
    dataset: Path,
    native: Path,
    request_count: int,
    request_timeout_seconds: int,
) -> list[str]:
    requests = setup["requests"]
    command = TOOL + [
        "profile",
        "--model",
        requests["model_id"],
        "--url",
        requests["base_url"],
        "--endpoint",
        "/v1/chat/completions",
        "--endpoint-type",
        "chat",
        "--streaming",
        "--tokenizer",
        setup["model"]["tokenizer"],
        "--use-server-token-count",
        "--request-count",
        str(request_count),
        "--concurrency",
        "1",
        "--input-file",
        str(dataset.resolve()),
        "--custom-dataset-type",
        "single_turn",
        "--random-seed",
        str(setup["benchmark"]["seed"]),
        "--request-timeout-seconds",
        str(request_timeout_seconds),
        "--export-level",
        "raw",
        "--no-server-metrics",
        "--no-gpu-telemetry",
        "--output-artifact-dir",
        str(native.resolve()),
    ]
    return command


def execute_tool(
    command: list[str],
    directory: Path,
    request_count: int,
    request_timeout_seconds: int,
    *,
    cancel_marker: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    save(directory / "command.json", command)
    timeout = request_count * request_timeout_seconds + 120
    if cancellation_requested(cancel_marker):
        raise RunCancelled("Cancelled before external tool invocation")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=(os.name == "posix"),
    )
    started = time.monotonic()
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired as exc:
                if cancellation_requested(cancel_marker):
                    raise RunCancelled("Cancelled during external tool invocation") from exc
                if native_endpoint_unreachable(directory):
                    raise EndpointUnreachable(
                        "Native request reports unavailable endpoint transport"
                    ) from exc
                if time.monotonic() - started >= timeout:
                    raise subprocess.TimeoutExpired(command, timeout) from exc
    except BaseException as exc:
        _terminate_owned_process(process)
        try:
            _remove_owned_container(command)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            stdout, stderr = process.communicate(timeout=0.1)
        except (OSError, subprocess.SubprocessError):
            stdout, stderr = "", ""
        save(
            directory / "execution.json",
            {
                "exit_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "timeout_seconds": timeout,
                "interrupted": isinstance(exc, (KeyboardInterrupt, RunCancelled)),
                "timed_out": isinstance(exc, subprocess.TimeoutExpired),
                "diagnostic": str(exc),
                "elapsed_seconds": round(time.monotonic() - started, 6),
            },
        )
        if isinstance(exc, KeyboardInterrupt):
            if cancel_marker is not None:
                save(cancel_marker, {"requested": True, "reason": "keyboard_interrupt"})
            raise RunCancelled("Cancelled during external tool invocation") from exc
        raise
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    save(
        directory / "execution.json",
        {
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timeout_seconds": timeout,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        },
    )
    if cancellation_requested(cancel_marker):
        raise RunCancelled("Cancelled after external tool invocation")
    if native_endpoint_unreachable(directory):
        raise EndpointUnreachable("Native request reports unavailable endpoint transport")
    return completed


def failure_outcomes(
    condition: FreshContextSpeedCondition, repetitions: int, diagnostic: str
) -> list[dict[str, Any]]:
    return [
        {
            "condition": condition.name,
            "scenario": "fresh-context",
            "input_tokens": condition.input_tokens,
            "output_target": condition.output_target,
            "repetition": repetition,
            "outcome": "failed",
            "diagnostic": diagnostic,
            "metrics": {},
        }
        for repetition in range(repetitions)
    ]


def condition_outcomes(
    condition: FreshContextSpeedCondition, normalized: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "condition": condition.name,
            "scenario": "fresh-context",
            "input_tokens": condition.input_tokens,
            "output_target": condition.output_target,
            "repetition": repetition,
            **outcome,
        }
        for repetition, outcome in enumerate(normalized)
    ]


def summarize_metric(outcomes: list[dict[str, Any]], name: str) -> dict[str, Any]:
    metric_rows = [row.get("metrics", {}).get(name) for row in outcomes]
    values = [
        metric["value"]
        for metric in metric_rows
        if metric is not None and isinstance(metric.get("value"), (int, float))
    ]
    unit = next(
        (metric["unit"] for metric in metric_rows if metric is not None),
        {
            "ttft": "ms",
            "completion_latency": "ms",
            "decode_rate": "tokens/sec/user",
            "effective_endpoint_observed_prefill_throughput": "tokens/sec/user",
        }[name],
    )
    return {
        "unit": unit,
        "values": values,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "unavailable": len(outcomes) - len(values),
    }


SPEED_METRICS = (
    "ttft",
    "completion_latency",
    "decode_rate",
    "effective_endpoint_observed_prefill_throughput",
)


def scenario_curve(scenario: str, outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curve = []
    for checkpoint in FRESH_CHECKPOINTS:
        for output_target in OUTPUT_TARGETS:
            rows = [
                row
                for row in outcomes
                if row["input_tokens"] == checkpoint and row["output_target"] == output_target
            ]
            curve.append(
                {
                    "scenario": scenario,
                    "input_tokens": checkpoint,
                    "output_target": output_target,
                    "outcomes": {
                        name: sum(row["outcome"] == name for row in rows)
                        for name in ("success", "failed", "missing", "unsupported", "cancelled")
                    },
                    "metrics": {name: summarize_metric(rows, name) for name in SPEED_METRICS},
                }
            )
    return curve


def fresh_context_curve(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return scenario_curve("fresh-context", outcomes)


@finished_on_exit
def execute_fresh_context(
    root: Path,
    submission: Submission,
    workload: FreshContextWorkload,
    run_id: str | None = None,
    cancel_marker: Path | None = None,
) -> dict[str, Any]:
    run_id, directory = create_run_directory(root, run_id)
    marker = cancel_marker or cancellation_path(directory)
    setup = submission.setup.declared
    conditions = [
        FreshContextSpeedCondition(
            name=f"fresh-context-{checkpoint}-{target}",
            input_tokens=checkpoint,
            output_target=target,
        )
        for checkpoint in workload.checkpoints
        for target in workload.output_targets
    ]
    save(directory / "setup.json", submission.setup.model_dump())
    save(directory / "workload.json", {"evidence": "run", **workload.model_dump()})
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "evidence": "run",
        "status": "partial",
        "default_workload_complete": False,
        "families": {"performance": "partial", "coding": "missing", "long_context": "missing"},
        "coverage": {"fresh_context": {"expected": 60, "accounted": 0, "successful": 0}},
        "cache": {
            "configuration": workload.cache_configuration,
            "observations": workload.cache_observations,
            "request_design": "distinct prefixes discourage reuse; cache state is not controlled",
        },
        "warmups": [],
        "tool_failures": [],
        "outcomes": [],
        "fresh_context_curve": [],
    }
    save(directory / "summary.json", summary)

    supported = [
        condition
        for condition in conditions
        if condition.input_tokens + condition.output_target <= setup["requests"]["context_window"]
    ]
    template_kwargs: dict[str, Any] = {}
    for request_group in (setup["requests"]["generation"], setup["requests"]["reasoning"]):
        if "chat_template_kwargs" in request_group:
            template_kwargs = request_group["chat_template_kwargs"]
    preparation_spec = {
        "tokenizer": setup["model"]["tokenizer"],
        "chat_template_kwargs": template_kwargs,
        "request_settings": mapped_request_settings(setup),
        "seed": setup["benchmark"]["seed"],
        "repetitions": workload.repetitions,
        "warmup_repetitions": workload.warmup_repetitions,
        "conditions": [condition.model_dump() for condition in supported],
    }
    preparation_spec_path = directory / "preparation-spec.json"
    save(preparation_spec_path, preparation_spec)
    preparation_command = TOKENIZER_TOOL + [
        str(Path(__file__).with_name("performance_inputs.py")),
        "--spec",
        str(preparation_spec_path.resolve()),
        "--directory",
        str(directory.resolve()),
    ]
    save(directory / "preparation-command.json", preparation_command)
    preparation_error = None
    preparation_failures: dict[str, str] = {}
    cancelled = False
    endpoint_outage = False
    if supported:
        try:
            prepared = execute_tool(
                preparation_command,
                directory,
                1,
                workload.request_timeout_seconds,
                cancel_marker=marker,
            )
            (directory / "execution.json").replace(directory / "preparation-execution.json")
            if prepared.returncode != 0:
                preparation_error = f"Input preparation exited {prepared.returncode}"
            else:
                preparation = json.loads((directory / "preparation.json").read_text())
                observations = preparation["observations"]
                failures = preparation.get("failures", [])
                supported_names = {condition.name for condition in supported}
                if any(failure["condition"] not in supported_names for failure in failures):
                    raise ValueError("input preparation reported an unknown condition")
                preparation_failures = {
                    failure["condition"]: failure["diagnostic"] for failure in failures
                }
                if len({row["nonce"] for row in observations}) != len(observations):
                    raise ValueError("fresh-context request prefixes are not distinct")
                expected_per_condition = workload.repetitions + workload.warmup_repetitions
                for condition in supported:
                    if condition.name in preparation_failures:
                        continue
                    condition_rows = [
                        row for row in observations if row["condition"] == condition.name
                    ]
                    if len(condition_rows) != expected_per_condition:
                        preparation_failures[condition.name] = (
                            "input preparation did not account for every request"
                        )
                    elif any(
                        row["local_input_tokens"] != condition.input_tokens
                        for row in condition_rows
                    ):
                        preparation_failures[condition.name] = (
                            "prepared input differs from exact context checkpoint"
                        )
        except RunCancelled:
            preparation_error = "Cancelled during input preparation"
            cancelled = True
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            preparation_error = str(exc)

    outcomes: list[dict[str, Any]] = []
    outage_recorded = False
    for condition in conditions:
        if cancellation_requested(marker):
            cancelled = True
            break
        if condition not in supported:
            outcomes.extend(
                condition_outcomes(
                    condition,
                    [
                        {
                            "outcome": "unsupported",
                            "diagnostic": "Input plus output exceeds declared context window",
                            "metrics": {},
                        }
                        for _ in range(workload.repetitions)
                    ],
                )
            )
            continue
        if preparation_error is not None:
            outcomes.extend(failure_outcomes(condition, workload.repetitions, preparation_error))
            continue
        if condition.name in preparation_failures:
            outcomes.extend(
                failure_outcomes(
                    condition,
                    workload.repetitions,
                    preparation_failures[condition.name],
                )
            )
            continue
        condition_directory = directory / condition.name
        warmup_directory = condition_directory / "warmup"
        try:
            warmup_command = api_key_args(
                aiperf_command(
                    setup,
                    warmup_directory / "input.jsonl",
                    warmup_directory / "native",
                    workload.warmup_repetitions,
                    workload.request_timeout_seconds,
                )
            )
            warmup = execute_tool(
                warmup_command,
                warmup_directory,
                workload.warmup_repetitions,
                workload.request_timeout_seconds,
                cancel_marker=marker,
            )
            summary["warmups"].append(
                {
                    "condition": condition.name,
                    "requests": workload.warmup_repetitions,
                    "excluded_from_measurements": True,
                    "exit_code": warmup.returncode,
                }
            )
            prepared_inputs = (condition_directory / "input.jsonl").read_text().splitlines()
            if len(prepared_inputs) != workload.repetitions:
                raise ValueError("prepared condition has an unexpected measured request count")
            normalized: list[dict[str, Any]] = []
            for repetition, input_row in enumerate(prepared_inputs):
                repetition_directory = condition_directory / f"repetition-{repetition:03d}"
                repetition_directory.mkdir()
                (repetition_directory / "input.jsonl").write_text(input_row + "\n")
                measured_command = api_key_args(
                    aiperf_command(
                        setup,
                        repetition_directory / "input.jsonl",
                        repetition_directory / "native",
                        1,
                        workload.request_timeout_seconds,
                    )
                )
                try:
                    measured = execute_tool(
                        measured_command,
                        repetition_directory,
                        1,
                        workload.request_timeout_seconds,
                        cancel_marker=marker,
                    )
                except EndpointUnreachable:
                    endpoint_outage = True
                    normalized.append(
                        {
                            "outcome": "failed",
                            "diagnostic": "Native request reports unavailable endpoint transport",
                            "metrics": {},
                        }
                    )
                    break
                except RunCancelled:
                    retained = normalize_many(
                        repetition_directory,
                        condition.input_tokens,
                        condition.output_target,
                        1,
                    )[0]
                    outcomes.extend(condition_outcomes(condition, normalized))
                    outcomes.extend(
                        [
                            {
                                "condition": condition.name,
                                "scenario": "fresh-context",
                                "input_tokens": condition.input_tokens,
                                "output_target": condition.output_target,
                                "repetition": repetition,
                                **retained,
                            }
                        ]
                        if retained["outcome"] == "success"
                        else [
                            {
                                "condition": condition.name,
                                "scenario": "fresh-context",
                                "input_tokens": condition.input_tokens,
                                "output_target": condition.output_target,
                                "repetition": repetition,
                                "outcome": "cancelled",
                                "diagnostic": "Cancelled during measured request",
                                "metrics": {},
                                "native_outcome": retained,
                            }
                        ]
                    )
                    cancelled = True
                    break
                except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
                    normalized.append({"outcome": "failed", "diagnostic": str(exc), "metrics": {}})
                    continue
                row = normalize_many(
                    repetition_directory,
                    condition.input_tokens,
                    condition.output_target,
                    1,
                )[0]
                if measured.returncode != 0:
                    summary["tool_failures"].append(
                        {
                            "condition": condition.name,
                            "repetition": repetition,
                            "exit_code": measured.returncode,
                        }
                    )
                    if row["outcome"] != "success":
                        row = {
                            "outcome": "failed",
                            "diagnostic": f"Tool exited {measured.returncode}",
                            "metrics": {},
                            "native_outcome": row,
                        }
                normalized.append(row)
                if cancellation_requested(marker):
                    outcomes.extend(condition_outcomes(condition, normalized))
                    cancelled = True
                    break
            if cancelled:
                break
            if endpoint_outage:
                outcomes.extend(condition_outcomes(condition, normalized))
                outage_recorded = True
                raise EndpointUnreachable("Endpoint became unreachable during measured requests")
            outcomes.extend(condition_outcomes(condition, normalized))
        except RunCancelled:
            cancelled = True
            save(directory / "outcomes.json", outcomes)
            break
        except EndpointUnreachable as exc:
            endpoint_outage = True
            if not outage_recorded:
                outcomes.extend(failure_outcomes(condition, workload.repetitions, str(exc)))
            save(directory / "outcomes.json", outcomes)
            break
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            outcomes.extend(failure_outcomes(condition, workload.repetitions, str(exc)))
        save(directory / "outcomes.json", outcomes)

    if cancelled or endpoint_outage:
        accounted_conditions = {(row["condition"], row["repetition"]) for row in outcomes}
        outcomes.extend(
            {
                "condition": condition.name,
                "scenario": "fresh-context",
                "input_tokens": condition.input_tokens,
                "output_target": condition.output_target,
                "repetition": repetition,
                "outcome": (
                    "unsupported"
                    if condition.input_tokens + condition.output_target
                    > setup["requests"]["context_window"]
                    else "missing"
                ),
                "diagnostic": (
                    "Input plus output exceeds declared context window"
                    if condition.input_tokens + condition.output_target
                    > setup["requests"]["context_window"]
                    else (
                        "Not attempted after cancellation"
                        if cancelled
                        else "Not attempted after endpoint became unreachable"
                    )
                ),
                "metrics": {},
            }
            for condition in conditions
            for repetition in range(workload.repetitions)
            if (condition.name, repetition) not in accounted_conditions
        )
        summary["diagnostic"] = "Cancelled" if cancelled else "Endpoint became unreachable"

    successful = sum(row["outcome"] == "success" for row in outcomes)
    accounted = len(outcomes)
    summary["coverage"]["fresh_context"] = {
        "expected": 60,
        "accounted": accounted,
        "successful": successful,
    }
    summary["outcomes"] = outcomes
    summary["fresh_context_curve"] = fresh_context_curve(outcomes)
    summary["status"] = (
        "completed"
        if accounted == 60
        and all(row["outcome"] in {"success", "unsupported"} for row in outcomes)
        and all(warmup["exit_code"] == 0 for warmup in summary["warmups"])
        and not summary["tool_failures"]
        else "partial"
    )
    save(directory / "outcomes.json", outcomes)
    save(directory / "summary.json", summary)
    sanitize_files(directory, run_secrets(submission))
    finish_run(directory)
    return read_result(root, run_id)


def assistant_from_raw(directory: Path) -> dict[str, Any] | None:
    """Accumulate the assistant reply from the retained raw export of a single request."""
    raw_export = read_jsonl(directory / "native/profile_export_raw.jsonl")
    if raw_export is None:
        return None
    raw_rows = raw_export[0]
    if len(raw_rows) != 1:
        return None
    raw = raw_rows[0]
    content = ""
    reasoning = ""
    finish = None
    done = False
    for response in raw.get("responses", []):
        for packet in response.get("packets", []):
            if packet.get("name") != "data":
                continue
            if packet["value"] == "[DONE]":
                done = True
                continue
            chunk = json.loads(packet["value"])
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                content += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
                finish = choice.get("finish_reason") or finish
    if raw.get("status") != 200 or finish is None or not done:
        return None
    assistant: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        assistant["reasoning_content"] = reasoning
    return assistant


def growing_structural(session: str, checkpoint: int, call_id: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": f"{session} checkpoint {checkpoint}. Read the log."},
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
        },
        {"role": "tool", "tool_call_id": call_id, "content": ""},
        {
            "role": "user",
            "content": "Summarize the log and explain a method for analyzing it in detail.",
        },
    ]


def prepare_growing_turn(
    turn_dir: Path,
    specification: dict[str, Any],
    request_timeout_seconds: int,
    cancel_marker: Path,
) -> tuple[list[dict[str, Any]] | None, float, str | None]:
    """Size one turn locally with the submitted tokenizer; return (messages, elapsed, error)."""
    turn_dir.mkdir(parents=True, exist_ok=True)
    spec_path = turn_dir / "sizing-spec.json"
    save(spec_path, specification)
    command = TOKENIZER_TOOL + [
        str(Path(__file__).with_name("performance_inputs.py")),
        "--mode",
        "grow",
        "--spec",
        str(spec_path.resolve()),
        "--directory",
        str(turn_dir.resolve()),
    ]
    save(turn_dir / "sizing-command.json", command)
    started = time.monotonic()
    try:
        completed = execute_tool(
            command,
            turn_dir,
            1,
            request_timeout_seconds,
            cancel_marker=cancel_marker,
        )
        (turn_dir / "execution.json").replace(turn_dir / "sizing-execution.json")
    except subprocess.TimeoutExpired as exc:
        return None, time.monotonic() - started, str(exc)
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        return None, elapsed, f"Turn sizing exited {completed.returncode}"
    try:
        row = json.loads((turn_dir / "input.jsonl").read_text())
        return row["extra"]["messages"], elapsed, None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, elapsed, str(exc)


def growing_outcome(
    session: str,
    output_target: int,
    repetition: int,
    turn: int,
    checkpoint: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "condition": session,
        "scenario": "growing-session",
        "session": session,
        "input_tokens": checkpoint,
        "output_target": output_target,
        "repetition": repetition,
        "turn": turn,
        **row,
    }


@finished_on_exit
def execute_growing_session(
    root: Path,
    submission: Submission,
    workload: GrowingSessionWorkload,
    run_id: str | None = None,
    cancel_marker: Path | None = None,
) -> dict[str, Any]:
    run_id, directory = create_run_directory(root, run_id)
    marker = cancel_marker or cancellation_path(directory)
    setup = submission.setup.declared
    save(directory / "setup.json", submission.setup.model_dump())
    save(directory / "workload.json", {"evidence": "run", **workload.model_dump()})
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "evidence": "run",
        "status": "partial",
        "default_workload_complete": False,
        "families": {"performance": "partial", "coding": "missing", "long_context": "missing"},
        "coverage": {"growing_session": {"expected": 60, "accounted": 0, "successful": 0}},
        "cache": {
            "configuration": workload.cache_configuration,
            "observations": workload.cache_observations,
            "request_design": "accumulated live history; cache state is not controlled",
        },
        "warmups": [],
        "construction": {
            "history_source": "live assistant responses plus authored synthetic tool results",
            "sizing_requests": 0,
            "construction_seconds": 0.0,
            "note": (
                "Each warm-up and measured turn is sized locally with the submitted tokenizer "
                "and chat template before a single AIPerf request. sizing_requests and "
                "construction_seconds count those local sizing steps (not endpoint model "
                "requests); no additional unscored endpoint requests are used. Warm-up requests "
                "are excluded from measurements and do not feed session history."
            ),
        },
        "outcomes": [],
        "growing_session_curve": [],
    }
    save(directory / "summary.json", summary)

    context_window = setup["requests"]["context_window"]
    template_kwargs: dict[str, Any] = {}
    for request_group in (setup["requests"]["generation"], setup["requests"]["reasoning"]):
        if "chat_template_kwargs" in request_group:
            template_kwargs = request_group["chat_template_kwargs"]
    request_settings = mapped_request_settings(setup)

    outcomes: list[dict[str, Any]] = []
    construction_requests = 0
    construction_seconds = 0.0
    endpoint_outage = False
    for repetition in range(workload.repetitions):
        if endpoint_outage:
            summary["diagnostic"] = "Endpoint became unreachable"
            break
        if cancellation_requested(marker):
            summary["diagnostic"] = "Cancelled"
            break
        for output_target in workload.output_targets:
            if endpoint_outage:
                summary["diagnostic"] = "Endpoint became unreachable"
                break
            if cancellation_requested(marker):
                summary["diagnostic"] = "Cancelled"
                break
            session = f"growing-session-{output_target}-rep{repetition}"
            session_dir = directory / session
            supported = [
                checkpoint
                for checkpoint in workload.checkpoints
                if checkpoint + output_target <= context_window
            ]
            if supported:
                session_dir.mkdir(parents=True, exist_ok=True)
                warmup_dir = session_dir / "warmup"
                warmup_checkpoint = supported[0]
                warmup_spec = {
                    "tokenizer": setup["model"]["tokenizer"],
                    "chat_template_kwargs": template_kwargs,
                    "request_settings": request_settings,
                    "history": [],
                    "structural": growing_structural(
                        session, warmup_checkpoint, f"log-{session}-warmup"
                    ),
                    "checkpoint": warmup_checkpoint,
                    "output_target": output_target,
                    "nonce": f"{session}-warmup",
                }
                try:
                    _, warmup_elapsed, warmup_error = prepare_growing_turn(
                        warmup_dir,
                        warmup_spec,
                        workload.request_timeout_seconds,
                        marker,
                    )
                except RunCancelled:
                    summary["diagnostic"] = "Cancelled"
                    break
                construction_requests += 1
                construction_seconds += warmup_elapsed
                if warmup_error is None:
                    try:
                        warmup_command = api_key_args(
                            aiperf_command(
                                setup,
                                warmup_dir / "input.jsonl",
                                warmup_dir / "native",
                                1,
                                workload.request_timeout_seconds,
                            )
                        )
                        warmup = execute_tool(
                            warmup_command,
                            warmup_dir,
                            1,
                            workload.request_timeout_seconds,
                            cancel_marker=marker,
                        )
                        warmup_exit: int | str = warmup.returncode
                    except RunCancelled:
                        summary["diagnostic"] = "Cancelled"
                        warmup_exit = "cancelled"
                    except EndpointUnreachable:
                        endpoint_outage = True
                        warmup_exit = "endpoint unavailable"
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        TypeError,
                        subprocess.TimeoutExpired,
                    ) as exc:
                        warmup_exit = f"error: {exc}"
                else:
                    warmup_exit = warmup_error
                summary["warmups"].append(
                    {
                        "session": session,
                        "checkpoint": warmup_checkpoint,
                        "requests": 1,
                        "excluded_from_measurements": True,
                        "exit_code": warmup_exit,
                        "elapsed_seconds": round(warmup_elapsed, 3),
                    }
                )

            if endpoint_outage:
                summary["diagnostic"] = "Endpoint became unreachable"
                break
            if cancellation_requested(marker):
                summary["diagnostic"] = "Cancelled"
                break

            history: list[dict[str, Any]] = []
            for turn, checkpoint in enumerate(workload.checkpoints):
                if checkpoint not in supported:
                    outcomes.append(
                        growing_outcome(
                            session,
                            output_target,
                            repetition,
                            turn,
                            checkpoint,
                            {
                                "outcome": "unsupported",
                                "diagnostic": "Input plus output exceeds declared context window",
                                "metrics": {},
                            },
                        )
                    )
                    continue
                session_dir.mkdir(parents=True, exist_ok=True)
                turn_dir = session_dir / f"turn-{checkpoint}"
                turn_spec = {
                    "tokenizer": setup["model"]["tokenizer"],
                    "chat_template_kwargs": template_kwargs,
                    "request_settings": request_settings,
                    "history": history,
                    "structural": growing_structural(
                        session, checkpoint, f"log-{session}-turn{turn}"
                    ),
                    "checkpoint": checkpoint,
                    "output_target": output_target,
                    "nonce": f"{session}-turn{turn}",
                }
                try:
                    messages, sizing_elapsed, sizing_error = prepare_growing_turn(
                        turn_dir,
                        turn_spec,
                        workload.request_timeout_seconds,
                        marker,
                    )
                except RunCancelled:
                    summary["diagnostic"] = "Cancelled"
                    break
                construction_requests += 1
                construction_seconds += sizing_elapsed
                if sizing_error is not None:
                    outcomes.append(
                        growing_outcome(
                            session,
                            output_target,
                            repetition,
                            turn,
                            checkpoint,
                            {"outcome": "failed", "diagnostic": sizing_error, "metrics": {}},
                        )
                    )
                    save(directory / "outcomes.json", outcomes)
                try:
                    command = api_key_args(
                        aiperf_command(
                            setup,
                            turn_dir / "input.jsonl",
                            turn_dir / "native",
                            1,
                            workload.request_timeout_seconds,
                        )
                    )
                    measured = execute_tool(
                        command,
                        turn_dir,
                        1,
                        workload.request_timeout_seconds,
                        cancel_marker=marker,
                    )
                    row = normalize_many(turn_dir, checkpoint, output_target, 1)[0]
                    if measured.returncode != 0:
                        row = {
                            "outcome": "failed",
                            "diagnostic": f"Tool exited {measured.returncode}",
                            "metrics": {},
                            "native_outcome": row,
                        }
                    if row["outcome"] == "success":
                        if messages is None:
                            row = {
                                "outcome": "failed",
                                "diagnostic": "Turn preparation produced no messages",
                                "metrics": {},
                                "native_outcome": row,
                            }
                        else:
                            assistant = assistant_from_raw(turn_dir)
                            if assistant is None:
                                row = {
                                    "outcome": "failed",
                                    "diagnostic": "No usable assistant response in native export",
                                    "metrics": {},
                                    "native_outcome": row,
                                }
                            else:
                                history = list(messages) + [assistant]
                    outcomes.append(
                        growing_outcome(session, output_target, repetition, turn, checkpoint, row)
                    )
                except RunCancelled:
                    retained = normalize_many(turn_dir, checkpoint, output_target, 1)[0]
                    summary["diagnostic"] = "Cancelled"
                    outcomes.append(
                        growing_outcome(
                            session,
                            output_target,
                            repetition,
                            turn,
                            checkpoint,
                            retained
                            if retained["outcome"] == "success"
                            else {
                                "outcome": "cancelled",
                                "diagnostic": "Cancelled",
                                "metrics": {},
                                "native_outcome": retained,
                            },
                        )
                    )
                    break
                except EndpointUnreachable as exc:
                    endpoint_outage = True
                    outcomes.append(
                        growing_outcome(
                            session,
                            output_target,
                            repetition,
                            turn,
                            checkpoint,
                            {"outcome": "failed", "diagnostic": str(exc), "metrics": {}},
                        )
                    )
                    break
                except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
                    outcomes.append(
                        growing_outcome(
                            session,
                            output_target,
                            repetition,
                            turn,
                            checkpoint,
                            {"outcome": "failed", "diagnostic": str(exc), "metrics": {}},
                        )
                    )
                save(directory / "outcomes.json", outcomes)
            if cancellation_requested(marker):
                summary["diagnostic"] = "Cancelled"
                break
            if endpoint_outage:
                summary["diagnostic"] = "Endpoint became unreachable"
                break
    summary["construction"]["sizing_requests"] = construction_requests
    summary["construction"]["construction_seconds"] = round(construction_seconds, 3)
    successful = sum(row["outcome"] == "success" for row in outcomes)
    accounted = len(outcomes)
    summary["coverage"]["growing_session"] = {
        "expected": 60,
        "accounted": accounted,
        "successful": successful,
    }
    summary["outcomes"] = outcomes
    summary["growing_session_curve"] = scenario_curve("growing-session", outcomes)
    summary["status"] = (
        "completed"
        if accounted == 60
        and all(row["outcome"] in {"success", "unsupported"} for row in outcomes)
        and all(
            isinstance(warmup["exit_code"], int) and warmup["exit_code"] == 0
            for warmup in summary["warmups"]
        )
        else "partial"
    )
    save(directory / "outcomes.json", outcomes)
    save(directory / "summary.json", summary)
    sanitize_files(directory, run_secrets(submission))
    finish_run(directory)
    return read_result(root, run_id)


SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
)


def mapped_request_settings(setup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The outgoing request settings: generation merged with reasoning,
    renamed per the serving engine's delivered mapping.

    llama.cpp renames the shared repeated-penalty setting to its own
    ``repeat_penalty``; vLLM, SGLang, ExLlamaV3, and engines without a
    delivered mapping keep the shared names. The retained setup record always keeps
    the shared per-run names; only the outgoing requests and their
    prepared inputs are renamed.
    """
    from . import compatibility

    settings = {**setup["requests"]["generation"], **setup["requests"]["reasoning"]}
    return compatibility.apply_setting_mappings(settings, compatibility.setting_mappings(setup))


def aiperf_request_settings(
    setup: dict[str, dict[str, Any]], settings: dict[str, Any]
) -> dict[str, Any]:
    """Add TabbyAPI's opt-in usage accounting to outgoing AIPerf inputs."""
    if str(setup["serving"]["engine"]).strip().lower() in {"exllamav3", "exl3", "exllama v3"}:
        return {**settings, "stream_options": {"include_usage": True}}
    return settings


def resolve_run_config(submission: Submission) -> Setup:
    """Resolve per-launch choices into the retained setup record.

    The sampling parameters merge over the declared ``requests.generation``
    (a null records no override, a value overrides, and non-sampling keys such
    as ``seed`` are preserved). ``reasoning_effort`` merges into
    ``requests.reasoning["chat_template_kwargs"]`` alongside any existing
    template kwargs. ``off`` also disables ``enable_thinking`` there; the
    outbound engine mapping removes the effort marker while retaining it in the
    setup record. The result is re-validated so a conflicting launch choice
    (overlap, reserved key, malformed object) fails before execution rather
    than silently dropping a requested setting.
    """
    if submission.run_config is None:
        raise ValueError("run_config is required at launch")
    run_config = submission.run_config

    setup = submission.setup.model_copy(deep=True)
    requests = setup.declared["requests"]
    generation = dict(requests["generation"])
    reasoning = dict(requests["reasoning"])

    for key in SAMPLING_KEYS:
        value = getattr(run_config, key)
        if value is None:
            generation.pop(key, None)
        else:
            generation[key] = value

    template_kwargs = dict(reasoning.get("chat_template_kwargs") or {})
    template_kwargs["reasoning_effort"] = run_config.reasoning_effort
    if run_config.reasoning_effort == "off":
        reasoning.pop("reasoning_effort", None)
        reasoning.pop("enable_thinking", None)
        template_kwargs["enable_thinking"] = False
    if template_kwargs:
        reasoning["chat_template_kwargs"] = template_kwargs
    else:
        reasoning.pop("chat_template_kwargs", None)

    requests["generation"] = generation
    requests["reasoning"] = reasoning
    return Setup.model_validate({"declared": setup.declared, "observed": setup.observed})


def execute(
    root: Path,
    submission: Submission,
    run_id: str | None = None,
    setup_directory: Path | None = None,
    restrict_paths: bool = False,
) -> dict[str, Any]:
    submission.setup = resolve_run_config(submission)
    # The final settings are validated against the endpoint before any scored
    # work: an unsupported setting or a rejected key stops the launch with an
    # actionable error instead of being silently dropped or run blind. Engines
    # without a delivered mapping keep their existing pass-through.
    from . import compatibility, quality_preparation, setup_state, speed_preparation

    evidence: dict[str, Any] | None = None
    if compatibility.gated_engine(submission.setup.declared):
        evidence = compatibility.launch_gate(submission.setup.declared)
    # The speed profiles launch from prepared inputs: when a setup state exists
    # in the given directory the gate refuses to launch until the pinned tools
    # and tokenizer still verify against the submitted configuration, so stale
    # inputs cannot pass as ready. The sampled-RULER profile sizes its prompts
    # with the same tokenizer, so it is gated and swapped the same way. Without
    # a state directory (or state) the launch keeps the live-sizing path. Either
    # way the decision is retained with the run.
    speed_evidence: dict[str, Any] | None = None
    if setup_directory is not None and isinstance(
        submission.workload,
        (
            FreshContextWorkload,
            GrowingSessionWorkload,
            SampledRulerWorkload,
            DefaultSuiteWorkload,
        ),
    ):
        state = setup_state.load_state(setup_directory)
        speed_evidence = speed_preparation.launch_gate(
            submission.setup.model_dump(), state, setup_directory
        )
        if speed_evidence.get("prepared") is True:
            # Size against the verified local tokenizer instead of re-fetching
            # the declared source at launch; the resolved record is re-validated.
            prepared_path = str(speed_preparation.tokenizer_directory(setup_directory))
            submission.setup.declared["model"]["tokenizer"] = prepared_path
            submission.setup = Setup.model_validate(
                {"declared": submission.setup.declared, "observed": submission.setup.observed}
            )
    # The coding workload launches from prepared inputs: when a setup state
    # exists in the given directory the gate verifies the prepared dataset
    # copy and grading container still hold against the pinned inputs, so a
    # tampered or stale resource cannot pass as ready. Without a state
    # directory (or state) the launch keeps the submitted dataset path.
    # Either way the decision is retained with the run.
    quality_evidence: dict[str, Any] | None = None
    if setup_directory is not None and isinstance(
        submission.workload, (HumanEvalWorkload, DefaultSuiteWorkload)
    ):
        state = setup_state.load_state(setup_directory)
        quality_evidence = quality_preparation.launch_gate(state, setup_directory)
        if quality_evidence.get("prepared") is True:
            # Grade against the verified local dataset copy instead of the
            # submitted path (nested under the suite's coding phase).
            prepared_problems = str(quality_preparation.problems_path(setup_directory))
            if isinstance(submission.workload, DefaultSuiteWorkload):
                submission.workload.coding.problems_path = prepared_problems
            else:
                submission.workload.problems_path = prepared_problems
    # The long-context workload launches from a prepared RULER source: when a
    # setup state exists in the given directory the gate verifies the prepared
    # checkout still holds at the pinned revision with its data manifest, so a
    # tampered or stale resource cannot pass as ready. Without a state directory
    # (or state) the launch keeps the submitted source path. The 117 prompts are
    # (re)prepared by the long-context runner against the final resolved
    # configuration, so a changed tokenizer, template, or reasoning choice
    # re-prepares before scored execution. Either way the decision is retained
    # with the run.
    long_context_evidence: dict[str, Any] | None = None
    if setup_directory is not None and isinstance(
        submission.workload, (SampledRulerWorkload, DefaultSuiteWorkload)
    ):
        from . import ruler_preparation

        state = setup_state.load_state(setup_directory)
        long_context_evidence = ruler_preparation.launch_gate(state, setup_directory)
        if long_context_evidence.get("prepared") is True:
            # Generate from the verified local checkout instead of the
            # submitted path (nested under the suite's long-context phase).
            prepared_source = str(ruler_preparation.source_path(setup_directory))
            if isinstance(submission.workload, DefaultSuiteWorkload):
                submission.workload.long_context.source_path = prepared_source
            else:
                submission.workload.source_path = prepared_source
    if restrict_paths and setup_directory is not None:
        # Untrusted (API) submissions may only read prepared inputs under the setup directory.
        workload = submission.workload
        supplied = {
            HumanEvalWorkload: lambda w: [w.problems_path],
            SampledRulerWorkload: lambda w: [w.source_path],
            DefaultSuiteWorkload: lambda w: [w.coding.problems_path, w.long_context.source_path],
        }.get(type(workload))
        for value in supplied(workload) if supplied else []:
            try:
                Path(value).resolve().relative_to(setup_directory.resolve())
            except ValueError:
                raise ValueError(
                    "problems_path and source_path must be prepared inputs in the setup directory"
                ) from None
    if isinstance(submission.workload, DefaultSuiteWorkload):
        from .default_suite import execute_default_suite

        result = execute_default_suite(root, submission, submission.workload, run_id)
    elif isinstance(submission.workload, HumanEvalWorkload):
        from .humaneval import execute_coding

        result = execute_coding(root, submission, submission.workload, run_id)
    elif isinstance(submission.workload, SampledRulerWorkload):
        from .ruler_runs import execute_ruler

        result = execute_ruler(root, submission, submission.workload, run_id)
    elif isinstance(submission.workload, FreshContextWorkload):
        result = execute_fresh_context(root, submission, submission.workload, run_id)
    elif isinstance(submission.workload, GrowingSessionWorkload):
        result = execute_growing_session(root, submission, submission.workload, run_id)
    else:
        result = execute_single(root, submission, submission.workload, run_id)
    if evidence is not None:
        save(root / str(result["run_id"]) / "compatibility.json", evidence)
    if speed_evidence is not None:
        save(root / str(result["run_id"]) / "speed-preparation.json", speed_evidence)
    if quality_evidence is not None:
        save(root / str(result["run_id"]) / "quality-preparation.json", quality_evidence)
    if long_context_evidence is not None:
        save(root / str(result["run_id"]) / "long-context-preparation.json", long_context_evidence)
    return result
