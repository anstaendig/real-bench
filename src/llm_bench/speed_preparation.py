"""Preparation of the speed workloads: pinned tools, tokenizer, and the launch gate.

The speed workloads size every prompt locally with the declared tokenizer and
chat template, so a launch needs two reusable resources that setup prepares
ahead of time:

- the pinned performance tools (``aiperf`` plus ``transformers`` at the
  declared benchmark versions), verified by loading them in the pinned
  environment and reading the installed versions back;
- a local copy of the declared tokenizer, saved with a sha256 manifest so the
  launch can verify the download instead of re-fetching it.

Everything here is ordinary ``subprocess.run`` against the project's pinned
environment (already cached by ``uv``); no model weights ever reach the runner
and no estimate is fabricated when a step is incomplete. The launch gate
(:func:`launch_gate`) enforces the prepared state for the two speed profile
workloads before any scored work; a submission that was prepared under a
different configuration cannot pass as ready.
"""

import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .ordinary_runs import FRESH_CHECKPOINTS, OUTPUT_TARGETS, TOKENIZER_TOOL, redact
from .setup_state import Preparation, setup_record, state_path, value_hash

PERFORMANCE_TOOL_TIMEOUT = 600
TOKENIZER_FETCH_TIMEOUT = 3600

MANIFEST_NAME = "manifest.json"
PERFORMANCE_TOOLS_EVIDENCE = "performance-tools.json"
SPEED_DIRECTORY = "speed"
SUBMISSIONS_DIRECTORY = "submissions"

VERSION_PIN = re.compile(r"[0-9][0-9A-Za-z.+!-]*")
PROBE_COMMAND = (
    "import importlib.metadata as m; print(m.version('aiperf')); print(m.version('transformers'))"
)


def speed_directory(directory: Path) -> Path:
    return directory / SPEED_DIRECTORY


def tokenizer_directory(directory: Path) -> Path:
    return speed_directory(directory) / "tokenizer"


def performance_tools_inputs(record: dict[str, Any]) -> tuple[Any, ...]:
    benchmark = record.get("declared", {}).get("benchmark") or {}
    return (
        benchmark.get("tool"),
        benchmark.get("version"),
        benchmark.get("transformers_version"),
    )


def _benchmark_pins(record: dict[str, Any]) -> tuple[str, str]:
    benchmark = record.get("declared", {}).get("benchmark") or {}
    pins = (str(benchmark["version"]), str(benchmark["transformers_version"]))
    for pin in pins:
        if not VERSION_PIN.fullmatch(pin):
            raise ValueError(f"invalid benchmark version {pin!r}: expected a plain version number")
    return pins


def _performance_command(pins: tuple[str, str]) -> list[str]:
    return [
        "uv",
        "run",
        "--with",
        f"aiperf=={pins[0]}",
        "--with",
        f"transformers=={pins[1]}",
        "python",
        "-c",
        PROBE_COMMAND,
    ]


def _helper_path() -> Path:
    from . import speed_prepare

    return Path(speed_prepare.__file__)


def verify_performance_tools(record: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Verify the pinned performance tools load and match the declared pins.

    Loads both packages in the pinned environment and reads the installed
    versions back; the evidence is retained under the state directory. Raises
    ``ValueError`` naming the mismatch when the environment does not provide
    the declared versions.
    """
    pins = _benchmark_pins(record)
    command = _performance_command(pins)
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=PERFORMANCE_TOOL_TIMEOUT
        )
    except FileNotFoundError:
        # The pinned environment is not on the PATH: name the missing
        # prerequisite so the preparation note is actionable, not a raw errno.
        raise ValueError(
            f"the pinned environment {command[0]} was not found on the PATH; install uv"
        ) from None
    except subprocess.TimeoutExpired:
        raise ValueError(
            f"loading the pinned performance tools timed out after "
            f"{PERFORMANCE_TOOL_TIMEOUT} seconds"
        ) from None
    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    observed = (
        lines[0] if len(lines) > 0 else None,
        lines[1] if len(lines) > 1 else None,
    )
    evidence: dict[str, Any] = {
        "command": command,
        "exit_code": completed.returncode,
        "observed": {"aiperf": observed[0], "transformers": observed[1]},
        "declared": {"aiperf": pins[0], "transformers": pins[1]},
    }
    if completed.returncode != 0 or observed != pins:
        raise ValueError(
            "pinned performance tools do not match the declared benchmark "
            f"(declared aiperf={pins[0]} transformers={pins[1]}, "
            f"observed aiperf={observed[0]} transformers={observed[1]}, "
            f"exit code {completed.returncode}); check the uv environment and network access"
        )
    target = speed_directory(directory) / PERFORMANCE_TOOLS_EVIDENCE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return evidence


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_verified_tokenizer(directory: Path, source: str) -> dict[str, Any] | None:
    """Read the fetched tokenizer manifest and verify every file on disk.

    Returns the manifest when the saved copy still verifies (source matches,
    every listed file exists and hashes to its recorded value); ``None``
    otherwise. Reading never repairs or refetches.
    """
    root = tokenizer_directory(directory)
    try:
        manifest = json.loads((root / MANIFEST_NAME).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("source") != source:
        return None
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return None
    try:
        for relative, expected in files.items():
            path = root / relative
            if not path.is_file() or path.is_symlink():
                return None
            if _hash_file(path) != expected:
                return None
    except OSError:
        return None
    return manifest


def prepare_tokenizer(record: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Fetch the declared tokenizer into the state directory with a manifest.

    Reuses a verified download: when the saved manifest matches the declared
    source and every file still hashes to its recorded value, no fetch occurs.
    The fetch runs in the pinned environment, so a tokenizer the performance
    tools cannot load fails the preparation instead of passing as ready.
    """
    source = str(record["declared"]["model"]["tokenizer"])
    verified = read_verified_tokenizer(directory, source)
    if verified is not None:
        return verified
    command = [
        *TOKENIZER_TOOL,
        str(_helper_path()),
        f"--source={source}",
        f"--directory={tokenizer_directory(directory)}",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=TOKENIZER_FETCH_TIMEOUT
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"tokenizer preparation failed: {exc}") from exc
    detail = (completed.stderr or completed.stdout or "").strip()
    if completed.returncode == 3:
        message = "tokenizer preparation needs a human-only prerequisite"
        if detail:
            message = f"{message}: {detail}"
        raise ValueError(message)
    if completed.returncode != 0:
        raise ValueError(f"tokenizer preparation failed: {detail or 'unknown error'}")
    verified = read_verified_tokenizer(directory, source)
    if verified is None:
        raise ValueError("tokenizer preparation finished but its manifest could not be verified")
    return verified


def run_speed_preparation(state: Any, directory: Path) -> None:
    """Prepare the speed workload resources for a complete setup state.

    Each step persists as it goes: the entry moves to in-progress before the
    (long) subprocess starts, so an interrupted session resumes the same
    preparation and verified downloads are never re-downloaded. A failure
    keeps the entry honest (pending with an actionable note) instead of
    reporting complete; the session itself is not failed by a preparation
    problem, and the prepared submissions exist only while both speed
    preparations pass.
    """
    from .setup_state import save_state

    record = setup_record(state)
    run_step(
        state,
        directory,
        "performance_tools",
        record,
        verify_performance_tools,
        verify_performance_tools,
    )
    run_step(state, directory, "tokenizer", record, prepare_tokenizer, _verify_tokenizer)
    both_complete = all(
        state.preparations[name].status == "complete"
        and not state.preparations[name].needs_revalidation
        for name in ("performance_tools", "tokenizer")
    )
    if both_complete:
        write_prepared_submissions(state, directory)
    else:
        _remove_prepared_submissions(directory)
    save_state(state, directory)


def _verify_tokenizer(record: dict[str, Any], directory: Path) -> None:
    """Lightly re-check the completed tokenizer without fetching.

    Only hashes the saved files against their manifest; raises
    ``ValueError`` when the verified resource no longer holds, which the
    caller turns into a re-preparation.
    """
    source = str(record["declared"]["model"]["tokenizer"])
    if read_verified_tokenizer(directory, source) is None:
        raise ValueError("the prepared tokenizer no longer verifies")


def run_step(
    state: Any,
    directory: Path,
    name: str,
    record: dict[str, Any],
    step: Callable[..., Any],
    verify: Callable[[dict[str, Any], Path], Any],
) -> None:
    """Advance one preparation step with honest, resumable state transitions.

    A completed step is re-verified cheaply first (``verify``); only when the
    verified resource no longer holds does the step re-run. The entry moves
    to in-progress and is persisted before the (possibly long) subprocess
    starts, so an interruption leaves a resumable in-progress entry. A
    failure keeps the entry honest (pending with an actionable note) instead
    of failing the session; a successful step records complete with the note
    cleared.
    """
    from .setup_state import save_state

    entry = state.preparations[name]
    if entry.status == "complete" and not entry.needs_revalidation:
        try:
            verify(record, directory)
        except (ValueError, OSError, subprocess.TimeoutExpired):
            # The verified resource no longer holds: fall through to the
            # in-progress path, which re-prepares it and persists the
            # transition before the (possibly long) subprocess starts, so an
            # interruption leaves a resumable in-progress entry.
            pass
        else:
            entry.note = ""
            save_state(state, directory)
            return
    entry.status = "in-progress"
    entry.note = "in progress"
    save_state(state, directory)
    try:
        step(record, directory)
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        state.preparations[name] = Preparation(
            status="pending",
            hash=entry.hash,
            needs_revalidation=False,
            note=str(exc),
        )
        save_state(state, directory)
        return
    state.preparations[name] = Preparation(
        status="complete",
        hash=entry.hash,
        needs_revalidation=False,
        note="",
    )
    save_state(state, directory)


def speed_readiness(state: Any, directory: Path | None = None) -> dict[str, bool]:
    """Report the speed-preparation state without doing any work.

    ``speed_ready`` is true only when both speed preparations are complete
    against the current configuration and the downloaded tokenizer still
    verifies on disk. ``default_suite_ready`` additionally requires the
    remaining workloads' preparations (datasets, grading container, long
    context); nothing unfinished is advertised as ready.
    """
    record = setup_record(state)
    declared = record["declared"]
    preparations = state.preparations
    tools = preparations.get("performance_tools")
    tools_complete = (
        tools is not None
        and tools.status == "complete"
        and not tools.needs_revalidation
        and tools.hash == value_hash(*performance_tools_inputs(record))
    )
    tokenizer = preparations.get("tokenizer")
    tokenizer_inputs = (
        declared["model"]["tokenizer"],
        declared["model"]["template_overrides"],
    )
    tokenizer_complete = (
        tokenizer is not None
        and tokenizer.status == "complete"
        and not tokenizer.needs_revalidation
        and tokenizer.hash == value_hash(*tokenizer_inputs)
    )
    if tokenizer_complete and directory is not None:
        tokenizer_complete = (
            read_verified_tokenizer(directory, str(declared["model"]["tokenizer"])) is not None
        )
    speed_ready = tools_complete and tokenizer_complete
    suite_ready = speed_ready
    for name in ("datasets", "grading_container", "long_context"):
        entry = preparations.get(name)
        suite_ready = suite_ready and entry is not None and entry.status == "complete"
    if suite_ready and directory is not None:
        # The sampled-RULER checkout is fetched, so a deleted or tampered
        # resource must not stay advertised ready.
        from .ruler_preparation import read_verified_source

        suite_ready = read_verified_source(directory) is not None
    return {"speed_ready": speed_ready, "default_suite_ready": suite_ready}


def speed_workload(profile: str, scenario: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "scenario": scenario,
        "checkpoints": list(FRESH_CHECKPOINTS),
        "output_targets": list(OUTPUT_TARGETS),
        "repetitions": 5,
        "warmup_repetitions": 1,
        "request_timeout_seconds": 900,
        "cache_configuration": {},
        "cache_observations": [],
    }


def prepared_submission_paths(directory: Path) -> dict[str, Path]:
    return {
        "fresh-context": directory / SUBMISSIONS_DIRECTORY / "fresh-context.json",
        "growing-session": directory / SUBMISSIONS_DIRECTORY / "growing-session.json",
    }


def write_prepared_submissions(state: Any, directory: Path) -> None:
    """Write the fixed speed-workload submissions once both preparations pass.

    The submissions are the workloads exactly as the speed profiles define
    them: fixed checkpoints, output targets, repetitions, and warm-up
    exclusion, with ``run_config`` left for the launch boundary to complete.
    They carry the redacted setup record, never a secret.
    """
    record = redact(setup_record(state))
    for name, profile in (
        ("fresh-context", "fresh-context-speed"),
        ("growing-session", "growing-session-speed"),
    ):
        path = prepared_submission_paths(directory)[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"setup": record, "workload": speed_workload(profile, name), "run_config": None}
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _remove_prepared_submissions(directory: Path) -> None:
    for path in prepared_submission_paths(directory).values():
        if path.is_file():
            path.unlink()


def suite_submission_path(directory: Path) -> Path:
    return directory / SUBMISSIONS_DIRECTORY / "default-suite.json"


def write_suite_submission(state: Any, directory: Path) -> None:
    """Compose the complete default-suite submission once every workload
    resource is prepared.

    The four fixed phases are the same workloads the individual prepared
    submissions carry: fixed speed matrices, the pinned HumanEval+ dataset
    from the prepared copy, and the sampled-RULER prompts from the prepared
    source checkout. ``run_config`` is left for the launch boundary to
    complete, and the record is redacted so no secret is persisted. The
    launch gate resolves the nested dataset and source paths against the
    verified prepared resources.
    """
    from . import quality_preparation, ruler_preparation

    record = redact(setup_record(state))
    path = suite_submission_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    workload = {
        "profile": "default-suite",
        "fresh_context": speed_workload("fresh-context-speed", "fresh-context"),
        "growing_session": speed_workload("growing-session-speed", "growing-session"),
        "coding": quality_preparation.coding_workload(directory),
        "long_context": ruler_preparation.long_context_workload(directory),
    }
    payload = {"setup": record, "workload": workload, "run_config": None}
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def remove_suite_submission(directory: Path) -> None:
    path = suite_submission_path(directory)
    if path.is_file():
        path.unlink()


def launch_gate(
    record: dict[str, Any], state: Any | None, directory: Path | None
) -> dict[str, Any]:
    """The speed launch gate: prepared inputs only, for the speed profiles.

    Runs on the resolved setup record (the ``run_config`` already merged in).
    Without a complete setup state in the given directory there is nothing
    prepared to enforce, so the launch keeps the existing live-sizing path.
    With a complete state, the gate refuses to launch until the prepared
    resources still match the configuration that submitted them: it re-runs
    the pinned version probe, re-verifies the tokenizer manifest on disk, and
    compares each preparation's hash against the resolved record, so stale
    inputs cannot pass as ready. Any refusal raises ``ValueError`` before any
    work directory is created; the returned evidence is retained with the run.
    """
    from .setup_state import critical_complete

    if state is None or directory is None or not state_path(directory).is_file():
        return {"prepared": False, "reason": "no setup state; live sizing"}
    if not critical_complete(setup_record(state), state.discovery):
        # The setup is still incomplete: there is nothing prepared to enforce,
        # so this launch keeps the live-sizing path instead of refusing.
        return {"prepared": False, "reason": "setup state incomplete; live sizing"}
    declared = record["declared"]
    tokenizer_source = str(declared["model"]["tokenizer"])
    problems: list[str] = []
    tools = state.preparations.get("performance_tools")
    if tools is None or tools.status != "complete" or tools.needs_revalidation:
        problems.append(
            "the pinned performance tools are not verified for this configuration"
            + (f" ({tools.note})" if tools is not None and tools.note else "")
        )
    elif tools.hash != value_hash(*performance_tools_inputs(record)):
        problems.append("the declared benchmark pins changed since the tools were verified")
    tokenizer = state.preparations.get("tokenizer")
    tokenizer_inputs = (tokenizer_source, declared["model"]["template_overrides"])
    if tokenizer is None or tokenizer.status != "complete" or tokenizer.needs_revalidation:
        problems.append(
            "the tokenizer is not prepared for this configuration"
            + (f" ({tokenizer.note})" if tokenizer is not None and tokenizer.note else "")
        )
    elif tokenizer.hash != value_hash(*tokenizer_inputs):
        problems.append("the declared tokenizer or template changed since it was prepared")
    if problems:
        raise ValueError(
            "speed workload launch refused: "
            + "; ".join(problems)
            + ". Re-run bench setup to prepare, and no work was started."
        )
    try:
        verify_performance_tools(record, directory)
    except ValueError:
        raise ValueError(
            "speed workload launch refused: the pinned performance tools no longer load "
            "in the pinned environment. Re-run bench setup; no work was started."
        ) from None
    manifest = read_verified_tokenizer(directory, tokenizer_source)
    if manifest is None:
        raise ValueError(
            "speed workload launch refused: the prepared tokenizer no longer verifies "
            "against its manifest. Re-run bench setup; no work was started."
        )
    return {
        "prepared": True,
        "performance_tools": {"declared": list(_benchmark_pins(record))},
        "tokenizer": {
            "source": tokenizer_source,
            "path": str(tokenizer_directory(directory)),
            "files": len(manifest["files"]),
            "total_bytes": manifest["total_bytes"],
        },
    }
