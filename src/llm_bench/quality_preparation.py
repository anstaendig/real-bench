"""Preparation of the quality workloads: the pinned dataset, the grading
container, and the coding launch gate.

The HumanEval+ coding workload grades every generated solution in the pinned
EvalPlus Docker image, so a launch needs two reusable resources that setup
prepares ahead of time:

- the pinned HumanEval+ v0.1.10 dataset (committed to the repository),
  copied into the state directory and verified against its pinned
  decompressed sha256, row count, and task ids;
- the pinned EvalPlus grading container, present on a Docker daemon whose
  linux/amd64 runtime is proven by running the image once.

Everything here is ordinary ``subprocess.run`` against the Docker CLI; no
network fetch happens except the one-time image pull, and a verified
resource is reused, never re-fetched. The launch gate (:func:`launch_gate`)
verifies — it does not fetch — and refuses to launch the coding workload
until the prepared resources still hold; a refusal names the gap and starts
no work.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .humaneval import DOCKER_PREREQUISITE_INSTRUCTIONS, IMAGE, SOURCE_SHA256, TASK_IDS
from .ordinary_runs import redact
from .setup_state import setup_record, state_path, value_hash

QUALITY_DIRECTORY = "quality"
PROBLEMS_FILE = "HumanEvalPlus-v0.1.10.jsonl.gz"
PROBLEMS_EVIDENCE = "problems.json"
CONTAINER_EVIDENCE = "grading-container.json"
SUBMISSIONS_DIRECTORY = "submissions"

DOCKER_COMMAND_TIMEOUT = 300
DOCKER_PULL_TIMEOUT = 3600

#: The pinned HumanEval+ dataset shipped inside the package (the setup source
#: of truth); its sha256 is verified when prepared.
PINNED_PROBLEMS = Path(__file__).resolve().parent / "datasets" / PROBLEMS_FILE


def problems_path(directory: Path) -> Path:
    """The prepared, verified problems file a coding launch runs against."""
    return Path(directory) / QUALITY_DIRECTORY / PROBLEMS_FILE


def read_problems(directory: Path) -> list[dict[str, Any]] | None:
    """Return the decoded problems when the prepared file is verified.

    Verification is the load-bearing check: the compressed archive must
    decompress to the pinned sha256, hold the 164 HumanEval+ tasks, and
    carry the pinned task ids in order. ``None`` means the file is missing,
    unreadable, or no longer matches the pins.
    """
    path = problems_path(directory)
    try:
        raw = gzip.decompress(path.read_bytes())
    except (OSError, ValueError, EOFError):
        # OSError covers a missing or unreadable file; ValueError covers a
        # corrupt gzip member; EOFError covers a truncated archive.
        return None
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        return None
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != len(TASK_IDS):
        return None
    if [row.get("task_id") for row in rows] != TASK_IDS:
        return None
    return rows


def _problems_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source": str(PINNED_PROBLEMS),
        "source_sha256": SOURCE_SHA256,
        "version": "HumanEval+ v0.1.10",
        "tasks": len(rows),
        "first_task_id": rows[0]["task_id"],
        "last_task_id": rows[-1]["task_id"],
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def prepare_datasets(record: dict[str, Any], directory: Path) -> None:
    """Copy the pinned dataset into the state directory and verify it.

    Raises ``ValueError`` when the committed source file is missing or the
    copy no longer matches the pinned decompressed sha256, row count, and
    task ids, so the preparation loop records an honest pending note.
    """
    if not PINNED_PROBLEMS.is_file():
        raise ValueError(
            f"the pinned HumanEval+ dataset is missing from the repository: {PINNED_PROBLEMS}"
        )
    target = problems_path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(PINNED_PROBLEMS.read_bytes())
    rows = read_problems(directory)
    if rows is None:
        raise ValueError(
            "the prepared HumanEval+ dataset no longer verifies against its "
            "pinned sha256, row count, and task ids"
        )
    (directory / QUALITY_DIRECTORY / PROBLEMS_EVIDENCE).write_text(
        json.dumps(_problems_evidence(rows), indent=2) + "\n"
    )


def _verify_datasets(record: dict[str, Any], directory: Path) -> None:
    """Lightly re-check the completed dataset without copying again.

    Only reads the saved file and re-verifies it against the pins; raises
    ``ValueError`` when the verified resource no longer holds, which the
    caller turns into a re-preparation.
    """
    if read_problems(directory) is None:
        raise ValueError("the prepared HumanEval+ dataset no longer verifies")


def _docker(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        # Docker is not installed on the PATH: name the missing prerequisite
        # so the preparation note is actionable, not a raw errno.
        raise _grading_failure(
            "the Docker CLI was not found on the PATH; install Docker."
        ) from None
    except subprocess.TimeoutExpired as exc:
        raise _grading_failure(
            f"the Docker command {command[1] if len(command) > 1 else 'run'} timed "
            f"out after {exc.timeout} seconds"
        ) from None


def _grading_failure(detail: str) -> ValueError:
    return ValueError(detail + " " + DOCKER_PREREQUISITE_INSTRUCTIONS)


def _require_daemon() -> None:
    daemon = _docker(["docker", "version"], DOCKER_COMMAND_TIMEOUT)
    if daemon.returncode != 0:
        raise _grading_failure(
            f"Docker is not available: {(daemon.stderr or daemon.stdout).strip()!r}"
        )


def _grading_present() -> bool:
    """Whether the grading image is present locally (no pull)."""
    inspect = _docker(["docker", "image", "inspect", IMAGE], DOCKER_COMMAND_TIMEOUT)
    if inspect.returncode != 0:
        return False
    try:
        return bool(json.loads(inspect.stdout or "[]"))
    except ValueError:
        return False


def _probe_machine() -> str:
    """Run the image once and report the machine it actually runs on."""
    probe = _docker(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            f"real-bench-quality-{uuid.uuid4().hex}",
            "--platform",
            "linux/amd64",
            IMAGE,
            "uname",
            "-m",
        ],
        DOCKER_COMMAND_TIMEOUT,
    )
    if probe.returncode != 0:
        raise _grading_failure(
            "the grading container would not run under linux/amd64: "
            f"{(probe.stderr or probe.stdout).strip()!r}"
        )
    machine = (probe.stdout or "").strip()
    if machine != "x86_64":
        raise _grading_failure(
            f"the grading container requires linux/amd64 but the image ran as "
            f"machine '{machine}', not x86_64"
        )
    return machine


def _container_evidence(pulled: bool, machine: str) -> dict[str, Any]:
    return {
        "source": str(IMAGE),
        "platform": "linux/amd64",
        "machine": machine,
        "pulled": pulled,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def prepare_grading_container(record: dict[str, Any], directory: Path) -> None:
    """Make the pinned grading container present and run linux/amd64.

    Uses the Docker CLI directly: the daemon must answer, the pinned image
    must be present (pulled once, for linux/amd64, when absent), and running
    the image must report x86_64. Raises an actionable ``ValueError`` when
    Docker is not usable, so the preparation loop records an honest pending
    note instead of failing the session.
    """
    _require_daemon()
    pulled = False
    if not _grading_present():
        pull = _docker(["docker", "pull", "--platform", "linux/amd64", IMAGE], DOCKER_PULL_TIMEOUT)
        if pull.returncode != 0:
            raise _grading_failure(
                f"pulling the grading container failed: {(pull.stderr or pull.stdout).strip()!r}"
            )
        pulled = True
    (directory / QUALITY_DIRECTORY / CONTAINER_EVIDENCE).write_text(
        json.dumps(_container_evidence(pulled, _probe_machine()), indent=2) + "\n"
    )


def _grading_check() -> str:
    """Verify the grading container is usable; return the machine it runs as.

    Re-runs the daemon check, the presence check, and the linux/amd64 probe
    — never the pull — and raises an actionable ``ValueError`` when the
    container no longer holds.
    """
    _require_daemon()
    if not _grading_present():
        raise _grading_failure(f"the grading container {IMAGE} is no longer present locally")
    return _probe_machine()


def _verify_grading_container(record: dict[str, Any], directory: Path) -> None:
    """Lightly re-check the completed container without pulling again.

    Raises ``ValueError`` when the container no longer holds, which the
    caller turns into a re-preparation.
    """
    _grading_check()


def run_quality_preparation(state: Any, directory: Path) -> None:
    """Prepare the quality workload resources for a complete setup state.

    Each step persists as it goes: the entry moves to in-progress before the
    (long) subprocess starts, so an interrupted session resumes the same
    preparation and verified resources are never re-fetched. A failure keeps
    the entry honest (pending with an actionable note) instead of reporting
    complete; the session itself is not failed by a preparation problem, and
    the prepared coding submission exists only while both quality
    preparations pass.
    """
    from .setup_state import save_state
    from .speed_preparation import run_step

    record = setup_record(state)
    run_step(state, directory, "datasets", record, prepare_datasets, _verify_datasets)
    run_step(
        state,
        directory,
        "grading_container",
        record,
        prepare_grading_container,
        _verify_grading_container,
    )
    both_complete = all(
        state.preparations[name].status == "complete"
        and not state.preparations[name].needs_revalidation
        for name in ("datasets", "grading_container")
    )
    if both_complete:
        write_prepared_submissions(state, directory)
    else:
        _remove_prepared_submissions(directory)
    save_state(state, directory)


def coding_workload(directory: Path) -> dict[str, Any]:
    return {
        "profile": "humaneval-plus",
        "problems_path": str(problems_path(directory)),
        "solutions_per_task": 1,
        "request_timeout_seconds": 900,
        "grading_timeout_seconds": 3600,
    }


def prepared_submission_paths(directory: Path) -> dict[str, Path]:
    return {"humaneval-plus": directory / SUBMISSIONS_DIRECTORY / "humaneval-plus.json"}


def write_prepared_submissions(state: Any, directory: Path) -> None:
    """Write the fixed coding submission once both quality preparations pass.

    The submission launches the coding workload against the prepared dataset
    copy with the fixed coding budgets; ``run_config`` is left for the launch
    boundary to complete. It carries the redacted setup record, never a
    secret.
    """
    record = redact(setup_record(state))
    path = prepared_submission_paths(directory)["humaneval-plus"]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"setup": record, "workload": coding_workload(directory), "run_config": None}
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _remove_prepared_submissions(directory: Path) -> None:
    for path in prepared_submission_paths(directory).values():
        if path.is_file():
            path.unlink()


def preparation_inputs(name: str) -> tuple[Any, ...]:
    """The pinned inputs one quality preparation entry is hashed against."""
    if name == "datasets":
        return (SOURCE_SHA256, len(TASK_IDS), "HumanEvalPlus", "v0.1.10")
    return (IMAGE, "linux/amd64")


def launch_gate(state: Any | None, directory: Path | None) -> dict[str, Any]:
    """The quality launch gate: prepared inputs only, for the coding profile.

    Without a complete setup state in the given directory there is nothing
    prepared to enforce, so the launch keeps the submitted dataset path.
    With a complete state, the gate verifies — it does not fetch — until the
    prepared resources still hold: both entries complete against the pinned
    inputs (hashes), the dataset copy still decompresses to the pinned
    sha256 with the pinned tasks, and the grading image is still present and
    still runs as x86_64 (no pull). Any refusal raises ``ValueError`` before
    any work directory is created; the returned evidence is retained with
    the run.
    """
    from .setup_state import critical_complete

    if state is None or directory is None or not state_path(directory).is_file():
        return {"prepared": False, "reason": "no setup state"}
    if not critical_complete(setup_record(state), state.discovery):
        # The setup is still incomplete: there is nothing prepared to
        # enforce, so this launch keeps the submitted dataset path.
        return {"prepared": False, "reason": "setup state incomplete"}
    problems: list[str] = []
    for name in ("datasets", "grading_container"):
        entry = state.preparations.get(name)
        if entry is None or entry.status != "complete" or entry.needs_revalidation:
            problems.append(
                f"the {name} preparation is not complete"
                + (f" ({entry.note})" if entry is not None and entry.note else "")
            )
        elif entry.hash != value_hash(*preparation_inputs(name)):
            problems.append(f"the {name} preparation no longer matches the pinned inputs")
    if problems:
        raise ValueError(
            "coding workload launch refused: "
            + "; ".join(problems)
            + ". Re-run bench setup to prepare, and no work was started."
        )
    rows = read_problems(directory)
    if rows is None:
        raise ValueError(
            "coding workload launch refused: the prepared HumanEval+ dataset no "
            "longer verifies against its pinned sha256, row count, and task ids. "
            "Re-run bench setup; no work was started."
        )
    try:
        machine = _grading_check()
    except ValueError as exc:
        raise ValueError(f"coding workload launch refused: {exc}") from None
    return {
        "prepared": True,
        "datasets": _problems_evidence(rows),
        "grading_container": {
            "source": str(IMAGE),
            "platform": "linux/amd64",
            "machine": machine,
        },
    }
