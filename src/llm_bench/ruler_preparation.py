"""Preparation of the sampled-RULER long-context workload: the pinned upstream
source and its hydrated data.

The sampled-RULER workload generates its 117 prompts with the pinned upstream
RULER synthetic generators, so a launch needs one reusable resource that setup
prepares ahead of time:

- a local checkout of the pinned upstream RULER revision, hydrated with the
  synthetic-data JSON inputs the generators read (SQuAD, HotpotQA, the Paul
  Graham essays, and the committed word list), verified against the pinned
  revision and the sha256 of every hydrated data file.

Everything here is ordinary ``subprocess.run`` in the pinned RULER runtime
environment; the fetch is a small git checkout plus a few public data files, and
no model weights reach the runner. A verified resource is reused, never
re-fetched. The launch gate (:func:`launch_gate`) verifies — it does not fetch —
and refuses to launch the long-context workload until the prepared source still
holds; a refusal names the gap and starts no work.

The 117 prompts are generated per launch by the long-context runner against the
final run configuration (the resolved tokenizer, chat template, and reasoning
choice), so a change to any of those re-prepares the inputs before scored
execution; this preparation supplies and guards the shared source they generate
from.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .ordinary_runs import redact
from .paths import source_checkout_root
from .setup_state import setup_record, state_path, value_hash

LONG_CONTEXT_DIRECTORY = "long-context"
SOURCE_SUBDIR = "source"
MANIFEST_NAME = "manifest.json"
EVIDENCE = "source.json"
SUBMISSIONS_DIRECTORY = "submissions"

# The pinned upstream source, recorded in ``third_party/lock.yaml``.
SOURCE_URL = "https://github.com/NVIDIA/RULER"
REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"

#: The data files the pinned generators read from the source checkout.
DATA_FILES = ("squad.json", "hotpotqa.json", "PaulGrahamEssays.json", "english_words.json")

SOURCE_PREPARE_TIMEOUT = 4200


def source_path(directory: Path) -> Path:
    """The prepared, verified RULER checkout a long-context launch runs from."""
    return Path(directory) / LONG_CONTEXT_DIRECTORY / SOURCE_SUBDIR


def _helper_path() -> Path:
    return source_checkout_root() / "scripts" / "ruler_source_prepare.py"


def read_verified_source(directory: Path) -> dict[str, Any] | None:
    """Read the prepared source manifest and verify the checkout on disk.

    Returns the manifest when the saved manifest still verifies: the checkout
    is a git repo at the pinned revision with no modified tracked
    generator/scorer scripts (the same unmodified-source check the upstream
    pilot applies), and every listed data file exists and hashes to its
    recorded value. ``None`` otherwise. Reading never repairs or refetches.
    """
    root = source_path(directory)
    manifest_file = Path(directory) / LONG_CONTEXT_DIRECTORY / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_file.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("source") != SOURCE_URL or manifest.get("revision") != REVISION:
        return None
    if not root.is_dir() or not (root / ".git").is_dir():
        return None
    if not _checkout_holds(root):
        return None
    data = manifest.get("data")
    if not isinstance(data, dict):
        return None
    json_dir = root / "scripts" / "data" / "synthetic" / "json"
    try:
        for name in DATA_FILES:
            expected = data.get(name)
            if not isinstance(expected, str):
                return None
            path = json_dir / name
            if not path.is_file() or path.is_symlink():
                return None
            if _sha256_file(path) != expected:
                return None
    except OSError:
        return None
    return manifest


def _checkout_holds(root: Path) -> bool:
    """The checkout is still at the pinned revision with unmodified tracked
    scripts: the generator and scorer sources that size and score the 117
    requests must hold exactly as pinned, or a modified tree could prepare
    inputs the pinned source never would have.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        clean = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "*.py", "*.yaml"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if head.returncode != 0 or (head.stdout or "").strip() != REVISION:
        return False
    return clean.returncode == 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": SOURCE_URL,
        "revision": REVISION,
        "data": dict(manifest.get("data", {})),
        "fetched_at": manifest.get("fetched_at"),
    }


def prepare_source(record: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Clone the pinned RULER revision, hydrate its data, and verify the result.

    Reuses a verified checkout: when the saved manifest matches the pinned
    source and the checkout still hashes to it, no fetch occurs. The fetch runs
    in the pinned RULER runtime environment. Raises ``ValueError`` naming the
    failure so the preparation loop records an honest pending note.
    """
    verified = read_verified_source(directory)
    if verified is not None:
        return verified
    (directory / LONG_CONTEXT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    command = [
        "uv",
        "run",
        "--project",
        str(source_checkout_root() / "third_party" / "ruler"),
        "python",
        str(_helper_path()),
        "--source-url",
        SOURCE_URL,
        "--revision",
        REVISION,
        "--directory",
        str(source_path(directory)),
        "--manifest",
        str(directory / LONG_CONTEXT_DIRECTORY / MANIFEST_NAME),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=SOURCE_PREPARE_TIMEOUT
        )
    except FileNotFoundError:
        raise ValueError(
            "the pinned environment uv was not found on the PATH; install uv. "
            "Re-run bench setup to prepare the sampled RULER source."
        ) from None
    except subprocess.TimeoutExpired:
        raise ValueError(
            f"preparing the sampled RULER source timed out after {SOURCE_PREPARE_TIMEOUT} seconds. "
            "Re-run bench setup; the preparation is resumable."
        ) from None
    detail = (completed.stderr or completed.stdout or "").strip()
    if completed.returncode == 3:
        message = "sampled RULER source preparation needs a human-only prerequisite"
        if detail:
            message = f"{message}: {detail}"
        raise ValueError(message + " Re-run bench setup after fixing it; no work was started.")
    if completed.returncode != 0:
        raise ValueError(
            f"sampled RULER source preparation failed: {detail or 'unknown error'}. "
            "Re-run bench setup; no work was started."
        )
    verified = read_verified_source(directory)
    if verified is None:
        raise ValueError(
            "sampled RULER source preparation finished but its checkout could not be verified. "
            "Re-run bench setup; no work was started."
        )
    _write_evidence(verified, directory)
    return verified


def _write_evidence(manifest: dict[str, Any], directory: Path) -> None:
    (directory / LONG_CONTEXT_DIRECTORY / EVIDENCE).write_text(
        json.dumps(_source_evidence(manifest), indent=2, allow_nan=False) + "\n"
    )


def _verify_source(record: dict[str, Any], directory: Path) -> None:
    """Lightly re-check the completed source without fetching.

    Only hashes the saved data files and re-reads the checkout revision; raises
    ``ValueError`` when the verified resource no longer holds, which the caller
    turns into a re-preparation.
    """
    if read_verified_source(directory) is None:
        raise ValueError("the prepared sampled RULER source no longer verifies")


def run_long_context_preparation(state: Any, directory: Path) -> None:
    """Prepare the long-context workload resources for a complete setup state.

    The entry moves to in-progress before the (long) clone starts, so an
    interrupted session resumes the same preparation and a verified checkout is
    never re-fetched. A failure keeps the entry honest (pending with an
    actionable note) instead of failing the session, and the prepared
    long-context submission exists only while the preparation passes.
    """
    from .setup_state import save_state
    from .speed_preparation import run_step

    record = setup_record(state)
    run_step(state, directory, "long_context", record, prepare_source, _verify_source)
    entry = state.preparations["long_context"]
    if entry.status == "complete" and not entry.needs_revalidation:
        write_prepared_submissions(state, directory)
    else:
        _remove_prepared_submissions(directory)
    save_state(state, directory)


def long_context_workload(directory: Path) -> dict[str, Any]:
    return {
        "profile": "sampled-ruler",
        "source_path": str(source_path(directory)),
        "request_timeout_seconds": 900,
    }


def prepared_submission_paths(directory: Path) -> dict[str, Path]:
    return {"sampled-ruler": directory / SUBMISSIONS_DIRECTORY / "sampled-ruler.json"}


def write_prepared_submissions(state: Any, directory: Path) -> None:
    """Write the fixed long-context submission once the preparation passes.

    The submission launches the sampled-RULER workload against the prepared
    source checkout with the fixed long-context output budget (``max_tokens`` is
    left out, so the workload's fixed budget applies); ``run_config`` is left
    for the launch boundary to complete. It carries the redacted setup record,
    never a secret.
    """
    record = redact(setup_record(state))
    path = prepared_submission_paths(directory)["sampled-ruler"]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "setup": record,
        "workload": long_context_workload(directory),
        "run_config": None,
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _remove_prepared_submissions(directory: Path) -> None:
    for path in prepared_submission_paths(directory).values():
        if path.is_file():
            path.unlink()


def preparation_inputs(name: str) -> tuple[Any, ...]:
    """The pinned inputs one long-context preparation entry is hashed against."""
    if name == "long_context":
        return (SOURCE_URL, REVISION)
    return ()


def launch_gate(state: Any | None, directory: Path | None) -> dict[str, Any]:
    """The long-context launch gate: prepared source only, for the RULER profile.

    Without a complete setup state in the given directory there is nothing
    prepared to enforce, so the launch keeps the submitted source path.
    With a complete state, the gate verifies — it
    does not fetch — until the prepared checkout still holds: the entry is
    complete against the pinned inputs, the checkout is still at the pinned
    revision, and every data file still hashes to its manifest, so a tampered or
    missing resource is refused with no work started. The returned evidence is
    retained with the run.
    """
    from .setup_state import critical_complete

    if state is None or directory is None or not state_path(directory).is_file():
        return {"prepared": False, "reason": "no setup state"}
    if not critical_complete(setup_record(state), state.discovery):
        return {"prepared": False, "reason": "setup state incomplete"}
    entry = state.preparations.get("long_context")
    if entry is None or entry.status != "complete" or entry.needs_revalidation:
        raise ValueError(
            "long-context workload launch refused: the sampled RULER source is not prepared"
            + (f" ({entry.note})" if entry is not None and entry.note else "")
            + ". Re-run bench setup to prepare, and no work was started."
        )
    if entry.hash != value_hash(*preparation_inputs("long_context")):
        raise ValueError(
            "long-context workload launch refused: the sampled RULER source no longer "
            "matches the pinned inputs. Re-run bench setup; no work was started."
        )
    manifest = read_verified_source(directory)
    if manifest is None:
        raise ValueError(
            "long-context workload launch refused: the prepared sampled RULER source no "
            "longer verifies against its pinned revision and data manifest. Re-run bench "
            "setup; no work was started."
        )
    return {
        "prepared": True,
        "source": _source_evidence(manifest),
        "path": str(source_path(directory)),
        "data_files": len(manifest.get("data", {})),
    }
