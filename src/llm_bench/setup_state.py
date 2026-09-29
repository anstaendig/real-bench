"""Resumable manual setup state for the guided setup flow.

The setup state keeps four things distinct:

- the submitted setup record, where unavailable descriptive metadata is
  preserved explicitly as ``unknown`` while execution-critical details are
  required before any run is possible;
- the nonsecret answers actually supplied, preserved across interruptions so a
  resumed session never re-asks completed questions;
- preparation entries, which report real status only. Unimplemented or
  unfinished preparation is never reported as complete.
- automatic discovery of endpoint facts, a choice inside the flow whose findings
  are presented for review with their source before they become declarations.
A changed configuration marks the preparations that depend on it for revalidation
instead of silently keeping them valid. This module does not introduce resumption
of an interrupted benchmark run; it is reusable setup state only.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError

from .discovery import discover as discover_instance
from .ordinary_runs import CRITICAL_SETUP, DEFAULT_BENCHMARK, REQUIRED, Setup, redact

STATE_FILE = "setup-state.json"
SCHEMA_VERSION = 1

# Descriptive metadata that may be recorded as unknown without preventing an
# otherwise executable run; unavailable values are preserved explicitly.
UNKNOWN_FIELDS = [
    ("hardware", field) for field in ("accelerator", "memory", "cpu", "ram", "os", "driver")
]
UNKNOWN_FIELDS += [("model", field) for field in ("source", "revision", "quantized_artifact")]
UNKNOWN_FIELDS += [("serving", field) for field in ("engine", "version", "launch_command")]
UNKNOWN_DEFAULT_FIELDS = set(UNKNOWN_FIELDS)

# Prompt order for guided entry. Benchmark fields prompt last with pinned defaults.
PROMPT_ORDER = [
    ("hardware", "accelerator"),
    ("hardware", "count"),
    ("hardware", "memory"),
    ("hardware", "cpu"),
    ("hardware", "ram"),
    ("hardware", "os"),
    ("hardware", "driver"),
    ("model", "source"),
    ("model", "revision"),
    ("model", "quantized_artifact"),
    ("model", "tokenizer"),
    ("model", "template_overrides"),
    ("serving", "engine"),
    ("serving", "version"),
    ("serving", "launch_command"),
    ("serving", "environment"),
    ("requests", "base_url"),
    ("requests", "model_id"),
    ("requests", "context_window"),
    ("requests", "generation"),
    ("requests", "reasoning"),
    ("benchmark", "tool"),
    ("benchmark", "version"),
    ("benchmark", "transformers_version"),
    ("benchmark", "seed"),
    ("benchmark", "settings"),
]

OBJECT_FIELDS = {
    ("model", "template_overrides"),
    ("serving", "environment"),
    ("requests", "generation"),
    ("requests", "reasoning"),
    ("benchmark", "settings"),
}

INTEGER_FIELDS = {
    ("hardware", "count"),
    ("requests", "context_window"),
    ("benchmark", "seed"),
}

PreparationStatus = Literal["pending", "in-progress", "complete", "stale", "unimplemented"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Preparation(Record):
    status: PreparationStatus
    hash: str
    needs_revalidation: bool = False
    note: str = ""


class DiscoveryRecord(Record):
    status: Literal["review", "unavailable", "stale"]
    endpoint: str | None = None
    findings: dict[str, dict[str, Any]]
    accepted: list[str]
    rejected: list[str]
    held: list[str]
    notes: list[str]
    at: str


class SetupState(Record):
    schema_version: int
    updated: str
    setup: dict[str, Any]
    provided: dict[str, Any]
    preparations: dict[str, Preparation]
    discovery: DiscoveryRecord | None = None


def state_path(directory: Path) -> Path:
    return directory / STATE_FILE


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dig(mapping: Any, *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _setdig(mapping: dict[str, Any], path: list[str], value: Any) -> None:
    node: dict[str, Any] = mapping
    for key in path[:-1]:
        child: dict[str, Any] = node.setdefault(key, {})
        node = child
    node[path[-1]] = value


def new_state() -> SetupState:
    declared: dict[str, dict[str, Any]] = {category: {} for category in REQUIRED}
    declared["benchmark"] = dict(DEFAULT_BENCHMARK)
    return SetupState(
        schema_version=SCHEMA_VERSION,
        updated=_now(),
        setup={"declared": declared, "observed": {}},
        provided={},
        preparations={},
        discovery=None,
    )


def setup_record(state: SetupState) -> dict[str, Any]:
    """Return the submitted setup record for the saved answers."""
    declared = {category: dict(fields) for category, fields in state.setup["declared"].items()}
    return {"declared": declared, "observed": dict(state.setup.get("observed", {}))}


def _critical_gaps(record: dict[str, Any]) -> list[str]:
    """Execution-critical field paths the declarations still lack."""
    declared = record.get("declared", {})
    return [
        f"{category}.{field}"
        for category, field in CRITICAL_SETUP
        if _dig(declared, category, field) in (None, "", "unknown")
    ]


def validate_critical(
    record: dict[str, Any], discovery: DiscoveryRecord | None = None
) -> list[str]:
    """List execution-critical fields that block a run, without pydantic echoes.

    Descriptive fields are not blockers: an unavailable descriptive value is
    preserved explicitly as unknown rather than stopping the session. A live
    discovery review still offering a value for a missing critical field
    blocks no run: the value is pending review, not fabricated.
    """
    fillable = _discovery_fillable(discovery)
    return [
        f"{path} is execution-critical and required"
        for path in _critical_gaps(record)
        if path not in fillable
    ]


def critical_complete(record: dict[str, Any], discovery: DiscoveryRecord | None = None) -> bool:
    """A record is launchable once every execution-critical value is supplied.

    ``discovery`` is a pending review of the current endpoint: a critical
    field it is offering still counts as fillable, so a record that awaits
    discovery review is not reported incomplete.
    """
    return not validate_critical(record, discovery)


def value_hash(*values: Any) -> str:
    return hashlib.sha256(
        json.dumps(list(values), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _preparation_inputs(record: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    """Inputs each preparation depends on, once its inputs are known.

    A preparation is not tracked until its inputs are fully supplied, so
    entering answers does not churn preparation entries; only a change to a
    fully known configuration affects them.
    """
    declared = record.get("declared", {})
    from . import quality_preparation, ruler_preparation  # local: they import setup_state

    inputs: dict[str, tuple[Any, ...]] = {
        "datasets": quality_preparation.preparation_inputs("datasets"),
        "grading_container": quality_preparation.preparation_inputs("grading_container"),
        # The sampled-RULER source and hydrated data are pinned, not
        # configuration-dependent: the entry tracks the pinned source, while
        # the configuration-dependent 117 prompts are (re)prepared at launch
        # against the resolved run configuration.
        "long_context": ruler_preparation.preparation_inputs("long_context"),
    }
    benchmark = declared.get("benchmark") or {}
    performance_tools = (
        benchmark.get("tool"),
        benchmark.get("version"),
        benchmark.get("transformers_version"),
    )
    if all(value is not None for value in performance_tools):
        inputs["performance_tools"] = performance_tools
    tokenizer = (
        _dig(declared, "model", "tokenizer"),
        _dig(declared, "model", "template_overrides"),
    )
    if all(value is not None for value in tokenizer):
        inputs["tokenizer"] = tokenizer
    compatibility = (
        _dig(declared, "requests", "base_url"),
        _dig(declared, "requests", "model_id"),
        _dig(declared, "requests", "context_window"),
        _dig(declared, "requests", "generation"),
        _dig(declared, "requests", "reasoning"),
    )
    if all(value is not None for value in compatibility):
        inputs["compatibility"] = compatibility
    return inputs


def default_preparations(record: dict[str, Any]) -> dict[str, Preparation]:
    """Honest initial preparation statuses.

    The speed workload resources (pinned performance tools, tokenizer), the
    quality workload resources (pinned dataset, grading container), and the
    long-context (sampled-RULER) resources (pinned upstream source and hydrated
    data) are prepared once the configuration is complete, so they start
    pending instead of unimplemented. The endpoint compatibility check stays
    unimplemented: the state must not report unimplemented preparation as
    finished.
    """
    inputs = _preparation_inputs(record)
    statuses: dict[str, PreparationStatus] = {
        "datasets": "pending",
        "grading_container": "pending",
        "long_context": "pending",
        "compatibility": "unimplemented",
        "performance_tools": "pending",
        "tokenizer": "pending",
    }
    return {
        name: Preparation(status=statuses[name], hash=value_hash(*inputs[name])) for name in inputs
    }


def _ensure_preparations(state: SetupState, record: dict[str, Any]) -> dict[str, Preparation]:
    """Keep preparation entries current against the supplied record.

    A changed configuration keeps each affected preparation's honest status
    but marks it for revalidation; unaffected entries are left as-is. Entries
    whose inputs are no longer fully known are kept exactly as they were, so a
    flag raised by an earlier change survives the gap while the operator
    re-enters the missing values; the entry reconciles against a new hash once
    every input is known again.
    """
    inputs = _preparation_inputs(record)
    preparations = dict(state.preparations)
    for name, input_values in inputs.items():
        new_hash = value_hash(*input_values)
        existing = state.preparations.get(name)
        if existing is None:
            preparations[name] = default_preparations(record)[name]
        elif existing.hash == new_hash:
            continue
        else:
            preparations[name] = Preparation(
                status=existing.status,
                hash=new_hash,
                needs_revalidation=True,
                note="configuration changed; revalidate before use",
            )
    return preparations


def apply_answer(state: SetupState, path: str, value: Any) -> bool:
    """Validate and store one setup answer; return whether it changed the record.

    Every input surface routes through here, so manual and agent-driven entry
    share one validation boundary: invalid values are rejected before they
    touch the saved record, and a resume can never be trapped by an invalid
    persisted answer.
    """
    keys = str(path).split(".")
    if len(keys) != 2 or keys[0] not in REQUIRED or keys[1] not in REQUIRED[keys[0]]:
        raise ValueError(f"setup answer path must name a category and field: {path}")
    if value == "keep":
        return False
    if tuple(keys) in INTEGER_FIELDS:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{path} must be a positive integer")
    if tuple(keys) in OBJECT_FIELDS and not isinstance(value, dict):
        raise ValueError(f"{path} must be an object (empty is explicit)")
    if tuple(keys) in CRITICAL_SETUP and value in ("", "unknown"):
        raise ValueError(f"{path} is execution-critical and cannot be unknown")
    if tuple(keys) == ("requests", "base_url"):
        url = urlsplit(value)
        if (
            not isinstance(value, str)
            or url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(f"{path} must be HTTP(S), without credentials, query or fragment")
    current = _dig(state.setup, "declared", *keys)
    if current == value:
        return False
    _setdig(state.setup, ["declared", *keys], value)
    _setdig(state.provided, list(keys), value)
    state.preparations = _ensure_preparations(state, setup_record(state))
    state.updated = _now()
    return True


def merge_provided(state: SetupState, provided: dict[str, Any]) -> list[str]:
    """Merge nonsecret answers into the saved state; return changed field paths."""
    changed = [
        str(path) for path, value in provided.items() if apply_answer(state, str(path), value)
    ]
    if changed:
        state.updated = _now()
    return changed


def load_state(directory: Path) -> SetupState | None:
    path = state_path(directory)
    if not path.is_file() or path.is_symlink():
        return None
    data = json.loads(path.read_text())
    state = SetupState.model_validate(data)
    if state.schema_version != SCHEMA_VERSION:
        raise ValueError(f"setup state schema version {state.schema_version} is not supported")
    return state


def save_state(state: SetupState, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = state_path(directory)
    # The state stores nonsecret answers; redact any value matching the
    # production credential pattern so a misnamed answer cannot persist a secret.
    payload = redact(state.model_dump(mode="json"))
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def unknown_fields(record: dict[str, Any]) -> list[str]:
    """List descriptive fields explicitly preserved as unknown."""
    declared = record.get("declared", {})
    return [
        f"{category}.{field}"
        for category, fields in REQUIRED.items()
        for field in sorted(fields)
        if (category, field) not in CRITICAL_SETUP
        and declared.get(category, {}).get(field) == "unknown"
    ]


def summarise(state: SetupState, directory: Path | None = None) -> dict[str, Any]:
    """Report the saved state: record, gaps, honest preparation, and readiness.

    ``speed_ready`` advertises the prepared speed workloads only when both of
    their preparations are complete against the current configuration;
    ``default_suite_ready`` additionally requires the quality workloads'
    preparations, so the full suite is never advertised ready before every
    workload is prepared. When a directory is given, the prepared submissions
    and their launch commands are listed (present only while the owning
    preparations pass).
    """
    record = redact(setup_record(state))
    # Findings and notes are the endpoint's report; the operator's own
    # accepted/rejected/held choices are saved state, not review output.
    discovery_dump = (
        {
            "status": state.discovery.status,
            "endpoint": state.discovery.endpoint,
            "findings": state.discovery.findings,
            "notes": state.discovery.notes,
            "at": state.discovery.at,
        }
        if state.discovery is not None
        else None
    )
    discovery = redact(discovery_dump) if discovery_dump is not None else None
    # A live review lists what it is offering; anything else is history, and
    # the operator needs to see every path the record tracked (including what
    # a stale review cleared), not just what the old endpoint said.
    if state.discovery is not None and state.discovery.status == "review":
        discovered: list[str] = sorted(state.discovery.findings)
    else:
        discovered = _discovery_paths(state.discovery)
    result: dict[str, Any] = {
        "state_file": STATE_FILE,
        "updated": state.updated,
        "critical_complete": critical_complete(record, state.discovery),
        "missing_critical": validate_critical(record, state.discovery),
        "unknown": unknown_fields(record),
        "setup": record,
        "preparations": [
            {
                "name": name,
                "status": entry.status,
                "needs_revalidation": entry.needs_revalidation,
                "note": entry.note,
            }
            for name, entry in state.preparations.items()
        ],
        "discovery": discovery,
        "discovered": discovered,
    }
    from . import (  # local: they import setup_state
        quality_preparation,
        ruler_preparation,
        speed_preparation,
    )

    readiness = speed_preparation.speed_readiness(state, directory)
    result["speed_ready"] = readiness["speed_ready"]
    result["default_suite_ready"] = readiness["default_suite_ready"]
    if directory is not None:
        submissions: dict[str, str] = {}
        commands: dict[str, str] = {}

        def launch_command(submission: Path) -> str:
            # shlex.join keeps the advertised command runnable when the setup
            # directory or the submission path contains spaces or shell
            # metacharacters.
            return shlex.join(
                ["bench", "run", str(submission), "--setup-directory", str(directory)]
            )

        for name, path in speed_preparation.prepared_submission_paths(directory).items():
            if path.is_file():
                submissions[name] = str(path)
                commands[name] = launch_command(path)
        for name, path in quality_preparation.prepared_submission_paths(directory).items():
            if path.is_file():
                submissions[name] = str(path)
                commands[name] = launch_command(path)
        for name, path in ruler_preparation.prepared_submission_paths(directory).items():
            if path.is_file():
                submissions[name] = str(path)
                commands[name] = launch_command(path)
        suite_path = speed_preparation.suite_submission_path(directory)
        if suite_path.is_file():
            submissions["default-suite"] = str(suite_path)
            commands["default-suite"] = launch_command(suite_path)
        if submissions:
            result["prepared_submissions"] = submissions
            result["launch_commands"] = commands
    return result


# --- Discovery -------------------------------------------------------------

# Findings are reviewed in a fixed order of use: the model id routes every
# request, then the serving version, then the context window.
_DISCOVERY_REVIEW_ORDER = [
    "requests.model_id",
    "serving.version",
    "requests.context_window",
]


def _review_order(paths: list[str]) -> list[str]:
    rank = {path: i for i, path in enumerate(_DISCOVERY_REVIEW_ORDER)}
    return sorted(paths, key=lambda path: (rank.get(path, len(rank)), path))


def _discovery_paths(discovery: DiscoveryRecord | None) -> list[str]:
    """Every setup path a discovery record knows about, ordered."""
    if discovery is None:
        return []
    paths = set(discovery.findings)
    paths.update(discovery.accepted)
    paths.update(discovery.rejected)
    paths.update(discovery.held)
    return sorted(paths)


def _discovery_fillable(discovery: DiscoveryRecord | None) -> set[str]:
    """Critical paths a live discovery review can still fill.

    Only an unsettled review offers anything: a stale or failed record is
    history, and a path is fillable while it is presented, held, or already
    accepted by the review.
    """
    if discovery is None or discovery.status != "review":
        return set()
    return set(_discovery_paths(discovery))


def _store_observation(
    state: SetupState, path: str, finding: dict[str, Any], note: str | None = None
) -> None:
    """Record one sourced observation alongside the declarations.

    An observation is what the endpoint reported, with its source and date:
    it is stored whether or not the operator accepted the value, so the
    provenance of every reviewed finding survives the choice.
    """
    category, field = path.split(".")
    entry = dict(finding)
    if note:
        entry["note"] = note
    state.setup["observed"].setdefault(category, {})[field] = entry


def _invalidate_discovery(state: SetupState, output_fn: Any) -> list[str]:
    """Return discovery-filled declarations to missing when the endpoint moved.

    A value accepted from an endpoint is evidence about that endpoint. Once
    ``requests.base_url`` points elsewhere, any review of the old endpoint is
    history: it is marked stale whether or not it still holds findings, so an
    old endpoint's pending findings can never unblock the new one. The
    declarations discovery filled are cleared (declared and provided alike)
    unless the operator has since set the value to something other than the
    endpoint's report; the sourced observations remain as history of what the
    old endpoint said, and keep their endpoint identity so a cleared value
    still shows where it came from.
    """
    discovery = state.discovery
    if discovery is None:
        return []

    def _replaced(category: str, field: str, current: Any) -> bool:
        # The last report the endpoint made for this path is the durable
        # reference: it survives an unchanged re-run that confirms the finding
        # silently and re-presents nothing. A current value that differs from
        # it is the operator's own answer, not discovery's.
        observed = state.setup.get("observed", {}).get(category, {})
        entry = observed.get(field)
        reported = entry.get("value") if isinstance(entry, dict) else None
        return reported is not None and current is not None and current != reported

    cleared: list[str] = []
    for path in discovery.accepted:
        category, field = path.split(".")
        declared = state.setup["declared"].setdefault(category, {})
        current = declared.get(field)
        if _replaced(category, field, current):
            continue
        if current is not None:
            declared[field] = None
        provided = state.provided.get(category)
        if isinstance(provided, dict):
            provided.pop(field, None)
        cleared.append(path)
    if discovery.status != "review" and not cleared:
        return []
    state.discovery = DiscoveryRecord(
        status="stale",
        endpoint=discovery.endpoint,
        findings=discovery.findings,
        accepted=discovery.accepted,
        rejected=discovery.rejected,
        held=discovery.held,
        notes=[
            *discovery.notes,
            "requests.base_url changed; discovery-accepted values were cleared "
            "and must be re-entered or re-discovered",
        ],
        at=discovery.at,
    )
    state.updated = _now()
    if cleared:
        output_fn(f"Endpoint changed; discovery-accepted values cleared: {', '.join(cleared)}")
    else:
        output_fn("Endpoint changed; the previous discovery review no longer applies.")
    return cleared


def _review_discovery(
    state: SetupState,
    discovery: dict[str, Any],
    provided: dict[str, Any],
    *,
    interactive: bool,
    input_fn: Any,
    output_fn: Any,
    directory: Path | None = None,
) -> DiscoveryRecord:
    """Present the findings for review; store accepted values with provenance.

    A finding that confirms the existing declaration is re-observed silently
    and never re-asked. Every other finding is presented with its source:
    the operator accepts the discovered value, keeps or replaces the
    declaration (a contradiction shows both values and is never resolved
    silently), or leaves the field for guided entry. In a non-interactive
    session a finding is accepted only when this session's answers supply the
    same value; anything else is held for review, never guessed. The record
    reflects this review only: findings confirmed against a declaration are
    not re-listed, so an unchanged endpoint re-run reports no new findings.
    """
    findings = discovery.get("findings", {})
    presented: dict[str, dict[str, Any]] = {}
    # Ownership, not presentation, is what endpoint invalidation relies on. A
    # value this review accepts is discovery-owned until the endpoint or the
    # value changes; on an unchanged re-run the previous ownership carries
    # through, so a later endpoint change still knows what discovery filled.
    accepted: set[str] = set()
    previous = state.discovery
    if previous is not None and previous.status == "review":
        accepted.update(previous.accepted)
    rejected: set[str] = set()
    held: list[str] = []

    def _record() -> DiscoveryRecord:
        # The status mirrors the endpoint's readability: ``discover`` returns
        # ``review`` when it obtained at least one finding, ``unavailable``
        # when nothing could be read. Ownership (``accepted``), not
        # presentation, is what a later endpoint change relies on.
        return DiscoveryRecord(
            status=cast(Literal["review", "unavailable"], discovery.get("status")),
            endpoint=discovery.get("endpoint"),
            findings=presented,
            accepted=sorted(accepted),
            rejected=sorted(rejected),
            held=sorted(held),
            notes=list(discovery.get("notes", [])),
            at=_now(),
        )

    def _persist_progress() -> None:
        # Every resolved finding is progress: persist the growing record so an
        # interruption leaves a resumable review, not a blank one.
        if directory is None:
            return
        state.discovery = _record()
        save_state(state, directory)

    def _accept(path: str) -> None:
        # Accepting is one move: store the value as a declaration with its
        # provenance and take ownership, so a later endpoint change can still
        # clear what discovery filled.
        apply_answer(state, path, value)
        _store_observation(state, path, finding)
        accepted.add(path)

    for path in _review_order(list(findings)):
        category, field = path.split(".")
        finding = findings[path]
        value = finding["value"]
        current = _dig(state.setup, "declared", category, field)
        if current is not None and current == value:
            # Re-observed and confirmed: silent, and ownership is kept so a
            # later endpoint change can still clear what discovery filled.
            accepted.add(path)
            _store_observation(state, path, finding)
            continue
        accepted.discard(path)
        presented[path] = dict(finding)
        supplied = provided.get(path)
        if current is None:
            if interactive:
                output_fn(f"Discovered {path}: {value!r} (source: {finding['source']})")
                output_fn(f"{path} is not declared yet; accept the discovered value?")
                choice = input_fn(f"{path}: [a]ccept / [s]kip for manual entry: ").strip().lower()
                if choice in ("a", "accept", "y", "yes"):
                    _accept(path)
                else:
                    _store_observation(state, path, finding, note="left for manual entry")
                    held.append(path)
            elif supplied == value:
                _accept(path)
            else:
                _store_observation(
                    state, path, finding, note="awaiting review; not stored as a declaration"
                )
                held.append(path)
            _persist_progress()
            continue
        # Contradiction: the declaration differs from the endpoint's report.
        if interactive:
            output_fn(f"Discovered {path}: {value!r} (source: {finding['source']})")
            output_fn(
                f"CONFLICT: declared {path} is {current!r}, the endpoint "
                f"reports {value!r} (source: {finding['source']})"
            )
            choice = (
                input_fn(f"{path}: [k]eep declared / [a]ccept discovery / [r]eplace: ")
                .strip()
                .lower()
            )
            if choice in ("a", "accept"):
                _accept(path)
            elif choice in ("r", "replace"):
                replacement = _prompt_value(
                    (category, field), current, input_fn=input_fn, output_fn=output_fn
                )
                apply_answer(state, path, replacement)
                _store_observation(
                    state,
                    path,
                    finding,
                    note=f"operator replaced the declaration with {replacement!r}",
                )
                rejected.add(path)
            else:
                _store_observation(
                    state, path, finding, note=f"operator kept the declared value {current!r}"
                )
                rejected.add(path)
        elif supplied == value:
            _accept(path)
        else:
            _store_observation(
                state,
                path,
                finding,
                note=f"conflicts with declared value {current!r}; awaiting review",
            )
            held.append(path)
        _persist_progress()
    if interactive:
        if not findings:
            output_fn("Discovery is unavailable for this endpoint; entering details manually:")
        for note in discovery.get("notes", []):
            output_fn(f"Discovery note: {note}")
    state.discovery = _record()
    return state.discovery


# --- Guided entry ---------------------------------------------------------


def _prompt_value(
    field_path: tuple[str, str],
    current: Any,
    *,
    input_fn: Any = None,
    output_fn: Any = None,
) -> Any:
    input_fn = input if input_fn is None else input_fn
    output_fn = print if output_fn is None else output_fn
    category, field = field_path
    label = f"{category}.{field}"
    if (category, field) in OBJECT_FIELDS:
        value = dict(current) if isinstance(current, dict) else {}
        output_fn(f"{label} (existing: {json.dumps(value, sort_keys=True)})")
        while True:
            key = input_fn(f"{label} key (blank when done): ").strip()
            if not key:
                return value
            raw = input_fn(f"{label}.{key} value (JSON or plain text): ").strip()
            try:
                value[key] = json.loads(raw)
            except ValueError:
                value[key] = raw
    if (category, field) in INTEGER_FIELDS:
        while True:
            hint = f" {current}" if isinstance(current, int) else ""
            raw = input_fn(f"{label}{hint}: ").strip()
            if not raw and isinstance(current, int):
                return current
            try:
                return int(raw)
            except ValueError:
                output_fn(f"{label} must be an integer")
    unknownable = (category, field) in UNKNOWN_DEFAULT_FIELDS
    while True:
        if isinstance(current, str) and current != "":
            raw = input_fn(f"{label} (currently {current!r}); blank keeps: ").strip()
            if not raw:
                return current
        else:
            suffix = " (blank records unknown)" if unknownable else ""
            raw = input_fn(f"{label}{suffix}: ").strip()
            if not raw:
                if unknownable:
                    return "unknown"
                continue
        return raw


def collect_answers(
    state: SetupState,
    directory: Path,
    *,
    input_fn: Any = input,
    output_fn: Any = print,
) -> None:
    """Prompt for every missing field until the record is complete.

    Answers are validated and saved after each prompt through the same
    boundary agent-driven answers use, so an interruption leaves a resumable
    state and an invalid answer is corrected on the spot, never persisted.
    """
    record = setup_record(state)
    for field_path in PROMPT_ORDER:
        current = _dig(record, "declared", *field_path)
        if current is not None:
            continue
        path = ".".join(field_path)
        while True:
            value = _prompt_value(field_path, current, input_fn=input_fn, output_fn=output_fn)
            try:
                apply_answer(state, path, value)
                break
            except ValueError as exc:
                output_fn(f"{exc}; enter a replacement")
        save_state(state, directory)
        record = setup_record(state)
    output_fn(
        f"Setup record is complete; {len(unknown_fields(record))} descriptive "
        "fields are recorded as unknown"
    )


def finalise(state: SetupState, directory: Path, output_fn: Any = print) -> dict[str, Any]:
    """Validate, persist, and report a complete setup state.

    Missing descriptive fields are preserved explicitly instead of left as
    holes: objects become empty (the documented explicit value) and every
    other string becomes the literal unknown. Missing execution-critical
    fields are never filled: an incomplete record is reported with its gaps
    instead of being reported complete, and the session's progress is saved
    so the gaps survive into the next attempt. A record whose only gaps are
    offered by a live discovery review is persisted without the strict launch
    validation: the values await review, not fabrication.
    """
    from . import (  # local: they import setup_state
        quality_preparation,
        ruler_preparation,
        speed_preparation,
    )

    declared = state.setup["declared"]
    for category, fields in REQUIRED.items():
        for field in fields:
            if declared.get(category, {}).get(field) is None:
                if (category, field) in CRITICAL_SETUP:
                    continue
                default = {} if (category, field) in OBJECT_FIELDS else "unknown"
                declared.setdefault(category, {})[field] = default
    record = setup_record(state)
    save_state(state, directory)
    problems = validate_critical(record, state.discovery)
    if problems:
        # A launchable record is required to compose the suite; invalidate any
        # prepared for a previous configuration so it is never advertised.
        speed_preparation.remove_suite_submission(directory)
        raise ValueError("; ".join(problems))
    if _critical_gaps(record):
        # Every remaining gap is offered by a pending discovery review; the
        # record is not launchable yet, so a previously composed suite (which
        # may name a superseded endpoint) is not advertised.
        speed_preparation.remove_suite_submission(directory)
        return summarise(state, directory)
    try:
        Setup.model_validate(record)
    except ValidationError as exc:
        speed_preparation.remove_suite_submission(directory)
        raise ValueError(
            "; ".join(
                error["msg"] for error in exc.errors(include_input=False, include_context=False)
            )
        ) from exc
    state.preparations = _ensure_preparations(state, record)
    save_state(state, directory)
    # The record is launchable: prepare the speed workload resources (pinned
    # tools, tokenizer) so a launch can reuse verified downloads. A
    # preparation problem keeps its entry honest instead of failing the
    # session; an interruption here resumes the same preparation.

    try:
        output_fn(
            "Preparing speed-workload resources: verifying the pinned performance tools "
            "(aiperf 0.12.0, transformers 5.17.0) and downloading the declared tokenizer "
            "(tokenizer files only, no model weights) into the setup directory; a verified "
            "tokenizer is reused, never re-fetched."
        )
        speed_preparation.run_speed_preparation(state, directory)
        output_fn(
            "Preparing quality-workload resources: copying the pinned HumanEval+ v0.1.10 "
            "dataset into the setup directory and verifying the pinned EvalPlus grading "
            "container (pulled once for linux/amd64 when absent); verified resources are "
            "reused, never re-fetched."
        )
        quality_preparation.run_quality_preparation(state, directory)
        output_fn(
            "Preparing long-context resources: cloning the pinned upstream RULER revision "
            "and hydrating its synthetic data (SQuAD, HotpotQA, Paul Graham essays; a small "
            "git checkout plus public data files, no model weights) into the setup "
            "directory; a verified checkout is reused, never re-fetched."
        )
        ruler_preparation.run_long_context_preparation(state, directory)
    except (KeyboardInterrupt, EOFError):
        # An interrupted re-preparation must not leave a suite prepared for the
        # previous, now-superseded configuration advertised to the operator.
        speed_preparation.remove_suite_submission(directory)
        raise
    # The complete default suite is composable only while every workload
    # resource is prepared and still matches the configuration. The four
    # phases are the fixed workloads, so the composed submission is written
    # only while all five workload preparations pass and removed otherwise,
    # so an incomplete or revalidation-pending setup never offers a suite
    # launch.
    suite_complete = all(
        state.preparations[name].status == "complete"
        and not state.preparations[name].needs_revalidation
        for name in (
            "performance_tools",
            "tokenizer",
            "datasets",
            "grading_container",
            "long_context",
        )
    )
    if suite_complete:
        speed_preparation.write_suite_submission(state, directory)
    else:
        speed_preparation.remove_suite_submission(directory)
    return summarise(state, directory)


def run_setup(
    directory: Path,
    provided: dict[str, Any],
    *,
    interactive: bool,
    discover: bool = False,
    input_fn: Any = None,
    output_fn: Any = None,
) -> dict[str, Any]:
    """One guided setup session: load, merge, discover if chosen, prompt, persist.

    Discovery is a choice within the flow, never a requirement: it runs only
    when requested, uses only endpoint reads, and its findings go through the
    same answer boundary as every other value. I/O resolves at call time so
    the public command boundary remains the prompt surface; agent-driven
    callers pass ``interactive=False`` and get the same input/state behavior
    with no prompts.
    """
    input_fn = input if input_fn is None else input_fn
    output_fn = print if output_fn is None else output_fn
    state = load_state(directory)
    resumed = state is not None
    if state is None:
        state = new_state()
    # merge_provided rejects invalid answers before they touch the record, so
    # only valid progress persists: an agent-driven session may accumulate
    # answers across calls, and the saved state stays launchable once the
    # remaining gaps close.
    changed = merge_provided(state, provided)
    if "requests.base_url" in changed:
        # A value accepted from one endpoint says nothing about a new one.
        cleared = _invalidate_discovery(state, output_fn)
        if cleared:
            save_state(state, directory)
    # Discovery is a choice within the flow, never a requirement: it runs only
    # when requested, uses only endpoint reads, and its findings are reviewed
    # before any missing field is prompted, so a discovered value is never
    # asked for by hand afterward. It needs the endpoint it reads; when the
    # operator chose discovery but the endpoint is unknown yet, the endpoint
    # is the first question, then discovery runs.
    discovery: dict[str, Any] | None = None
    if discover:
        base_url = _dig(state.setup, "declared", "requests", "base_url")
        if interactive and not (isinstance(base_url, str) and base_url):
            while not (isinstance(base_url, str) and base_url):
                entered = _prompt_value(
                    ("requests", "base_url"),
                    base_url,
                    input_fn=input_fn,
                    output_fn=output_fn,
                )
                try:
                    apply_answer(state, "requests.base_url", entered)
                except ValueError as exc:
                    output_fn(f"{exc}; enter a replacement")
                    continue
                base_url = _dig(state.setup, "declared", "requests", "base_url")
            save_state(state, directory)
        if isinstance(base_url, str) and base_url:
            discovery = discover_instance(base_url)
    if interactive:
        # Every valid merge is progress; persist it before asking anything,
        # so an interrupted review still finds the supplied answers saved.
        if changed:
            save_state(state, directory)
        missing = any(
            _dig(setup_record(state), "declared", *field_path) is None
            for field_path in PROMPT_ORDER
        )
        if missing:
            if resumed:
                output_fn("Resuming; saved answers are kept unless you change them.")
            else:
                output_fn("Answer the setup record; progress saves after each answer.")
        try:
            if discovery is not None:
                output_fn("Discovery: reading the endpoint (read-only); findings will be reviewed.")
                _review_discovery(
                    state,
                    discovery,
                    provided,
                    interactive=interactive,
                    input_fn=input_fn,
                    output_fn=output_fn,
                    directory=directory,
                )
                # The review record persists as it grows (see _review_discovery);
                # the manual prompts that follow save their progress too, so an
                # interruption resumes into the same review instead of a blank one.
                save_state(state, directory)
            if missing:
                collect_answers(state, directory, input_fn=input_fn, output_fn=output_fn)
        except (KeyboardInterrupt, EOFError):
            state.updated = _now()
            save_state(state, directory)
            raise
    elif discovery is not None:
        _review_discovery(
            state,
            discovery,
            provided,
            interactive=interactive,
            input_fn=input_fn,
            output_fn=output_fn,
            directory=directory,
        )
        save_state(state, directory)
    result = finalise(state, directory, output_fn)
    result["resumed"] = resumed
    result["changed"] = changed
    return result
