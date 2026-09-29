"""Explicit per-launch configuration: historical review and interactive entry.

Historical sampling values are recommendations for review only; they never
pre-select a reasoning choice. Reasoning effort stays an explicit, unselected
choice at every launch, including when saved setup or retained results exist.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pydantic

from .ordinary_runs import SAMPLING_KEYS, RunConfig

# Historical review presents the sampling parameters for acceptance or
# editing; reasoning effort is deliberately excluded so it stays unselected.
REVIEW_ORDER = SAMPLING_KEYS


NO_OVERRIDE = "none"


def historical_sampling(results: Path) -> dict[str, object]:
    """Sampling values from the most recent retained run, for review.

    Reads the newest retained run's ``setup.json`` (the resolved record as
    executed). A parameter the newest run left absent is not recommended, so a
    later explicit no-override is never silently resurrected from an older run.
    Only the sampling keys are returned; reasoning effort stays a fresh choice.
    """
    if not results.is_dir():
        return {}
    for path in sorted(results.glob("*/setup.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        generation = record.get("declared", {}).get("requests", {}).get("generation")
        if not isinstance(generation, dict):
            continue
        return {key: generation[key] for key in SAMPLING_KEYS if key in generation}
    return {}


def _parse_value(value: str) -> float | int | None:
    text = value.strip()
    if text in ("", NO_OVERRIDE):
        return None
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"must be a number, '{NO_OVERRIDE}', or blank: {value!r}") from None
    return int(number) if number.is_integer() and "." not in text else number


def collect_run_config(
    history: dict[str, object],
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> RunConfig:
    """Prompt for every per-run choice until a complete, valid RunConfig exists.

    Sampling values show the historical recommendation and accept it on blank;
    a typed value edits it, and blank records an explicit no-override once
    history exists. Reasoning effort has no recommendation, including ``off``:
    it is explicitly entered at every launch. Invalid answers are corrected
    on the spot and never used.
    """
    input_fn = input if input_fn is None else input_fn
    output_fn = print if output_fn is None else output_fn
    values: dict[str, float | int | None] = {}
    for key in REVIEW_ORDER:
        recommendation = history.get(key)
        while True:
            if recommendation is None:
                prompt = f"{key} (blank records no override): "
            else:
                prompt = (
                    f"{key} (previous {json.dumps(recommendation)}; blank keeps, 'none' clears): "
                )
            raw = input_fn(prompt)
            if raw.strip() == "" and recommendation is not None:
                value = cast("float | int | None", recommendation)
            else:
                try:
                    value = _parse_value(raw)
                except ValueError as exc:
                    output_fn(f"{key} {exc}; enter a replacement")
                    continue
            values[key] = value
            break
    while True:
        raw = input_fn("reasoning_effort (required; 'off' disables thinking): ")
        try:
            run_config = RunConfig(
                reasoning_effort=raw.strip(),
                **{key: values[key] for key in REVIEW_ORDER},
            )
        except pydantic.ValidationError as exc:
            output_fn(f"{exc.errors()[0]['msg']}; enter a replacement")
            continue
        return run_config
