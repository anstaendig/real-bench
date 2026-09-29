"""Bound transcript payloads in browser reads without changing retained evidence."""

from __future__ import annotations

from typing import Any

_TRANSCRIPT_FIELDS = {"prompt", "content", "reasoning_content", "prediction"}


def compact_report(value: Any, *, outcomes: bool = False) -> Any:
    """Keep all metric/coverage data; omit only outcome transcript strings.

    Setup and workload parameters remain exact for reproducibility and comparison.
    Full transcripts remain available through ordinary result reads and run files.
    """
    if isinstance(value, list):
        return [compact_report(item, outcomes=outcomes) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: (
            "[Transcript retained in run artifacts]"
            if outcomes and key in _TRANSCRIPT_FIELDS and isinstance(item, str)
            else compact_report(
                item, outcomes=outcomes or key in {"outcomes", "live_outcomes", "observed_outcomes"}
            )
        )
        for key, item in value.items()
    }
