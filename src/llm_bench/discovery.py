"""Automatic discovery of instance details from an endpoint's read routes.

Discovery uses only reads on the OpenAI-compatible serving API: the model
list (``GET /v1/models``) and the vLLM loads route (``GET /v1/loads``). It
never sends a completion, never launches or reconfigures the service, and
never guesses: a field the endpoint does not report stays unavailable for
guided entry instead of becoming fabricated metadata.

Each finding is a sourced observation — a value, the route it came from, and
a date. Findings are presented for operator review; accepting one stores it
with its provenance in ``setup.observed`` alongside the operator's
declarations, so the two never blur together.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any, cast

from .ordinary_runs import redact, runtime_api_key


def _get_json(url: str, key: str | None) -> tuple[int, Any]:
    headers = {"accept": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
        return 0, None
    try:
        return status, json.loads(raw.decode())
    except (ValueError, UnicodeDecodeError):
        return status, None


def discover(base_url: str, *, key: str | None = None, when: str | None = None) -> dict[str, Any]:
    """Find instance facts the endpoint reports itself, with provenance.

    Returns a discovery record: ``status`` is ``review`` when at least one
    finding was obtained and ``unavailable`` when the discovery routes could
    not be read at all. ``findings`` maps each setup path to its value,
    source, and date. The two routes are read independently: a broken model
    list never throws away a readable loads route. No deployment fact is
    inferred: nothing the endpoint did not report is returned, and malformed
    entries are left for manual entry instead of being coerced. No request is
    sent that the endpoint would treat as inference or mutation.
    """
    key = key if key is not None else runtime_api_key()
    when = when if when is not None else datetime.now(UTC).isoformat(timespec="seconds")
    root = str(base_url).rstrip("/")

    findings: dict[str, dict[str, Any]] = {}
    notes: list[str] = []

    status, body = _get_json(root + "/v1/models", key)
    if status in (401, 403):
        return {
            "status": "unavailable",
            "endpoint": root,
            "findings": {},
            "accepted": [],
            "rejected": [],
            "notes": [
                f"the endpoint rejected the supplied credentials with HTTP {status}; "
                "check the bearer key, then re-run discovery or enter the details manually"
            ],
        }
    if status != 200 or not isinstance(body, dict) or not isinstance(body.get("data"), list):
        notes.append(
            "the model list route (GET /v1/models) did not return a readable model list; "
            "enter requests.model_id manually"
        )
    else:
        listed = [
            item["id"]
            for item in body["data"]
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
        ]
        if len(listed) == 1:
            findings["requests.model_id"] = {
                "value": listed[0],
                "source": "GET /v1/models",
                "date": when,
            }
        elif listed:
            shown = ", ".join(listed[:4]) + (", ..." if len(listed) > 4 else "")
            notes.append(
                f"the endpoint lists {len(listed)} model ids ({shown}); "
                "supply requests.model_id manually"
            )
        else:
            notes.append(
                "the model list route (GET /v1/models) did not return a readable "
                "model list; enter requests.model_id manually"
            )

    status, body = _get_json(root + "/v1/loads", key)
    if status != 200 or not isinstance(body, dict):
        notes.append(
            "the vLLM loads route (GET /v1/loads) is not readable; "
            "requests.context_window and serving.version are unavailable from discovery"
        )
    else:
        version = body.get("vllm_version")
        if isinstance(version, str) and version.strip():
            findings["serving.version"] = {
                "value": version.strip(),
                "source": "GET /v1/loads",
                "date": when,
            }
        context_window = None
        model = body.get("model")
        if isinstance(model, dict):
            reported = model.get("max_model_len")
            if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
                context_window = reported
        if context_window is not None:
            findings["requests.context_window"] = {
                "value": context_window,
                "source": "GET /v1/loads",
                "date": when,
            }
        if "requests.context_window" not in findings:
            notes.append(
                "no context window was reported on GET /v1/loads; "
                "enter requests.context_window manually"
            )
        if "serving.version" not in findings:
            notes.append(
                "no vLLM version was reported on GET /v1/loads; enter serving.version manually"
            )

    return {
        "status": "review" if findings else "unavailable",
        "endpoint": root,
        "findings": cast(dict[str, Any], redact(findings)),
        "accepted": [],
        "rejected": [],
        "notes": notes,
    }
