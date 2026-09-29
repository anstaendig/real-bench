"""Setup and launch compatibility checks for externally operated routes.

These checks use only reads (``GET /v1/models``) and inference
(``POST /v1/chat/completions``); they never launch, stop, flush, tune, or
reconfigure the externally operated service. A single HTTP 200 is not proof
that a request setting was applied: the vLLM, SGLang, and TabbyAPI
OpenAI-compatible servers answer a chat completion with 200 even when a
parameter it cannot honour is present (SGLang's and TabbyAPI's request
models drop fields they do not register), and the llama.cpp server ignores
fields outside its request schema, so a request carrying a specific setting
is the evidence that the setting is accepted, and a 400 that names the
parameter is evidence that it is not.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, cast

from .ordinary_runs import API_KEY_ENV, SAMPLING_KEYS, redact, runtime_api_key

# Request settings whose vLLM OpenAI-protocol support is documented per
# release, with the earliest release observed to accept them. A declared
# serving version below a floor is recorded as an advisory in the evidence;
# the live request probe below remains the decisive, per-instance signal.
MINIMUM_VERSION = {
    # min_p / repetition_penalty: accepted by the OpenAI-compatible server
    # from v0.2.4 onward (PR #1868, December 2023).
    "min_p": (0, 2, 4),
    "repetition_penalty": (0, 2, 4),
    # chat_template_kwargs (and therefore reasoning_effort / enable_thinking)
    # on the OpenAI-compatible server from at least v0.5.4.
    "reasoning_effort": (0, 5, 4),
}

_VERSION_RE = re.compile(r"(\d+(?:\.\d+)*)")

# The llama.cpp server is identified by its ``b<NNNN>`` build tag, not a
# dotted release number. The per-run setting mapping and the advisory below
# are recorded against the pinned build; the live probe remains the
# per-instance authority, and a declared version that cannot be parsed is
# reported as advisory rather than as a floor.
LLAMA_VERSION_EVIDENCE = "b6150"

# The OpenAI-compatible llama.cpp server (``llama-server``) registers the
# repeated-penalty setting as ``repeat_penalty``; the shared per-run surface
# names it ``repetition_penalty`` (the vLLM name). Every other per-run
# setting keeps its name, and ``chat_template_kwargs`` (and the reasoning
# values it carries) reach the Jinja template unchanged. Mapping to a name
# the server does not register is not a capability failure: the server
# silently ignores unregistered fields, so the probe would record 200 while
# the setting never applied.
LLAMA_SETTING_MAPPINGS = {
    "repetition_penalty": "repeat_penalty",
}


# The SGLang OpenAI-compatible server is identified by its dotted release
# version. The per-run setting mapping (none: it registers every shared
# name) and the reasoning pass-through below are recorded against the
# pinned release; the live probe remains the per-instance authority, and a
# declared version that cannot be parsed is reported as advisory rather
# than as a floor.
SGLANG_VERSION_EVIDENCE = "0.5.20"

# ExLlamaV3 is served by TabbyAPI, its official OpenAI-compatible server,
# which is a rolling release with no dotted release number: the mapping is
# recorded against the pinned source revision (third_party/lock.yaml). The
# per-run mapping moves reasoning_effort out of chat_template_kwargs because
# the deployed ExLlama OpenAI server reads it only at the top level.
EXLLAMAV3_ENGINE_ALIASES = ("exllamav3", "exl3", "exllama v3")
EXLLAMAV3_VERSION_EVIDENCE = "v1.5.0 (revision 0740edc2da56)"
TABBYAPI_VERSION_EVIDENCE = "53da7919d4e4"
EXLLAMAV3_SETTING_MAPPINGS = {
    "chat_template_kwargs.reasoning_effort": "reasoning_effort",
}


def vllm_engine(setup: dict[str, Any]) -> bool:
    """Whether the declared serving engine is vLLM (case-insensitive)."""
    return str(setup.get("serving", {}).get("engine", "")).strip().lower() == "vllm"


def llama_engine(setup: dict[str, Any]) -> bool:
    """Whether the declared serving engine is llama.cpp (case-insensitive)."""
    return str(setup.get("serving", {}).get("engine", "")).strip().lower() == "llama.cpp"


def sglang_engine(setup: dict[str, Any]) -> bool:
    """Whether the declared serving engine is SGLang (case-insensitive)."""
    return str(setup.get("serving", {}).get("engine", "")).strip().lower() == "sglang"


def exllamav3_engine(setup: dict[str, Any]) -> bool:
    """Whether the declared serving engine is ExLlamaV3 (case-insensitive).

    Accepts the engine name and its common short forms (``exl3``,
    ``exllama v3``); an ExLlamaV3 route is served by TabbyAPI.
    """
    engine = str(setup.get("serving", {}).get("engine", "")).strip().lower()
    return engine in EXLLAMAV3_ENGINE_ALIASES


def gated_engine(setup: dict[str, Any]) -> bool:
    """Whether the declared engine has a delivered compatibility mapping.

    vLLM, llama.cpp, SGLang, and ExLlamaV3 probe their endpoints before
    scored work; engines without a delivered mapping keep their existing
    pass-through rather than being probed with an unverified parameter set.
    """
    return (
        vllm_engine(setup) or llama_engine(setup) or sglang_engine(setup) or exllamav3_engine(setup)
    )


def setting_mappings(setup: dict[str, Any]) -> dict[str, str]:
    """The engine-appropriate renames for the shared per-run setting names.

    vLLM keeps every shared name; llama.cpp renames the repeated-penalty
    setting; ExLlamaV3 moves reasoning_effort from the shared template kwargs
    into the top-level field its OpenAI server reads. SGLang keeps every
    shared name. Engines without a delivered mapping have no mapping.
    """
    if llama_engine(setup):
        return dict(LLAMA_SETTING_MAPPINGS)
    if exllamav3_engine(setup):
        return dict(EXLLAMAV3_SETTING_MAPPINGS)
    return {}


def apply_setting_mappings(settings: dict[str, Any], mappings: dict[str, str]) -> dict[str, Any]:
    """Apply engine-specific request-setting mappings to a copied request.

    Dotted source keys address nested settings. An ``off`` effort is retained
    in the setup for comparison, but the outgoing request carries only the
    no-thinking switch: template kwargs on shared engines, top-level on
    ExLlamaV3.
    """
    mapped: dict[str, Any] = {}
    template_kwargs = settings.get("chat_template_kwargs")
    off = isinstance(template_kwargs, dict) and template_kwargs.get("reasoning_effort") == "off"
    for name, value in settings.items():
        if off and name in {"reasoning_effort", "enable_thinking"}:
            continue
        if name == "chat_template_kwargs" and isinstance(value, dict):
            nested = dict(value)
            if off:
                nested.pop("reasoning_effort")
                nested["enable_thinking"] = False
                if "chat_template_kwargs.reasoning_effort" in mappings:
                    mapped["enable_thinking"] = nested.pop("enable_thinking")
            for nested_name in list(nested):
                target = mappings.get(f"{name}.{nested_name}")
                if target is None:
                    continue
                mapped[target] = nested.pop(nested_name)
            if off and not nested:
                continue
            value = nested
        mapped[mappings.get(name, name)] = value
    return mapped


def _http_failure(exc: BaseException) -> tuple[int | None, str]:
    """The HTTP status and server message carried by a failed request, if any.

    A 401/403 is the endpoint's authentication answer; without its status the
    gate cannot tell "no key supplied" from "key rejected".
    """
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = json.loads(exc.read())
        except (OSError, ValueError):
            body = None
        message = _error_message(exc.code, body) or str(exc)
        return exc.code, message
    if isinstance(exc, urllib.error.URLError):
        return None, str(exc.reason)
    return None, str(exc)


def parse_version(text: str) -> tuple[int, ...] | None:
    """Extract the leading dotted numeric version, if any.

    vLLM reports dev and container digests (e.g.
    ``0.11.2.dev280+gilded...b12xcd3ce19``) alongside or instead of a release
    number; when no leading number is present the version is unusable and the
    live probe alone is authoritative.
    """
    match = _VERSION_RE.search(text or "")
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _post(url: str, payload: dict[str, Any], key: str | None) -> tuple[int, Any]:
    """One inference request. Returns the HTTP status and parsed body."""
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
    try:
        return status, json.loads(raw.decode())
    except (ValueError, UnicodeDecodeError):
        return status, {"raw": redact(raw.decode(errors="replace")[:2048])}


def _error_message(status: int, body: Any) -> str | None:
    """The server's own error text, truncated so the evidence stays bounded."""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        elif isinstance(error, str):
            message = error
        elif body.get("message"):
            message = str(body.get("message"))
        else:
            message = ""
    else:
        message = ""
    message = str(redact(message))
    return (message[:512] + "...") if len(message) > 512 else message


def _failures(status: int, body: Any) -> list[str]:
    """Name the condition this response proves: auth, a setting, or a failure.

    Any non-2xx response is a failure, so a 404, 422 or 500 is never accepted
    as proof that a request setting is honoured; only an HTTP success with no
    named rejection is recorded as ok.
    """
    if status in (401, 403):
        return ["authentication"]
    if 200 <= status < 300:
        return []
    if status == 400:
        message = _error_message(status, body) or ""
        lowered = message.lower()
        for name in list(SAMPLING_KEYS) + [
            "reasoning_effort",
            "enable_thinking",
            "chat_template_kwargs",
        ]:
            if name.lower() in lowered:
                return [f"setting:{name}"]
        return ["setting"]
    return ["error"]


def _settings_for(setup: dict[str, Any], mappings: dict[str, str]) -> dict[str, Any]:
    """The final request settings to probe: generation plus reasoning.

    Every key the scored request carries is probed, because the workloads
    merge both groups into the outgoing body; reasoning keys other than
    ``chat_template_kwargs`` ride at the top level. The engine-appropriate
    renames are applied so the probe carries exactly the names the scored
    request will carry.
    """
    requests = setup["requests"]
    settings = dict(requests.get("generation", {}))
    for name, value in requests.get("reasoning", {}).items():
        if name == "chat_template_kwargs" and isinstance(value, dict):
            settings["chat_template_kwargs"] = dict(value)
        else:
            settings[name] = value
    return apply_setting_mappings(settings, mappings)


def _probes(settings: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """One request per individual setting, so a failure names its parameter.

    A combined request is answered by the server even when a single setting
    within it is unsupported, so each requested setting is probed on its own:
    the response that fails is the evidence for the setting it carried.
    """
    probes = []
    for name, value in settings.items():
        if name == "chat_template_kwargs":
            for kw, kw_value in value.items():
                probes.append(
                    (f"chat_template_kwargs.{kw}", {"chat_template_kwargs": {kw: kw_value}})
                )
        else:
            probes.append((name, {name: value}))
    return probes


def _scrub(value: Any, key: str | None) -> Any:
    """Rebuild a JSON-shaped value with the supplied key removed from every
    string, including unlabelled echoes the endpoint may have returned.

    The name-based redaction cannot catch a raw echo of the bearer key, so the
    live key is replaced by value before the evidence is retained.
    """
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, dict):
        return {name: _scrub(item, key) for name, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, key) for item in value]
    return value


def check(
    setup: dict[str, Any],
    *,
    include_settings: bool = False,
    key: str | None = None,
) -> dict[str, Any]:
    """Probe the endpoint and report which final settings it accepts.

    Only ``GET /v1/models`` and ``POST /v1/chat/completions`` are used. The
    result is the retained evidence for the setup or launch that asked for the
    check, so every value is bounded and free of the supplied key.
    """
    requests = setup["requests"]
    base_url = str(requests["base_url"]).rstrip("/")
    model_id = requests["model_id"]
    key = key if key is not None else runtime_api_key()
    mappings = setting_mappings(setup)

    evidence: dict[str, Any] = {
        "engine": setup.get("serving", {}).get("engine"),
        "version": setup.get("serving", {}).get("version"),
        "authenticated": key is not None,
        "model": model_id,
        "setting_mappings": mappings,
        "checks": {},
        "unsupported": [],
        "advisory": [],
    }

    # Routing + a minimal completion, without any request-setting override.
    headers: dict[str, str] = {}
    if key:
        headers["authorization"] = f"Bearer {key}"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(base_url + "/v1/models", headers=headers), timeout=30
        ) as response:
            models = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        status, message = _http_failure(exc)
        evidence["checks"]["models"] = {
            "ok": False,
            "http": status,
            "error": redact(message),
        }
        return cast(dict[str, Any], _scrub(evidence, key))
    listed = [m.get("id") for m in models.get("data", []) if isinstance(m, dict)]
    # The model list is endpoint-supplied text; redact it like every other
    # externally echoed value before the evidence is retained.
    evidence["checks"]["models"] = {
        "ok": model_id in listed,
        "models": [redact(str(identifier)) for identifier in listed[:16]],
    }

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "max_tokens": 16,
    }
    status, body = _post(base_url + "/v1/chat/completions", payload, key)
    failure = _failures(status, body)
    evidence["checks"]["completion"] = {
        "ok": not failure,
        "http": status,
        "error": _error_message(status, body) if failure else None,
    }

    if not include_settings:
        return cast(dict[str, Any], _scrub(evidence, key))

    settings = _settings_for(setup, mappings)
    probes = _probes(settings)
    for name, extra in probes:
        probe_status, probe_body = _post(
            base_url + "/v1/chat/completions", {**payload, **extra}, key
        )
        probe_failure = _failures(probe_status, probe_body)
        evidence["checks"][name] = {
            "ok": not probe_failure,
            "http": probe_status,
            "error": _error_message(probe_status, probe_body) if probe_failure else None,
        }
        if probe_failure:
            evidence["unsupported"].append(name)

    # Advisory only: the live probe above is what decides for this instance.
    # A vLLM version below a documented floor is recorded with the floor; a
    # llama.cpp build is recorded against the pinned build the mapping is
    # evidence for, with its limitations.
    version_text = str(setup.get("serving", {}).get("version") or "")
    if vllm_engine(setup):
        version = parse_version(version_text)
        for name, floor in MINIMUM_VERSION.items():
            if version is not None and version < floor:
                evidence["advisory"].append(
                    f"{name}: serving version {version} is below the documented vLLM "
                    f"minimum {'.'.join(map(str, floor))}"
                )
    if llama_engine(setup):
        evidence["advisory"].append(
            "llama.cpp: parameter and reasoning mappings were established against "
            f"build {LLAMA_VERSION_EVIDENCE} (llama-server, OpenAI-compatible "
            f"server); declared version: {version_text or 'not declared'}. The live "
            "probe is the per-instance signal; a request carrying a setting it "
            "accepts does not prove every parameter was applied."
        )
    if sglang_engine(setup):
        evidence["advisory"].append(
            "SGLang: parameter and reasoning mappings were established against "
            f"release {SGLANG_VERSION_EVIDENCE} (sglang OpenAI-compatible "
            "server); declared version: "
            f"{version_text or 'not declared'}. SGLang's request model drops "
            "unregistered fields, so a request carrying a setting it accepts "
            "does not prove every parameter was applied; the live probe is "
            "the per-instance signal."
        )
    if exllamav3_engine(setup):
        evidence["advisory"].append(
            "ExLlamaV3: parameter and reasoning mappings were established against "
            f"TabbyAPI revision {TABBYAPI_VERSION_EVIDENCE} and exllamav3 "
            f"{EXLLAMAV3_VERSION_EVIDENCE} (TabbyAPI is the official "
            "OpenAI-compatible server); declared version: "
            f"{version_text or 'not declared'}. TabbyAPI's request model drops "
            "unregistered fields, so a request carrying a setting it accepts "
            "does not prove every parameter was applied; the live probe is "
            "the per-instance signal."
        )
    return cast(dict[str, Any], _scrub(evidence, key))


def launch_gate(setup: dict[str, Any], *, key: str | None = None) -> dict[str, Any]:
    """Stop the launch until the endpoint accepts every final setting.

    Raises ``ValueError`` naming each unsupported setting (or an
    authentication/routing problem) instead of silently dropping it. Returns
    the evidence when every final setting is accepted.
    """
    key = key if key is not None else runtime_api_key()
    evidence = check(setup, include_settings=True, key=key)
    problems: list[str] = []
    for name in ("models", "completion"):
        result = evidence["checks"].get(name)
        if result is None or result.get("ok"):
            continue
        status = result.get("http")
        message = result.get("error") or ""
        if status in (401, 403):
            if key:
                problems.append(
                    "the endpoint rejected the bearer key "
                    f"({API_KEY_ENV}); verify the key is correct for this route"
                )
            else:
                problems.append(
                    "the endpoint requires authentication; set the bearer key in "
                    f"the {API_KEY_ENV} environment variable"
                )
        else:
            problems.append(f"the endpoint {name} check failed ({message or status})")
    for name in evidence.get("unsupported", []):
        detail = evidence["checks"].get(name, {}).get("error")
        problems.append(f"setting '{name}' is unsupported by the endpoint {detail or ''}")
    if problems:
        raise ValueError(
            f"{evidence.get('engine') or 'engine'} compatibility check failed before "
            "scored execution: "
            + "; ".join(problems)
            + ". Correct the settings or the endpoint and re-run; no work was started."
        )
    return evidence
