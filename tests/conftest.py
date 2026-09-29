"""Opt-in deterministic transports; live OCI checks remain separate evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest


def option_value(command, name: str) -> str:
    """Return an option's value from either ``name value`` or ``name=value`` arguments."""
    args = [str(part) for part in command]
    for position, part in enumerate(args):
        if part == name:
            return args[position + 1]
        if part.startswith(name + "="):
            return part.split("=", 1)[1]
    raise ValueError(f"{name} is not in the command")


@pytest.fixture
def external_tool_process_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let retained-export fixtures drive the owned-process public seam.

    Ordinary-run tests replace ``subprocess.run`` with a controlled external
    tool. Production now owns a ``Popen`` process group, so this test-only
    adapter presents that controlled completed process through Popen's public
    surface instead of bypassing the production execution boundary.
    """
    standard_run = subprocess.run
    standard_popen = subprocess.Popen

    class ControlledProcess:
        def __init__(self, command, **kwargs) -> None:
            self.args = command
            self._completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=kwargs.get("timeout"),
            )
            self.returncode = self._completed.returncode
            self.pid = 0

        def communicate(self, timeout=None):
            return self._completed.stdout, self._completed.stderr

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def send_signal(self, signal):
            return None

        def terminate(self):
            return None

        def kill(self):
            return None

    def create_process(command, **kwargs):
        if subprocess.run is standard_run:
            return standard_popen(command, **kwargs)
        return ControlledProcess(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", create_process)


def fake_preparation(command: list[Any], **kwargs: Any) -> subprocess.CompletedProcess:
    """Serve the guided-setup preparation subprocesses deterministically.

    The speed preparation fetches the pinned speed tools and tokenizer; the
    quality preparation verifies the Docker daemon and the pinned grading
    image; the long-context preparation clones the pinned RULER source and
    hydrates its data. All run in-memory: no network, no daemon, no git remote.
    """

    args = [str(part) for part in command]
    if any(p.endswith("ruler_source_prepare.py") for p in args):
        target = Path(args[args.index("--directory") + 1])
        manifest_path = Path(args[args.index("--manifest") + 1])
        revision = args[args.index("--revision") + 1]
        source_url = args[args.index("--source-url") + 1]
        json_dir = target / "scripts" / "data" / "synthetic" / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        (target / ".git").mkdir(parents=True, exist_ok=True)
        (target / ".git" / "FAKE_REV").write_text(revision + "\n")
        # The scoring configuration the long-context runner reads from the
        # source checkout, from the committed pilot fixtures.
        fixture = Path(__file__).parent / "fixtures" / "ruler-upstream"
        (target / "scripts").mkdir(parents=True, exist_ok=True)
        (target / "scripts").joinpath("synthetic.yaml").write_bytes(
            (fixture / "synthetic.yaml").read_bytes()
        )
        (target / "scripts" / "eval" / "synthetic").mkdir(parents=True, exist_ok=True)
        (target / "scripts" / "eval" / "synthetic" / "constants.py").write_bytes(
            (fixture / "constants.py.txt").read_bytes()
        )
        data = {
            "squad.json": '{"fake": "squad"}',
            "hotpotqa.json": '{"fake": "hotpotqa"}',
            "PaulGrahamEssays.json": '{"fake": "essays"}',
            "english_words.json": '{"fake": "words"}',
        }
        digests = {}
        for name, content in data.items():
            raw = content.encode()
            (json_dir / name).write_bytes(raw)
            digests[name] = hashlib.sha256(raw).hexdigest()
        manifest = {
            "source": source_url,
            "revision": revision,
            "data": digests,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return subprocess.CompletedProcess(
            command, 0, "", f"RULER source ready: revision {revision}"
        )
    if args and args[0] == "git" and "-C" in args:
        repository = Path(args[args.index("-C") + 1])
        if "rev-parse" in args:
            marker = repository / ".git" / "FAKE_REV"
            if marker.is_file():
                return subprocess.CompletedProcess(command, 0, marker.read_text(), "")
            return subprocess.CompletedProcess(command, 1, "", "not a git repository")
        if "diff" in args:
            # A `.git/DIRTY` marker emulates a modified tracked script.
            code = 1 if (repository / ".git" / "DIRTY").is_file() else 0
            return subprocess.CompletedProcess(command, code, "", "")
        return subprocess.CompletedProcess(command, 1, "", "unhandled git command")
    if "-c" in command:
        return subprocess.CompletedProcess(command, 0, stdout="0.12.0\n5.17.0\n", stderr="")
    if any(str(p).endswith("speed_prepare.py") for p in command):
        source = option_value(command, "--source")
        target = Path(option_value(command, "--directory"))
        target.mkdir(parents=True, exist_ok=True)
        files = {
            "tokenizer.json": '{"version":"1.0","model":{"type":"WordLevel"}}',
            "tokenizer_config.json": (
                '{"tokenizer_class":"PreTrainedTokenizerFast","chat_template":"x"}'
            ),
            # The explicit template the long-context preparation sizes with
            # when no override is declared.
            "chat_template.jinja": "{{ prompts }} {{ generation_prompt }}",
        }
        digest = {}
        total = 0
        for name, content in files.items():
            raw = content.encode()
            (target / name).write_bytes(raw)
            digest[name] = hashlib.sha256(raw).hexdigest()
            total += len(raw)
        manifest = {
            "source": source,
            "files": digest,
            "total_bytes": total,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return subprocess.CompletedProcess(
            command, 0, "", f"tokenizer ready: {len(files)} file(s), {total} bytes from {source}"
        )
    if str(command[0]) == "docker":
        if command[1:2] == ["version"]:
            return subprocess.CompletedProcess(command, 0, stdout=DOCKER_VERSION_OUTPUT, stderr="")
        if command[1:2] == ["image"] and command[2:3] == ["inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="[]\n", stderr="")
        if command[1:2] == ["pull"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="pulled")
        if command[1:2] == ["run"] and "uname" in " ".join(str(p) for p in command):
            return subprocess.CompletedProcess(command, 0, stdout="x86_64\n", stderr="")
        raise LookupError(f"unhandled preparation command: {command}")
    raise LookupError(f"unhandled preparation command: {command}")


DOCKER_VERSION_OUTPUT = (
    "Client:\n"
    " Version:           28.1.0\n"
    " API version:       1.49\n"
    " Go version:        go1.23.4\n"
    " OS/Arch:           darwin/arm64\n"
    "\n"
    "Server: Docker Engine - Community\n"
    " Version:           28.1.0\n"
    " API version:       1.49 (minimum version 1.24)\n"
    " OS/Arch:           linux/amd64\n"
)


@pytest.fixture
def preparation_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic guided-setup preparation: the pinned-tools probe, tokenizer
    fetch, and the quality docker commands never hit the network or a daemon.
    Tests overriding ``subprocess.run`` route preparation commands through
    :func:`fake_preparation`."""
    monkeypatch.setattr(subprocess, "run", fake_preparation)


@pytest.fixture(autouse=True)
def plain_cli_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI output plain and unwrapped so tests can match message text.

    Typer forces colored, boxed output when it detects CI (for example
    ``GITHUB_ACTIONS``), which splits phrases across lines.
    """
    for name in ("GITHUB_ACTIONS", "FORCE_COLOR", "GITLAB_CI", "TERMINAL_WIDTH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("_TYPER_FORCE_DISABLE_TERMINAL", "1")
