"""Public CLI for externally operated instances and ordinary run directories."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import uvicorn
from pydantic import ValidationError

from . import launch_config, setup_state
from .ordinary_runs import (
    SAMPLING_KEYS,
    RunConfig,
    Submission,
    execute,
    read_result,
    request_cancellation,
    runtime_api_key,
)

# Locals in crash output could include the runtime API key.
app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def run(
    submission: Path,
    results: Path = Path("results"),
    run_id: str | None = typer.Option(None, help="Stable id that may be cancelled while running."),
    setup_directory: Path = typer.Option(
        Path(".real-bench/setup"),
        help="Setup state directory whose prepared speed and quality inputs gate the launch.",
    ),
    no_interactive: bool = typer.Option(
        False, help="Never prompt; fail when per-run configuration choices are missing."
    ),
    run_config: str = typer.Option(
        "",
        help=(
            "JSON object supplying the per-run configuration choices, e.g. "
            '{"reasoning_effort": "high"}; required when a prepared submission '
            "carries no run_config"
        ),
    ),
) -> None:
    """Execute a submitted performance, coding, or long-context workload.

    A submission carrying a complete ``run_config`` runs as-is, which is the
    contract agent-driven and API submissions rely on. Missing per-run choices
    prompt this terminal with historical recommendations and an unselected
    reasoning choice; an unattended launch (--no-interactive or no terminal
    input) names the choices required instead of guessing or inheriting them.
    """
    required = ", ".join(["reasoning_effort", *SAMPLING_KEYS])

    def complete_run_config(payload: Submission) -> None:
        if payload.run_config is not None:
            return
        if run_config:
            parsed = json.loads(run_config)
            if not isinstance(parsed, dict):
                raise ValueError("--run-config must be a JSON object")
            payload.run_config = RunConfig.model_validate(parsed)
            return
        if no_interactive:
            raise ValueError(f"run_config is required at launch: supply {required}")
        typer.echo("Review the per-run configuration; historical values are recommendations only.")
        try:
            payload.run_config = launch_config.collect_run_config(
                launch_config.historical_sampling(results)
            )
        except (EOFError, KeyboardInterrupt):
            raise ValueError(f"run_config is required at launch: supply {required}") from None

    try:
        payload = Submission.model_validate_json(submission.read_text())
        complete_run_config(payload)
        typer.echo(
            json.dumps(execute(results, payload, run_id, setup_directory=setup_directory), indent=2)
        )
    except ValidationError as exc:
        raise typer.BadParameter(
            json.dumps(exc.errors(include_input=False, include_context=False))
        ) from exc
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def setup(
    directory: Path = typer.Option(Path(".real-bench/setup"), help="Setup state directory."),
    answers: str | None = typer.Option(
        None,
        help=(
            "JSON object of nonsecret setup answers, e.g. "
            '{"requests.base_url": "http://host:8080", "model.tokenizer": "/path"}'
        ),
    ),
    no_interactive: bool = typer.Option(
        False, help="Never prompt; fail on missing required setup fields."
    ),
    discover: bool = typer.Option(
        False,
        help=(
            "Read the declared endpoint's instance details (model id, serving "
            "version, context window) and present them for review before entry."
        ),
    ),
) -> None:
    """Guide manual setup entry, or apply agent-driven answers without prompts.

    Saved answers resume across interruptions; execution-critical fields are
    reported instead of guessed when --no-interactive is set.
    """
    supplied: dict[str, Any] = {}
    if answers is not None:
        try:
            parsed = json.loads(answers)
        except ValueError as exc:
            raise typer.BadParameter(f"--answers must be a JSON object: {exc}") from exc
        if not isinstance(parsed, dict):
            raise typer.BadParameter("--answers must be a JSON object")
        supplied = parsed
    try:
        result = setup_state.run_setup(
            directory, supplied, interactive=not no_interactive, discover=discover
        )
    except (KeyboardInterrupt, EOFError):
        typer.echo(f"Setup interrupted; saved progress remains in {directory}")
        raise typer.Exit(code=130) from None
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2))


@app.command()
def check(
    directory: Path = typer.Option(Path(".real-bench/setup"), help="Setup state directory."),
    settings: bool = typer.Option(
        False, help="Also probe the declared request settings, as a launch would."
    ),
) -> None:
    """Run the setup compatibility check against the declared endpoint.

    Reads the saved setup state and probes the endpoint with only a model list
    and a minimal completion, never launching or reconfiguring the service. The
    result is retained in the setup state directory, separate from any scored
    run.
    """
    from . import compatibility
    from .setup_state import load_state, setup_record

    try:
        state = load_state(directory)
        if state is None:
            raise typer.BadParameter(f"no saved setup state in {directory}")
        record = setup_record(state)
        evidence = compatibility.check(
            record["declared"], include_settings=settings, key=runtime_api_key()
        )
        evidence["saved_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        setup_dir = directory / "compatibility"
        setup_dir.mkdir(parents=True, exist_ok=True)
        setup_dir.joinpath("check.json").write_text(json.dumps(evidence, indent=2) + "\n")
    except ValidationError as exc:
        raise typer.BadParameter(f"unreadable setup state in {directory}") from exc
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(evidence, indent=2))


@app.command()
def result(run_id: str, results: Path = Path("results")) -> None:
    """Read a retained run without executing any work."""
    try:
        typer.echo(json.dumps(read_result(results, run_id), indent=2))
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("cancel")
def cancel(run_id: str, results: Path = Path("results")) -> None:
    """Persist cancellation for an identified in-progress ordinary run."""
    try:
        typer.echo(json.dumps(request_cancellation(results, run_id), indent=2))
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("rebuild-index")
def rebuild_index(results: Path = Path("results")) -> None:
    """Rebuild the disposable report index from ordinary run directories."""
    from .ordinary_reports import rebuild_index as rebuild

    typer.echo(json.dumps(rebuild(results), indent=2))


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    results: Path = Path("results"),
    allow_remote: bool = typer.Option(
        False,
        help="Allow a non-loopback --host and Host header. The API is unauthenticated.",
    ),
) -> None:
    """Serve the live dashboard, run catalog, comparisons, and submission API."""
    from .ordinary_api import LOOPBACK_HOSTS, create_app

    if host.strip("[]").lower() not in LOOPBACK_HOSTS and not allow_remote:
        raise typer.BadParameter(
            f"--host {host} is not loopback and the API is unauthenticated; "
            "pass --allow-remote to expose it",
            param_hint="--host",
        )
    try:
        api = create_app(results, allow_remote=allow_remote)
        uvicorn.run(api, host=host, port=port)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
