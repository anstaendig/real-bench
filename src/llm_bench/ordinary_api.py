"""Public ordinary-directory submission and result reads; no database required."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import setup_state
from .ordinary_reports import comparison, rebuild_index
from .ordinary_runs import Submission, execute, read_result, request_cancellation
from .report_projection import compact_report

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _host_name(netloc: str) -> str:
    """The host part of a Host header or Origin netloc, without port or brackets."""
    netloc = netloc.strip().lower()
    if netloc.startswith("["):
        return netloc[1:].split("]", 1)[0]
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


def _guard_requests(app: FastAPI, allow_remote: bool) -> None:
    """Reject DNS-rebinding and cross-site requests unless remote access was requested."""

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Any:
        host = request.headers.get("host", "")
        if not allow_remote and _host_name(host) not in LOOPBACK_HOSTS:
            return JSONResponse(status_code=400, content={"detail": "Host not allowed"})
        origin = request.headers.get("origin")
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and origin is not None:
            origin_netloc = origin.split("://", 1)[-1].split("/", 1)[0]
            if origin_netloc.strip().lower() != host.strip().lower():
                return JSONResponse(status_code=403, content={"detail": "Cross-origin request"})
        return await call_next(request)


def _within(path: str, directory: Path) -> bool:
    try:
        Path(path).resolve().relative_to(directory.resolve())
    except (ValueError, OSError):
        return False
    return True


def create_app(results: Path = Path("results"), allow_remote: bool | None = None) -> FastAPI:
    """Build the API. ``allow_remote=None`` skips request guards (in-process use and tests);
    ``bench serve`` passes a bool, which enables Host and Origin checks."""
    app = FastAPI(title="real-bench")
    setup_directory = results / "setup"
    if allow_remote is not None:
        _guard_requests(app, allow_remote)

    @app.exception_handler(RequestValidationError)
    async def invalid_submission(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's default response echoes the entire rejected setup, including credentials.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
                    for error in exc.errors()
                ]
            },
        )

    @app.get("/", response_class=HTMLResponse)
    @app.get("/reports", response_class=HTMLResponse)
    def report_view() -> str:
        return Path(__file__).with_name("report.html").read_text()

    @app.get("/assets/{asset}")
    def report_asset(asset: str) -> FileResponse:
        if asset not in {"report.js", "report.css"}:
            raise HTTPException(404, "Asset not found")
        return FileResponse(Path(__file__).with_name(asset))

    @app.get("/api/dashboard")
    def dashboard() -> dict[str, Any]:
        from .dashboard import dashboard_snapshot

        return dashboard_snapshot(results)

    @app.get("/api/reports")
    def reports(
        filter: list[str] = Query(default=[]),
        sort: str = "run_id",
        descending: bool = False,
        run_id: list[str] = Query(default=[]),
        compact: bool = False,
    ) -> dict[str, Any]:
        try:
            return comparison(results, filter, sort, descending, run_id, compact=compact)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/reports/rebuild")
    def rebuild(compact: bool = False) -> dict[str, Any]:
        data = rebuild_index(results)
        return compact_report(data) if compact else data

    @app.post("/api/setup")
    def setup(body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply nonsecret setup answers and report the resumable setup state.

        The API is agent-driven: answers are supplied in the body and missing
        critical fields are reported instead of prompting.
        """
        data = dict(body or {})
        data.pop("directory", None)  # the setup directory is fixed by the server
        discover_value = data.pop("discover", False)
        provided = data.pop("answers", None)
        if provided is None and data:
            provided = data
        if provided is not None and not isinstance(provided, dict):
            raise HTTPException(422, "answers must be an object of category.field paths")
        if not isinstance(discover_value, bool):
            raise HTTPException(422, "discover must be a boolean")
        try:
            return setup_state.run_setup(
                setup_directory, provided or {}, interactive=False, discover=discover_value
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/setup")
    def saved_setup() -> dict[str, Any]:
        """Read the saved resumable setup state without executing any work."""
        try:
            state = setup_state.load_state(setup_directory)
        except (ValueError, OSError) as exc:  # includes pydantic ValidationError
            raise HTTPException(422, "Saved setup state is unreadable") from exc
        if state is None:
            raise HTTPException(404, "No saved setup state")
        return setup_state.summarise(state, setup_directory)

    @app.post("/api/runs", status_code=201)
    def submit(submission: Submission, request: Request) -> dict[str, Any]:
        try:
            return execute(
                results,
                submission,
                request.headers.get("x-real-bench-run-id"),
                setup_directory=setup_directory,
                restrict_paths=True,
            )
        except (ValueError, OSError) as exc:
            detail = str(exc)
            status = 409 if detail == "run_id already exists" else 422
            raise HTTPException(status, detail) from exc

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str) -> dict[str, Any]:
        try:
            return request_cancellation(results, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Run not found") from exc

    @app.get("/api/runs/{run_id}")
    def result(run_id: str, compact: bool = False) -> dict[str, Any]:
        try:
            data = read_result(results, run_id)
            return compact_report(data) if compact else data
        except FileNotFoundError as exc:
            raise HTTPException(404, "Run not found") from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(422, "Run record is unreadable") from exc

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def view(run_id: str) -> str:
        data = result(run_id, compact=True)
        if data.get("workload", {}).get("profile") == "default-suite":
            heading = (
                "Complete default-suite results"
                if data.get("status") == "completed"
                else "Partial default-suite results"
            )
            description = (
                "Both performance scenarios, 164 HumanEval+ generations, and 117 sampled "
                "RULER requests, with phase timing, coverage, and partial outcomes."
            )
        elif data.get("coding"):
            heading = "HumanEval+ coding results"
            description = (
                "164 problems, one solution each. Accuracy and uncertainty cover graded tasks; "
                "inspect coverage and failures. Standalone code quality, not coding-agent success."
            )
        elif data.get("ruler"):
            heading = "Sampled RULER long-context results"
            description = (
                "117 requests across 13 task types and three nominal input ceilings. "
                "Sampled diagnostic coverage with limited precision; "
                "not a full official RULER score."
            )
        elif data.get("fresh_context_curve"):
            heading = "Fresh-context speed curve"
            description = (
                "Six context checkpoints, two output targets, and five measured repetitions."
            )
        elif data.get("growing_session_curve"):
            heading = "Growing-session speed curve"
            description = (
                "Accumulating history across turns: six context checkpoints, two output targets, "
                "and five measured repetitions per session."
            )
        else:
            heading = "Validation run"
            description = "One speed condition; not a completed benchmark."
        fallback = (
            "<noscript>"
            f"<h1>{heading}</h1><p>{description}</p>"
            "<pre>" + html.escape(json.dumps(data, indent=2)) + "</pre></noscript>"
        )
        return report_view().replace("</body>", fallback + "</body>")

    return app
