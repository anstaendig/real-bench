"""Serve fixture report data and accept local browser-test submissions."""

import os
import shutil
import tempfile
from pathlib import Path

import uvicorn

from llm_bench.ordinary_api import create_app

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="real-bench-report-") as directory:
        root = Path(directory) / "runs"
        root.mkdir()
        extra = os.environ.get("REAL_BENCH_REPORT_RUNS")
        if extra:
            shutil.copytree(Path(extra), root, dirs_exist_ok=True)
        shutil.copytree(Path("tests/fixtures/quality-readiness/runs"), root, dirs_exist_ok=True)
        shutil.copytree(Path("tests/fixtures/e2e-runs"), root, dirs_exist_ok=True)
        uvicorn.run(create_app(root), host="127.0.0.1", port=4174)
