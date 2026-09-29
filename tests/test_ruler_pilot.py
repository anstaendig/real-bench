"""Read preparation evidence through the pilot's public report command."""

import json
import shutil
import subprocess
import sys
from pathlib import Path


def test_report_does_not_promote_short_partial_preparation_to_validated_profile(
    tmp_path: Path,
) -> None:
    (tmp_path / "workload.json").write_text(json.dumps({"planned_requests": 117}))
    (tmp_path / "preparation.json").write_text(
        json.dumps(
            [
                {
                    "task": "niah_single_1",
                    "checkpoint": 196608,
                    "exit_code": 0,
                    "elapsed_seconds": 2,
                    "samples": [
                        {"index": 0, "input_tokens": 196582, "difference_from_checkpoint": -26}
                    ],
                }
            ]
        )
    )
    result = subprocess.run(
        [sys.executable, "scripts/ruler_pilot.py", "--report", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["planned_requests"] == 117
    assert report["prepared_requests"] == 1
    assert report["exact_checkpoint_requests"] == 0
    assert report["largest_shortfall_tokens"] == 26
    assert report["preparation_validated"] is False
    assert report["full_official_score"] is None


def test_report_accepts_upstream_ceilings_but_rejects_overshoot_and_missing_coverage(
    tmp_path: Path,
) -> None:
    source = Path("tests/fixtures/pilots/ruler-123/run")
    for name in ("workload.json", "preparation.json"):
        shutil.copyfile(source / name, tmp_path / name)

    def read_report() -> dict:
        result = subprocess.run(
            [sys.executable, "scripts/ruler_pilot.py", "--report", str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    report = read_report()
    assert report["preparation_validated"] is True
    assert report["prepared_requests"] == 117
    assert report["input_ranges"]["196608"] == {"samples": 39, "minimum": 188130, "maximum": 196608}
    assert report["full_profile_executed"] is False
    preparations = json.loads((tmp_path / "preparation.json").read_text())
    preparations[0]["samples"][0]["input_tokens"] = 32769
    (tmp_path / "preparation.json").write_text(json.dumps(preparations))
    assert read_report()["preparation_validated"] is False
    preparations[0] = preparations[1]
    (tmp_path / "preparation.json").write_text(json.dumps(preparations))
    assert read_report()["preparation_validated"] is False
