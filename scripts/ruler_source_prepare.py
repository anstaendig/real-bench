"""Obtain the pinned upstream RULER source and its hydrated data.

Executed through the pinned RULER runtime environment (``uv run --project
third_party/ruler``) so the generator dependencies (``bs4``, ``html2text``,
``tqdm``) are present. The helper clones the pinned RULER revision into a local
preparation directory, hydrates the synthetic-data JSON inputs using the
repository's own download helpers, and writes a sha256 manifest of the hydrated
data so a later launch can verify the checkout and its data were not tampered
with. No model weights are involved; this is a small git checkout plus a few
public data files. The source tracks ``english_words.json`` through git-lfs,
so a plain checkout leaves a pointer; the preparation fetches the object
after checkout and fails actionably when ``git lfs`` is unavailable.

A git-lfs installation is required in addition to git and network access.

The manifest is written *outside* the git checkout so the checkout stays clean
relative to its pinned revision.

Exit codes:

- ``0`` preparation complete;
- ``3`` a human-only prerequisite is missing (no network / gated source); the
  stderr message names the prerequisite and how to supply it;
- ``1`` any other clone, checkout, download, or save failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# The data files the pinned RULER synthetic generators read from
# ``scripts/data/synthetic/json/``. ``english_words.json`` is committed to the
# source; the other three are hydrated by the repository's own helpers.
DATA_FILES = ("squad.json", "hotpotqa.json", "PaulGrahamEssays.json", "english_words.json")
CLONE_TIMEOUT = 600
CHECKOUT_TIMEOUT = 300
DOWNLOAD_TIMEOUT = 3600


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(args: list[str], timeout: int) -> str:
    completed = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise subprocess.CalledProcessError(
            completed.returncode,
            ["git", *args],
            output=completed.stdout,
            stderr=detail or None,
        )
    return (completed.stdout or "").strip()


def _human_only(detail: str) -> bool:
    lowered = detail.lower()
    tokens = ("could not resolve host", "unrecognized host", "timed out", "network")
    return any(token in lowered for token in tokens)


def hydration_problems(json_dir: Path) -> list[str]:
    """Problems that keep a hydrated data directory from being trusted.

    The pinned helpers can exit 0 with an empty or incomplete output: the QA
    script's exit status is its final command (a failed SQuAD wget is masked
    by a successful HotpotQA download) and the essay helper catches per-URL
    failures. Every file must therefore actually hold the structure the
    upstream generators read before the manifest a later launch trusts is
    published. ``english_words.json`` is a git-lfs word list the checkout may
    leave as a pointer; it is validated by content, not JSON structure.
    """
    problems: list[str] = []
    for name in ("squad.json", "hotpotqa.json"):
        path = json_dir / name
        try:
            parsed = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            problems.append(f"{name}: not valid JSON ({exc})")
            continue
        if name == "hotpotqa.json":
            if not isinstance(parsed, list) or not parsed:
                problems.append(f"{name}: missing or empty data list")
            continue
        data = parsed.get("data") if isinstance(parsed, dict) else None
        if not isinstance(data, list) or not data:
            problems.append(f"{name}: missing or empty data list")
    path = json_dir / "PaulGrahamEssays.json"
    try:
        parsed = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        problems.append(f"PaulGrahamEssays.json: not valid JSON ({exc})")
    else:
        text = parsed.get("text") if isinstance(parsed, dict) else None
        if not isinstance(text, str) or len(text) < 1000:
            problems.append("PaulGrahamEssays.json: missing or incomplete essay text")
    words = json_dir / "english_words.json"
    try:
        content = words.read_bytes()
    except OSError:
        content = b""
    if len(content) < 200 or content.startswith(b"version https://git-lfs.github.com"):
        problems.append("english_words.json: word list missing or an unfetched git-lfs pointer")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", required=True, help="Pinned RULER git source.")
    parser.add_argument("--revision", required=True, help="Pinned RULER revision (full sha).")
    parser.add_argument("--directory", required=True, help="Directory to save the checkout into.")
    parser.add_argument("--manifest", required=True, help="Path to write the sha256 manifest.")
    args = parser.parse_args()

    target = Path(args.directory).resolve()
    manifest_path = Path(args.manifest).resolve()

    # A re-preparation replaces the previous checkout so a stale or tampered
    # tree (including a previously hydrated data file) cannot survive beside
    # the new manifest.
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        _git(["clone", args.source_url, str(target)], CLONE_TIMEOUT)
        _git(["-C", str(target), "checkout", args.revision], CHECKOUT_TIMEOUT)
        head = _git(["-C", str(target), "rev-parse", "HEAD"], CHECKOUT_TIMEOUT)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = str(exc)
        if isinstance(exc, subprocess.TimeoutExpired):
            detail = f"cloning the pinned RULER source timed out after {CLONE_TIMEOUT} seconds"
        elif isinstance(exc, OSError):
            detail = "git is not available on this host; install git"
        elif exc.stderr:
            detail = exc.stderr
        if _human_only(detail):
            print(
                "The pinned RULER source could not be reached. This host needs "
                "outbound network access to "
                f"{args.source_url}. Restore network access and re-run bench setup; "
                "the setup state keeps the interrupted preparation.",
                file=sys.stderr,
            )
            return 3
        print(f"RULER source clone failed: {detail}", file=sys.stderr)
        return 1

    try:
        # ``english_words.json`` is tracked through git-lfs: without the lfs
        # smudge filter the checkout holds a 132-byte pointer, so the word
        # list the generators read would be missing. Fetch it explicitly.
        _git(["-C", str(target), "lfs", "pull"], CHECKOUT_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(
            f"Fetching the RULER git-lfs word list timed out after {CHECKOUT_TIMEOUT} seconds. "
            "Re-run bench setup; the setup state keeps the interrupted preparation.",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.output or "").strip()
        print(
            "Fetching the RULER git-lfs word list (english_words.json) failed"
            + (f": {detail}" if detail else "")
            + ". Install git-lfs (for example ``brew install git-lfs`` or "
            "``apt-get install git-lfs``) and re-run bench setup; the setup "
            "state keeps the interrupted preparation.",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(
            f"git is not available on this host; install git: {exc}",
            file=sys.stderr,
        )
        return 1

    json_dir = target / "scripts" / "data" / "synthetic" / "json"
    if not json_dir.is_dir():
        print(
            "The pinned RULER source has no scripts/data/synthetic/json directory",
            file=sys.stderr,
        )
        return 1

    detail: str | None = None
    try:
        # squad.json + hotpotqa.json via the repository's own helper (wget with
        # an HTTPS fallback for the HotpotQA mirror); a failed helper can leave
        # an empty output file, so its exit status is authoritative.
        completed = subprocess.run(
            ["bash", str(json_dir / "download_qa_dataset.sh")],
            cwd=str(json_dir),
            capture_output=True,
            text=True,
            timeout=DOWNLOAD_TIMEOUT,
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                ["bash", "download_qa_dataset.sh"],
                output=completed.stdout,
                stderr=(completed.stderr or "").strip() or None,
            )
        # PaulGrahamEssays.json via the repository's own helper (needs the
        # pinned bs4 / html2text / tqdm, present in this environment).
        completed = subprocess.run(
            [sys.executable, str(json_dir / "download_paulgraham_essay.py")],
            cwd=str(json_dir),
            capture_output=True,
            text=True,
            timeout=DOWNLOAD_TIMEOUT,
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                ["python", "download_paulgraham_essay.py"],
                output=completed.stdout,
                stderr=(completed.stderr or "").strip() or None,
            )
    except subprocess.TimeoutExpired:
        detail = f"hydrating the RULER data timed out after {DOWNLOAD_TIMEOUT} seconds"
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.output or "").strip() or f"exited {exc.returncode}"
    except OSError as exc:
        detail = str(exc)
    if detail is not None:
        if _human_only(detail):
            print(
                "Hydrating the RULER data needs outbound network access (SQuAD, HotpotQA, "
                "Paul Graham essays). Restore network access and re-run bench setup; the "
                "setup state keeps the interrupted preparation.",
                file=sys.stderr,
            )
            return 3
        print(f"RULER data hydration failed: {detail}", file=sys.stderr)
        return 1

    problems = hydration_problems(json_dir)
    if problems:
        print(
            "The RULER data hydration left unusable files: "
            + "; ".join(problems)
            + ". Re-run bench setup to retry the download.",
            file=sys.stderr,
        )
        return 1

    data = {name: _sha256(json_dir / name) for name in DATA_FILES}
    manifest = {
        "source": args.source_url,
        "revision": head,
        "data": data,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"RULER source ready: revision {head}, {len(DATA_FILES)} data file(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
