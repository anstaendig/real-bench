"""Fetch a tokenizer for the pinned performance environment.

Executed through the project's pinned tool environment (``uv run --with
aiperf==<pin> --with transformers==<pin>``) so the runner never downloads model
weights and the tokenizer the speed workloads size with is exactly the one the
performance tools would load. The helper downloads tokenizer files only, saves
them into a local preparation directory, and writes a sha256 manifest of what
it wrote so a later launch can verify the download was not tampered with.

Exit codes:

- ``0`` preparation complete;
- ``3`` a human-only prerequisite is missing (gated source); the stderr
  message names the prerequisite and how to supply it;
- ``1`` any other fetch or save failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Tokenizer repo id or local directory.")
    parser.add_argument("--directory", required=True, help="Directory to save the tokenizer into.")
    args = parser.parse_args()

    source_path = Path(args.source).expanduser()
    source_is_local = source_path.is_dir()
    if not source_is_local:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            print(f"transformers is unavailable in the pinned environment: {exc}", file=sys.stderr)
            return 1
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.source)
        except Exception as exc:  # noqa: BLE001 - the message is the operator's report
            detail = str(exc)
            lowered = detail.lower()
            if "401" in detail or "403" in detail or "gated" in lowered or "token" in lowered:
                print(
                    "The tokenizer source is gated and needs a human-only Hugging Face token. "
                    "Create a token at https://huggingface.co/settings/tokens, then run "
                    "export HF_TOKEN=<token> in the terminal that runs the benchmark, and "
                    "re-run bench setup. The setup state keeps the interrupted preparation.",
                    file=sys.stderr,
                )
                return 3
            print(f"Tokenizer fetch failed: {detail}", file=sys.stderr)
            return 1

    target = Path(args.directory).resolve()
    if target.exists():
        # A re-preparation replaces the previous resource so a tokenizer
        # template removed between runs cannot survive as a stale loadable
        # file beside the new manifest.
        shutil.rmtree(target)
    if source_is_local:
        shutil.copytree(source_path, target)
    else:
        target.mkdir(parents=True)
        tokenizer.save_pretrained(target)

    # Manifest from the directory as written: every file under the target,
    # including nested chat-template files (``chat_templates/<name>.jinja``),
    # keyed by its path relative to the target so verification and launch
    # agree on where each file lives.
    files: dict[str, str] = {}
    total = 0
    for path in sorted(p for p in target.rglob("*") if p.is_file()):
        relative = path.relative_to(target).as_posix()
        files[relative] = _sha256(path)
        total += path.stat().st_size

    manifest = {
        "source": args.source,
        "files": files,
        "total_bytes": total,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(
        f"tokenizer ready: {len(files)} file(s), {total} bytes from {args.source}", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
