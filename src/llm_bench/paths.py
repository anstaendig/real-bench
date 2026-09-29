"""Locations of files that ship in the source checkout rather than the wheel."""

from __future__ import annotations

from pathlib import Path


def source_checkout_root() -> Path:
    """The repository root holding ``scripts/`` and ``third_party/ruler``.

    Raises ``RuntimeError`` when they are absent, which happens for a
    non-editable install.
    """
    root = Path(__file__).resolve().parents[2]
    for required in (root / "scripts", root / "third_party" / "ruler"):
        if not required.exists():
            raise RuntimeError(
                "real-bench must be installed from a source checkout: "
                f"`uv tool install --editable .`; missing {required}"
            )
    return root
