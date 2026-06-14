"""Prepare documentation for GitHub Pages deployment."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    """Validate that the documentation folder has the expected entry point."""
    docs_dir = Path("docs")
    index = docs_dir / "index.md"
    if not index.exists():
        raise SystemExit("docs/index.md was not found.")

    print("Documentation is ready for GitHub Pages from the docs/ folder.")
    print("Configure GitHub Pages to publish from the main branch /docs directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
