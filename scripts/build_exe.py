"""Build a Windows executable with PyInstaller."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    """Run PyInstaller with the recommended fikzPy options."""
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--windowed",
        "--name",
        "fikzPy",
        "fikzpy/main.py",
    ]
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
