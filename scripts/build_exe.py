"""Build a Windows executable with PyInstaller."""

from __future__ import annotations

import subprocess
import sys
import argparse
import os
from pathlib import Path


def main() -> int:
    """Run PyInstaller with the recommended fikzPy options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onedir", action="store_true", help="Build the offline installer's application directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onedir" if args.onedir else "--onefile",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        "fikzPy",
        "--paths", str(root),
        "--collect-all", "inkex",
        "--collect-all", "svg2tikz",
        "--collect-all", "skimage",
        "--copy-metadata", "scikit-image",
        str(root / "installer" / "entrypoint.py"),
    ]
    environment = os.environ.copy()
    if args.onedir:
        # A developer PATH can contain incompatible ICU/OpenSSL DLLs from other
        # applications. Resolve Windows system DLLs and wheel-local DLLs only.
        windows = Path(environment.get("SystemRoot", "C:/Windows"))
        environment["PATH"] = os.pathsep.join([
            str(Path(sys.executable).parent), str(Path(sys.base_prefix) / "DLLs"),
            str(windows / "System32"), str(windows),
        ])
    subprocess.run(command, cwd=root, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
