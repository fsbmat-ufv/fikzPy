"""Application entry point."""

from __future__ import annotations


def main() -> int:
    """Start the desktop application."""
    try:
        from fikzpy.gui.main_window import run_app
    except ImportError as exc:
        message = (
            "Could not import the graphical interface. Install the desktop "
            "dependencies with: python -m pip install -r requirements.txt"
        )
        raise SystemExit(message) from exc

    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
