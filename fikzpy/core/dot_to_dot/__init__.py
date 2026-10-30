"""Independent Ponto a Ponto (dot-to-dot) mode.

This subpackage implements a self-contained pipeline that turns an SVG figure
into a numbered dot-to-dot drawing. It reuses the intermediate SVG produced by
the project's Visual tracing stage (working on an independent copy) but has its
own geometry, sampling, label placement and TikZ export, so it never changes
the behaviour of the Classic, Visual or Contornos modes.
"""

from __future__ import annotations

from fikzpy.core.dot_to_dot.config import (
    DotToDotConfig,
    DotToDotGenerationMode,
    NumberingDirection,
    SelectedSegment,
)
from fikzpy.core.dot_to_dot.pipeline import DotToDotError, DotToDotResult, run_dot_to_dot

__all__ = [
    "DotToDotConfig",
    "DotToDotError",
    "DotToDotGenerationMode",
    "DotToDotResult",
    "NumberingDirection",
    "SelectedSegment",
    "run_dot_to_dot",
]
