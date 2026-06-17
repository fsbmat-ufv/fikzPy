"""Reusable TikZ examples shown in the application."""

from __future__ import annotations

from fikzpy.templates import arrows, basic_shapes, bezier, nodes


def get_template_groups() -> dict[str, dict[str, str]]:
    """Return templates grouped by menu section."""
    return {
        "Basic shapes": basic_shapes.TEMPLATES,
        "Bezier": bezier.TEMPLATES,
        "Nodes": nodes.TEMPLATES,
        "Arrows": arrows.TEMPLATES,
    }


__all__ = ["get_template_groups"]
