"""TikZ style conversion for semantic primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any

from fikzpy.core.semantic_geometry import FillStyle, RGBColor, StrokeStyle
from fikzpy.core.tikz_formatting import TikzCoordinateFormatter


class TikzStyleGroupingMode(Enum):
    """Supported style grouping strategies."""

    NONE = "none"
    SCOPES = "scopes"
    NAMED_STYLES = "named_styles"


class TikzFillMode(Enum):
    """Configured fill handling."""

    PRESERVE = "preserve"
    DISABLE = "disable"


@dataclass(frozen=True)
class TikzStyle:
    """A deterministic TikZ option list."""

    options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", tuple(str(option) for option in self.options if str(option).strip()))

    @property
    def empty(self) -> bool:
        """Return whether this style has no explicit TikZ options."""
        return not self.options

    @property
    def signature(self) -> tuple[str, ...]:
        """Return a stable style signature for grouping and named styles."""
        return self.options

    def option_text(self) -> str:
        """Return the TikZ bracket option text without surrounding brackets."""
        return ", ".join(self.options)

    def to_dict(self) -> dict[str, Any]:
        """Return serializable style diagnostics."""
        return {"options": list(self.options)}


@dataclass(frozen=True)
class TikzStyleConfig:
    """Minimal style settings consumed by the style converter."""

    style_precision: int = 3
    line_width_unit: str = "pt"
    default_line_width: float = 1.0
    default_stroke_color: RGBColor = field(default_factory=RGBColor.black)
    default_fill_color: RGBColor | None = None
    omit_default_styles: bool = True
    allow_fill: bool = True
    allow_opacity: bool = True
    allow_dash_patterns: bool = True

    def __post_init__(self) -> None:
        if int(self.style_precision) < 0:
            raise ValueError("style_precision must be non-negative.")
        object.__setattr__(self, "style_precision", int(self.style_precision))
        width = float(self.default_line_width)
        if not isfinite(width) or width <= 0.0:
            raise ValueError("default_line_width must be finite and positive.")
        object.__setattr__(self, "default_line_width", width)
        if not isinstance(self.default_stroke_color, RGBColor):
            raise TypeError("default_stroke_color must be RGBColor.")
        if self.default_fill_color is not None and not isinstance(self.default_fill_color, RGBColor):
            raise TypeError("default_fill_color must be RGBColor or None.")
        for name in ("omit_default_styles", "allow_fill", "allow_opacity", "allow_dash_patterns"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")


@dataclass(frozen=True)
class TikzStyleConversion:
    """Style conversion result for one primitive."""

    style: TikzStyle
    visible: bool
    draws_stroke: bool
    fills: bool


def color_to_tikz(color: RGBColor) -> str:
    """Return a robust TikZ RGB color expression."""
    if not isinstance(color, RGBColor):
        raise TypeError("color must be RGBColor.")
    return f"{{rgb,255:red,{color.red};green,{color.green};blue,{color.blue}}}"


def style_for_primitive(
    *,
    stroke: StrokeStyle,
    fill: FillStyle | None,
    opacity: float | None,
    metadata: Mapping[str, Any],
    config: TikzStyleConfig,
    formatter: TikzCoordinateFormatter,
    warn: Callable[[str, str], None],
) -> TikzStyleConversion:
    """Convert semantic stroke/fill fields into TikZ options."""
    if not isinstance(stroke, StrokeStyle):
        raise TypeError("stroke must be StrokeStyle.")
    if fill is not None and not isinstance(fill, FillStyle):
        raise TypeError("fill must be FillStyle or None.")

    options: list[str] = []
    stroke_opacity = 1.0 if stroke.opacity is None else float(stroke.opacity)
    overall_opacity = 1.0 if opacity is None else float(opacity)
    draws_stroke = stroke_opacity > 0.0 and overall_opacity > 0.0
    fills = bool(fill is not None and config.allow_fill and (fill.opacity is None or fill.opacity > 0.0) and overall_opacity > 0.0)

    if fill is not None and not config.allow_fill:
        warn("fill_disabled", "Fill style was ignored because allow_fill=False.")

    if not draws_stroke and fills:
        options.append("draw=none")
    elif draws_stroke:
        if not (config.omit_default_styles and stroke.color == config.default_stroke_color):
            options.append(f"draw={color_to_tikz(stroke.color)}")
        if not (config.omit_default_styles and abs(stroke.width - config.default_line_width) <= 1e-12):
            options.append(
                "line width="
                + formatter.format_length(stroke.width, unit=config.line_width_unit, precision=config.style_precision)
            )
        if stroke.line_cap:
            options.append(f"line cap={_sanitize_style_keyword(stroke.line_cap)}")
        if stroke.line_join:
            options.append(f"line join={_sanitize_style_keyword(stroke.line_join)}")
        if stroke.dash_pattern:
            if config.allow_dash_patterns:
                options.append(_dash_pattern(stroke.dash_pattern, formatter, config))
            else:
                warn("dash_pattern_disabled", "Dash pattern was ignored because allow_dash_patterns=False.")

    if fills and fill is not None:
        if not (config.omit_default_styles and config.default_fill_color is not None and fill.color == config.default_fill_color):
            options.append(f"fill={color_to_tikz(fill.color)}")
        if fill.opacity is not None and fill.opacity < 1.0:
            if config.allow_opacity:
                options.append(f"fill opacity={formatter.format_number(fill.opacity, precision=config.style_precision)}")
            else:
                warn("opacity_disabled", "Fill opacity was ignored because allow_opacity=False.")

    if draws_stroke and stroke.opacity is not None and stroke.opacity < 1.0:
        if config.allow_opacity:
            options.append(f"draw opacity={formatter.format_number(stroke.opacity, precision=config.style_precision)}")
        else:
            warn("opacity_disabled", "Stroke opacity was ignored because allow_opacity=False.")

    if opacity is not None and opacity < 1.0:
        if config.allow_opacity:
            options.append(f"opacity={formatter.format_number(opacity, precision=config.style_precision)}")
        else:
            warn("opacity_disabled", "Overall opacity was ignored because allow_opacity=False.")

    rounded_corners = _metadata_number(metadata, ("rounded_corners", "rounded_corner_radius"))
    if rounded_corners is not None:
        options.append(
            "rounded corners="
            + formatter.format_length(rounded_corners, unit=config.line_width_unit, precision=config.style_precision)
        )

    fill_rule = _fill_rule(metadata)
    if fill_rule:
        normalized = fill_rule.strip().lower().replace("-", "")
        if normalized in {"evenodd", "evenoddrule"}:
            options.append("even odd rule")
        elif normalized not in {"nonzero", "nonzerorule", "winding"}:
            warn("fill_rule_partial", f"Fill rule {fill_rule!r} is not fully supported.")

    visible = draws_stroke or fills
    return TikzStyleConversion(TikzStyle(tuple(options)), visible, draws_stroke, fills)


def coerce_style_grouping_mode(value: TikzStyleGroupingMode | str) -> TikzStyleGroupingMode:
    """Coerce style grouping values."""
    if isinstance(value, TikzStyleGroupingMode):
        return value
    normalized = str(value).strip().lower()
    for item in TikzStyleGroupingMode:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported TikZ style grouping mode: {value!r}")


def coerce_fill_mode(value: TikzFillMode | str) -> TikzFillMode:
    """Coerce fill mode values."""
    if isinstance(value, TikzFillMode):
        return value
    normalized = str(value).strip().lower()
    for item in TikzFillMode:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported TikZ fill mode: {value!r}")


def _dash_pattern(
    dash_pattern: tuple[float, ...],
    formatter: TikzCoordinateFormatter,
    config: TikzStyleConfig,
) -> str:
    parts: list[str] = []
    for index, value in enumerate(dash_pattern):
        prefix = "on" if index % 2 == 0 else "off"
        parts.append(f"{prefix} {formatter.format_length(value, unit=config.line_width_unit, precision=config.style_precision)}")
    return "dash pattern=" + " ".join(parts)


def _sanitize_style_keyword(value: str) -> str:
    text = str(value).strip().lower().replace("_", " ")
    allowed = "".join(char for char in text if char.isalnum() or char in {" ", "-"})
    return allowed or "round"


def _metadata_number(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = dict(metadata).get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(number) and number >= 0.0:
            return number
    return None


def _fill_rule(metadata: Mapping[str, Any]) -> str | None:
    data = dict(metadata)
    direct = data.get("fill_rule")
    if direct is not None:
        return str(direct)
    resolved = data.get("resolved_style")
    if isinstance(resolved, Mapping) and resolved.get("fill_rule") is not None:
        return str(resolved["fill_rule"])
    return None


__all__ = [
    "TikzFillMode",
    "TikzStyle",
    "TikzStyleConfig",
    "TikzStyleConversion",
    "TikzStyleGroupingMode",
    "coerce_fill_mode",
    "coerce_style_grouping_mode",
    "color_to_tikz",
    "style_for_primitive",
]
