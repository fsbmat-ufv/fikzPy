"""SVG style and color resolution for the semantic SVG parser."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
import re
from typing import Any, Mapping

from fikzpy.core.semantic_geometry import FillStyle, RGBColor, StrokeStyle


class SvgStyleError(ValueError):
    """Raised when an SVG style value cannot be parsed safely."""


_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_RGB_RE = re.compile(r"rgba?\(([^)]*)\)", re.IGNORECASE)

_NAMED_COLORS: dict[str, RGBColor] = {
    "black": RGBColor(0, 0, 0),
    "white": RGBColor(255, 255, 255),
    "red": RGBColor(255, 0, 0),
    "green": RGBColor(0, 128, 0),
    "lime": RGBColor(0, 255, 0),
    "blue": RGBColor(0, 0, 255),
    "yellow": RGBColor(255, 255, 0),
    "cyan": RGBColor(0, 255, 255),
    "aqua": RGBColor(0, 255, 255),
    "magenta": RGBColor(255, 0, 255),
    "fuchsia": RGBColor(255, 0, 255),
    "gray": RGBColor(128, 128, 128),
    "grey": RGBColor(128, 128, 128),
    "silver": RGBColor(192, 192, 192),
    "maroon": RGBColor(128, 0, 0),
    "olive": RGBColor(128, 128, 0),
    "purple": RGBColor(128, 0, 128),
    "teal": RGBColor(0, 128, 128),
    "navy": RGBColor(0, 0, 128),
    "orange": RGBColor(255, 165, 0),
}


@dataclass(frozen=True)
class SvgPaint:
    """Resolved SVG paint color plus intrinsic alpha."""

    color: RGBColor | None
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.color is not None and not isinstance(self.color, RGBColor):
            raise TypeError("color must be RGBColor or None.")
        object.__setattr__(self, "alpha", _unit_interval("paint alpha", self.alpha))

    @property
    def is_none(self) -> bool:
        """Return whether the paint is explicitly disabled."""
        return self.color is None

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary."""
        return {"color": self.color.to_dict() if self.color is not None else None, "alpha": self.alpha}


@dataclass(frozen=True)
class SvgStyle:
    """Resolved SVG presentation style for one element."""

    stroke: SvgPaint = field(default_factory=lambda: SvgPaint(RGBColor.black()))
    fill: SvgPaint = field(default_factory=lambda: SvgPaint(None))
    stroke_width: float = 1.0
    stroke_opacity: float = 1.0
    fill_opacity: float = 1.0
    opacity: float = 1.0
    line_cap: str | None = None
    line_join: str | None = None
    dash_array: tuple[float, ...] | None = None
    fill_rule: str | None = None
    color: RGBColor = field(default_factory=RGBColor.black)
    display: str | None = None
    visibility: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stroke, SvgPaint):
            raise TypeError("stroke must be SvgPaint.")
        if not isinstance(self.fill, SvgPaint):
            raise TypeError("fill must be SvgPaint.")
        if not isinstance(self.color, RGBColor):
            raise TypeError("color must be RGBColor.")
        width = float(self.stroke_width)
        if not isfinite(width) or width < 0.0:
            raise SvgStyleError("stroke_width must be finite and non-negative.")
        object.__setattr__(self, "stroke_width", width)
        object.__setattr__(self, "stroke_opacity", _unit_interval("stroke_opacity", self.stroke_opacity))
        object.__setattr__(self, "fill_opacity", _unit_interval("fill_opacity", self.fill_opacity))
        object.__setattr__(self, "opacity", _unit_interval("opacity", self.opacity))
        if self.dash_array is not None:
            values = tuple(float(value) for value in self.dash_array)
            if not values or any(not isfinite(value) or value < 0.0 for value in values):
                raise SvgStyleError("dash_array values must be finite and non-negative.")
            object.__setattr__(self, "dash_array", values)

    def stroke_style(self, *, stroke_scale: float = 1.0) -> StrokeStyle:
        """Return a semantic geometry stroke style."""
        color = self.stroke.color or RGBColor.black()
        opacity = self.effective_stroke_opacity()
        width = max(1e-9, self.stroke_width * max(float(stroke_scale), 1e-9))
        return StrokeStyle(
            color=color,
            width=width,
            opacity=opacity,
            line_cap=self.line_cap,
            line_join=self.line_join,
            dash_pattern=self.dash_array,
        )

    def fill_style(self) -> FillStyle | None:
        """Return a semantic geometry fill style, or ``None`` for no fill."""
        if self.fill.color is None:
            return None
        return FillStyle(self.fill.color, self.effective_fill_opacity())

    def effective_stroke_opacity(self) -> float:
        """Return stroke opacity after paint alpha and element opacity."""
        if self.stroke.color is None:
            return 0.0
        return _clamp01(self.opacity * self.stroke_opacity * self.stroke.alpha)

    def effective_fill_opacity(self) -> float:
        """Return fill opacity after paint alpha and element opacity."""
        if self.fill.color is None:
            return 0.0
        return _clamp01(self.opacity * self.fill_opacity * self.fill.alpha)

    def is_visible(self) -> bool:
        """Return whether the style can render visible paint."""
        if (self.display or "").strip().lower() == "none":
            return False
        if (self.visibility or "").strip().lower() in {"hidden", "collapse"}:
            return False
        if self.opacity <= 0.0:
            return False
        return self.effective_stroke_opacity() > 0.0 or self.effective_fill_opacity() > 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return serializable diagnostics."""
        return {
            "stroke": self.stroke.to_dict(),
            "fill": self.fill.to_dict(),
            "stroke_width": self.stroke_width,
            "stroke_opacity": self.stroke_opacity,
            "fill_opacity": self.fill_opacity,
            "opacity": self.opacity,
            "line_cap": self.line_cap,
            "line_join": self.line_join,
            "dash_array": list(self.dash_array) if self.dash_array is not None else None,
            "fill_rule": self.fill_rule,
            "color": self.color.to_dict(),
            "display": self.display,
            "visibility": self.visibility,
        }


def style_from_attributes(
    attributes: Mapping[str, str],
    inherited: SvgStyle | None = None,
) -> SvgStyle:
    """Resolve SVG presentation attributes and inline style."""
    style = inherited or SvgStyle()
    presentation = _presentation_attributes(attributes)
    inline = _parse_inline_style(attributes.get("style"))

    for values in (presentation, inline):
        style = _apply_style_values(style, values)
    return style


def parse_color(value: str, current_color: RGBColor | None = None) -> SvgPaint:
    """Parse an SVG color or paint value."""
    text = str(value).strip()
    if not text:
        raise SvgStyleError("Color value must not be empty.")
    normalized = text.lower()
    if normalized == "none":
        return SvgPaint(None)
    if normalized == "inherit":
        raise SvgStyleError("inherit must be resolved by style inheritance.")
    if normalized == "currentcolor":
        return SvgPaint(current_color or RGBColor.black())
    if normalized == "transparent":
        return SvgPaint(RGBColor.black(), 0.0)
    if normalized.startswith("#"):
        return _parse_hex_color(normalized)
    match = _RGB_RE.fullmatch(text)
    if match is not None:
        return _parse_rgb_color(match.group(1))
    try:
        return SvgPaint(_NAMED_COLORS[normalized])
    except KeyError as exc:
        raise SvgStyleError(f"Unsupported color value: {value!r}.") from exc


def _presentation_attributes(attributes: Mapping[str, str]) -> dict[str, str]:
    keys = {
        "stroke",
        "stroke-width",
        "stroke-opacity",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-dasharray",
        "fill",
        "fill-opacity",
        "fill-rule",
        "opacity",
        "color",
        "display",
        "visibility",
    }
    return {key: value for key, value in attributes.items() if key in keys}


def _parse_inline_style(style_text: str | None) -> dict[str, str]:
    if style_text is None:
        return {}
    result: dict[str, str] = {}
    for item in str(style_text).split(";"):
        if not item.strip():
            continue
        if ":" not in item:
            raise SvgStyleError(f"Invalid inline style item: {item!r}.")
        key, value = item.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _apply_style_values(style: SvgStyle, values: Mapping[str, str]) -> SvgStyle:
    result = style
    for key, value in values.items():
        normalized = key.strip().lower()
        text = str(value).strip()
        if text.lower() == "inherit":
            continue
        if normalized == "color":
            result = replace(result, color=parse_color(text, result.color).color or result.color)
        elif normalized == "stroke":
            result = replace(result, stroke=parse_color(text, result.color))
        elif normalized == "fill":
            result = replace(result, fill=parse_color(text, result.color))
        elif normalized == "stroke-width":
            result = replace(result, stroke_width=_parse_non_negative_number(text, "stroke-width"))
        elif normalized == "stroke-opacity":
            result = replace(result, stroke_opacity=_parse_opacity(text, "stroke-opacity"))
        elif normalized == "fill-opacity":
            result = replace(result, fill_opacity=_parse_opacity(text, "fill-opacity"))
        elif normalized == "opacity":
            result = replace(result, opacity=_parse_opacity(text, "opacity"))
        elif normalized == "stroke-linecap":
            result = replace(result, line_cap=text or None)
        elif normalized == "stroke-linejoin":
            result = replace(result, line_join=text or None)
        elif normalized == "stroke-dasharray":
            result = replace(result, dash_array=_parse_dash_array(text))
        elif normalized == "fill-rule":
            result = replace(result, fill_rule=text or None)
        elif normalized == "display":
            result = replace(result, display=text or None)
        elif normalized == "visibility":
            result = replace(result, visibility=text or None)
    return result


def _parse_hex_color(value: str) -> SvgPaint:
    digits = value[1:]
    if len(digits) == 3:
        channels = [int(item * 2, 16) for item in digits]
        return SvgPaint(RGBColor(*channels))
    if len(digits) == 4:
        channels = [int(item * 2, 16) for item in digits[:3]]
        alpha = int(digits[3] * 2, 16) / 255.0
        return SvgPaint(RGBColor(*channels), alpha)
    if len(digits) == 6:
        channels = [int(digits[index : index + 2], 16) for index in (0, 2, 4)]
        return SvgPaint(RGBColor(*channels))
    if len(digits) == 8:
        channels = [int(digits[index : index + 2], 16) for index in (0, 2, 4)]
        alpha = int(digits[6:8], 16) / 255.0
        return SvgPaint(RGBColor(*channels), alpha)
    raise SvgStyleError(f"Invalid hex color: {value!r}.")


def _parse_rgb_color(value: str) -> SvgPaint:
    parts = [part.strip() for part in re.split(r"[,\s]+", value.strip()) if part.strip()]
    if len(parts) not in {3, 4}:
        raise SvgStyleError("rgb() requires three channels.")
    channels = tuple(_parse_rgb_channel(part) for part in parts[:3])
    alpha = _parse_opacity(parts[3], "rgb alpha") if len(parts) == 4 else 1.0
    return SvgPaint(RGBColor(*channels), alpha)


def _parse_rgb_channel(value: str) -> int:
    if value.endswith("%"):
        number = float(value[:-1])
        if not isfinite(number) or number < 0.0 or number > 100.0:
            raise SvgStyleError("RGB percentage channels must be between 0 and 100.")
        return int(round(number * 255.0 / 100.0))
    number = float(value)
    if not isfinite(number) or number < 0.0 or number > 255.0:
        raise SvgStyleError("RGB channels must be between 0 and 255.")
    return int(round(number))


def _parse_dash_array(value: str) -> tuple[float, ...] | None:
    if value.strip().lower() == "none":
        return None
    numbers = tuple(float(match.group(0)) for match in _NUMBER_RE.finditer(value))
    if not numbers:
        raise SvgStyleError("stroke-dasharray requires numeric values or none.")
    if any(not isfinite(number) or number < 0.0 for number in numbers):
        raise SvgStyleError("stroke-dasharray values must be finite and non-negative.")
    return numbers


def _parse_non_negative_number(value: str, name: str) -> float:
    match = _NUMBER_RE.match(value)
    if match is None:
        raise SvgStyleError(f"{name} must be numeric.")
    number = float(match.group(0))
    if not isfinite(number) or number < 0.0:
        raise SvgStyleError(f"{name} must be finite and non-negative.")
    return number


def _parse_opacity(value: str, name: str) -> float:
    number = float(str(value).strip().rstrip("%"))
    if str(value).strip().endswith("%"):
        number /= 100.0
    return _unit_interval(name, number)


def _unit_interval(name: str, value: float) -> float:
    number = float(value)
    if not isfinite(number) or number < 0.0 or number > 1.0:
        raise SvgStyleError(f"{name} must be between 0 and 1.")
    return number


def _clamp01(value: float) -> float:
    if not isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


__all__ = ["SvgPaint", "SvgStyle", "SvgStyleError", "parse_color", "style_from_attributes"]
