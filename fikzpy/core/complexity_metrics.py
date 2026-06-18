"""Deterministic primitive and TikZ complexity metrics for semantic validation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
import json
import re
from typing import Any

from fikzpy.core.semantic_geometry import BezierPrimitive, CirclePrimitive, ClosedShapePrimitive
from fikzpy.core.semantic_geometry import EllipsePrimitive, FillStyle, LinePrimitive, Point2D, PointPrimitive
from fikzpy.core.semantic_geometry import PolylinePrimitive, Primitive, PrimitiveGroup, SemanticGeometry, StrokeStyle


class ComplexityMetricKind(Enum):
    """Families of complexity metrics emitted by this module."""

    PRIMITIVES = "primitives"
    TIKZ = "tikz"
    READABILITY = "readability"
    SEMANTIC_COMPACTNESS = "semantic_compactness"


@dataclass(frozen=True)
class ComplexityMetricsConfig:
    """Configuration for primitive and TikZ complexity analysis."""

    long_line_threshold: int = 120
    raw_path_coordinate_threshold: int = 12
    target_average_command_length: int = 96
    excessive_primitive_count: int = 512
    excessive_coordinate_count: int = 2048
    style_repetition_penalty_weight: float = 0.15
    raw_path_penalty_weight: float = 0.35

    def __post_init__(self) -> None:
        for name in ("long_line_threshold", "raw_path_coordinate_threshold", "target_average_command_length"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        for name in ("excessive_primitive_count", "excessive_coordinate_count"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        for name in ("style_repetition_penalty_weight", "raw_path_penalty_weight"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        """Return serializable configuration diagnostics."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class ComplexityMetrics:
    """Primitive and TikZ scalar diagnostics used by the fidelity score."""

    primitive_count: int = 0
    primitive_type_counts: Mapping[str, int] = field(default_factory=dict)
    group_count: int = 0
    max_group_depth: int = 0
    point_count: int = 0
    linear_segment_count: int = 0
    bezier_segment_count: int = 0
    closed_path_count: int = 0
    fill_count: int = 0
    distinct_style_count: int = 0
    optimization_operation_count: int = 0
    tikz_lines: int = 0
    tikz_characters: int = 0
    tikz_draw_commands: int = 0
    tikz_path_commands: int = 0
    tikz_coordinates: int = 0
    tikz_bezier_controls: int = 0
    tikz_named_styles: int = 0
    tikz_average_command_length: float = 0.0
    tikz_long_lines: int = 0
    tikz_repeated_style_count: int = 0
    tikz_semantic_primitive_count: int = 0
    tikz_raw_path_penalty: float = 0.0
    editability_score: float = 1.0
    semantic_compactness_score: float = 1.0
    complexity_score: float = 1.0
    config: ComplexityMetricsConfig = field(default_factory=ComplexityMetricsConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic serializable metrics."""
        data = dict(self.__dict__)
        data["primitive_type_counts"] = dict(sorted((str(k), int(v)) for k, v in self.primitive_type_counts.items()))
        data["config"] = self.config.to_dict()
        for key, value in list(data.items()):
            if isinstance(value, float):
                data[key] = _rounded(value)
        return data


def compute_complexity_metrics(
    primitives: Any = None,
    tikz_code: str | Any | None = None,
    config: ComplexityMetricsConfig | None = None,
) -> ComplexityMetrics:
    """Compute deterministic primitive and TikZ complexity metrics."""
    effective_config = config or ComplexityMetricsConfig()
    code = _coerce_tikz_code(tikz_code)
    items: tuple[SemanticGeometry, ...] = ()
    optimization_operation_count = 0

    if _looks_like_tikz_export_result(primitives) and code is None:
        code = _coerce_tikz_code(primitives)
        primitives = None
    if _looks_like_geometry_optimization_result(primitives):
        optimization_operation_count = len(getattr(primitives, "operations", ()) or ())
    if primitives is not None:
        items = _normalize_input(primitives)

    primitive_stats = _primitive_stats(items, optimization_operation_count)
    tikz_stats = _tikz_stats(code or "", effective_config)
    editability_score = _editability_score(primitive_stats, tikz_stats, effective_config)
    semantic_score = _semantic_compactness_score(primitive_stats, tikz_stats, effective_config)
    primitive_load = _primitive_load_score(primitive_stats, effective_config)
    complexity_score = _clamp01((editability_score * 0.45) + (semantic_score * 0.35) + (primitive_load * 0.20))

    return ComplexityMetrics(
        primitive_count=primitive_stats["primitive_count"],
        primitive_type_counts=primitive_stats["primitive_type_counts"],
        group_count=primitive_stats["group_count"],
        max_group_depth=primitive_stats["max_group_depth"],
        point_count=primitive_stats["point_count"],
        linear_segment_count=primitive_stats["linear_segment_count"],
        bezier_segment_count=primitive_stats["bezier_segment_count"],
        closed_path_count=primitive_stats["closed_path_count"],
        fill_count=primitive_stats["fill_count"],
        distinct_style_count=primitive_stats["distinct_style_count"],
        optimization_operation_count=primitive_stats["optimization_operation_count"],
        tikz_lines=tikz_stats["tikz_lines"],
        tikz_characters=tikz_stats["tikz_characters"],
        tikz_draw_commands=tikz_stats["tikz_draw_commands"],
        tikz_path_commands=tikz_stats["tikz_path_commands"],
        tikz_coordinates=tikz_stats["tikz_coordinates"],
        tikz_bezier_controls=tikz_stats["tikz_bezier_controls"],
        tikz_named_styles=tikz_stats["tikz_named_styles"],
        tikz_average_command_length=tikz_stats["tikz_average_command_length"],
        tikz_long_lines=tikz_stats["tikz_long_lines"],
        tikz_repeated_style_count=tikz_stats["tikz_repeated_style_count"],
        tikz_semantic_primitive_count=tikz_stats["tikz_semantic_primitive_count"],
        tikz_raw_path_penalty=tikz_stats["tikz_raw_path_penalty"],
        editability_score=editability_score,
        semantic_compactness_score=semantic_score,
        complexity_score=complexity_score,
        config=effective_config,
    )


def _primitive_stats(items: tuple[SemanticGeometry, ...], optimization_operation_count: int) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    styles: set[str] = set()
    stats = {
        "primitive_count": 0,
        "primitive_type_counts": {},
        "group_count": 0,
        "max_group_depth": 0,
        "point_count": 0,
        "linear_segment_count": 0,
        "bezier_segment_count": 0,
        "closed_path_count": 0,
        "fill_count": 0,
        "distinct_style_count": 0,
        "optimization_operation_count": int(optimization_operation_count),
    }

    def visit(item: SemanticGeometry, depth: int) -> None:
        if isinstance(item, PrimitiveGroup):
            stats["group_count"] += 1
            stats["max_group_depth"] = max(stats["max_group_depth"], depth)
            for child in item.items:
                visit(child, depth + 1)
            return
        stats["primitive_count"] += 1
        primitive_type = _primitive_type(item)
        counts[primitive_type] += 1
        styles.add(_style_signature(item))
        if item.fill is not None:
            stats["fill_count"] += 1
        if isinstance(item, PointPrimitive):
            stats["point_count"] += 1
        elif isinstance(item, LinePrimitive):
            stats["point_count"] += 2
            stats["linear_segment_count"] += 1
        elif isinstance(item, PolylinePrimitive):
            point_count = len(item.points)
            stats["point_count"] += point_count
            stats["linear_segment_count"] += max(point_count - 1, 0) + (1 if item.closed else 0)
            if item.closed:
                stats["closed_path_count"] += 1
        elif isinstance(item, CirclePrimitive):
            stats["point_count"] += 1
        elif isinstance(item, EllipsePrimitive):
            stats["point_count"] += 1
        elif isinstance(item, BezierPrimitive):
            stats["point_count"] += 4
            stats["bezier_segment_count"] += 1
        elif isinstance(item, ClosedShapePrimitive):
            stats["point_count"] += len(item.points)
            stats["linear_segment_count"] += len(item.points)
            stats["closed_path_count"] += 1

    for root in items:
        visit(root, 1)

    stats["primitive_type_counts"] = dict(sorted(counts.items()))
    stats["distinct_style_count"] = len(styles)
    return stats


def _tikz_stats(code: str, config: ComplexityMetricsConfig) -> dict[str, Any]:
    lines = code.splitlines() if code else []
    draw_commands = len(re.findall(r"\\draw\b", code))
    path_commands = len(re.findall(r"\\path\b", code))
    coordinates = len(re.findall(r"\([+-]?\d", code))
    controls = code.count(".. controls")
    named_styles = len(re.findall(r"/\.style=", code))
    semantic_primitives = len(re.findall(r"\b(circle|ellipse)\s*\[", code)) + controls
    long_lines = sum(1 for line in lines if len(line) > config.long_line_threshold)
    command_lengths = _command_lengths(code)
    avg_command_length = sum(command_lengths) / len(command_lengths) if command_lengths else 0.0
    repeated_styles = _repeated_style_count(code)
    raw_path_penalty = _raw_path_penalty(
        code,
        draw_commands=draw_commands,
        path_commands=path_commands,
        coordinates=coordinates,
        semantic_primitives=semantic_primitives,
        config=config,
    )
    return {
        "tikz_lines": len(lines),
        "tikz_characters": len(code),
        "tikz_draw_commands": draw_commands,
        "tikz_path_commands": path_commands,
        "tikz_coordinates": coordinates,
        "tikz_bezier_controls": controls,
        "tikz_named_styles": named_styles,
        "tikz_average_command_length": float(avg_command_length),
        "tikz_long_lines": long_lines,
        "tikz_repeated_style_count": repeated_styles,
        "tikz_semantic_primitive_count": semantic_primitives,
        "tikz_raw_path_penalty": raw_path_penalty,
    }


def _command_lengths(code: str) -> list[int]:
    lengths: list[int] = []
    current: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("\\draw") or stripped.startswith("\\path"):
            current = [stripped]
            if stripped.endswith(";"):
                lengths.append(len(" ".join(current)))
                current = []
        elif current:
            current.append(stripped)
            if stripped.endswith(";"):
                lengths.append(len(" ".join(current)))
                current = []
    return lengths


def _repeated_style_count(code: str) -> int:
    options = re.findall(r"\\draw\[([^\]]+)\]", code)
    counts = Counter(option.strip() for option in options if option.strip())
    return sum(count - 1 for count in counts.values() if count > 1)


def _raw_path_penalty(
    code: str,
    *,
    draw_commands: int,
    path_commands: int,
    coordinates: int,
    semantic_primitives: int,
    config: ComplexityMetricsConfig,
) -> float:
    command_count = max(draw_commands + path_commands, 1)
    path_component = path_commands / command_count
    coordinate_component = 0.0
    if coordinates > config.raw_path_coordinate_threshold and semantic_primitives == 0:
        coordinate_component = min(1.0, coordinates / max(config.raw_path_coordinate_threshold * 4.0, 1.0))
    raw_keyword_component = 1.0 if re.search(r"\b(?:svg|path\s*d=|controls\s+\+)", code, re.IGNORECASE) else 0.0
    return _clamp01((path_component * 0.45) + (coordinate_component * 0.45) + (raw_keyword_component * 0.10))


def _editability_score(
    primitive_stats: Mapping[str, Any],
    tikz_stats: Mapping[str, Any],
    config: ComplexityMetricsConfig,
) -> float:
    if tikz_stats["tikz_characters"] == 0 and primitive_stats["primitive_count"] == 0:
        return 1.0
    long_line_penalty = min(1.0, tikz_stats["tikz_long_lines"] / max(tikz_stats["tikz_lines"], 1))
    average_penalty = max(0.0, tikz_stats["tikz_average_command_length"] - config.target_average_command_length)
    average_penalty = min(1.0, average_penalty / max(config.target_average_command_length, 1))
    repeated_style_penalty = min(1.0, tikz_stats["tikz_repeated_style_count"] / max(tikz_stats["tikz_draw_commands"], 1))
    raw_penalty = tikz_stats["tikz_raw_path_penalty"]
    penalty = (
        long_line_penalty * 0.25
        + average_penalty * 0.25
        + repeated_style_penalty * config.style_repetition_penalty_weight
        + raw_penalty * config.raw_path_penalty_weight
    )
    return _clamp01(1.0 - penalty)


def _semantic_compactness_score(
    primitive_stats: Mapping[str, Any],
    tikz_stats: Mapping[str, Any],
    config: ComplexityMetricsConfig,
) -> float:
    command_count = tikz_stats["tikz_draw_commands"] + tikz_stats["tikz_path_commands"]
    if command_count == 0:
        return 1.0 if primitive_stats["primitive_count"] == 0 else 0.5
    semantic_ratio = tikz_stats["tikz_semantic_primitive_count"] / max(command_count, 1)
    raw_penalty = tikz_stats["tikz_raw_path_penalty"]
    coordinate_pressure = min(1.0, tikz_stats["tikz_coordinates"] / max(config.excessive_coordinate_count, 1))
    primitive_semantic_bonus = 0.0
    type_counts = primitive_stats["primitive_type_counts"]
    for key in ("circle", "ellipse", "bezier"):
        if type_counts.get(key, 0):
            primitive_semantic_bonus += 0.08
    return _clamp01(0.70 + semantic_ratio * 0.35 + primitive_semantic_bonus - raw_penalty * 0.45 - coordinate_pressure * 0.20)


def _primitive_load_score(stats: Mapping[str, Any], config: ComplexityMetricsConfig) -> float:
    primitive_pressure = min(1.0, stats["primitive_count"] / max(config.excessive_primitive_count, 1))
    coordinate_pressure = min(1.0, stats["point_count"] / max(config.excessive_coordinate_count, 1))
    return _clamp01(1.0 - primitive_pressure * 0.55 - coordinate_pressure * 0.45)


def _normalize_input(value: Any) -> tuple[SemanticGeometry, ...]:
    if value is None:
        return ()
    if isinstance(value, PrimitiveGroup):
        return (value,)
    if isinstance(value, _PRIMITIVE_TYPES):
        return (value,)
    if _looks_like_geometry_optimization_result(value):
        return tuple(value.primitives)
    if _looks_like_primitive_fit_result(value):
        return tuple(value.primitives)
    if _looks_like_svg_parse_result(value):
        return tuple(value.primitives)
    if _looks_like_centerline_result(value):
        return tuple(path.to_polyline_primitive() for path in value.paths)
    if _looks_like_centerline_path(value):
        return (value.to_polyline_primitive(),)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        output: list[SemanticGeometry] = []
        for item in value:
            output.extend(_normalize_input(item))
        return tuple(output)
    raise TypeError(f"Unsupported input for complexity metrics: {type(value).__name__}")


def _coerce_tikz_code(value: str | Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "code"):
        return str(value.code)
    return str(value)


def _style_signature(primitive: Primitive) -> str:
    payload = {
        "stroke": _stroke_to_dict(primitive.stroke),
        "fill": _fill_to_dict(primitive.fill),
        "opacity": primitive.opacity,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stroke_to_dict(stroke: StrokeStyle) -> dict[str, Any]:
    return stroke.to_dict()


def _fill_to_dict(fill: FillStyle | None) -> dict[str, Any] | None:
    return fill.to_dict() if fill is not None else None


def _primitive_type(primitive: Primitive) -> str:
    if isinstance(primitive, PointPrimitive):
        return "point"
    if isinstance(primitive, LinePrimitive):
        return "line"
    if isinstance(primitive, PolylinePrimitive):
        return "polyline"
    if isinstance(primitive, CirclePrimitive):
        return "circle"
    if isinstance(primitive, EllipsePrimitive):
        return "ellipse"
    if isinstance(primitive, BezierPrimitive):
        return "bezier"
    if isinstance(primitive, ClosedShapePrimitive):
        return "closed_shape"
    return type(primitive).__name__


def _looks_like_tikz_export_result(value: Any) -> bool:
    return hasattr(value, "code") and hasattr(value, "metrics") and hasattr(value, "deterministic_hash")


def _looks_like_geometry_optimization_result(value: Any) -> bool:
    return hasattr(value, "primitives") and hasattr(value, "operations") and hasattr(value, "deterministic_hash")


def _looks_like_primitive_fit_result(value: Any) -> bool:
    return hasattr(value, "primitives") and hasattr(value, "selected_kind") and hasattr(value, "candidates")


def _looks_like_svg_parse_result(value: Any) -> bool:
    return hasattr(value, "primitives") and hasattr(value, "document_info") and hasattr(value, "input_hash")


def _looks_like_centerline_result(value: Any) -> bool:
    return hasattr(value, "paths") and hasattr(value, "metrics") and type(value).__name__ == "CenterlineResult"


def _looks_like_centerline_path(value: Any) -> bool:
    return hasattr(value, "points") and hasattr(value, "closure") and hasattr(value, "to_polyline_primitive")


def _rounded(value: float, digits: int = 6) -> float:
    number = float(value)
    if not isfinite(number):
        return 0.0
    rounded = round(number, digits)
    return 0.0 if rounded == 0 else rounded


def _clamp01(value: float) -> float:
    if not isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


_PRIMITIVE_TYPES = (
    PointPrimitive,
    LinePrimitive,
    PolylinePrimitive,
    CirclePrimitive,
    EllipsePrimitive,
    BezierPrimitive,
    ClosedShapePrimitive,
)


__all__ = [
    "ComplexityMetricKind",
    "ComplexityMetrics",
    "ComplexityMetricsConfig",
    "compute_complexity_metrics",
]
