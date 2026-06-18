"""Semantic TikZ exporter for fitted internal primitives."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from typing import Any

from fikzpy.core.semantic_geometry import BezierPrimitive, CirclePrimitive, ClosedShapePrimitive
from fikzpy.core.semantic_geometry import EllipsePrimitive, FillStyle, LinePrimitive, Point2D, PointPrimitive
from fikzpy.core.semantic_geometry import PolylinePrimitive, Primitive, PrimitiveGroup, RGBColor
from fikzpy.core.semantic_geometry import SemanticGeometry, StrokeStyle
from fikzpy.core.tikz_formatting import TikzCoordinateFormatter, TikzFormattingError
from fikzpy.core.tikz_formatting import TikzPathBuilder, TikzPathMode, TikzYAxisMode
from fikzpy.core.tikz_formatting import coerce_path_mode, coerce_y_axis_mode
from fikzpy.core.tikz_styles import TikzFillMode, TikzStyle, TikzStyleConfig, TikzStyleGroupingMode
from fikzpy.core.tikz_styles import coerce_fill_mode, coerce_style_grouping_mode, style_for_primitive


class TikzExportError(ValueError):
    """Raised when semantic TikZ export cannot continue."""


class TikzCodeOutputMode(Enum):
    """Top-level code output shape."""

    FIG_ONLY = "figonly"
    TIKZPICTURE = "tikzpicture"


@dataclass(frozen=True)
class TikzExportConfig:
    """Configuration for semantic TikZ serialization."""

    include_tikzpicture_environment: bool = False
    include_scope_environment: bool = False
    code_output_mode: TikzCodeOutputMode | str = TikzCodeOutputMode.FIG_ONLY
    coordinate_precision: int = 3
    style_precision: int = 3
    unit: str = ""
    scale: float = 1.0
    y_axis_mode: TikzYAxisMode | str = TikzYAxisMode.CARTESIAN
    invert_y_axis: bool = False
    image_height: float | None = None
    normalize_coordinates: bool = False
    coordinate_origin: Point2D | tuple[float, float] | None = None
    group_styles: bool = False
    style_grouping_mode: TikzStyleGroupingMode | str = TikzStyleGroupingMode.NONE
    define_common_styles: bool = False
    use_named_styles: bool = False
    common_style_name_prefix: str = "fikzStyle"
    minimum_named_style_usage: int = 2
    line_width_unit: str = "pt"
    default_line_width: float = 1.0
    default_stroke_color: RGBColor = field(default_factory=RGBColor.black)
    default_fill_color: RGBColor | None = None
    omit_default_styles: bool = True
    preserve_draw_order: bool = True
    preserve_groups: bool = True
    indent: str = " "
    indent_size: int = 2
    max_points_per_line: int = 5
    split_long_paths: bool = True
    combine_compatible_paths: bool = True
    emit_comments: bool = False
    include_metadata_comments: bool = False
    strict: bool = False
    allow_fill: bool = True
    fill_mode: TikzFillMode | str = TikzFillMode.PRESERVE
    allow_opacity: bool = True
    allow_dash_patterns: bool = True
    use_cycle_for_closed_paths: bool = True
    use_relative_coordinates: bool = False
    use_tikz_ellipse_syntax: bool = True
    use_tikz_circle_syntax: bool = True
    use_bezier_syntax: bool = True
    escape_latex_comments: bool = True
    point_radius: float = 0.5
    point_radius_unit: str = "pt"
    path_mode: TikzPathMode | str = TikzPathMode.AUTO

    def __post_init__(self) -> None:
        for name in (
            "include_tikzpicture_environment",
            "include_scope_environment",
            "invert_y_axis",
            "normalize_coordinates",
            "group_styles",
            "define_common_styles",
            "use_named_styles",
            "omit_default_styles",
            "preserve_draw_order",
            "preserve_groups",
            "split_long_paths",
            "combine_compatible_paths",
            "emit_comments",
            "include_metadata_comments",
            "strict",
            "allow_fill",
            "allow_opacity",
            "allow_dash_patterns",
            "use_cycle_for_closed_paths",
            "use_relative_coordinates",
            "use_tikz_ellipse_syntax",
            "use_tikz_circle_syntax",
            "use_bezier_syntax",
            "escape_latex_comments",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")
        object.__setattr__(self, "code_output_mode", _coerce_output_mode(self.code_output_mode))
        object.__setattr__(self, "path_mode", coerce_path_mode(self.path_mode))
        object.__setattr__(self, "y_axis_mode", coerce_y_axis_mode(self.y_axis_mode))
        object.__setattr__(self, "style_grouping_mode", coerce_style_grouping_mode(self.style_grouping_mode))
        object.__setattr__(self, "fill_mode", coerce_fill_mode(self.fill_mode))
        if self.y_axis_mode is TikzYAxisMode.IMAGE:
            object.__setattr__(self, "invert_y_axis", True)
        if self.fill_mode is TikzFillMode.DISABLE:
            object.__setattr__(self, "allow_fill", False)
        for name in ("coordinate_precision", "style_precision", "indent_size"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, int(getattr(self, name)))
        if int(self.max_points_per_line) < 1:
            raise ValueError("max_points_per_line must be positive.")
        object.__setattr__(self, "max_points_per_line", int(self.max_points_per_line))
        if int(self.minimum_named_style_usage) < 2:
            raise ValueError("minimum_named_style_usage must be at least 2.")
        object.__setattr__(self, "minimum_named_style_usage", int(self.minimum_named_style_usage))
        for name in ("scale", "default_line_width", "point_radius"):
            number = float(getattr(self, name))
            if not isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, number)
        if self.image_height is not None:
            height = float(self.image_height)
            if not isfinite(height):
                raise ValueError("image_height must be finite when provided.")
            object.__setattr__(self, "image_height", height)
        if self.coordinate_origin is not None and not isinstance(self.coordinate_origin, Point2D):
            if not isinstance(self.coordinate_origin, tuple) or len(self.coordinate_origin) != 2:
                raise TypeError("coordinate_origin must be Point2D, an x/y tuple, or None.")
            object.__setattr__(
                self,
                "coordinate_origin",
                Point2D(float(self.coordinate_origin[0]), float(self.coordinate_origin[1])),
            )
        if not isinstance(self.default_stroke_color, RGBColor):
            raise TypeError("default_stroke_color must be RGBColor.")
        if self.default_fill_color is not None and not isinstance(self.default_fill_color, RGBColor):
            raise TypeError("default_fill_color must be RGBColor or None.")
        for name in ("unit", "line_width_unit", "common_style_name_prefix", "point_radius_unit"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string.")

    @property
    def indent_text(self) -> str:
        """Return one indentation step."""
        return self.indent * self.indent_size

    def to_dict(self) -> dict[str, Any]:
        """Return serializable configuration diagnostics."""
        data = dict(self.__dict__)
        data["code_output_mode"] = self.code_output_mode.value
        data["path_mode"] = self.path_mode.value
        data["y_axis_mode"] = self.y_axis_mode.value
        data["style_grouping_mode"] = self.style_grouping_mode.value
        data["fill_mode"] = self.fill_mode.value
        data["default_stroke_color"] = self.default_stroke_color.to_dict()
        data["default_fill_color"] = self.default_fill_color.to_dict() if self.default_fill_color else None
        data["coordinate_origin"] = self.coordinate_origin.to_dict() if self.coordinate_origin else None
        return data


@dataclass(frozen=True)
class TikzExportWarning:
    """A deterministic semantic TikZ export warning."""

    code: str
    message: str
    primitive_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return serializable warning diagnostics."""
        return {"code": self.code, "message": self.message, "primitive_index": self.primitive_index}


@dataclass
class TikzExportMetrics:
    """Scalar diagnostics for semantic TikZ export."""

    input_primitive_count: int = 0
    exported_primitive_count: int = 0
    skipped_primitive_count: int = 0
    draw_commands: int = 0
    path_commands: int = 0
    coordinates_written: int = 0
    bezier_segments_written: int = 0
    circles_written: int = 0
    ellipses_written: int = 0
    closed_paths_written: int = 0
    filled_paths_written: int = 0
    styles_written: int = 0
    named_styles_written: int = 0
    groups_written: int = 0
    warnings_count: int = 0
    code_lines: int = 0
    code_characters: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return serializable scalar diagnostics."""
        return {key: int(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class TikzExportResult:
    """Semantic TikZ code and export diagnostics."""

    code: str
    config: TikzExportConfig
    metrics: TikzExportMetrics
    warnings: tuple[TikzExportWarning, ...]
    style_definitions: tuple[str, ...]
    primitive_count: int
    command_count: int
    draw_count: int
    fill_count: int
    node_count: int
    line_count: int
    bezier_count: int
    circle_count: int
    ellipse_count: int
    closed_shape_count: int
    coordinate_count: int
    estimated_code_size: int
    deterministic_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return serializable export diagnostics."""
        return {
            "code": self.code,
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "style_definitions": list(self.style_definitions),
            "primitive_count": self.primitive_count,
            "command_count": self.command_count,
            "draw_count": self.draw_count,
            "fill_count": self.fill_count,
            "node_count": self.node_count,
            "line_count": self.line_count,
            "bezier_count": self.bezier_count,
            "circle_count": self.circle_count,
            "ellipse_count": self.ellipse_count,
            "closed_shape_count": self.closed_shape_count,
            "coordinate_count": self.coordinate_count,
            "estimated_code_size": self.estimated_code_size,
            "deterministic_hash": self.deterministic_hash,
        }


@dataclass(frozen=True)
class _CommandNode:
    fragments: tuple[str, ...]
    style: TikzStyle
    extra_options: tuple[str, ...] = ()
    comments: tuple[str, ...] = ()
    force_multiline: bool = False
    primitive_count: int = 1
    coordinate_count: int = 0
    bezier_count: int = 0
    circle_count: int = 0
    ellipse_count: int = 0
    closed_shape_count: int = 0
    line_count: int = 0
    filled: bool = False


@dataclass(frozen=True)
class _GroupNode:
    children: tuple["_Node", ...]
    name: str | None = None
    comments: tuple[str, ...] = ()


_Node = _CommandNode | _GroupNode


class SemanticTikzExporter:
    """Export semantic primitives to compact, readable TikZ."""

    def __init__(self, config: TikzExportConfig | None = None) -> None:
        self.config = config or TikzExportConfig()
        self.metrics = TikzExportMetrics()
        self.warnings: list[TikzExportWarning] = []
        self._primitive_index = 0
        self._formatter = TikzCoordinateFormatter()
        self._builder = TikzPathBuilder()

    def export(self, primitives: Any) -> TikzExportResult:
        """Export supported semantic primitive inputs to TikZ code."""
        items = _normalize_input(primitives)
        self.metrics.input_primitive_count = _object_count(items)
        origin = self._effective_origin(items)
        invert = self.config.invert_y_axis
        if invert and self.config.image_height is None:
            self._warn("y_axis_image_height_missing", "Y axis inversion requires image_height; coordinates were not inverted.")
            invert = False
        self._formatter = TikzCoordinateFormatter(
            precision=self.config.coordinate_precision,
            unit=self.config.unit,
            scale=self.config.scale,
            invert_y_axis=invert,
            image_height=self.config.image_height,
            origin=origin,
        )
        self._builder = TikzPathBuilder(
            indent=self.config.indent_text,
            mode=self.config.path_mode,
            max_points_per_line=self.config.max_points_per_line,
            split_long_paths=self.config.split_long_paths,
        )
        nodes = self._build_nodes(items)
        named_styles = self._named_styles(nodes)
        style_definitions = self._style_definition_lines(named_styles)
        code_lines = self._render_code(nodes, style_definitions, named_styles)
        code = "\n".join(code_lines)
        self.metrics.warnings_count = len(self.warnings)
        self.metrics.code_lines = len(code_lines)
        self.metrics.code_characters = len(code)
        self.metrics.named_styles_written = len(style_definitions)
        deterministic_hash = self._hash(code, style_definitions)
        return TikzExportResult(
            code=code,
            config=self.config,
            metrics=self.metrics,
            warnings=tuple(self.warnings),
            style_definitions=tuple(style_definitions),
            primitive_count=self.metrics.input_primitive_count,
            command_count=self.metrics.draw_commands,
            draw_count=self.metrics.draw_commands,
            fill_count=self.metrics.filled_paths_written,
            node_count=0,
            line_count=self.metrics.path_commands,
            bezier_count=self.metrics.bezier_segments_written,
            circle_count=self.metrics.circles_written,
            ellipse_count=self.metrics.ellipses_written,
            closed_shape_count=self.metrics.closed_paths_written,
            coordinate_count=self.metrics.coordinates_written,
            estimated_code_size=len(code.encode("utf-8")),
            deterministic_hash=deterministic_hash,
        )

    def _effective_origin(self, items: tuple[SemanticGeometry, ...]) -> Point2D:
        if self.config.coordinate_origin is not None:
            return self.config.coordinate_origin
        if not self.config.normalize_coordinates:
            return Point2D(0.0, 0.0)
        points = _all_points(items)
        if not points:
            return Point2D(0.0, 0.0)
        return Point2D(min(point.x for point in points), min(point.y for point in points))

    def _build_nodes(self, items: tuple[SemanticGeometry, ...]) -> tuple[_Node, ...]:
        nodes: list[_Node] = []
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, PrimitiveGroup):
                if self.config.preserve_groups:
                    children = self._build_nodes(tuple(item.items))
                    if not children:
                        self._warn("empty_group", "PrimitiveGroup is empty.", None)
                        index += 1
                        continue
                    comments = self._group_comments(item)
                    nodes.append(_GroupNode(children, name=item.name, comments=comments))
                else:
                    nodes.extend(self._build_nodes(item.flatten()))
                index += 1
                continue

            if (
                self.config.combine_compatible_paths
                and isinstance(item, BezierPrimitive)
                and self.config.use_bezier_syntax
            ):
                run = self._bezier_run(items, index)
                node = self._command_for_bezier_run(run)
                if node is not None:
                    nodes.append(node)
                index += len(run)
                continue

            node = self._command_for_primitive(item)
            if node is not None:
                nodes.append(node)
            index += 1
        return tuple(nodes)

    def _command_for_primitive(self, primitive: Primitive) -> _CommandNode | None:
        primitive_index = self._next_primitive_index()
        try:
            conversion = self._style(primitive, primitive_index)
            if not conversion.visible:
                self.metrics.skipped_primitive_count += 1
                self._warn("invisible_primitive", "Primitive has neither visible draw nor fill paint.", primitive_index)
                return None
            comments = self._primitive_comments(primitive)
            if isinstance(primitive, PointPrimitive):
                node = self._point_node(primitive, conversion.style, comments)
            elif isinstance(primitive, LinePrimitive):
                node = self._line_node(primitive, conversion.style, comments, conversion.fills)
            elif isinstance(primitive, PolylinePrimitive):
                node = self._polyline_node(primitive, conversion.style, comments, conversion.fills)
            elif isinstance(primitive, CirclePrimitive):
                node = self._circle_node(primitive, conversion.style, comments, conversion.fills)
            elif isinstance(primitive, EllipsePrimitive):
                node = self._ellipse_node(primitive, conversion.style, comments, conversion.fills)
            elif isinstance(primitive, BezierPrimitive):
                node = self._command_for_bezier_run((primitive,), primitive_index=primitive_index)
                return node
            elif isinstance(primitive, ClosedShapePrimitive):
                node = self._closed_shape_node(primitive, conversion.style, comments, conversion.fills)
            else:
                raise TypeError(f"Unsupported semantic primitive: {type(primitive).__name__}")
            self._record_command(node)
            return node
        except Exception as exc:
            if self.config.strict:
                raise TikzExportError(str(exc)) from exc
            self.metrics.skipped_primitive_count += 1
            self._warn("geometry_degenerate", f"Primitive skipped: {exc}", primitive_index)
            return None

    def _command_for_bezier_run(
        self,
        run: tuple[BezierPrimitive, ...],
        *,
        primitive_index: int | None = None,
    ) -> _CommandNode | None:
        if not self.config.use_bezier_syntax:
            self._warn("generic_path_fallback", "Bezier syntax is disabled; Bezier primitive was skipped.", primitive_index)
            self.metrics.skipped_primitive_count += len(run)
            return None
        first_index = primitive_index if primitive_index is not None else self._next_primitive_index()
        if primitive_index is None:
            for _item in run[1:]:
                self._next_primitive_index()
        try:
            conversion = self._style(run[0], first_index)
            if not conversion.visible:
                self.metrics.skipped_primitive_count += len(run)
                self._warn("invisible_primitive", "Bezier run has neither visible draw nor fill paint.", first_index)
                return None
            fragments = [self._formatter.format_point(run[0].start)]
            for primitive in run:
                fragments.append(
                    ".. controls "
                    + self._formatter.format_point(primitive.control1)
                    + " and "
                    + self._formatter.format_point(primitive.control2)
                    + " .. "
                    + self._formatter.format_point(primitive.end)
                )
            node = _CommandNode(
                fragments=tuple(fragments),
                style=conversion.style,
                comments=self._primitive_comments(run[0]),
                force_multiline=len(run) > 1,
                primitive_count=len(run),
                coordinate_count=1 + 3 * len(run),
                bezier_count=len(run),
                filled=conversion.fills,
            )
            self._record_command(node)
            return node
        except Exception as exc:
            if self.config.strict:
                raise TikzExportError(str(exc)) from exc
            self.metrics.skipped_primitive_count += len(run)
            self._warn("geometry_degenerate", f"Bezier run skipped: {exc}", first_index)
            return None

    def _point_node(self, primitive: PointPrimitive, style: TikzStyle, comments: tuple[str, ...]) -> _CommandNode:
        radius = self._formatter.format_length(
            self.config.point_radius,
            unit=self.config.point_radius_unit,
            precision=self.config.style_precision,
        )
        return _CommandNode(
            fragments=(f"{self._formatter.format_point(primitive.point)} circle[radius={radius}]",),
            style=style,
            comments=comments,
            coordinate_count=1,
            circle_count=1,
        )

    def _line_node(
        self,
        primitive: LinePrimitive,
        style: TikzStyle,
        comments: tuple[str, ...],
        filled: bool,
    ) -> _CommandNode:
        return _CommandNode(
            fragments=(
                self._formatter.format_point(primitive.start),
                "-- " + self._formatter.format_point(primitive.end),
            ),
            style=style,
            comments=comments,
            coordinate_count=2,
            line_count=1,
            filled=filled,
        )

    def _polyline_node(
        self,
        primitive: PolylinePrimitive,
        style: TikzStyle,
        comments: tuple[str, ...],
        filled: bool,
    ) -> _CommandNode:
        points = _without_repeated_close(tuple(primitive.points)) if primitive.closed else tuple(primitive.points)
        fragments = self._path_fragments(points, closed=primitive.closed)
        return _CommandNode(
            fragments=fragments,
            style=style,
            comments=comments,
            force_multiline=len(points) > self.config.max_points_per_line,
            coordinate_count=len(points),
            closed_shape_count=1 if primitive.closed else 0,
            line_count=1,
            filled=filled,
        )

    def _closed_shape_node(
        self,
        primitive: ClosedShapePrimitive,
        style: TikzStyle,
        comments: tuple[str, ...],
        filled: bool,
    ) -> _CommandNode:
        self._closed_shape_warnings(primitive)
        points = _without_repeated_close(tuple(primitive.points))
        return _CommandNode(
            fragments=self._path_fragments(points, closed=True),
            style=style,
            comments=comments,
            force_multiline=len(points) > self.config.max_points_per_line,
            coordinate_count=len(points),
            closed_shape_count=1,
            line_count=1,
            filled=filled,
        )

    def _circle_node(
        self,
        primitive: CirclePrimitive,
        style: TikzStyle,
        comments: tuple[str, ...],
        filled: bool,
    ) -> _CommandNode:
        radius = self._formatter.format_length(primitive.radius * self.config.scale)
        fragment = f"{self._formatter.format_point(primitive.center)} circle[radius={radius}]"
        return _CommandNode(
            fragments=(fragment,),
            style=style,
            comments=comments,
            coordinate_count=1,
            circle_count=1,
            filled=filled,
        )

    def _ellipse_node(
        self,
        primitive: EllipsePrimitive,
        style: TikzStyle,
        comments: tuple[str, ...],
        filled: bool,
    ) -> _CommandNode:
        center = self._formatter.format_point(primitive.center)
        rx = self._formatter.format_length(primitive.radius_x * self.config.scale)
        ry = self._formatter.format_length(primitive.radius_y * self.config.scale)
        extra: tuple[str, ...] = ()
        if abs(primitive.rotation) > 1e-12:
            angle = self._formatter.format_number(primitive.rotation, precision=self.config.style_precision)
            extra = (f"rotate around={{{angle}:{center}}}",)
        return _CommandNode(
            fragments=(f"{center} ellipse[x radius={rx}, y radius={ry}]",),
            style=style,
            extra_options=extra,
            comments=comments,
            force_multiline=bool(extra),
            coordinate_count=1,
            ellipse_count=1,
            filled=filled,
        )

    def _path_fragments(self, points: tuple[Point2D, ...], *, closed: bool) -> tuple[str, ...]:
        if len(points) < 2:
            raise TikzFormattingError("path requires at least two points.")
        fragments = [self._formatter.format_point(points[0])]
        fragments.extend("-- " + self._formatter.format_point(point) for point in points[1:])
        if closed:
            if self.config.use_cycle_for_closed_paths:
                fragments.append("-- cycle")
            else:
                fragments.append("-- " + self._formatter.format_point(points[0]))
        return tuple(fragments)

    def _style(self, primitive: Primitive, primitive_index: int | None) -> Any:
        style_config = TikzStyleConfig(
            style_precision=self.config.style_precision,
            line_width_unit=self.config.line_width_unit,
            default_line_width=self.config.default_line_width,
            default_stroke_color=self.config.default_stroke_color,
            default_fill_color=self.config.default_fill_color,
            omit_default_styles=self.config.omit_default_styles,
            allow_fill=self.config.allow_fill,
            allow_opacity=self.config.allow_opacity,
            allow_dash_patterns=self.config.allow_dash_patterns,
        )
        return style_for_primitive(
            stroke=primitive.stroke,
            fill=primitive.fill,
            opacity=primitive.opacity,
            metadata=primitive.metadata,
            config=style_config,
            formatter=self._formatter,
            warn=lambda code, message: self._warn(code, message, primitive_index),
        )

    def _record_command(self, node: _CommandNode) -> None:
        self.metrics.exported_primitive_count += node.primitive_count
        self.metrics.draw_commands += 1
        self.metrics.path_commands += 1
        self.metrics.coordinates_written += node.coordinate_count
        self.metrics.bezier_segments_written += node.bezier_count
        self.metrics.circles_written += node.circle_count
        self.metrics.ellipses_written += node.ellipse_count
        self.metrics.closed_paths_written += node.closed_shape_count
        if node.filled:
            self.metrics.filled_paths_written += 1
        if not node.style.empty or node.extra_options:
            self.metrics.styles_written += 1

    def _bezier_run(self, items: tuple[SemanticGeometry, ...], start_index: int) -> tuple[BezierPrimitive, ...]:
        first = items[start_index]
        if not isinstance(first, BezierPrimitive):
            return ()
        run = [first]
        cursor = start_index + 1
        while cursor < len(items):
            current = items[cursor]
            previous = run[-1]
            if not isinstance(current, BezierPrimitive):
                break
            if current.start != previous.end:
                break
            if not _styles_equal(previous, current):
                break
            run.append(current)
            cursor += 1
        return tuple(run)

    def _render_code(
        self,
        nodes: tuple[_Node, ...],
        style_definitions: list[str],
        named_styles: dict[tuple[str, ...], str],
    ) -> list[str]:
        lines: list[str] = []
        lines.extend(style_definitions)
        if style_definitions and nodes:
            lines.append("")
        include_picture = self.config.include_tikzpicture_environment or self.config.code_output_mode is TikzCodeOutputMode.TIKZPICTURE
        base_level = 0
        if include_picture:
            lines.append("\\begin{tikzpicture}[scale=1]")
            base_level = 1
        body_lines = self._render_sequence(nodes, level=base_level, named_styles=named_styles)
        if self.config.include_scope_environment:
            prefix = self.config.indent_text * base_level
            lines.append(f"{prefix}\\begin{{scope}}[line cap=round, line join=round]")
            lines.extend(self.config.indent_text + line for line in body_lines)
            lines.append(f"{prefix}\\end{{scope}}")
            self.metrics.groups_written += 1
        else:
            lines.extend(body_lines)
        if include_picture:
            lines.append("\\end{tikzpicture}")
        return lines

    def _render_sequence(
        self,
        nodes: tuple[_Node, ...],
        *,
        level: int,
        named_styles: dict[tuple[str, ...], str],
    ) -> list[str]:
        if not self.config.group_styles:
            return self._render_nodes(nodes, level=level, named_styles=named_styles)
        lines: list[str] = []
        index = 0
        while index < len(nodes):
            node = nodes[index]
            if not isinstance(node, _CommandNode) or node.style.empty:
                lines.extend(self._render_nodes((node,), level=level, named_styles=named_styles))
                index += 1
                continue
            signature = node.style.signature
            run = [node]
            cursor = index + 1
            while cursor < len(nodes):
                next_node = nodes[cursor]
                if not isinstance(next_node, _CommandNode) or next_node.style.signature != signature:
                    break
                run.append(next_node)
                cursor += 1
            if len(run) < 2:
                lines.extend(self._render_nodes((node,), level=level, named_styles=named_styles))
                index += 1
                continue
            style_text = named_styles.get(signature) or node.style.option_text()
            prefix = self.config.indent_text * level
            lines.append(f"{prefix}\\begin{{scope}}[{style_text}]")
            self.metrics.groups_written += 1
            for grouped in run:
                lines.extend(self._render_command(grouped, level=level + 1, named_styles=named_styles, omit_base_style=True))
            lines.append(f"{prefix}\\end{{scope}}")
            index = cursor
        return lines

    def _render_nodes(
        self,
        nodes: tuple[_Node, ...],
        *,
        level: int,
        named_styles: dict[tuple[str, ...], str],
    ) -> list[str]:
        lines: list[str] = []
        for node in nodes:
            if isinstance(node, _CommandNode):
                lines.extend(self._render_command(node, level=level, named_styles=named_styles))
            else:
                prefix = self.config.indent_text * level
                lines.extend(f"{prefix}{comment}" for comment in node.comments)
                option = f"[{_escape_option_text(node.name)}]" if node.name else ""
                lines.append(f"{prefix}\\begin{{scope}}{option}")
                self.metrics.groups_written += 1
                lines.extend(self._render_sequence(node.children, level=level + 1, named_styles=named_styles))
                lines.append(f"{prefix}\\end{{scope}}")
        return lines

    def _render_command(
        self,
        node: _CommandNode,
        *,
        level: int,
        named_styles: dict[tuple[str, ...], str],
        omit_base_style: bool = False,
    ) -> list[str]:
        lines: list[str] = []
        prefix = self.config.indent_text * level
        lines.extend(f"{prefix}{comment}" for comment in node.comments)
        options: list[str] = []
        if not omit_base_style:
            named = named_styles.get(node.style.signature)
            if named:
                options.append(named)
            elif not node.style.empty:
                options.extend(node.style.options)
        options.extend(node.extra_options)
        lines.extend(
            self._builder.draw_lines(
                list(node.fragments),
                options=", ".join(options),
                level=level,
                force_multiline=node.force_multiline,
            )
        )
        return lines

    def _named_styles(self, nodes: tuple[_Node, ...]) -> dict[tuple[str, ...], str]:
        if not (self.config.define_common_styles or self.config.use_named_styles):
            return {}
        signatures = [node.style.signature for node in _command_nodes(nodes) if node.style.signature]
        counts = Counter(signatures)
        output: dict[tuple[str, ...], str] = {}
        for signature in signatures:
            if signature in output:
                continue
            if counts[signature] < self.config.minimum_named_style_usage:
                continue
            output[signature] = f"{self.config.common_style_name_prefix}{len(output)}"
        return output

    def _style_definition_lines(self, named_styles: dict[tuple[str, ...], str]) -> list[str]:
        if not named_styles:
            return []
        lines = ["\\tikzset{"]
        items = list(named_styles.items())
        for index, (signature, name) in enumerate(items):
            suffix = "," if index < len(items) - 1 else ""
            lines.append(f"{self.config.indent_text}{name}/.style={{{', '.join(signature)}}}{suffix}")
        lines.append("}")
        return lines

    def _primitive_comments(self, primitive: Primitive) -> tuple[str, ...]:
        if not self.config.emit_comments:
            return ()
        comments = [f"% primitive: {_primitive_type(primitive)}"]
        if self.config.include_metadata_comments and primitive.metadata:
            comments.append("% metadata: " + _comment_mapping(primitive.metadata, escape=self.config.escape_latex_comments))
        return tuple(comments)

    def _group_comments(self, group: PrimitiveGroup) -> tuple[str, ...]:
        if not self.config.emit_comments:
            return ()
        comments = [f"% group: {group.name or 'unnamed'}"]
        if self.config.include_metadata_comments and group.metadata:
            comments.append("% metadata: " + _comment_mapping(group.metadata, escape=self.config.escape_latex_comments))
        return tuple(comments)

    def _closed_shape_warnings(self, primitive: ClosedShapePrimitive) -> None:
        metadata = dict(primitive.metadata)
        if getattr(primitive, "closed", True) is not True:
            self._warn("open_closed_shape", "ClosedShapePrimitive has closed=False; exporting with cycle.")
        for key in ("holes", "hole_count"):
            if metadata.get(key):
                self._warn("holes_partial", "Closed shape holes are preserved only as metadata comments.")
                break
        if metadata.get("subpaths"):
            self._warn("subpaths_partial", "Closed shape subpaths may not be exported as independent TikZ subpaths.")

    def _next_primitive_index(self) -> int:
        index = self._primitive_index
        self._primitive_index += 1
        return index

    def _warn(self, code: str, message: str, primitive_index: int | None = None) -> None:
        if self.config.strict:
            raise TikzExportError(f"{code}: {message}")
        self.warnings.append(TikzExportWarning(code, message, primitive_index))

    def _hash(self, code: str, style_definitions: list[str]) -> str:
        payload = {
            "code": code,
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "style_definitions": style_definitions,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return sha256(encoded.encode("utf-8")).hexdigest()


def export_primitives_to_tikz(primitives: Any, config: TikzExportConfig | None = None) -> TikzExportResult:
    """Export semantic primitives or supported result wrappers to TikZ."""
    return SemanticTikzExporter(config).export(primitives)


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
    raise TypeError(f"Unsupported input for semantic TikZ export: {type(value).__name__}")


def _coerce_output_mode(value: TikzCodeOutputMode | str) -> TikzCodeOutputMode:
    if isinstance(value, TikzCodeOutputMode):
        return value
    normalized = str(value).strip().lower()
    aliases = {"body": "figonly", "body_only": "figonly", "figure_only": "figonly"}
    normalized = aliases.get(normalized, normalized)
    for item in TikzCodeOutputMode:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported TikZ output mode: {value!r}")


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


def _all_points(items: tuple[SemanticGeometry, ...]) -> tuple[Point2D, ...]:
    points: list[Point2D] = []
    for item in items:
        if isinstance(item, PrimitiveGroup):
            points.extend(_all_points(tuple(item.items)))
        elif isinstance(item, PointPrimitive):
            points.append(item.point)
        elif isinstance(item, LinePrimitive):
            points.extend((item.start, item.end))
        elif isinstance(item, PolylinePrimitive):
            points.extend(item.points)
        elif isinstance(item, CirclePrimitive):
            points.append(item.center)
        elif isinstance(item, EllipsePrimitive):
            points.append(item.center)
        elif isinstance(item, BezierPrimitive):
            points.extend((item.start, item.control1, item.control2, item.end))
        elif isinstance(item, ClosedShapePrimitive):
            points.extend(item.points)
    return tuple(points)


def _object_count(items: tuple[SemanticGeometry, ...]) -> int:
    count = 0
    for item in items:
        if isinstance(item, PrimitiveGroup):
            count += _object_count(tuple(item.items))
        else:
            count += 1
    return count


def _without_repeated_close(points: tuple[Point2D, ...]) -> tuple[Point2D, ...]:
    if len(points) > 1 and points[0] == points[-1]:
        return points[:-1]
    return points


def _styles_equal(first: Primitive, second: Primitive) -> bool:
    return first.stroke == second.stroke and first.fill == second.fill and first.opacity == second.opacity


def _command_nodes(nodes: tuple[_Node, ...]) -> tuple[_CommandNode, ...]:
    output: list[_CommandNode] = []
    for node in nodes:
        if isinstance(node, _CommandNode):
            output.append(node)
        else:
            output.extend(_command_nodes(node.children))
    return tuple(output)


def _primitive_type(primitive: Primitive) -> str:
    return primitive.to_dict().get("type", type(primitive).__name__)


def _comment_mapping(mapping: Mapping[str, Any], *, escape: bool) -> str:
    text = json.dumps(_compact_mapping(mapping), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    text = text.replace("\r", " ").replace("\n", " ")
    if escape:
        text = text.replace("%", "\\%")
    return text


def _compact_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in dict(mapping).items():
        if isinstance(value, Mapping):
            result[key] = _compact_mapping(value)
        elif isinstance(value, Enum):
            result[key] = value.value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            result[key] = [_compact_value(item) for item in list(value)[:12]]
            if len(value) > 12:
                result[f"{key}_truncated"] = True
        else:
            result[key] = repr(value)
    return result


def _compact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _compact_mapping(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _escape_option_text(value: str) -> str:
    text = str(value).strip()
    return "".join(char for char in text if char.isalnum() or char in {" ", "-", "_", ":"})[:80]


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
    "SemanticTikzExporter",
    "TikzCodeOutputMode",
    "TikzExportConfig",
    "TikzExportError",
    "TikzExportMetrics",
    "TikzExportResult",
    "TikzExportWarning",
    "export_primitives_to_tikz",
]
