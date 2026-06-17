"""Parse traced SVG into semantic geometry primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import acos, ceil, cos, degrees, hypot, isclose, isfinite, pi, radians, sin, sqrt, tan
from pathlib import Path
import re
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from fikzpy.core.diagnostics import log_event
from fikzpy.core.semantic_geometry import BezierPrimitive, CirclePrimitive, ClosedShapePrimitive
from fikzpy.core.semantic_geometry import EllipsePrimitive, FillStyle, LinePrimitive, Point2D
from fikzpy.core.semantic_geometry import PointPrimitive, PolylinePrimitive, Primitive
from fikzpy.core.semantic_geometry import PrimitiveGroup, RGBColor, SemanticGeometry
from fikzpy.core.semantic_geometry import StrokeStyle
from fikzpy.core.svg_styles import SvgStyle, SvgStyleError, style_from_attributes
from fikzpy.core.svg_transforms import SvgTransform, SvgTransformError, parse_transform_list
from fikzpy.core.tracers.base import TracerResult


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
PARSER_BACKEND = "stdlib_elementtree"
_COMMAND_RE = re.compile(r"[AaCcHhLlMmQqSsTtVvZz]")
_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_TOKEN_RE = re.compile(r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_LENGTH_RE = re.compile(r"^\s*([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)\s*([A-Za-z%]*)\s*$")
_KAPPA = 4.0 * (sqrt(2.0) - 1.0) / 3.0
_EPSILON = 1e-9


class SvgParseError(ValueError):
    """Raised when an SVG document cannot be parsed."""


class SvgPathError(SvgParseError):
    """Raised when an SVG path is malformed."""


class SvgReferenceError(SvgParseError):
    """Raised when a safe SVG reference cannot be resolved."""


@dataclass(frozen=True)
class SvgParserConfig:
    """Configuration for isolated semantic SVG parsing."""

    strict: bool = False
    apply_transforms: bool = True
    inherit_styles: bool = True
    convert_arcs_to_beziers: bool = True
    convert_quadratics_to_cubics: bool = True
    preserve_groups: bool = True
    preserve_metadata: bool = True
    normalize_units: bool = True
    target_unit: str = "px"
    decimal_precision: int = 6
    ignore_invisible_elements: bool = True
    preserve_unsupported_elements_as_warnings: bool = True
    maximum_reference_depth: int = 8
    resolve_use_elements: bool = True
    allow_external_references: bool = False
    default_dpi: float = 96.0
    flatten_nested_groups: bool = False
    preserve_subpaths: bool = True
    include_source_element_ids: bool = True

    def __post_init__(self) -> None:
        for name in (
            "strict",
            "apply_transforms",
            "inherit_styles",
            "convert_arcs_to_beziers",
            "convert_quadratics_to_cubics",
            "preserve_groups",
            "preserve_metadata",
            "normalize_units",
            "ignore_invisible_elements",
            "preserve_unsupported_elements_as_warnings",
            "resolve_use_elements",
            "allow_external_references",
            "flatten_nested_groups",
            "preserve_subpaths",
            "include_source_element_ids",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")
        if int(self.decimal_precision) < 0:
            raise ValueError("decimal_precision must be non-negative.")
        object.__setattr__(self, "decimal_precision", int(self.decimal_precision))
        if int(self.maximum_reference_depth) < 0:
            raise ValueError("maximum_reference_depth must be non-negative.")
        object.__setattr__(self, "maximum_reference_depth", int(self.maximum_reference_depth))
        if not isfinite(float(self.default_dpi)) or float(self.default_dpi) <= 0.0:
            raise ValueError("default_dpi must be finite and positive.")
        object.__setattr__(self, "default_dpi", float(self.default_dpi))
        if self.target_unit not in {"px", "pt", "pc", "mm", "cm", "in"}:
            raise ValueError("target_unit must be one of px, pt, pc, mm, cm, or in.")

    def to_dict(self) -> dict[str, Any]:
        """Return serializable parser configuration."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class SvgLength:
    """An SVG length normalized to the parser target unit."""

    value: float
    unit: str = "px"

    def __post_init__(self) -> None:
        if not isfinite(float(self.value)):
            raise SvgParseError("SVG length value must be finite.")
        object.__setattr__(self, "value", float(self.value))

    def to_user_units(self, config: SvgParserConfig) -> float:
        """Return the length converted into the configured target unit."""
        return _convert_units(self.value, self.unit or "px", config.target_unit, config.default_dpi)

    def to_dict(self) -> dict[str, Any]:
        """Return serializable length diagnostics."""
        return {"value": self.value, "unit": self.unit}


@dataclass(frozen=True)
class SvgViewBox:
    """Parsed SVG viewBox."""

    min_x: float
    min_y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name in ("min_x", "min_y", "width", "height"):
            number = float(getattr(self, name))
            if not isfinite(number):
                raise SvgParseError("viewBox values must be finite.")
            object.__setattr__(self, name, number)
        if self.width <= 0.0 or self.height <= 0.0:
            raise SvgParseError("viewBox width and height must be positive.")

    def to_dict(self) -> dict[str, float]:
        """Return serializable viewBox diagnostics."""
        return {"min_x": self.min_x, "min_y": self.min_y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class SvgDocumentInfo:
    """Lightweight SVG document metadata."""

    width: SvgLength | None = None
    height: SvgLength | None = None
    viewbox: SvgViewBox | None = None
    preserve_aspect_ratio: str | None = None
    source_unit: str | None = None
    target_unit: str = "px"
    namespace: str | None = None
    version: str | None = None
    title: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return serializable document diagnostics."""
        return {
            "width": self.width.to_dict() if self.width else None,
            "height": self.height.to_dict() if self.height else None,
            "viewBox": self.viewbox.to_dict() if self.viewbox else None,
            "preserveAspectRatio": self.preserve_aspect_ratio,
            "source_unit": self.source_unit,
            "target_unit": self.target_unit,
            "namespace": self.namespace,
            "version": self.version,
            "title": self.title,
            "description": self.description,
        }


@dataclass(frozen=True)
class SvgParseWarning:
    """A deterministic parser warning."""

    code: str
    message: str
    tag: str | None = None
    element_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable warning."""
        return {"code": self.code, "message": self.message, "tag": self.tag, "element_id": self.element_id}


@dataclass
class SvgParseMetrics:
    """Counters gathered during SVG parsing."""

    elements_total: int = 0
    elements_parsed: int = 0
    elements_ignored: int = 0
    unsupported_elements: int = 0
    groups: int = 0
    use_elements: int = 0
    references_resolved: int = 0
    paths: int = 0
    subpaths: int = 0
    lines: int = 0
    polylines: int = 0
    polygons: int = 0
    rectangles: int = 0
    circles: int = 0
    ellipses: int = 0
    line_primitives: int = 0
    polyline_primitives: int = 0
    circle_primitives: int = 0
    ellipse_primitives: int = 0
    bezier_primitives: int = 0
    closed_shape_primitives: int = 0
    primitive_groups: int = 0
    transform_count: int = 0
    inherited_style_count: int = 0
    arc_count: int = 0
    arcs_converted: int = 0
    warnings_count: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return serializable parser metrics."""
        return {key: int(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class SvgParseResult:
    """Semantic primitives and diagnostics from an SVG parse."""

    primitives: tuple[SemanticGeometry, ...]
    document_info: SvgDocumentInfo
    metrics: SvgParseMetrics
    warnings: tuple[SvgParseWarning, ...]
    config: SvgParserConfig
    input_hash: str
    backend: str
    source_type: str
    tracer_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics without storing the full SVG input."""
        return {
            "primitives": [primitive.to_dict() for primitive in self.primitives],
            "document_info": self.document_info.to_dict(),
            "metrics": self.metrics.to_dict(),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "configuration": self.config.to_dict(),
            "input_hash": self.input_hash,
            "backend": self.backend,
            "source_type": self.source_type,
            "tracer_metadata": self.tracer_metadata,
        }


@dataclass(frozen=True)
class _Source:
    text: str
    source_type: str
    path: Path | None = None
    tracer_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class _PathSegment:
    kind: str
    start: Point2D
    end: Point2D
    control1: Point2D | None = None
    control2: Point2D | None = None


@dataclass(frozen=True)
class _Subpath:
    segments: tuple[_PathSegment, ...]
    closed: bool
    start: Point2D | None = None


class SvgSemanticParser:
    """Parse SVG content into semantic geometry primitives."""

    def __init__(self, config: SvgParserConfig | None = None) -> None:
        self.config = config or SvgParserConfig()
        self.metrics = SvgParseMetrics()
        self.warnings: list[SvgParseWarning] = []
        self.definitions: dict[str, ET.Element] = {}
        self.document_info = SvgDocumentInfo(target_unit=self.config.target_unit)
        self.root_transform = SvgTransform.identity()
        self.order_index = 0

    def parse(self, svg_source: str | bytes | Path | TracerResult) -> SvgParseResult:
        """Parse an SVG source into semantic geometry primitives."""
        source = _read_svg_source(svg_source)
        input_hash = sha256(source.text.encode("utf-8")).hexdigest()
        root = self._parse_xml(source.text)
        root_tag = _local_name(root.tag)
        if root_tag != "svg":
            raise SvgParseError(f"Root element must be svg, got {root_tag!r}.")

        namespace = _namespace(root.tag)
        if namespace not in {None, "", SVG_NAMESPACE}:
            self._warn("namespace", f"Unsupported SVG namespace {namespace!r}.", root)
        elif namespace in {None, ""}:
            self._warn("namespace", "SVG root has no namespace.", root)

        self.metrics.elements_total = sum(1 for _ in root.iter())
        self.definitions = self._collect_definitions(root)
        self.document_info = self._document_info(root, namespace)
        self.root_transform = self._document_transform(self.document_info, root)

        base_style = SvgStyle()
        inherited_transform = self.root_transform
        if self.config.apply_transforms:
            root_element_transform = self._element_transform(root)
            inherited_transform = self.root_transform.multiply(root_element_transform)
        root_style = self._element_style(root, base_style)
        items = self._parse_children(root, root_style, inherited_transform, (), skip_defs=True)
        primitives = tuple(items)
        self.metrics.warnings_count = len(self.warnings)

        log_event("SVG Parser", f"source={source.source_type}")
        if source.tracer_metadata:
            log_event("SVG Parser", f"tracer={source.tracer_metadata.get('effective_tracer')}")
        log_event("SVG Parser", f"backend={PARSER_BACKEND}")
        log_event("SVG Parser", f"elements={self.metrics.elements_total}")
        log_event("SVG Parser", f"primitives={sum(1 for item in _flatten_items(primitives))}")
        log_event("SVG Parser", f"paths={self.metrics.paths}")
        log_event("SVG Parser", f"beziers={self.metrics.bezier_primitives}")
        log_event("SVG Parser", f"transforms={self.metrics.transform_count}")
        log_event("SVG Parser", f"ignored={self.metrics.elements_ignored}")
        log_event("SVG Parser", f"warnings={len(self.warnings)}")

        return SvgParseResult(
            primitives=primitives,
            document_info=self.document_info,
            metrics=self.metrics,
            warnings=tuple(self.warnings),
            config=self.config,
            input_hash=input_hash,
            backend=PARSER_BACKEND,
            source_type=source.source_type,
            tracer_metadata=source.tracer_metadata,
        )

    def _parse_xml(self, text: str) -> ET.Element:
        stripped = text.strip()
        if not stripped:
            raise SvgParseError("SVG input must not be empty.")
        lowered = stripped[:512].lower()
        if "<!doctype" in lowered or "<!entity" in lowered:
            raise SvgParseError("SVG input must not declare DTD or entities.")
        try:
            return ET.fromstring(stripped)
        except ET.ParseError as exc:
            raise SvgParseError(f"Malformed SVG XML: {exc}") from exc

    def _collect_definitions(self, root: ET.Element) -> dict[str, ET.Element]:
        definitions: dict[str, ET.Element] = {}
        for element in root.iter():
            element_id = _attributes(element).get("id")
            if element_id:
                definitions[element_id] = element
        return definitions

    def _document_info(self, root: ET.Element, namespace: str | None) -> SvgDocumentInfo:
        attrs = _attributes(root)
        width = self._parse_optional_length(attrs.get("width"), "width", allow_percentage=True)
        height = self._parse_optional_length(attrs.get("height"), "height", allow_percentage=True)
        viewbox = self._parse_viewbox(attrs.get("viewBox") or attrs.get("viewbox"))
        title = _first_child_text(root, "title")
        description = _first_child_text(root, "desc")
        source_unit = width.unit if width is not None else height.unit if height is not None else None
        return SvgDocumentInfo(
            width=width,
            height=height,
            viewbox=viewbox,
            preserve_aspect_ratio=attrs.get("preserveAspectRatio", "xMidYMid meet"),
            source_unit=source_unit,
            target_unit=self.config.target_unit,
            namespace=namespace,
            version=attrs.get("version"),
            title=title,
            description=description,
        )

    def _document_transform(self, info: SvgDocumentInfo, root: ET.Element) -> SvgTransform:
        if not self.config.normalize_units or info.viewbox is None:
            return SvgTransform.identity()
        width = info.width.to_user_units(self.config) if info.width is not None and info.width.unit != "%" else None
        height = info.height.to_user_units(self.config) if info.height is not None and info.height.unit != "%" else None
        if width is None:
            width = info.viewbox.width
            self._warn("dimension", "SVG width missing or percentage; using viewBox width.", root)
        if height is None:
            height = info.viewbox.height
            self._warn("dimension", "SVG height missing or percentage; using viewBox height.", root)
        return _preserve_aspect_ratio_transform(info.viewbox, width, height, info.preserve_aspect_ratio or "xMidYMid meet")

    def _parse_children(
        self,
        element: ET.Element,
        inherited_style: SvgStyle,
        inherited_transform: SvgTransform,
        reference_stack: tuple[str, ...],
        *,
        skip_defs: bool = False,
    ) -> list[SemanticGeometry]:
        items: list[SemanticGeometry] = []
        for child in list(element):
            if skip_defs and _local_name(child.tag) == "defs":
                continue
            parsed = self._parse_element(child, inherited_style, inherited_transform, reference_stack)
            items.extend(parsed)
        return items

    def _parse_element(
        self,
        element: ET.Element,
        inherited_style: SvgStyle,
        inherited_transform: SvgTransform,
        reference_stack: tuple[str, ...],
    ) -> list[SemanticGeometry]:
        tag = _local_name(element.tag)
        attrs = _attributes(element)
        style = self._element_style(element, inherited_style)
        transform = inherited_transform
        if self.config.apply_transforms:
            transform = inherited_transform.multiply(self._element_transform(element))
        metadata = self._metadata(element, transform, style)
        order = self.order_index
        self.order_index += 1
        if self.config.preserve_metadata:
            metadata["order"] = order

        if not style.is_visible() and self.config.ignore_invisible_elements and tag not in {"g", "defs", "symbol"}:
            self.metrics.elements_ignored += 1
            self._warn("invisible", "Element ignored because it has no visible paint.", element)
            return []

        try:
            if tag == "g":
                return self._parse_group(element, style, transform, reference_stack, metadata)
            if tag == "defs":
                return []
            if tag == "use":
                return self._parse_use(element, style, transform, reference_stack, metadata)
            if tag == "line":
                self.metrics.lines += 1
                return self._one(self._parse_line(attrs, style, transform, metadata), element)
            if tag == "polyline":
                self.metrics.polylines += 1
                return self._one(self._parse_polyline(attrs, style, transform, metadata, closed=False), element)
            if tag == "polygon":
                self.metrics.polygons += 1
                return self._one(self._parse_polyline(attrs, style, transform, metadata, closed=True), element)
            if tag == "rect":
                self.metrics.rectangles += 1
                return self._one(self._parse_rect(attrs, style, transform, metadata), element)
            if tag == "circle":
                self.metrics.circles += 1
                return self._one(self._parse_circle(attrs, style, transform, metadata), element)
            if tag == "ellipse":
                self.metrics.ellipses += 1
                return self._one(self._parse_ellipse(attrs, style, transform, metadata), element)
            if tag == "path":
                self.metrics.paths += 1
                return self._one(self._parse_path(attrs, style, transform, metadata), element)
            if tag in {"metadata", "title", "desc", "symbol", "clipPath"}:
                self.metrics.elements_ignored += 1
                return []
            self.metrics.unsupported_elements += 1
            self.metrics.elements_ignored += 1
            self._warn("unsupported", f"Unsupported SVG element <{tag}> ignored.", element)
            return []
        except (SvgParseError, SvgPathError, SvgStyleError, SvgTransformError, ValueError, TypeError) as exc:
            if self.config.strict:
                raise
            self.metrics.elements_ignored += 1
            self._warn("element_error", f"Element <{tag}> ignored: {exc}", element)
            return []

    def _parse_group(
        self,
        element: ET.Element,
        style: SvgStyle,
        transform: SvgTransform,
        reference_stack: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> list[SemanticGeometry]:
        self.metrics.groups += 1
        children = self._parse_children(element, style, transform, reference_stack)
        if not children:
            self.metrics.elements_ignored += 1
            return []
        self.metrics.elements_parsed += 1
        if self.config.flatten_nested_groups or not self.config.preserve_groups:
            return children
        group = PrimitiveGroup(tuple(children), name=metadata.get("id"), metadata=metadata)
        self.metrics.primitive_groups += 1
        return [group]

    def _parse_use(
        self,
        element: ET.Element,
        style: SvgStyle,
        transform: SvgTransform,
        reference_stack: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> list[SemanticGeometry]:
        self.metrics.use_elements += 1
        if not self.config.resolve_use_elements:
            self._warn("use_disabled", "use element ignored because reference resolution is disabled.", element)
            return []
        attrs = _attributes(element)
        href = attrs.get("href") or attrs.get("xlink:href")
        if not href:
            self._warn("missing_reference", "use element has no href.", element)
            return []
        if not href.startswith("#"):
            if self.config.allow_external_references:
                self._warn("external_reference", "External use reference is not loaded by this parser.", element)
            else:
                self._warn("external_reference", "External use reference rejected.", element)
            return []
        ref_id = href[1:]
        if ref_id in reference_stack:
            raise SvgReferenceError(f"Circular SVG reference detected: {' -> '.join((*reference_stack, ref_id))}.")
        if len(reference_stack) >= self.config.maximum_reference_depth:
            raise SvgReferenceError("Maximum SVG reference depth exceeded.")
        target = self.definitions.get(ref_id)
        if target is None:
            self._warn("missing_reference", f"Reference {href!r} was not found.", element)
            return []
        x = self._length_value(attrs.get("x"), 0.0, "x")
        y = self._length_value(attrs.get("y"), 0.0, "y")
        use_transform = transform.multiply(SvgTransform.translate(x, y)) if self.config.apply_transforms else transform
        self.metrics.references_resolved += 1
        resolved = self._parse_element(target, style, use_transform, (*reference_stack, ref_id))
        if self.config.preserve_groups and self.config.preserve_metadata and resolved:
            group = PrimitiveGroup(tuple(resolved), name=metadata.get("id"), metadata={**metadata, "href": href})
            self.metrics.primitive_groups += 1
            return [group]
        return resolved

    def _parse_line(
        self,
        attrs: dict[str, str],
        style: SvgStyle,
        transform: SvgTransform,
        metadata: dict[str, Any],
    ) -> LinePrimitive:
        start = transform.apply_xy(self._length_value(attrs.get("x1"), 0.0, "x1"), self._length_value(attrs.get("y1"), 0.0, "y1"))
        end = transform.apply_xy(self._length_value(attrs.get("x2"), 0.0, "x2"), self._length_value(attrs.get("y2"), 0.0, "y2"))
        return LinePrimitive(start=start, end=end, **self._primitive_style(style, transform, metadata))

    def _parse_polyline(
        self,
        attrs: dict[str, str],
        style: SvgStyle,
        transform: SvgTransform,
        metadata: dict[str, Any],
        *,
        closed: bool,
    ) -> PolylinePrimitive | ClosedShapePrimitive:
        points = tuple(transform.apply(point) for point in _parse_points(attrs.get("points", "")))
        if closed:
            return ClosedShapePrimitive(points=_remove_repeated_close(points), **self._primitive_style(style, transform, metadata))
        return PolylinePrimitive(points=points, closed=False, **self._primitive_style(style, transform, metadata))

    def _parse_rect(
        self,
        attrs: dict[str, str],
        style: SvgStyle,
        transform: SvgTransform,
        metadata: dict[str, Any],
    ) -> SemanticGeometry:
        x = self._length_value(attrs.get("x"), 0.0, "x")
        y = self._length_value(attrs.get("y"), 0.0, "y")
        width = self._required_positive_length(attrs.get("width"), "width")
        height = self._required_positive_length(attrs.get("height"), "height")
        rx_raw = attrs.get("rx")
        ry_raw = attrs.get("ry")
        rx = self._length_value(rx_raw, 0.0, "rx") if rx_raw is not None else None
        ry = self._length_value(ry_raw, 0.0, "ry") if ry_raw is not None else None
        if rx is None and ry is not None:
            rx = ry
        if ry is None and rx is not None:
            ry = rx
        rx = min(max(rx or 0.0, 0.0), width / 2.0)
        ry = min(max(ry or 0.0, 0.0), height / 2.0)
        if rx <= 0.0 and ry <= 0.0:
            points = (
                transform.apply_xy(x, y),
                transform.apply_xy(x + width, y),
                transform.apply_xy(x + width, y + height),
                transform.apply_xy(x, y + height),
            )
            return ClosedShapePrimitive(points=points, **self._primitive_style(style, transform, metadata))
        return self._rounded_rect_group(x, y, width, height, rx, ry, style, transform, metadata)

    def _parse_circle(
        self,
        attrs: dict[str, str],
        style: SvgStyle,
        transform: SvgTransform,
        metadata: dict[str, Any],
    ) -> SemanticGeometry:
        center = Point2D(self._length_value(attrs.get("cx"), 0.0, "cx"), self._length_value(attrs.get("cy"), 0.0, "cy"))
        radius = self._required_positive_length(attrs.get("r"), "r")
        return self._circle_or_ellipse_from_transform(center, radius, radius, style, transform, metadata, source_type="circle")

    def _parse_ellipse(
        self,
        attrs: dict[str, str],
        style: SvgStyle,
        transform: SvgTransform,
        metadata: dict[str, Any],
    ) -> SemanticGeometry:
        center = Point2D(self._length_value(attrs.get("cx"), 0.0, "cx"), self._length_value(attrs.get("cy"), 0.0, "cy"))
        rx = self._required_positive_length(attrs.get("rx"), "rx")
        ry = self._required_positive_length(attrs.get("ry"), "ry")
        return self._circle_or_ellipse_from_transform(center, rx, ry, style, transform, metadata, source_type="ellipse")

    def _parse_path(
        self,
        attrs: dict[str, str],
        style: SvgStyle,
        transform: SvgTransform,
        metadata: dict[str, Any],
    ) -> SemanticGeometry:
        data = attrs.get("d", "")
        subpaths = _parse_path_data(data, self)
        self.metrics.subpaths += len(subpaths)
        subpath_items: list[SemanticGeometry] = []
        for index, subpath in enumerate(subpaths):
            sub_metadata = {**metadata, "subpath": index}
            primitive = self._subpath_to_primitive(subpath, style, transform, sub_metadata)
            if primitive is not None:
                subpath_items.append(primitive)
        if not subpath_items:
            raise SvgPathError("Path contains no drawable subpaths.")
        if len(subpath_items) == 1:
            return subpath_items[0]
        if self.config.preserve_subpaths:
            return PrimitiveGroup(tuple(subpath_items), name=metadata.get("id"), metadata={**metadata, "subpaths": len(subpath_items)})
        return PrimitiveGroup(tuple(_flatten_items(subpath_items)), name=metadata.get("id"), metadata=metadata)

    def _subpath_to_primitive(
        self,
        subpath: _Subpath,
        style: SvgStyle,
        transform: SvgTransform,
        metadata: dict[str, Any],
    ) -> SemanticGeometry | None:
        primitive_style = self._primitive_style(style, transform, metadata)
        if not subpath.segments:
            if subpath.start is None:
                return None
            return PointPrimitive(point=transform.apply(subpath.start), **primitive_style)
        if all(segment.kind == "line" for segment in subpath.segments):
            points = [subpath.segments[0].start]
            points.extend(segment.end for segment in subpath.segments)
            transformed = tuple(transform.apply(point) for point in points)
            if subpath.closed:
                closed_points = _remove_repeated_close(transformed)
                if len(closed_points) >= 3:
                    return ClosedShapePrimitive(points=closed_points, **primitive_style)
                return PolylinePrimitive(points=transformed, closed=True, **primitive_style)
            if len(transformed) == 2:
                return LinePrimitive(start=transformed[0], end=transformed[1], **primitive_style)
            return PolylinePrimitive(points=transformed, closed=False, **primitive_style)

        items: list[Primitive] = []
        for segment in subpath.segments:
            if segment.kind == "line":
                items.append(LinePrimitive(start=transform.apply(segment.start), end=transform.apply(segment.end), **primitive_style))
            else:
                assert segment.control1 is not None and segment.control2 is not None
                items.append(
                    BezierPrimitive(
                        start=transform.apply(segment.start),
                        control1=transform.apply(segment.control1),
                        control2=transform.apply(segment.control2),
                        end=transform.apply(segment.end),
                        **primitive_style,
                    )
                )
        if len(items) == 1:
            return items[0]
        return PrimitiveGroup(tuple(items), name=metadata.get("id"), metadata={**metadata, "closed": subpath.closed})

    def _circle_or_ellipse_from_transform(
        self,
        center: Point2D,
        radius_x: float,
        radius_y: float,
        style: SvgStyle,
        transform: SvgTransform,
        metadata: dict[str, Any],
        *,
        source_type: str,
    ) -> SemanticGeometry:
        primitive_style = self._primitive_style(style, transform, metadata)
        transformed_center = transform.apply(center)
        sx, sy = transform.axis_scales()
        if source_type == "circle" and transform.is_similarity():
            return CirclePrimitive(center=transformed_center, radius=radius_x * sx, **primitive_style)
        if transform.has_orthogonal_axes() and source_type == "circle":
            return EllipsePrimitive(
                center=transformed_center,
                radius_x=radius_x * sx,
                radius_y=radius_x * sy,
                rotation=transform.rotation_degrees(),
                **primitive_style,
            )
        if transform.has_orthogonal_axes() and source_type == "ellipse":
            return EllipsePrimitive(
                center=transformed_center,
                radius_x=radius_x * sx,
                radius_y=radius_y * sy,
                rotation=transform.rotation_degrees(),
                **primitive_style,
            )
        self._warn("general_transform", f"{source_type} converted to Beziers under general transform.", None)
        items = _ellipse_bezier_segments(center, radius_x, radius_y)
        return PrimitiveGroup(
            tuple(
                BezierPrimitive(
                    start=transform.apply(segment.start),
                    control1=transform.apply(segment.control1 or segment.start),
                    control2=transform.apply(segment.control2 or segment.end),
                    end=transform.apply(segment.end),
                    **primitive_style,
                )
                for segment in items
            ),
            name=metadata.get("id"),
            metadata={**metadata, "converted_from": source_type, "reason": "general_transform"},
        )

    def _rounded_rect_group(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        rx: float,
        ry: float,
        style: SvgStyle,
        transform: SvgTransform,
        metadata: dict[str, Any],
    ) -> PrimitiveGroup:
        primitive_style = self._primitive_style(style, transform, metadata)
        raw_segments = [
            _PathSegment("line", Point2D(x + rx, y), Point2D(x + width - rx, y)),
            _PathSegment(
                "bezier",
                Point2D(x + width - rx, y),
                Point2D(x + width, y + ry),
                Point2D(x + width - rx + _KAPPA * rx, y),
                Point2D(x + width, y + ry - _KAPPA * ry),
            ),
            _PathSegment("line", Point2D(x + width, y + ry), Point2D(x + width, y + height - ry)),
            _PathSegment(
                "bezier",
                Point2D(x + width, y + height - ry),
                Point2D(x + width - rx, y + height),
                Point2D(x + width, y + height - ry + _KAPPA * ry),
                Point2D(x + width - rx + _KAPPA * rx, y + height),
            ),
            _PathSegment("line", Point2D(x + width - rx, y + height), Point2D(x + rx, y + height)),
            _PathSegment(
                "bezier",
                Point2D(x + rx, y + height),
                Point2D(x, y + height - ry),
                Point2D(x + rx - _KAPPA * rx, y + height),
                Point2D(x, y + height - ry + _KAPPA * ry),
            ),
            _PathSegment("line", Point2D(x, y + height - ry), Point2D(x, y + ry)),
            _PathSegment(
                "bezier",
                Point2D(x, y + ry),
                Point2D(x + rx, y),
                Point2D(x, y + ry - _KAPPA * ry),
                Point2D(x + rx - _KAPPA * rx, y),
            ),
        ]
        items: list[Primitive] = []
        for segment in raw_segments:
            if segment.kind == "line":
                items.append(LinePrimitive(start=transform.apply(segment.start), end=transform.apply(segment.end), **primitive_style))
            else:
                assert segment.control1 is not None and segment.control2 is not None
                items.append(
                    BezierPrimitive(
                        start=transform.apply(segment.start),
                        control1=transform.apply(segment.control1),
                        control2=transform.apply(segment.control2),
                        end=transform.apply(segment.end),
                        **primitive_style,
                    )
                )
        return PrimitiveGroup(tuple(items), name=metadata.get("id"), metadata={**metadata, "shape": "rounded_rect"})

    def _element_style(self, element: ET.Element, inherited: SvgStyle) -> SvgStyle:
        try:
            base = inherited if self.config.inherit_styles else SvgStyle()
            style = style_from_attributes(_attributes(element), base)
            if style is not inherited:
                self.metrics.inherited_style_count += 1
            return style
        except SvgStyleError:
            if self.config.strict:
                raise
            self._warn("style", "Invalid style ignored; inherited style used.", element)
            return inherited

    def _element_transform(self, element: ET.Element) -> SvgTransform:
        attrs = _attributes(element)
        value = attrs.get("transform")
        if not value:
            return SvgTransform.identity()
        self.metrics.transform_count += 1
        return parse_transform_list(value)

    def _primitive_style(self, style: SvgStyle, transform: SvgTransform, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "stroke": style.stroke_style(stroke_scale=transform.stroke_scale()),
            "fill": style.fill_style(),
            "opacity": style.opacity if style.opacity < 1.0 else None,
            "metadata": metadata if self.config.preserve_metadata else {},
        }

    def _metadata(self, element: ET.Element | None, transform: SvgTransform, style: SvgStyle) -> dict[str, Any]:
        if not self.config.preserve_metadata:
            return {}
        metadata: dict[str, Any] = {"source": "svg_semantic_parser"}
        if element is not None:
            attrs = _attributes(element)
            metadata["tag"] = _local_name(element.tag)
            if self.config.include_source_element_ids and attrs.get("id"):
                metadata["id"] = attrs["id"]
            if attrs.get("class"):
                metadata["class"] = attrs["class"]
            if attrs.get("transform"):
                metadata["transform"] = attrs["transform"]
            if attrs.get("style"):
                metadata["style"] = attrs["style"]
        metadata["transform_matrix"] = transform.to_dict()
        metadata["resolved_style"] = style.to_dict()
        return metadata

    def _one(self, item: SemanticGeometry, element: ET.Element) -> list[SemanticGeometry]:
        self.metrics.elements_parsed += 1
        self._add_primitive_metrics(item)
        return [item]

    def _add_primitive_metrics(self, item: SemanticGeometry) -> None:
        if isinstance(item, PrimitiveGroup):
            self.metrics.primitive_groups += 1
            for child in item.items:
                self._add_primitive_metrics(child)
        elif isinstance(item, LinePrimitive):
            self.metrics.line_primitives += 1
        elif isinstance(item, PolylinePrimitive):
            self.metrics.polyline_primitives += 1
        elif isinstance(item, CirclePrimitive):
            self.metrics.circle_primitives += 1
        elif isinstance(item, EllipsePrimitive):
            self.metrics.ellipse_primitives += 1
        elif isinstance(item, BezierPrimitive):
            self.metrics.bezier_primitives += 1
        elif isinstance(item, ClosedShapePrimitive):
            self.metrics.closed_shape_primitives += 1

    def _warn(self, code: str, message: str, element: ET.Element | None) -> None:
        tag = _local_name(element.tag) if element is not None else None
        element_id = _attributes(element).get("id") if element is not None else None
        warning = SvgParseWarning(code=code, message=message, tag=tag, element_id=element_id)
        if self.config.strict:
            raise SvgParseError(f"{code}: {message}")
        self.warnings.append(warning)

    def _parse_optional_length(self, value: str | None, name: str, *, allow_percentage: bool = False) -> SvgLength | None:
        if value is None or not str(value).strip():
            return None
        try:
            return _parse_length(value, self.config, allow_percentage=allow_percentage)
        except SvgParseError as exc:
            self._warn("length", f"Invalid {name}: {exc}", None)
            return None

    def _length_value(self, value: str | None, default: float, name: str) -> float:
        if value is None or not str(value).strip():
            return float(default)
        length = _parse_length(value, self.config, allow_percentage=False)
        return length.to_user_units(self.config) if self.config.normalize_units else length.value

    def _required_positive_length(self, value: str | None, name: str) -> float:
        if value is None:
            raise SvgParseError(f"{name} is required.")
        length = self._length_value(value, 0.0, name)
        if length <= 0.0:
            raise SvgParseError(f"{name} must be positive.")
        return length

    def _parse_viewbox(self, value: str | None) -> SvgViewBox | None:
        if value is None or not str(value).strip():
            return None
        numbers = _numbers(value)
        if len(numbers) != 4:
            self._warn("viewBox", "viewBox must contain four numbers.", None)
            return None
        try:
            return SvgViewBox(*numbers)
        except SvgParseError as exc:
            self._warn("viewBox", str(exc), None)
            return None


def parse_svg_to_primitives(
    svg_source: str | bytes | Path | TracerResult,
    config: SvgParserConfig | None = None,
) -> SvgParseResult:
    """Parse SVG into semantic primitives without generating TikZ."""
    return SvgSemanticParser(config).parse(svg_source)


def _read_svg_source(svg_source: str | bytes | Path | TracerResult) -> _Source:
    if isinstance(svg_source, TracerResult):
        if not svg_source.success:
            raise SvgParseError("TracerResult must be successful.")
        text = svg_source.svg_text
        path_text = None
        path = Path(svg_source.svg_path) if svg_source.svg_path else None
        if path is not None:
            if not path.exists():
                raise FileNotFoundError(f"SVG path from TracerResult does not exist: {path}")
            path_text = path.read_text(encoding="utf-8")
        if text is None and path_text is None:
            raise SvgParseError("TracerResult does not contain SVG text or path.")
        if text is not None and path_text is not None and text.strip() != path_text.strip():
            raise SvgParseError("TracerResult svg_text and svg_path content differ.")
        final_text = text if text is not None else path_text
        assert final_text is not None
        computed_hash = sha256(final_text.encode("utf-8")).hexdigest()
        if svg_source.svg_sha256 and svg_source.svg_sha256 != computed_hash:
            raise SvgParseError("TracerResult SVG hash is inconsistent with parsed content.")
        metadata = {
            "requested_tracer": svg_source.requested_tracer.value,
            "effective_tracer": svg_source.effective_tracer.value,
            "backend": svg_source.backend,
            "version": svg_source.version,
            "svg_sha256": computed_hash,
        }
        return _Source(final_text, "tracer_result", path, metadata)
    if isinstance(svg_source, bytes):
        if not svg_source:
            raise SvgParseError("SVG bytes must not be empty.")
        return _Source(svg_source.decode("utf-8-sig"), "bytes")
    if isinstance(svg_source, Path):
        if not svg_source.exists():
            raise FileNotFoundError(f"SVG file not found: {svg_source}")
        return _Source(svg_source.read_text(encoding="utf-8"), "path", svg_source)
    if isinstance(svg_source, str):
        if not svg_source.strip():
            raise SvgParseError("SVG string must not be empty.")
        if "<svg" in svg_source.lower() or svg_source.lstrip().startswith("<?xml"):
            return _Source(svg_source, "string")
        path = Path(svg_source)
        if path.exists():
            return _Source(path.read_text(encoding="utf-8"), "path", path)
        raise FileNotFoundError(f"SVG path not found and string does not contain SVG content: {svg_source}")
    raise TypeError("Unsupported SVG source type.")


def _parse_path_data(data: str, parser: SvgSemanticParser) -> tuple[_Subpath, ...]:
    tokens = _path_tokens(data)
    if not tokens:
        raise SvgPathError("Path data must not be empty.")
    index = 0
    command = ""
    current = Point2D(0.0, 0.0)
    start_point: Point2D | None = None
    segments: list[_PathSegment] = []
    subpaths: list[_Subpath] = []
    last_cubic_control: Point2D | None = None
    last_quad_control: Point2D | None = None
    previous_command = ""

    def finish_subpath(*, closed: bool = False) -> None:
        nonlocal segments, start_point
        if segments or start_point is not None:
            subpaths.append(_Subpath(tuple(segments), closed=closed, start=start_point))
        segments = []
        start_point = None

    while index < len(tokens):
        token = tokens[index]
        if _is_command(token):
            command = token
            index += 1
        elif not command:
            raise SvgPathError("Path data begins with numbers before a command.")

        absolute = command.isupper()
        lower = command.lower()
        if lower == "z":
            closed_start = start_point
            if start_point is not None and not _same_point(current, start_point):
                segments.append(_PathSegment("line", current, start_point))
            finish_subpath(closed=True)
            current = closed_start or current
            last_cubic_control = None
            last_quad_control = None
            previous_command = command
            command = ""
            continue

        if lower == "m":
            first = True
            while index < len(tokens) and not _is_command(tokens[index]):
                x = _take_number(tokens, index, "M x")
                y = _take_number(tokens, index + 1, "M y")
                index += 2
                point = _absolute_point(current, x, y, absolute)
                if first:
                    finish_subpath(closed=False)
                    current = point
                    start_point = point
                    first = False
                else:
                    segments.append(_PathSegment("line", current, point))
                    current = point
            command = "L" if absolute else "l"
            last_cubic_control = None
            last_quad_control = None
            previous_command = command
            continue

        if start_point is None:
            raise SvgPathError("Path segment appears before a move command.")

        if lower == "l":
            while index < len(tokens) and not _is_command(tokens[index]):
                point = _absolute_point(current, _take_number(tokens, index, "L x"), _take_number(tokens, index + 1, "L y"), absolute)
                index += 2
                segments.append(_PathSegment("line", current, point))
                current = point
            last_cubic_control = None
            last_quad_control = None
        elif lower == "h":
            while index < len(tokens) and not _is_command(tokens[index]):
                x = _take_number(tokens, index, "H x")
                index += 1
                point = Point2D(x if absolute else current.x + x, current.y)
                segments.append(_PathSegment("line", current, point))
                current = point
            last_cubic_control = None
            last_quad_control = None
        elif lower == "v":
            while index < len(tokens) and not _is_command(tokens[index]):
                y = _take_number(tokens, index, "V y")
                index += 1
                point = Point2D(current.x, y if absolute else current.y + y)
                segments.append(_PathSegment("line", current, point))
                current = point
            last_cubic_control = None
            last_quad_control = None
        elif lower == "c":
            while index < len(tokens) and not _is_command(tokens[index]):
                c1 = _absolute_point(current, _take_number(tokens, index, "C x1"), _take_number(tokens, index + 1, "C y1"), absolute)
                c2 = _absolute_point(current, _take_number(tokens, index + 2, "C x2"), _take_number(tokens, index + 3, "C y2"), absolute)
                end = _absolute_point(current, _take_number(tokens, index + 4, "C x"), _take_number(tokens, index + 5, "C y"), absolute)
                index += 6
                segments.append(_PathSegment("bezier", current, end, c1, c2))
                current = end
                last_cubic_control = c2
                last_quad_control = None
        elif lower == "s":
            while index < len(tokens) and not _is_command(tokens[index]):
                c1 = _reflect_point(current, last_cubic_control) if previous_command.lower() in {"c", "s"} else current
                c2 = _absolute_point(current, _take_number(tokens, index, "S x2"), _take_number(tokens, index + 1, "S y2"), absolute)
                end = _absolute_point(current, _take_number(tokens, index + 2, "S x"), _take_number(tokens, index + 3, "S y"), absolute)
                index += 4
                segments.append(_PathSegment("bezier", current, end, c1, c2))
                current = end
                last_cubic_control = c2
                last_quad_control = None
        elif lower == "q":
            while index < len(tokens) and not _is_command(tokens[index]):
                q = _absolute_point(current, _take_number(tokens, index, "Q x1"), _take_number(tokens, index + 1, "Q y1"), absolute)
                end = _absolute_point(current, _take_number(tokens, index + 2, "Q x"), _take_number(tokens, index + 3, "Q y"), absolute)
                index += 4
                c1, c2 = _quadratic_to_cubic(current, q, end)
                segments.append(_PathSegment("bezier", current, end, c1, c2))
                current = end
                last_quad_control = q
                last_cubic_control = None
        elif lower == "t":
            while index < len(tokens) and not _is_command(tokens[index]):
                q = _reflect_point(current, last_quad_control) if previous_command.lower() in {"q", "t"} else current
                end = _absolute_point(current, _take_number(tokens, index, "T x"), _take_number(tokens, index + 1, "T y"), absolute)
                index += 2
                c1, c2 = _quadratic_to_cubic(current, q, end)
                segments.append(_PathSegment("bezier", current, end, c1, c2))
                current = end
                last_quad_control = q
                last_cubic_control = None
        elif lower == "a":
            while index < len(tokens) and not _is_command(tokens[index]):
                rx = _take_number(tokens, index, "A rx")
                ry = _take_number(tokens, index + 1, "A ry")
                rotation = _take_number(tokens, index + 2, "A rotation")
                large_arc = _arc_flag(_take_number(tokens, index + 3, "A large-arc-flag"))
                sweep = _arc_flag(_take_number(tokens, index + 4, "A sweep-flag"))
                end = _absolute_point(current, _take_number(tokens, index + 5, "A x"), _take_number(tokens, index + 6, "A y"), absolute)
                index += 7
                parser.metrics.arc_count += 1
                if rx == 0.0 or ry == 0.0 or _same_point(current, end):
                    if not _same_point(current, end):
                        segments.append(_PathSegment("line", current, end))
                    current = end
                elif parser.config.convert_arcs_to_beziers:
                    arc_segments = _arc_to_cubic_segments(current, end, rx, ry, rotation, large_arc, sweep)
                    parser.metrics.arcs_converted += 1
                    segments.extend(arc_segments)
                    current = end
                else:
                    parser._warn("arc", "Arc command ignored because arc conversion is disabled.", None)
                    current = end
                last_cubic_control = segments[-1].control2 if segments and segments[-1].kind == "bezier" else None
                last_quad_control = None
        else:
            raise SvgPathError(f"Unsupported path command: {command}")
        previous_command = command

    finish_subpath(closed=False)
    return tuple(subpaths)


def _path_tokens(data: str) -> list[str]:
    text = str(data)
    tokens: list[str] = []
    position = 0
    for match in _TOKEN_RE.finditer(text):
        skipped = text[position : match.start()]
        if skipped.replace(",", " ").strip():
            raise SvgPathError(f"Invalid path data near {skipped!r}.")
        tokens.append(match.group(0))
        position = match.end()
    if text[position:].replace(",", " ").strip():
        raise SvgPathError(f"Invalid path data near {text[position:]!r}.")
    return tokens


def _take_number(tokens: list[str], index: int, name: str) -> float:
    if index >= len(tokens) or _is_command(tokens[index]):
        raise SvgPathError(f"Missing number for {name}.")
    number = float(tokens[index])
    if not isfinite(number):
        raise SvgPathError(f"{name} must be finite.")
    return number


def _arc_flag(value: float) -> bool:
    if value not in {0.0, 1.0}:
        raise SvgPathError("Arc flags must be 0 or 1.")
    return bool(int(value))


def _arc_to_cubic_segments(
    start: Point2D,
    end: Point2D,
    rx: float,
    ry: float,
    x_axis_rotation: float,
    large_arc: bool,
    sweep: bool,
) -> list[_PathSegment]:
    rx = abs(float(rx))
    ry = abs(float(ry))
    phi = radians(x_axis_rotation % 360.0)
    cos_phi = cos(phi)
    sin_phi = sin(phi)
    dx = (start.x - end.x) / 2.0
    dy = (start.y - end.y) / 2.0
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy
    if rx <= _EPSILON or ry <= _EPSILON:
        return [_PathSegment("line", start, end)]

    lambda_value = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lambda_value > 1.0:
        scale = sqrt(lambda_value)
        rx *= scale
        ry *= scale
    numerator = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    denominator = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    factor = 0.0 if denominator == 0.0 else sqrt(max(0.0, numerator / denominator))
    if large_arc == sweep:
        factor = -factor
    cxp = factor * rx * y1p / ry
    cyp = -factor * ry * x1p / rx
    cx = cos_phi * cxp - sin_phi * cyp + (start.x + end.x) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (start.y + end.y) / 2.0

    def angle_between(u: tuple[float, float], v: tuple[float, float]) -> float:
        dot = u[0] * v[0] + u[1] * v[1]
        length = hypot(*u) * hypot(*v)
        if length == 0.0:
            return 0.0
        sign = -1.0 if u[0] * v[1] - u[1] * v[0] < 0 else 1.0
        return sign * acos(max(-1.0, min(1.0, dot / length)))

    theta1 = angle_between((1.0, 0.0), ((x1p - cxp) / rx, (y1p - cyp) / ry))
    delta = angle_between(((x1p - cxp) / rx, (y1p - cyp) / ry), ((-x1p - cxp) / rx, (-y1p - cyp) / ry))
    if not sweep and delta > 0:
        delta -= 2.0 * pi
    elif sweep and delta < 0:
        delta += 2.0 * pi

    segment_count = max(1, int(ceil(abs(delta) / (pi / 2.0))))
    segment_delta = delta / segment_count
    segments: list[_PathSegment] = []
    current = start
    for index in range(segment_count):
        t1 = theta1 + index * segment_delta
        t2 = t1 + segment_delta
        bezier = _arc_segment_to_cubic(cx, cy, rx, ry, phi, t1, t2)
        if index == 0:
            bezier = _PathSegment("bezier", current, bezier.end, bezier.control1, bezier.control2)
        if index == segment_count - 1:
            bezier = _PathSegment("bezier", bezier.start, end, bezier.control1, bezier.control2)
        segments.append(bezier)
        current = bezier.end
    return segments


def _arc_segment_to_cubic(cx: float, cy: float, rx: float, ry: float, phi: float, t1: float, t2: float) -> _PathSegment:
    delta = t2 - t1
    alpha = 4.0 / 3.0 * tan(delta / 4.0)
    p0 = (cos(t1), sin(t1))
    p3 = (cos(t2), sin(t2))
    p1 = (p0[0] - alpha * p0[1], p0[1] + alpha * p0[0])
    p2 = (p3[0] + alpha * p3[1], p3[1] - alpha * p3[0])

    def map_point(point: tuple[float, float]) -> Point2D:
        x = rx * point[0]
        y = ry * point[1]
        return Point2D(cx + cos(phi) * x - sin(phi) * y, cy + sin(phi) * x + cos(phi) * y)

    return _PathSegment("bezier", map_point(p0), map_point(p3), map_point(p1), map_point(p2))


def _ellipse_bezier_segments(center: Point2D, rx: float, ry: float) -> tuple[_PathSegment, ...]:
    x = center.x
    y = center.y
    return (
        _PathSegment("bezier", Point2D(x + rx, y), Point2D(x, y + ry), Point2D(x + rx, y + _KAPPA * ry), Point2D(x + _KAPPA * rx, y + ry)),
        _PathSegment("bezier", Point2D(x, y + ry), Point2D(x - rx, y), Point2D(x - _KAPPA * rx, y + ry), Point2D(x - rx, y + _KAPPA * ry)),
        _PathSegment("bezier", Point2D(x - rx, y), Point2D(x, y - ry), Point2D(x - rx, y - _KAPPA * ry), Point2D(x - _KAPPA * rx, y - ry)),
        _PathSegment("bezier", Point2D(x, y - ry), Point2D(x + rx, y), Point2D(x + _KAPPA * rx, y - ry), Point2D(x + rx, y - _KAPPA * ry)),
    )


def _parse_points(value: str) -> tuple[Point2D, ...]:
    numbers = _numbers(value)
    if len(numbers) < 4 or len(numbers) % 2 != 0:
        raise SvgParseError("points must contain at least two x/y pairs.")
    return tuple(Point2D(numbers[index], numbers[index + 1]) for index in range(0, len(numbers), 2))


def _numbers(value: str) -> tuple[float, ...]:
    numbers = tuple(float(match.group(0)) for match in _NUMBER_RE.finditer(str(value)))
    consumed = _NUMBER_RE.sub(" ", str(value))
    if consumed.replace(",", " ").strip():
        raise SvgParseError(f"Invalid numeric list: {value!r}.")
    if not all(isfinite(number) for number in numbers):
        raise SvgParseError("Numbers must be finite.")
    return numbers


def _parse_length(value: str, config: SvgParserConfig, *, allow_percentage: bool) -> SvgLength:
    match = _LENGTH_RE.match(str(value))
    if match is None:
        raise SvgParseError(f"Invalid SVG length: {value!r}.")
    number = float(match.group(1))
    unit = match.group(2) or "px"
    if not isfinite(number):
        raise SvgParseError("Length must be finite.")
    if unit == "%" and not allow_percentage:
        raise SvgParseError("Percentage length is not supported here.")
    if unit not in {"", "px", "pt", "pc", "mm", "cm", "in", "%"}:
        raise SvgParseError(f"Unsupported SVG length unit: {unit}.")
    return SvgLength(number, unit or "px")


def _convert_units(value: float, source: str, target: str, dpi: float) -> float:
    if source == "%":
        return value
    source_px = value * _unit_to_px(source, dpi)
    return source_px / _unit_to_px(target, dpi)


def _unit_to_px(unit: str, dpi: float) -> float:
    if unit in {"", "px"}:
        return 1.0
    if unit == "in":
        return dpi
    if unit == "cm":
        return dpi / 2.54
    if unit == "mm":
        return dpi / 25.4
    if unit == "pt":
        return dpi / 72.0
    if unit == "pc":
        return dpi / 6.0
    raise SvgParseError(f"Unsupported unit: {unit}")


def _preserve_aspect_ratio_transform(
    viewbox: SvgViewBox,
    viewport_width: float,
    viewport_height: float,
    preserve_aspect_ratio: str,
) -> SvgTransform:
    text = (preserve_aspect_ratio or "xMidYMid meet").strip()
    parts = text.split()
    align = parts[0] if parts else "xMidYMid"
    mode = parts[1] if len(parts) > 1 else "meet"
    sx = viewport_width / viewbox.width
    sy = viewport_height / viewbox.height
    if align == "none":
        return SvgTransform.translate(0.0, 0.0).multiply(SvgTransform.scale(sx, sy)).multiply(SvgTransform.translate(-viewbox.min_x, -viewbox.min_y))
    scale = max(sx, sy) if mode == "slice" else min(sx, sy)
    extra_x = viewport_width - viewbox.width * scale
    extra_y = viewport_height - viewbox.height * scale
    x_align = 0.0
    y_align = 0.0
    if "xMid" in align:
        x_align = extra_x / 2.0
    elif "xMax" in align:
        x_align = extra_x
    if "YMid" in align:
        y_align = extra_y / 2.0
    elif "YMax" in align:
        y_align = extra_y
    return SvgTransform.translate(x_align, y_align).multiply(SvgTransform.scale(scale)).multiply(SvgTransform.translate(-viewbox.min_x, -viewbox.min_y))


def _attributes(element: ET.Element | None) -> dict[str, str]:
    if element is None:
        return {}
    attrs: dict[str, str] = {}
    for key, value in element.attrib.items():
        local = _local_name(key)
        if _namespace(key) == XLINK_NAMESPACE and local == "href":
            attrs["xlink:href"] = value
        attrs[local] = value
    return attrs


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[1] if "}" in tag else tag


def _namespace(tag: str) -> str | None:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") and "}" in tag else None


def _first_child_text(element: ET.Element, name: str) -> str | None:
    for child in list(element):
        if _local_name(child.tag) == name and child.text:
            return child.text.strip()
    return None


def _is_command(token: str) -> bool:
    return bool(_COMMAND_RE.fullmatch(token))


def _absolute_point(current: Point2D, x: float, y: float, absolute: bool) -> Point2D:
    return Point2D(x, y) if absolute else Point2D(current.x + x, current.y + y)


def _reflect_point(origin: Point2D, control: Point2D | None) -> Point2D:
    if control is None:
        return origin
    return Point2D(2.0 * origin.x - control.x, 2.0 * origin.y - control.y)


def _quadratic_to_cubic(start: Point2D, control: Point2D, end: Point2D) -> tuple[Point2D, Point2D]:
    c1 = Point2D(start.x + (2.0 / 3.0) * (control.x - start.x), start.y + (2.0 / 3.0) * (control.y - start.y))
    c2 = Point2D(end.x + (2.0 / 3.0) * (control.x - end.x), end.y + (2.0 / 3.0) * (control.y - end.y))
    return c1, c2


def _same_point(first: Point2D, second: Point2D) -> bool:
    return isclose(first.x, second.x, abs_tol=1e-9) and isclose(first.y, second.y, abs_tol=1e-9)


def _remove_repeated_close(points: tuple[Point2D, ...]) -> tuple[Point2D, ...]:
    if len(points) > 1 and _same_point(points[0], points[-1]):
        return points[:-1]
    return points


def _flatten_items(items: Iterable[SemanticGeometry]) -> tuple[Primitive, ...]:
    flattened: list[Primitive] = []
    for item in items:
        if isinstance(item, PrimitiveGroup):
            flattened.extend(item.flatten())
        else:
            flattened.append(item)
    return tuple(flattened)


__all__ = [
    "PARSER_BACKEND",
    "SVG_NAMESPACE",
    "SvgDocumentInfo",
    "SvgLength",
    "SvgParseError",
    "SvgParseMetrics",
    "SvgParseResult",
    "SvgParseWarning",
    "SvgParserConfig",
    "SvgPathError",
    "SvgReferenceError",
    "SvgSemanticParser",
    "SvgViewBox",
    "parse_svg_to_primitives",
]
