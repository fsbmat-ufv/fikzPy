"""Conservative path-only SVG optimization using existing fikzPy algorithms.

Only d attributes change. Compound paths are never split into painted objects.
Private parser helpers are reused deliberately in this isolated prototype.
"""

from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import io
import re
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np

from fikzpy.core.bezier_fit import fit_cubic_beziers
from fikzpy.core.geometry_error import point_distances_to_polyline, signed_area, simplify_polyline_rdp
from fikzpy.core.primitive_fitting import PrimitiveFittingConfig, fit_primitive
from fikzpy.core.semantic_geometry import CirclePrimitive, EllipsePrimitive, Point2D, PolylinePrimitive
from fikzpy.core.svg_semantic_parser import SvgSemanticParser, _parse_path_data, _PathSegment
from fikzpy.core.vector_objects import BezierCurve, Line, Polyline

from .validation import compare_images, passes_visual_gate, render_svg, viewport

ET.register_namespace("", "http://www.w3.org/2000/svg")


@dataclass(frozen=True)
class OptimizationConfig:
    """Tolerances are in viewBox user units; transformed documents are passed through."""

    tolerance: float = 0.25
    corner_degrees: float = 35.0
    render_scale: float = 4.0
    minimum_iou: float = 0.99
    maximum_mae: float = 0.5
    sample_step: float = 0.1

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Configuration must be finite.")
        if self.tolerance <= 0 or self.sample_step <= 0 or self.sample_step >= self.tolerance:
            raise ValueError("Require 0 < sample_step < tolerance.")
        if self.render_scale <= 0 or not 0 < self.minimum_iou <= 1 or self.maximum_mae < 0:
            raise ValueError("Invalid visual validation configuration.")
        if not 0 < self.corner_degrees < 180:
            raise ValueError("Invalid corner threshold.")


@dataclass(frozen=True)
class OptimizationResult:
    svg: str
    changed: bool
    diagnostics: dict
    candidate_svg: str | None = None


def _number(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".") or "0"


def _xy(point: Point2D) -> str:
    return f"{_number(point.x)} {_number(point.y)}"


def parse_paths(data: str):
    """Reuse the official SVG command parser while retaining subpath boundaries."""
    return _parse_path_data(data, SvgSemanticParser())


def _array(point) -> np.ndarray:
    return np.array([point.x, point.y], dtype=float)


def _dense_polyline(points: np.ndarray, step: float) -> np.ndarray:
    chunks = []
    for a, b in zip(points[:-1], points[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        chunks.append(a + np.arange(count)[:, None] / count * (b - a))
    return np.vstack([*chunks, points[-1:]])


def _flatten_cubic(segment: _PathSegment, flatness: float) -> np.ndarray:
    """Flatten by De Casteljau with a control-hull distance bound to the chord."""
    controls = np.array([_array(p) for p in (segment.start, segment.control1, segment.control2, segment.end)])

    def walk(p, depth=0):
        error = point_distances_to_polyline(p[1:3], p[[0, 3]]).max()
        if error <= flatness:
            return [p[0], p[3]]
        if depth >= 24:
            raise ValueError("Cubic flattening could not meet the error bound.")
        a, b, c = (p[:-1] + p[1:]) / 2
        d, e = (np.array([a, b]) + np.array([b, c])) / 2
        middle = (d + e) / 2
        return walk(np.array([p[0], a, d, middle]), depth + 1)[:-1] + walk(np.array([middle, e, c, p[3]]), depth + 1)

    return np.array(walk(controls))


def flatten(segments, flatness: float = 0.01) -> np.ndarray:
    """Flatten connected segments, keeping original line vertices exactly."""
    chunks = []
    for segment in segments:
        points = (np.array([_array(segment.start), _array(segment.end)]) if segment.kind == "line"
                  else _flatten_cubic(segment, flatness))
        chunks.append(points[:-1])
    return np.vstack([*chunks, _array(segments[-1].end)[None, :]])


def bounded_error(before: np.ndarray, after: np.ndarray, config: OptimizationConfig, flatness: float) -> float:
    """Bound continuous polyline deviation using dense samples and distance's Lipschitz bound.

    Each direction includes half the sampling step and both curve flattening
    bounds. This is conservative, not an exact Hausdorff computation.
    """
    error = 0.0
    for source, target in ((before, after), (after, before)):
        samples = _dense_polyline(source, config.sample_step)
        error = max(error, float(point_distances_to_polyline(samples, target).max(initial=0)))
    return error + config.sample_step / 2 + 2 * flatness


def _cost(segments) -> int:
    return sum(3 if s.kind == "bezier" else 1 for s in segments)


def _from_points(points: np.ndarray) -> list[_PathSegment]:
    return [_PathSegment("line", Point2D(*a), Point2D(*b)) for a, b in zip(points[:-1], points[1:])]


def _fitted_segments(points: np.ndarray, config: OptimizationConfig) -> list[_PathSegment]:
    output = []
    # Use a tighter fitting budget than the admission ceiling: broad fits that
    # pass geometry alone visibly move narrow strokes in the independent renderer.
    fitted = fit_cubic_beziers(points, config.tolerance * 0.1, min_bezier_length=1,
                              min_points_for_bezier=4, straightness_tolerance=min(config.tolerance * 0.01, .001))
    for primitive in fitted:
        if isinstance(primitive, BezierCurve):
            output.append(_PathSegment("bezier", Point2D(primitive.start.x, primitive.start.y),
                                       Point2D(primitive.end.x, primitive.end.y),
                                       Point2D(primitive.control1.x, primitive.control1.y),
                                       Point2D(primitive.control2.x, primitive.control2.y)))
        elif isinstance(primitive, Line):
            output.extend(_from_points(np.array([primitive.start.as_tuple(), primitive.end.as_tuple()])))
        elif isinstance(primitive, Polyline):
            output.extend(_from_points(np.array([p.as_tuple() for p in primitive.points])))
    return output


def _optimize_run(run, locked, config):
    points = np.array([_array(run[0].start), *[_array(s.end) for s in run]])
    cuts = [0]
    for i in range(1, len(points) - 1):
        a, b = points[i] - points[i - 1], points[i + 1] - points[i]
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        angle = np.degrees(np.arccos(np.clip(np.dot(a, b) / denominator, -1, 1))) if denominator else 180
        if angle >= config.corner_degrees or tuple(points[i]) in locked:
            cuts.append(i)
    cuts.append(len(points) - 1)
    output, maximum_error, operations = [], 0.0, Counter()
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        original = run[lo:hi]
        sample = points[lo:hi + 1]
        best = list(original)
        best_error = 0.0
        operation = "unchanged"
        if len(sample) > 2:
            candidates = [("rdp", _from_points(simplify_polyline_rdp(sample, config.tolerance * 0.45)))]
            if len(sample) >= 4:
                candidates.append(("cubic_fit", _fitted_segments(sample, config)))
            for name, candidate in candidates:
                if not candidate or _cost(candidate) >= _cost(best):
                    continue
                flatness = min(.01, config.tolerance * .04)
                error = bounded_error(sample, flatten(candidate, flatness), config, flatness)
                # Reserve most of the ceiling for rendering/serialization effects;
                # a cheaper but less accurate RDP candidate must not displace a
                # more faithful cubic merely because it uses fewer parameters.
                if error <= config.tolerance * .4:
                    best, best_error, operation = candidate, error, name
        output.extend(best)
        maximum_error = max(maximum_error, best_error)
        operations[operation] += 1
    return output, maximum_error, operations


def _serialize(subpath, segments):
    commands = ["M " + _xy(subpath.start)]
    for index, segment in enumerate(segments):
        if segment.kind == "line":
            if subpath.closed and index == len(segments) - 1 and segment.end == subpath.start:
                continue  # Z already represents this exact closing segment.
            commands.append("L " + _xy(segment.end))
        else:
            commands.append("C " + " ".join(_xy(p) for p in (segment.control1, segment.control2, segment.end)))
    if subpath.closed:
        commands.append("Z")
    return " ".join(commands)


def _recognize_ring(subpath, locked, config):
    """Reuse semantic recognition; retain a ring as arcs in the SAME filled path."""
    if not subpath.closed or any(tuple(_array(s.start)) in locked for s in subpath.segments):
        return None
    reference = flatten(subpath.segments)
    if len(reference) < 12:
        return None
    fitting = PrimitiveFittingConfig(enable_point_fit=False, enable_line_fit=False,
        enable_polyline_fit=False, enable_bezier_fit=False, normalize_error_by_scale=False,
        circle_error_tolerance=config.tolerance * 0.3, ellipse_error_tolerance=config.tolerance * 0.3)
    with redirect_stdout(io.StringIO()):
        result = fit_primitive(PolylinePrimitive(tuple(Point2D(*p) for p in reference[:-1]), closed=True), fitting)
    if len(result.primitives) != 1 or not isinstance(result.primitives[0], (CirclePrimitive, EllipsePrimitive)):
        return None
    shape = result.primitives[0]
    rx, ry = (shape.radius, shape.radius) if isinstance(shape, CirclePrimitive) else (shape.radius_x, shape.radius_y)
    rotation = 0.0 if isinstance(shape, CirclePrimitive) else shape.rotation
    angle = np.radians(rotation)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    sweep = 1 if signed_area(reference) > 0 else 0
    parameters = np.linspace(0, (1 if sweep else -1) * 2 * np.pi,
                             max(128, int(np.ceil(2 * np.pi * max(rx, ry) / config.sample_step))) + 1)
    candidate = np.column_stack([rx * np.cos(parameters), ry * np.sin(parameters)]) @ matrix.T + _array(shape.center)
    # Curved arc/chord deviation is bounded by max radius times angular-step squared / 8.
    arc_flatness = max(rx, ry) * (2 * np.pi / (len(parameters) - 1)) ** 2 / 8
    error = bounded_error(reference, candidate, config, max(0.01, arc_flatness))
    if error > config.tolerance:
        return None
    p0 = Point2D(*candidate[0])
    # Use the analytic opposite endpoint, independent of an odd sample count.
    p1 = Point2D(*(-np.array([rx, 0.0]) @ matrix.T + _array(shape.center)))
    arc = f"A {_number(rx)} {_number(ry)} {_number(rotation)} 0 {sweep} "
    data = "M " + _xy(p0) + " " + arc + _xy(p1) + " " + arc + _xy(p0) + " Z"
    if _cost(subpath.segments) <= 6:
        return None
    return data, error, type(shape).__name__


def optimize_svg(svg: str, config: OptimizationConfig | None = None) -> OptimizationResult:
    """Optimize app-generated paths, retaining unsupported input unchanged with diagnostics."""
    config = config or OptimizationConfig()
    started = perf_counter()
    root = ET.fromstring(svg)
    viewport(svg)
    diagnostics = {"config": asdict(config), "warnings": [], "operations": {}, "maximum_error_bound": 0.0}
    # No claims about CSS, marker placement, affine stroke scaling or reuse semantics.
    safe_tags = {"svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "title", "desc"}
    unsafe_attributes = {"transform", "clip-path", "mask", "filter", "marker-start", "marker-mid", "marker-end", "class", "pathLength", "vector-effect"}
    if any(e.tag.rsplit("}", 1)[-1] not in safe_tags or unsafe_attributes.intersection(e.attrib)
           or (e is not root and e.tag.rsplit("}", 1)[-1] == "svg")
           or any("url(" in v or "transform:" in v.replace(" ", "") for v in e.attrib.values()) for e in root.iter()):
        diagnostics["warnings"].append("unsupported_document_passed_through")
        diagnostics["seconds"] = perf_counter() - started
        return OptimizationResult(svg, False, diagnostics)
    paths = [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "path"]
    parsed = []
    for path in paths:
        data = path.get("d", "")
        # The app emits M/L/C/Z. H/V and relative equivalents are also safe to normalize.
        if set(re.findall(r"[A-DF-Za-df-z]", data)) - set("MmLlHhVvCcZz"):
            diagnostics["warnings"].append("unsupported_commands_passed_through")
            parsed.append(None)
        else:
            parsed.append(parse_paths(data))
    owners = Counter()
    for subpaths in parsed:
        for subpath in subpaths or []:
            owners.update(set(tuple(_array(p)) for s in subpath.segments for p in (s.start, s.end)))
    locked = {point for point, count in owners.items() if count > 1}
    operations = Counter()
    changed = False
    for path, subpaths in zip(paths, parsed):
        if subpaths is None:
            continue
        parts = []
        for subpath in subpaths:
            if not subpath.segments:
                parts.append("M " + _xy(subpath.start))
                continue
            recognized = _recognize_ring(subpath, locked, config)
            if recognized:
                data, error, kind = recognized
                parts.append(data)
                operations[kind] += 1
                diagnostics["maximum_error_bound"] = max(diagnostics["maximum_error_bound"], error)
                continue
            optimized, i = [], 0
            while i < len(subpath.segments):
                if subpath.segments[i].kind != "line":
                    optimized.append(subpath.segments[i])
                    i += 1
                    continue
                end = i + 1
                while end < len(subpath.segments) and subpath.segments[end].kind == "line":
                    end += 1
                run, error, counts = _optimize_run(subpath.segments[i:end], locked, config)
                optimized.extend(run)
                operations.update(counts)
                diagnostics["maximum_error_bound"] = max(diagnostics["maximum_error_bound"], error)
                i = end
            parts.append(_serialize(subpath, optimized))
        data = " ".join(parts)
        if data != path.get("d"):
            path.set("d", data)
            changed = True
    candidate = ET.tostring(root, encoding="unicode") if changed else svg
    visual = compare_images(render_svg(svg, config.render_scale), render_svg(candidate, config.render_scale), rgba=True)
    accepted = passes_visual_gate(visual, config.minimum_iou, config.maximum_mae)
    # Check each paint object as well: global compositing can hide a changed
    # region behind an opaque foreground object or background rectangle.
    local_metrics = []
    if accepted and len(paths) > 1:
        for index in range(len(paths)):
            isolated = []
            for source in (svg, candidate):
                tree = ET.fromstring(source)
                path_index = 0
                for parent in tree.iter():
                    for child in list(parent):
                        tag = child.tag.rsplit("}", 1)[-1]
                        if tag == "path":
                            if path_index != index:
                                parent.remove(child)
                            path_index += 1
                        elif tag in {"rect", "circle", "ellipse", "line", "polyline", "polygon"}:
                            parent.remove(child)
                isolated.append(render_svg(ET.tostring(tree, encoding="unicode"), config.render_scale))
            metrics = compare_images(*isolated, rgba=True)
            local_metrics.append(metrics)
            accepted = accepted and passes_visual_gate(metrics, config.minimum_iou, config.maximum_mae)
    diagnostics.update(operations=dict(operations), svg_visual_metrics=visual, candidate_accepted=accepted,
                       per_paint_metrics=local_metrics, seconds=perf_counter() - started)
    if not accepted:
        diagnostics["warnings"].append("candidate_rejected_visual_or_topology_gate")
        return OptimizationResult(svg, False, diagnostics, candidate)
    return OptimizationResult(candidate, changed, diagnostics, candidate)
