"""Phase-two probes/adapters; existing fitting and parsing algorithms are reused."""

from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
import inspect
import io
import xml.etree.ElementTree as ET
from unittest.mock import patch

import cv2
import numpy as np

from fikzpy.core.dot_to_dot.models import CubicSegment, Vec2
from fikzpy.core.geometry_error import point_distances_to_polyline, signed_area, simplify_polyline_rdp
from fikzpy.core.primitive_fitting import PrimitiveFittingConfig, fit_primitive
from fikzpy.core.semantic_geometry import Point2D, PolylinePrimitive
from fikzpy.core.svg_semantic_parser import _PathSegment
from fikzpy.core import bezier_fit
from fikzpy.core import visual_pipeline
from fikzpy.core.tikz_generator import TikzOptions
from fikzpy.core.visual_pipeline import _contour_to_svg_path

from .optimizer import OptimizationConfig, _array, _cost, _fitted_segments, _from_points
from .optimizer import _serialize, bounded_error, flatten, parse_paths


def tangent(segment, at_end: bool) -> np.ndarray:
    """Return a nonzero endpoint tangent, falling back to the endpoint chord."""
    if segment.kind == "bezier":
        vector = (_array(segment.end) - _array(segment.control2) if at_end
                  else _array(segment.control1) - _array(segment.start))
        if np.linalg.norm(vector) > 1e-10:
            return vector
    return _array(segment.end) - _array(segment.start)


def turn(first, second) -> float:
    a, b = tangent(first, True), tangent(second, False)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denominator, -1, 1)))) if denominator else 180.0


def subdivide_svg(svg: str, levels: int) -> str:
    """Change cubic representation by exact De Casteljau splits, not new fitting.

    Arc-to-cubic conversion comes from the existing parser and is approximate;
    only subsequent cubic subdivision is algebraically exact.
    """
    root = ET.fromstring(svg)
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "path":
            continue
        parts = []
        for subpath in parse_paths(element.get("d", "")):
            output = []
            for segment in subpath.segments:
                if segment.kind != "bezier":
                    output.append(segment)
                    continue
                curves = [CubicSegment(*[Vec2(p.x, p.y) for p in
                                        (segment.start, segment.control1, segment.control2, segment.end)])]
                for _ in range(levels):
                    curves = [piece for curve in curves for piece in curve.split(.5)]
                output.extend(_PathSegment("bezier", Point2D(c.start.x, c.start.y), Point2D(c.end.x, c.end.y),
                                           Point2D(c.control1.x, c.control1.y), Point2D(c.control2.x, c.control2.y))
                              for c in curves)
            parts.append(_serialize(subpath, output))
        element.set("d", " ".join(parts))
    return ET.tostring(root, encoding="unicode")


def inventory(svg: str) -> dict:
    """Count representation constraints; none of these counts claims a global minimum."""
    subpaths = [s for e in ET.fromstring(svg).iter() if e.tag.rsplit("}", 1)[-1] == "path"
                for s in parse_paths(e.get("d", ""))]
    lengths, pairs, angles, cubic_runs, all_runs = [], Counter(), [], [], []
    line_count = cubic_count = explicit_closings = eligible = 0
    micro_rings = micro_segments = 0
    for subpath in subpaths:
        segments = subpath.segments
        if not segments:
            continue
        if subpath.closed and abs(signed_area(flatten(segments, .02))) <= 4:
            micro_rings += 1
            micro_segments += len(segments)
        explicit_closings += int(subpath.closed and segments[-1].kind == "line" and segments[-1].end == subpath.start)
        current, curve_run = 1, 0
        for i, segment in enumerate(segments):
            if segment.kind == "line":
                line_count += 1
                lengths.append(float(np.linalg.norm(_array(segment.end) - _array(segment.start))))
                if curve_run:
                    cubic_runs.append(curve_run)
                curve_run = 0
            else:
                cubic_count += 1
                curve_run += 1
            if i:
                angle = turn(segments[i - 1], segment)
                angles.append(angle)
                pairs[segments[i - 1].kind + "_" + segment.kind] += 1
                eligible += int(angle < 8 and (segment.kind == "bezier" or segments[i - 1].kind == "bezier"))
                if angle >= 35:
                    all_runs.append(current)
                    current = 1
                else:
                    current += 1
        all_runs.append(current)
        if curve_run:
            cubic_runs.append(curve_run)
    return {"subpaths": len(subpaths), "closed_subpaths": sum(s.closed for s in subpaths),
            "segments": line_count + cubic_count, "line_segments_including_closure": line_count,
            "cubic_segments": cubic_count, "closure_segments": explicit_closings,
            "line_length_lt_1": sum(v < 1 for v in lengths),
            "line_length_lt_2": sum(v < 2 for v in lengths),
            "turns_ge_35_degrees": sum(v >= 35 for v in angles),
            "turns_lt_8_degrees": sum(v < 8 for v in angles),
            "junction_type_counts": dict(pairs), "smooth_cubic_adjacent_joins": eligible,
            "cubic_runs": dict(Counter(cubic_runs)), "corner_delimited_runs": dict(Counter(all_runs)),
            "runs_with_4_or_more_segments": sum(n >= 4 for n in all_runs),
            "rings_area_le_4_user_units2": micro_rings, "segments_in_rings_area_le_4": micro_segments}


def recognition_reasons(svg: str) -> dict:
    """Inspect why complete closed rings fail existing circle/ellipse fitting."""
    reasons, selected = Counter(), Counter()
    config = PrimitiveFittingConfig(enable_point_fit=False, enable_line_fit=False,
        enable_polyline_fit=False, enable_bezier_fit=False, normalize_error_by_scale=False,
        circle_error_tolerance=.075, ellipse_error_tolerance=.075)
    for e in ET.fromstring(svg).iter():
        if e.tag.rsplit("}", 1)[-1] != "path":
            continue
        for s in parse_paths(e.get("d", "")):
            if not s.closed or not s.segments:
                continue
            points = flatten(s.segments, .02)
            with redirect_stdout(io.StringIO()):
                result = fit_primitive(PolylinePrimitive(tuple(Point2D(*p) for p in points[:-1]), closed=True), config)
            selected[result.selected_kind.value] += 1
            for candidate in result.candidates:
                if candidate.kind.value in {"circle", "ellipse"}:
                    reasons[f"{candidate.kind.value}: {candidate.rejection_reason or 'accepted'}"] += 1
    return {"selected": dict(selected), "candidate_reasons": dict(reasons)}


def tracing_provenance(image, traced, settings, high_resolution_mask) -> dict:
    """Instrument the existing fitter without modifying its return values or source."""
    counts, output_lines = Counter(), Counter()
    original = bezier_fit._fallback_small_path

    def observed(points, straightness_tolerance):
        local = inspect.currentframe().f_back.f_locals
        if len(points) < local.get("min_points_for_bezier", 0):
            reason = "insufficient_points_for_cubic"
        elif bezier_fit._path_length(points) < local.get("min_bezier_length", 0):
            reason = "short_path"
        elif local.get("max_depth") == 0:
            reason = "depth_limit"
        else:
            reason = "unstable_split"
        primitive = original(points, straightness_tolerance)
        counts[reason] += 1
        output_lines[reason] += max(0, len(primitive.points) - 1) if hasattr(primitive, "points") else 1
        return primitive

    with patch.object(bezier_fit, "_fallback_small_path", observed):
        generated = [_contour_to_svg_path(c, settings) for c in traced.contours]
    mask = high_resolution_mask.astype(np.uint8)
    raw, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    scale = max(1, settings.upsample_factor)
    areas = stats[1:, cv2.CC_STAT_AREA] / (scale * scale)
    small_contours = [c for c in raw if abs(cv2.contourArea(c)) / (scale * scale) <= 4]
    return {"image_pixels": int(image.shape[0] * image.shape[1]), "mask_components": int(n - 1),
            "mask_components_le_4_px2": int(np.count_nonzero(areas <= 4)),
            "mask_components_le_16_px2": int(np.count_nonzero(areas <= 16)),
            "raw_contours": len(raw), "raw_contour_points": sum(len(c) for c in raw),
            "raw_points_on_contours_le_4_px2": sum(len(c) for c in small_contours),
            "simplified_contours": len(traced.contours),
            "simplified_contour_points": sum(len(c.points) for c in traced.contours),
            "fitter_fallback_calls": dict(counts), "fitter_fallback_line_segments": dict(output_lines),
            "generated_subpaths": len(generated)}


def minimum_points_probe(contours, image_shape, settings, minimum_points: int = 4) -> str:
    """Change one existing fitter argument temporarily; reuse its SVG serializer too."""
    original = visual_pipeline.fit_cubic_beziers

    def controlled(*args, **kwargs):
        kwargs["min_points_for_bezier"] = minimum_points
        return original(*args, **kwargs)

    with patch.object(visual_pipeline, "fit_cubic_beziers", controlled):
        return visual_pipeline.contours_to_svg(contours, image_shape, options=TikzOptions(), settings=settings)


def svg_error_bound(source: str, candidate: str) -> float:
    """Bound corresponding subpath deviations; do not match unrelated components."""
    def paths(svg):
        return [s for e in ET.fromstring(svg).iter() if e.tag.rsplit("}", 1)[-1] == "path"
                for s in parse_paths(e.get("d", ""))]
    old, new = paths(source), paths(candidate)
    if len(old) != len(new):
        return float("inf")
    config = OptimizationConfig(tolerance=.25, sample_step=.025)
    maximum = 0.0
    for a, b in zip(old, new):
        if a.closed != b.closed:
            return float("inf")
        if a.segments == b.segments:
            continue
        maximum = max(maximum, bounded_error(flatten(a.segments, .002), flatten(b.segments, .002), config, .002))
    return maximum


def refit_svg(svg: str, epsilon: float, *, include_curves: bool) -> tuple[str, dict]:
    """Try stricter RDP/global fitting across smooth runs, preserving paint objects.

    This adapter targets the controlled corpus only. It does not handle arbitrary
    transforms/CSS. Existing geometry and fitting routines provide all candidates.
    """
    root = ET.fromstring(svg)
    path_elements = [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "path"]
    groups = [parse_paths(e.get("d", "")) for e in path_elements]
    owners = Counter()
    for group in groups:
        for subpath in group:
            owners.update(set(tuple(_array(p)) for s in subpath.segments for p in (s.start, s.end)))
    locks = {p for p, count in owners.items() if count > 1}
    operations = Counter()
    error_bound = 0.0
    config = OptimizationConfig(tolerance=.25, sample_step=.025)
    for element, group in zip(path_elements, groups):
        parts = []
        for subpath in group:
            segments = subpath.segments
            if not segments:
                parts.append(element.get("d", ""))
                continue
            cuts = [0]
            for i in range(1, len(segments)):
                if turn(segments[i - 1], segments[i]) >= 35 or tuple(_array(segments[i].start)) in locks:
                    cuts.append(i)
                elif not include_curves and (segments[i - 1].kind != "line" or segments[i].kind != "line"):
                    cuts.append(i)
            cuts.append(len(segments))
            output = []
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                run = list(segments[lo:hi])
                best = run
                run_error = 0.0
                operations["runs_examined"] += 1
                if len(run) >= 2 and (include_curves or all(s.kind == "line" for s in run)):
                    operations["runs_with_fitting_opportunity"] += 1
                    reference = flatten(run, .002)
                    candidates = [("rdp", _from_points(simplify_polyline_rdp(reference, epsilon)))]
                    if include_curves:
                        candidates.append(("global_cubic", _fitted_segments(reference, replace(config, tolerance=.25))))
                    for name, proposed in candidates:
                        if not proposed or _cost(proposed) >= _cost(best):
                            operations[name + "_not_more_compact"] += 1
                            continue
                        error = bounded_error(reference, flatten(proposed, .002), config, .002)
                        if error <= .1:
                            best, run_error = proposed, error
                            operations[name] += 1
                        else:
                            operations[name + "_error_budget_rejected"] += 1
                output.extend(best)
                error_bound = max(error_bound, run_error)
            if subpath.closed:
                reference, proposed = flatten(segments, .002), flatten(output, .002)
                if np.sign(signed_area(reference)) != np.sign(signed_area(proposed)):
                    output = list(segments)
                    operations["orientation_rejected"] += 1
            parts.append(_serialize(subpath, output))
        element.set("d", " ".join(parts))
    return ET.tostring(root, encoding="unicode"), {"operations": dict(operations), "maximum_error_bound": error_bound,
                                                 "epsilon": epsilon, "include_curves": include_curves}
