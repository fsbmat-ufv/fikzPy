"""Pre-RDP contour reconstruction adapters; never imported by the application."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from unittest.mock import patch
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from fikzpy.core import bezier_fit as legacy
from fikzpy.core import bezier_fitting as semantic
from fikzpy.core import visual_pipeline as visual
from fikzpy.core.geometry_error import corner_indices, signed_area
from .optimizer import (_array, _cost, _serialize, bounded_error,
                        flatten, OptimizationConfig, parse_paths)


@dataclass(frozen=True)
class ReconstructionConfig:
    """Distances are original image pixels, not upsampled mask pixels."""

    strategy: str = "A"
    fit_error: float = .72
    geometry_budget: float = .25
    corner_angle: float = 60.0
    corner_scales: tuple[float, float] = (.5, 1.0)

    def __post_init__(self):
        if self.strategy not in {"A", "B"}:
            raise ValueError("Only strategies A and B are supported.")
        values = [self.fit_error, self.geometry_budget, self.corner_angle, *self.corner_scales]
        if not np.isfinite(values).all() or self.fit_error <= 0 or not 0 < self.geometry_budget <= .25:
            raise ValueError("Finite positive tolerances and geometry budget <= .25 are required.")
        if len(self.corner_scales) != 2 or min(self.corner_scales) <= 0 or not 0 < self.corner_angle < 180:
            raise ValueError("Two positive spatial scales and a valid corner angle are required.")


def capture_contours(image, settings):
    """Observe the official RDP call, keeping its filtering and ordering exactly."""
    scale = max(1, int(settings.upsample_factor))
    bgr = visual._ensure_bgr(image)
    mask = visual._build_high_resolution_mask(visual._resize_for_trace(bgr, scale), settings, scale)
    observed = defaultdict(deque)
    original = cv2.approxPolyDP

    def observe(raw, epsilon, closed):
        approx = original(raw, epsilon, closed)
        points = approx.reshape(-1, 2).astype(np.float64) / scale
        observed[points.tobytes()].append(raw.reshape(-1, 2).astype(np.float64) / scale)
        return approx

    with patch.object(cv2, "approxPolyDP", observe):
        contours = visual._extract_visual_contours(mask, scale=scale, settings=settings)
    raw = [observed[c.points.tobytes()].popleft() for c in contours]
    return contours, raw, mask


def persistent_corners(points, config):
    """Keep angular vertices present at two spatial RDP scales; use raw samples for fitting.

    This is an operational corner detector, not semantic classification. Small
    rings are retained later. RDP is used only to locate anchors in this adapter.
    """
    sets = []
    for epsilon in config.corner_scales:
        reduced = cv2.approxPolyDP(np.asarray(points, np.float32), epsilon, True).reshape(-1, 2)
        indices = corner_indices(reduced, threshold_degrees=config.corner_angle, closed=True)
        sets.append({int(np.argmin(np.linalg.norm(points - reduced[i], axis=1))) for i in indices})
    return sets[0] & sets[1]


def _objects_to_segments(objects):
    """Reuse the official serializer instead of introducing another primitive exporter."""
    dummy = visual.Contour(points=np.array([[0., 0.], [1., 0.], [1., 1.]]), closed=True, area=.5, perimeter=3.)
    with patch.object(visual, "fit_cubic_beziers", return_value=tuple(objects)):
        data = visual._contour_to_svg_path(dummy, visual.VisualTracingSettings())
    # Serialization always adds Z; discard only the synthetic closing edge for
    # an open fitting span. The returned objects themselves retain their ends.
    return list(parse_paths(data.removesuffix("Z"))[0].segments)


def fit_span(points, config):
    """Compare the unchanged fitter with an adapter that propagates split tangents."""
    if config.strategy == "A":
        objects = legacy.fit_cubic_beziers(points, config.fit_error, min_bezier_length=.9,
            min_points_for_bezier=6, straightness_tolerance=min(.1, config.fit_error / 2), max_depth=18)
    elif config.strategy == "B":
        objects = tangent_fit(points, config.fit_error)
    else:
        raise ValueError("Only strategies A and B are supported.")
    return _objects_to_segments(objects) if objects else []


def tangent_fit(points, error, *, depth=18, start_tangent=None, end_tangent=None):
    """Reuse legacy LS/evaluation/fallback, passing correctly signed central tangents.

    Minimum six points and minimum length .9 match the Visual fitter. The
    averaged tangent window spans about one original pixel, avoiding a single
    staircase pixel as an endpoint direction. Corners are split by the caller.
    G1 is guaranteed only at accepted cubic/cubic recursive splits; line and
    polyline fallbacks do not acquire an artificial smoothness guarantee.
    """
    points = np.asarray(points, dtype=float)
    straight = min(.1, error / 2)
    if len(points) < 6 or legacy._path_length(points) < .9:
        return [legacy._fallback_small_path(points, straight)]
    if legacy._max_line_distance(points) <= straight:
        return [legacy._line_from_array(points[0], points[-1])]
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    left = min(len(points) - 1, max(1, int(np.searchsorted(distances, 1.))))
    right = max(0, min(len(points) - 2, int(np.searchsorted(distances, distances[-1] - 1.))))
    ta = legacy._unit(points[left] - points[0]) if start_tangent is None else start_tangent
    tb = legacy._unit(points[right] - points[-1]) if end_tangent is None else end_tangent
    curve = legacy._generate_cubic(points, ta, tb)
    maximum, split = legacy._max_bezier_error(points, curve)
    # Reuse the semantic fitter's Newton and LS kernels. Keep the refined
    # parameters when measuring error, rather than reverting to chord u.
    parameters = legacy._chord_length_parameters(points)
    for _ in range(3):
        if maximum <= error:
            break
        refined = np.array([semantic._newton_parameter(curve, p, float(u))
                            for p, u in zip(points, parameters, strict=True)])
        if not np.isfinite(refined).all() or np.any(np.diff(refined) <= 0):
            break
        refined[0], refined[-1] = 0., 1.
        if refined.min() < 0 or refined.max() > 1:
            break
        trial = semantic._generate_cubic(points, refined, ta, tb)
        errors = np.linalg.norm(legacy._bezier_values(trial, refined) - points, axis=1)
        index = int(np.argmax(errors[1:-1]) + 1)
        if errors[index] >= maximum:
            break
        curve, parameters, maximum, split = trial, refined, float(errors[index]), index
    if maximum <= error and legacy._is_valid_curve(curve, 1e-4, .9):
        return [legacy._bezier_from_arrays(*curve)]
    if depth == 0 or split <= 1 or split >= len(points) - 2:
        return [legacy._fallback_small_path(points, straight)]
    lo = max(0, int(np.searchsorted(distances, distances[split] - 1.)))
    hi = min(len(points) - 1, int(np.searchsorted(distances, distances[split] + 1.)))
    center = legacy._unit(points[hi] - points[lo])
    if np.linalg.norm(center) == 0:
        center = ta
    return (tangent_fit(points[:split + 1], error, depth=depth - 1, start_tangent=ta, end_tangent=-center)
            + tangent_fit(points[split:], error, depth=depth - 1, start_tangent=center, end_tangent=tb))


def reconstruct_svg(source, raw_contours, config):
    """Fit raw spans, admitting only cheaper replacements within a continuous bound.

    Baseline segment endpoints locate raw samples. Persistent raw corners and
    shared endpoints are locked. Unsuccessful spans subdivide on BASELINE
    boundaries and retain baseline geometry. This conservative hybrid does not
    claim to solve the minimum-segment problem or identify semantic objects.
    """
    root = ET.fromstring(source)
    paths = [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "path"]
    if len(paths) != 1 or paths[0].get("fill-rule") != "evenodd" or paths[0].get("stroke") != "none":
        raise ValueError("Adapter expects the official Visual compound filled path.")
    rings = parse_paths(paths[0].get("d", ""))
    if len(rings) != len(raw_contours):
        raise ValueError("Raw contours must correspond one-to-one to baseline subpaths.")
    owners = defaultdict(set)
    self_contacts = defaultdict(set)
    for i, raw in enumerate(raw_contours):
        seen = set()
        for point in raw:
            key = tuple(point)
            if key in seen:
                self_contacts[i].add(key)
            seen.add(key)
            owners[key].add(i)
    shared = {p for p, ids in owners.items() if len(ids) > 1}
    bounds = OptimizationConfig(sample_step=.025)
    stats = {"strategy": config.strategy, "raw_points": sum(map(len, raw_contours)),
             "persistent_corners": 0, "shared_vertices": len(shared), "replacements": 0,
             "maximum_error_bound": 0., "attempted_spans": 0, "cheaper_spans": 0,
             "rejected_geometric_bounds": [], "rings": []}
    parts, proposals = [], []
    for ring_index, (ring, raw) in enumerate(zip(rings, raw_contours, strict=True)):
        segments = list(ring.segments)
        corners = persistent_corners(raw, config)
        stats["persistent_corners"] += len(corners)
        contact_indices = {j for j, p in enumerate(raw) if tuple(p) in shared or tuple(p) in self_contacts[ring_index]}
        anchors = corners | contact_indices
        # Count/noise is not evidence that a micro-region is unimportant.
        keep = abs(signed_area(raw)) <= 4
        mapping = [int(np.argmin(np.linalg.norm(raw - _array(s.start), axis=1))) for s in segments]
        distances = [float(np.linalg.norm(raw[j] - _array(s.start))) for j, s in zip(mapping, segments)]
        # Baseline serialization rounds thirds to three decimals. No snapping
        # beyond that known rounding bound, and no reversed/ambiguous traversal.
        increments = [(b - a) % len(raw) for a, b in zip(mapping, mapping[1:] + mapping[:1])]
        if max(distances, default=0) > .001 or sum(increments) != len(raw):
            keep = True
        cuts = {0, len(segments)}
        for k, j in enumerate(mapping):
            if any(min((j - c) % len(raw), (c - j) % len(raw)) <= 1 for c in anchors):
                cuts.add(k)
        # Limit closed span degeneracy with an existing vertex opposite the start.
        if len(cuts) == 2 and len(segments) > 2:
            cuts.add(len(segments) // 2)
        # Also retain the unguarded raw proposal as a negative/control artifact.
        # It has the SAME corner anchors and fitter, without the baseline guard.
        raw_cuts = sorted({0, len(raw), *anchors})
        if len(raw_cuts) == 2:
            raw_cuts.insert(1, len(raw) // 2)
        closed_raw = np.vstack([raw, raw[0]])
        proposal = segments if keep else [s for lo, hi in zip(raw_cuts[:-1], raw_cuts[1:])
            for s in fit_span(closed_raw[lo:hi + 1], config)]
        # Use its own raw start, keeping region identity and orientation.
        from dataclasses import replace
        proposals.append(_serialize(replace(ring, start=proposal[0].start), proposal))
        accepted = []
        ring_errors = []

        def fit(lo, hi):
            old = segments[lo:hi]
            if hi - lo < 2:
                return old
            start = mapping[lo]
            length = sum(increments[lo:hi])
            sample = raw[(np.arange(length + 1) + start) % len(raw)].copy()
            sample[0], sample[-1] = _array(old[0].start), _array(old[-1].end)
            candidate = fit_span(sample, config)
            stats["attempted_spans"] += 1
            if candidate and _cost(candidate) < _cost(old):
                stats["cheaper_spans"] += 1
                old_flat, new_flat = flatten(old, .002), flatten(candidate, .002)
                error = bounded_error(old_flat, new_flat, bounds, .002)
                if error <= config.geometry_budget:
                    # Keep persistent raw corners inside a baseline segment too;
                    # if no existing vertex coincides, retain that span.
                    interior_corners = [c for c in anchors if 0 < (c - start) % len(raw) < length]
                    vertices = np.array([_array(s.start) for s in candidate] + [_array(candidate[-1].end)])
                    if any(np.min(np.linalg.norm(vertices - raw[c], axis=1)) > .001 for c in interior_corners):
                        mid = (lo + hi) // 2
                        return fit(lo, mid) + fit(mid, hi)
                    accepted.append((lo, hi))
                    ring_errors.append(error)
                    return candidate
                stats["rejected_geometric_bounds"].append(error)
            mid = (lo + hi) // 2
            return fit(lo, mid) + fit(mid, hi)

        result = segments if keep else [s for lo, hi in zip(sorted(cuts)[:-1], sorted(cuts)[1:]) for s in fit(lo, hi)]
        # Never reverse a region's winding even though Visual uses evenodd.
        if np.sign(signed_area(flatten(result))) != np.sign(signed_area(raw)):
            result, accepted, ring_errors = segments, [], []
        stats["replacements"] += len(accepted)
        stats["maximum_error_bound"] = max(stats["maximum_error_bound"], max(ring_errors, default=0))
        stats["rings"].append({"raw_points": len(raw), "before_segments": len(segments),
            "after_segments": len(result), "persistent_corners": len(corners), "kept_small_or_contact_or_mapping": keep,
            "contact_anchors": len(contact_indices),
            "replaced_spans": accepted, "maximum_error_bound": max(ring_errors, default=0)})
        parts.append(_serialize(ring, result))
    paths[0].set("d", " ".join(parts))
    guarded = ET.tostring(root, encoding="unicode")
    paths[0].set("d", " ".join(proposals))
    return guarded, stats, ET.tostring(root, encoding="unicode")
