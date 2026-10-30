"""Calibrated frame and region-identity checks for the reconstruction experiment."""

from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .optimizer import _dense_polyline, flatten, parse_paths
from .phase2_reference import svg_at_fractional_frame
from .validation import white_rgb
from fikzpy.core.geometry_error import signed_area


def rings(svg):
    return [s for e in ET.fromstring(svg).iter() if e.tag.endswith("path") for s in parse_paths(e.get("d", ""))]


def closed_error_bound(a, b, *, step=.025, flatness=.002):
    """Bound both directions using OpenCV's exact closed-polyline distance kernel.

    Add half the arclength sampling step, both flattening budgets, and the
    float32 contour conversion error. This is a conservative Hausdorff bound,
    not a correspondence metric or a proof of topology.
    """
    maximum = 0.
    rounding = 0.
    for source, target in ((a, b), (b, a)):
        contour = np.asarray(target, np.float32).reshape(-1, 1, 2)
        rounding = max(rounding, float(np.linalg.norm(target - contour[:, 0].astype(float), axis=1).max(initial=0)))
        samples = _dense_polyline(source, step)
        maximum = max(maximum, max((abs(cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), True))
                                    for p in samples), default=0.))
    return maximum + step / 2 + 2 * flatness + 2 * rounding


def geometry_metrics(source, candidate, raw, baseline_raw_bounds=None):
    """Retain corresponding rings rather than letting a nearby ring hide a loss."""
    before, after = rings(source), rings(candidate)
    if len(before) != len(after) or len(before) != len(raw):
        raise ValueError("Subpath identities must be retained.")
    baseline_raw_bounds = baseline_raw_bounds or [None] * len(raw)
    values = []
    for a, b, points, cached in zip(before, after, raw, baseline_raw_bounds, strict=True):
        aa, bb = flatten(a.segments, .002), flatten(b.segments, .002)
        original = np.vstack([points, points[0]])
        raw_before = closed_error_bound(original, aa) if cached is None else cached
        unchanged = a.segments == b.segments
        values.append({"baseline_to_candidate_bound": 0. if unchanged else closed_error_bound(aa, bb),
            "raw_to_baseline_bound": raw_before,
            "raw_to_candidate_bound": raw_before if unchanged else closed_error_bound(original, bb),
            "winding_equal": bool(np.sign(signed_area(aa)) == np.sign(signed_area(bb))),
            "closed_equal": a.closed == b.closed,
            "geometry_unchanged": unchanged})
    return {"rings": values, "maximum_error_bound": max(x["baseline_to_candidate_bound"] for x in values),
        "maximum_raw_baseline_bound": max(x["raw_to_baseline_bound"] for x in values),
        "maximum_raw_candidate_bound": max(x["raw_to_candidate_bound"] for x in values),
        "changed_rings": sum(not x["geometry_unchanged"] for x in values),
        "winding_equal": all(x["winding_equal"] for x in values),
        "closed_equal": all(x["closed_equal"] for x in values)}


def _region_bijection(a, b, *, holes=False):
    """Require overlap to identify EVERY region, including small enclosed holes."""
    if holes:
        a, b = np.pad(~a, 1, constant_values=True), np.pad(~b, 1, constant_values=True)
    na, aa = cv2.connectedComponents(a.astype(np.uint8), connectivity=4 if holes else 8)
    nb, bb = cv2.connectedComponents(b.astype(np.uint8), connectivity=4 if holes else 8)
    ignored_a, ignored_b = {0}, {0}
    if holes:
        ignored_a.add(int(aa[0, 0]))
        ignored_b.add(int(bb[0, 0]))
    expected_a = set(range(na)) - ignored_a
    expected_b = set(range(nb)) - ignored_b
    pairs = np.unique(np.column_stack([aa.ravel(), bb.ravel()]), axis=0)
    forward = {i: set() for i in expected_a}
    backward = {i: set() for i in expected_b}
    for i, j in pairs:
        if i in forward and j in backward:
            forward[i].add(int(j))
            backward[j].add(int(i))
    return all(len(v) == 1 for v in forward.values()) and all(len(v) == 1 for v in backward.values())


def region_identity(before, after):
    a, b = (cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) < 245 for im in (before, after))
    return {"component_identity": _region_bijection(a, b), "hole_identity": _region_bijection(a, b, holes=True)}


def physical_frame(pdf, png, pdfinfo):
    """Read the actual MediaBox extent at 180 dpi, retaining fractional pixels."""
    metadata = subprocess.check_output([pdfinfo, str(pdf)], text=True)
    match = re.search(r"Page size:\s*([\d.]+) x ([\d.]+) pts", metadata)
    if not match:
        raise RuntimeError("Cannot read PDF physical frame.")
    target = tuple(float(v) * 180 / 72 for v in match.groups())
    return (png.shape[1], png.shape[0]), target


def raster_at_frame(image, canvas, target):
    """Map raster pixel EDGES onto the calibrated frame, without feature alignment."""
    width, height = canvas
    tw, th = target
    sy, sx = image.shape[:2]
    # OpenCV coordinates refer to pixel centers. The half-pixel translation
    # makes input and physical frame pixel edges agree (resize convention).
    transform = np.array([[tw / sx, 0., (tw / sx - 1) / 2], [0., th / sy, (th / sy - 1) / 2]])
    return cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def svg_at_frame(svg, canvas, target):
    return white_rgb(svg_at_fractional_frame(svg, canvas, target))
