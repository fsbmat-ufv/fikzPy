"""Safety and geometry tests for the isolated pre-RDP adapters."""

import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from experiments.svg_tikz.optimizer import flatten, parse_paths
from experiments.svg_tikz.phase3_metrics import closed_error_bound, raster_at_frame, region_identity
from experiments.svg_tikz.reconstruction import (ReconstructionConfig, _objects_to_segments,
    capture_contours, persistent_corners, reconstruct_svg, tangent_fit)
from experiments.svg_tikz.run import svg_document
from fikzpy.core.image_processor import ProcessingSettings, visual_settings_from_processing
from fikzpy.core.visual_pipeline import contours_to_svg, trace_visual_contours


def test_capture_reuses_exact_mask_rdp_and_order_without_mutation():
    image = np.full((60, 80, 3), 255, np.uint8)
    cv2.circle(image, (25, 30), 16, (0, 0, 0), 2)
    cv2.line(image, (50, 20), (70, 40), (0, 0, 0), 2)
    original_image, original_rdp = image.copy(), cv2.approxPolyDP
    settings = visual_settings_from_processing(ProcessingSettings(vectorization_mode="visual"))
    baseline = trace_visual_contours(image, settings)
    contours, raw, mask = capture_contours(image, settings)
    assert cv2.approxPolyDP is original_rdp
    assert np.array_equal(image, original_image)
    assert len(contours) == len(raw) == len(baseline.contours)
    assert sum(map(len, raw)) > sum(len(c.points) for c in contours)
    for a, b in zip(contours, baseline.contours, strict=True):
        assert np.array_equal(a.points, b.points)
        assert (a.area, a.perimeter) == (b.area, b.perimeter)
    assert mask.shape == (180, 240)


def test_recursive_split_propagates_tangent_with_correct_sign():
    x = np.linspace(0, 100, 501)
    points = np.column_stack([x, 10 * np.sin(x / 9)])
    fitted = tangent_fit(points, .3)
    # Inspect objects before the existing three-decimal serializer rounds them.
    from fikzpy.core.vector_objects import BezierCurve
    adjacent = [(a, b) for a, b in zip(fitted[:-1], fitted[1:])
                if isinstance(a, BezierCurve) and isinstance(b, BezierCurve)]
    assert len(adjacent) >= 2
    for a, b in adjacent:
        assert a.end == b.start
        va = np.array(a.end.as_tuple()) - a.control2.as_tuple()
        vb = np.array(b.control1.as_tuple()) - b.start.as_tuple()
        assert np.dot(va, vb) > 0
        assert abs(va[0] * vb[1] - va[1] * vb[0]) < 1e-8
    segments = _objects_to_segments(fitted)
    assert np.allclose([segments[0].start.x, segments[0].start.y], points[0], atol=.001)
    assert np.allclose([segments[-1].end.x, segments[-1].end.y], points[-1], atol=.001)


def test_multiscale_corners_retain_rectangle_corners():
    points = np.array([(x, 0) for x in range(20)] + [(20, y) for y in range(20)]
                      + [(x, 20) for x in range(20, 0, -1)] + [(0, y) for y in range(20, 0, -1)], float)
    corners = persistent_corners(points, ReconstructionConfig())
    assert {tuple(points[i]) for i in corners} == {(0, 0), (20, 0), (20, 20), (0, 20)}


def test_continuous_bound_sees_between_vertex_overshoot_in_both_directions():
    line = np.array([[0., 0.], [10., 0.], [0., 0.]])
    curve = flatten(parse_paths("M0 0C0 40 10 40 10 0L0 0Z")[0].segments, .002)
    assert closed_error_bound(line, curve) > 29.9
    assert closed_error_bound(curve, line) > 29.9


def test_region_identity_rejects_replaced_detail_with_same_component_count():
    a = np.full((50, 50, 3), 255, np.uint8)
    a[5:15, 5:15] = 0
    a[30:34, 30:34] = 0
    b = a.copy()
    b[30:34, 30:34] = 255
    b[40:44, 40:44] = 0
    assert region_identity(a, a) == {"component_identity": True, "hole_identity": True}
    assert not region_identity(a, b)["component_identity"]


def test_calibrated_raster_frame_uses_pixel_edges_and_white_padding():
    source = np.zeros((2, 2, 3), np.uint8)
    result = raster_at_frame(source, (4, 4), (2., 2.))
    assert np.array_equal(result[:2, :2], source)
    assert np.all(result[2:, :] == 255) and np.all(result[:, 2:] == 255)


@pytest.mark.parametrize("strategy", ["A", "B"])
def test_compound_micro_hole_is_retained(strategy):
    raw = [np.array([[10., 10.], [50., 10.], [50., 50.], [10., 50.]]),
           np.array([[20., 20.], [21., 20.], [21., 21.], [20., 21.]])]
    source = svg_document('<path fill="black" stroke="none" fill-rule="evenodd" opacity=".7" '
        'd="M10 10L50 10L50 50L10 50Z M20 20L21 20L21 21L20 21Z"/>')
    result, details, proposal = reconstruct_svg(source, raw, ReconstructionConfig(strategy=strategy))
    for svg in (result, proposal):
        path = ET.fromstring(svg)[0]
        assert path.get("fill-rule") == "evenodd" and path.get("opacity") == ".7"
        rings = parse_paths(path.get("d"))
        assert len(rings) == 2 and all(s.closed for s in rings)
        assert rings[1].segments == parse_paths(ET.fromstring(source)[0].get("d"))[1].segments
    assert details["maximum_error_bound"] <= .25


def test_shared_raw_contact_is_an_explicit_anchor_in_both_regions():
    raw = [np.array([[10., 10.], [30., 10.], [30., 30.], [10., 30.]]),
           np.array([[30., 30.], [50., 30.], [50., 50.], [30., 50.]])]
    source = svg_document('<path fill="black" stroke="none" fill-rule="evenodd" '
        'd="M10 10L30 10L30 30L10 30Z M30 30L50 30L50 50L30 50Z"/>')
    guarded, details, proposal = reconstruct_svg(source, raw, ReconstructionConfig(strategy="B"))
    assert details["shared_vertices"] == 1
    assert [r["contact_anchors"] for r in details["rings"]] == [1, 1]
    for svg in (guarded, proposal):
        for ring in parse_paths(ET.fromstring(svg)[0].get("d")):
            assert (30., 30.) in {(s.start.x, s.start.y) for s in ring.segments}


def test_adapters_reject_invalid_inputs_instead_of_hiding_failure():
    with pytest.raises(ValueError):
        ReconstructionConfig(strategy="C")
    with pytest.raises(ValueError):
        ReconstructionConfig(geometry_budget=.3)
    with pytest.raises(ValueError):
        reconstruct_svg(svg_document('<path d="M0 0L10 0" stroke="black"/>'), [], ReconstructionConfig())
