"""Isolated tests for renderer calibration and phase-two geometry adapters."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from experiments.svg_tikz.optimizer import flatten, parse_paths
from experiments.svg_tikz.phase2 import coverage, snapshot, stroker_circle
from experiments.svg_tikz.phase2_geometry import inventory, minimum_points_probe, refit_svg, subdivide_svg
from experiments.svg_tikz.phase2_geometry import svg_error_bound, tangent, turn, tracing_provenance
from experiments.svg_tikz.phase2_reference import svg_at_fractional_frame
from experiments.svg_tikz.run import svg_document
from experiments.svg_tikz.validation import compare_images, render_svg
from fikzpy.core import bezier_fit
from fikzpy.core.dot_to_dot.models import CubicSegment, Vec2
from fikzpy.core.image_processor import ProcessingSettings, visual_settings_from_processing
from fikzpy.core.visual_pipeline import _build_high_resolution_mask, _resize_for_trace, trace_visual_contours


def test_exact_subdivision_preserves_every_evaluated_curve_point():
    source = svg_document('<path d="M10 20C20 0 80 0 90 20" fill="none" stroke="black"/>')
    result = subdivide_svg(source, 2)
    before = parse_paths(ET.fromstring(source)[0].get("d"))[0].segments[0]
    pieces = parse_paths(ET.fromstring(result)[0].get("d"))[0].segments
    def curve(s):
        return CubicSegment(*[Vec2(p.x, p.y) for p in (s.start, s.control1, s.control2, s.end)])
    assert len(pieces) == 4
    for index, piece in enumerate(pieces):
        for u in np.linspace(0, 1, 51):
            a = curve(before).point((index + u) / 4)
            b = curve(piece).point(u)
            assert a.distance_to(b) < 1e-8
    assert pieces[0].start == before.start
    assert pieces[-1].end == before.end


def test_subdivision_keeps_compound_fill_order_and_closed_flags():
    source = svg_document('<path id="shape" fill="red" fill-rule="evenodd" opacity=".7" '
                          'd="M10 20C20 0 80 0 90 20L90 90L10 90Z M30 30H50V50H30Z"/>')
    result = ET.fromstring(subdivide_svg(source, 2))[0]
    assert {k: v for k, v in result.attrib.items() if k != "d"} == {
        k: v for k, v in ET.fromstring(source)[0].attrib.items() if k != "d"}
    assert [s.closed for s in parse_paths(result.get("d"))] == [True, True]


def test_refit_can_cross_existing_line_curve_boundaries_without_moving_anchors():
    source = svg_document('<path d="M10 30C20 30 30 30 40 30L50 30L60 30" fill="none" stroke="black"/>')
    candidate, diagnostics = refit_svg(source, .02, include_curves=True)
    segments = parse_paths(ET.fromstring(candidate)[0].get("d"))[0].segments
    assert len(segments) == 1
    assert (segments[0].start.x, segments[0].end.x) == (10, 60)
    assert diagnostics["maximum_error_bound"] <= .1


def test_strict_rdp_keeps_junction_and_sharp_corner():
    source = svg_document('<path d="M10 30L20 30L30 30L50 30L50 60" fill="none" stroke="black"/>'
                          '<path d="M30 30L30 10" fill="none" stroke="red"/>')
    candidate, _ = refit_svg(source, .05, include_curves=True)
    points = {(s.end.x, s.end.y) for s in parse_paths(ET.fromstring(candidate)[0].get("d"))[0].segments}
    assert (30, 30) in points and (50, 30) in points


def test_inventory_distinguishes_geometric_segments_and_redundant_closure():
    svg = svg_document('<path d="M0 0L10 0L10 10L0 0Z"/>')
    result = inventory(svg)
    assert result["segments"] == 3
    assert result["closure_segments"] == 1
    assert result["cubic_segments"] == 0
    assert result["closed_subpaths"] == 1


def test_tangent_at_degenerate_control_uses_chord():
    s = parse_paths('M0 0C0 0 10 0 10 0')[0].segments[0]
    assert np.allclose(tangent(s, False), [10, 0])
    assert np.allclose(tangent(s, True), [10, 0])
    second = parse_paths('M10 0L20 0')[0].segments[0]
    assert turn(s, second) == 0


def test_tracing_probe_restores_official_fitter_and_returns_same_contours():
    import cv2
    image = np.full((40, 60, 3), 255, np.uint8)
    cv2.rectangle(image, (10, 10), (40, 30), (0, 0, 0), 2)
    settings = visual_settings_from_processing(ProcessingSettings(vectorization_mode="visual"))
    traced = trace_visual_contours(image, settings)
    mask = _build_high_resolution_mask(_resize_for_trace(image, settings.upsample_factor), settings, settings.upsample_factor)
    previous = bezier_fit._fallback_small_path
    points_before = [c.points.copy() for c in traced.contours]
    stats = tracing_provenance(image, traced, settings, mask)
    assert bezier_fit._fallback_small_path is previous
    assert all(np.array_equal(a, b.points) for a, b in zip(points_before, traced.contours))
    assert stats["raw_contour_points"] >= stats["simplified_contour_points"]


def test_snapshot_detects_even_one_byte_change(tmp_path):
    p = tmp_path / "before.svg"
    p.write_bytes(b"before")
    before = snapshot(tmp_path)
    p.write_bytes(b"after")
    assert snapshot(tmp_path) != before


def test_high_precision_stroker_matches_known_annulus_area():
    area = coverage(stroker_circle(.001), 8)["area_user_units2"]
    assert abs(area - 2 * np.pi * 45 * 2) / (2 * np.pi * 45 * 2) < .005


def test_qt611_renderer_control_reproduces_representation_sensitive_stroke():
    from PySide6.QtCore import qVersion
    if qVersion() != "6.11.1":
        pytest.skip("Diagnostic control is specific to the measured Qt 6.11.1 build")
    ordinary, precise = stroker_circle(.25), stroker_circle(.001)
    assert coverage(ordinary, 8)["area_user_units2"] < coverage(precise, 8)["area_user_units2"] * .98
    assert compare_images(ordinary, precise, rgba=True)["foreground_iou"] < .99


def test_minimum_point_probe_restores_the_official_serializer_dependency():
    from fikzpy.core import visual_pipeline
    from fikzpy.core.contour_detector import Contour
    settings = visual_settings_from_processing(ProcessingSettings(vectorization_mode="visual"))
    contour = Contour(points=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float), closed=True)
    previous = visual_pipeline.fit_cubic_beziers
    result = minimum_points_probe([contour], (20, 20, 3), settings)
    assert visual_pipeline.fit_cubic_beziers is previous
    assert ET.fromstring(result)[0].get("fill-rule") == "evenodd"


def test_geometry_gate_rejects_missing_or_moved_rings():
    original = svg_document('<path d="M10 10H100V100H10Z M40 40H60V60H40Z" fill-rule="evenodd"/>')
    missing = svg_document('<path d="M10 10H100V100H10Z"/>')
    moved = original.replace('M40 40H60V60H40', 'M45 40H65V60H45')
    assert np.isinf(svg_error_bound(original, missing))
    assert svg_error_bound(original, moved) > 4.9


def test_fractional_physical_frame_does_not_stretch_to_rounded_canvas():
    svg = svg_document('<rect width="200" height="140" fill="black"/>')
    pixels = svg_at_fractional_frame(svg, (102, 72), (100.0, 70.0))
    assert pixels[:, 101, 3].max() == 0
    assert pixels[71, :, 3].max() == 0
    with pytest.raises(ValueError):
        svg_at_fractional_frame(svg, (100, 70), (101.0, 70.0))
