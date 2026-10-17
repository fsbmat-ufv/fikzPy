"""Regression tests for the opt-in experiment, independent of GUI routing."""

import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from experiments.svg_tikz.optimizer import OptimizationConfig, bounded_error, optimize_svg, parse_paths
from experiments.svg_tikz.run import _image, _render_pdf, convert, fixtures, framed_document, run, svg_document
from experiments.svg_tikz.validation import compare_images, render_svg, topology


def test_redundant_line_reduces_coordinates_and_preserves_endpoints():
    _, svg, _ = next(fixtures())
    result = optimize_svg(svg)
    before = parse_paths(ET.fromstring(svg)[0].get("d"))[0]
    after = parse_paths(ET.fromstring(result.svg)[0].get("d"))[0]
    assert result.changed
    assert after.segments[0].start == before.segments[0].start
    assert after.segments[-1].end == before.segments[-1].end
    assert len(after.segments) == 1
    assert result.diagnostics["maximum_error_bound"] <= .25


def test_compound_fill_and_style_are_not_split():
    svg = svg_document('<g opacity=".8"><path id="ink" style="fill:#2458a0;stroke:none;fill-rule:evenodd" '
                       'd="M10 10 L30 10 L50 10 L100 10 L100 100 L10 100Z '
                       'M40 40L60 40L60 60L40 60Z M47 47L48 47L48 48L47 48Z"/></g>')
    result = optimize_svg(svg)
    before = ET.fromstring(svg)[0][0]
    after = ET.fromstring(result.svg)[0][0]
    assert {k: v for k, v in before.attrib.items() if k != "d"} == {k: v for k, v in after.attrib.items() if k != "d"}
    assert len(parse_paths(after.get("d"))) == 3
    assert all(s.closed for s in parse_paths(after.get("d")))
    metrics = compare_images(render_svg(svg), render_svg(result.svg), rgba=True)
    assert metrics["topology_equal"] and metrics["alpha_topology_equal"]


@pytest.mark.parametrize("body", [
    '<g transform="scale(2)"><path d="M0 0L1 0L2 0"/></g>',
    '<defs><clipPath id="c"><rect width="10" height="10"/></clipPath></defs><path d="M0 0L5 0L10 0" clip-path="url(#c)"/>',
    '<style>path{stroke:red}</style><path d="M0 0L1 0L2 0"/>',
    '<path d="M0 0L1 0L2 0" marker-mid="url(#arrow)"/>',
])
def test_unsupported_document_is_returned_byte_for_byte(body):
    svg = svg_document(body)
    result = optimize_svg(svg)
    assert result.svg == svg
    assert not result.changed
    assert "unsupported_document_passed_through" in result.diagnostics["warnings"]


def test_existing_cubic_is_not_refitted():
    svg = svg_document('<path d="M10 20C20 0 80 0 90 20L100 20L110 20L120 20" fill="none" stroke="black"/>')
    result = optimize_svg(svg)
    a = parse_paths(ET.fromstring(svg)[0].get("d"))[0].segments[0]
    b = parse_paths(ET.fromstring(result.svg)[0].get("d"))[0].segments[0]
    assert a == b


def test_shared_vertex_and_corner_are_retained():
    svg = svg_document('<path d="M10 30L30 30L50 30L50 50L50 80" fill="none" stroke="black"/>'
                       '<path d="M30 30L30 10" fill="none" stroke="red"/>')
    result = optimize_svg(svg)
    subpath = parse_paths(ET.fromstring(result.svg)[0].get("d"))[0]
    points = {(s.end.x, s.end.y) for s in subpath.segments}
    assert (30, 30) in points
    assert (50, 30) in points


def test_reverse_directed_distance_detects_shortcut():
    source = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], float)
    shortcut = np.array([[0, 0], [10, 0]], float)
    assert bounded_error(source, shortcut, OptimizationConfig(), .01) > 9


def test_hole_counter_and_visual_regression_detect_deleted_detail():
    mask = np.zeros((30, 30), bool)
    mask[2:28, 2:28] = True
    mask[10:12, 10:12] = False
    assert topology(mask) == {"components": 1, "holes": 1}
    svg = svg_document('<path fill-rule="evenodd" d="M10 10H100V100H10Z M40 40H42V42H40Z"/>')
    bad = svg_document('<path d="M10 10H100V100H10Z"/>')
    assert not compare_images(render_svg(svg), render_svg(bad), rgba=True)["topology_equal"]


def test_geometric_candidate_is_rejected_when_visual_gate_fails(monkeypatch):
    _, svg, _ = next(fixtures())
    monkeypatch.setattr("experiments.svg_tikz.optimizer.passes_visual_gate", lambda *a: False)
    result = optimize_svg(svg)
    assert result.svg == svg and not result.changed
    assert not result.diagnostics["candidate_accepted"]


def test_svg_to_tikz_keeps_compound_path_and_custom_color_definition():
    pytest.importorskip("svg2tikz")
    svg = svg_document('<path fill="#2458a0" fill-rule="evenodd" d="M10 10H100V100H10Z M40 40H60V60H40Z"/>')
    code, _ = convert(svg)
    assert code.count("\\draw[") == 1
    assert code.count("-- cycle") == 2
    assert "even odd rule" in code
    assert "\\definecolor{c2458a0}" in code
    document = framed_document(code, svg)
    assert "\\end{scope}\n\\pgfresetboundingbox" in document


def test_compiled_colored_compound_path_has_valid_frame_and_hole(tmp_path):
    from fikzpy.core.latex_compiler import compile_latex_document, resolve_latex_executable
    try:
        engine = resolve_latex_executable()
    except FileNotFoundError:
        pytest.skip("LaTeX is optional; compilation unavailable")
    poppler = shutil.which("pdftoppm")
    if poppler is None:
        pytest.skip("Poppler unavailable")
    svg = svg_document('<path fill="#2458a0" fill-rule="evenodd" d="M10 10H100V100H10Z M40 40H60V60H40Z"/>')
    code, _ = convert(svg)
    tex = tmp_path / "colored.tex"
    tex.write_text(framed_document(code, svg), encoding="utf-8")
    compiled = compile_latex_document(tex, manual_path=engine)
    assert compiled.returncode == 0, compiled.output[-2000:]
    pixels = _image(_render_pdf(compiled.pdf_path, poppler))
    assert min(pixels.shape[:2]) > 500
    assert pixels.min() < 100
    assert topology(pixels.mean(axis=2) < 245) == {"components": 1, "holes": 1}


def test_counterclockwise_nonzero_rings_keep_their_winding():
    svg = svg_document('<path fill-rule="nonzero" d="M10 10H100V100H10Z M40 40V60H60V40Z"/>')
    result = optimize_svg(svg)
    from fikzpy.core.geometry_error import signed_area
    from experiments.svg_tikz.optimizer import flatten
    before = parse_paths(ET.fromstring(svg)[0].get("d"))
    after = parse_paths(ET.fromstring(result.svg)[0].get("d"))
    assert [np.sign(signed_area(flatten(s.segments))) for s in before] == [np.sign(signed_area(flatten(s.segments))) for s in after]


def test_output_directory_cannot_be_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path, OptimizationConfig(), compile_pdf=False)


@pytest.mark.parametrize("kwargs", [{"tolerance": 0}, {"sample_step": .5}, {"minimum_iou": 2}, {"render_scale": float("nan")}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        OptimizationConfig(**kwargs)
