from __future__ import annotations

from hashlib import sha256
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fikzpy.core.image_processor as image_processor
import fikzpy.core.svg_semantic_parser as svg_semantic_parser
from fikzpy.core.image_processor import ProcessingSettings, process_image
from fikzpy.core.semantic_geometry import BezierPrimitive, CirclePrimitive, ClosedShapePrimitive
from fikzpy.core.semantic_geometry import EllipsePrimitive, LinePrimitive, Point2D, PolylinePrimitive
from fikzpy.core.semantic_geometry import PrimitiveGroup, RGBColor
from fikzpy.core.svg_semantic_parser import SvgParseError, SvgParserConfig, parse_svg_to_primitives
from fikzpy.core.svg_styles import SvgStyle, parse_color, style_from_attributes
from fikzpy.core.svg_transforms import SvgTransform, parse_transform_list
from fikzpy.core.tracers.base import TracerConfig, TracerKind, TracerResult
from fikzpy.core.vectorization_config import config_for_mode


def _svg(body: str, attrs: str = 'width="100" height="100" viewBox="0 0 100 100"') -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" {attrs}>{body}</svg>'


def _parse(body: str, config: SvgParserConfig | None = None):
    return parse_svg_to_primitives(_svg(body), config=config)


def _flat(result) -> tuple:
    items = []
    for primitive in result.primitives:
        if isinstance(primitive, PrimitiveGroup):
            items.extend(primitive.flatten())
        else:
            items.append(primitive)
    return tuple(items)


@pytest.mark.parametrize(
    "source",
    [
        "",
        b"",
        "<svg>",
        "<html></html>",
        "<svg><path></svg>",
        '<svg><path d="M 0 0 L"/></svg>',
        '<svg><line x1="NaN" y1="0" x2="1" y2="0"/></svg>',
        '<svg><line transform="translate(bad)" x2="1"/></svg>',
        '<svg><line stroke="notacolor" x2="1"/></svg>',
        '<svg><path d="M0 0 A 1 1 0 2 0 1 1"/></svg>',
    ],
)
def test_invalid_svg_inputs_are_reported(source) -> None:
    with pytest.raises((SvgParseError, FileNotFoundError)):
        parse_svg_to_primitives(source, SvgParserConfig(strict=True))


def test_svg_without_namespace_is_tolerated_with_warning() -> None:
    result = parse_svg_to_primitives('<svg><line x1="0" y1="0" x2="1" y2="0"/></svg>')

    assert isinstance(result.primitives[0], LinePrimitive)
    assert any(warning.code == "namespace" for warning in result.warnings)


def test_svg_namespace_is_recorded() -> None:
    result = _parse('<line x1="0" y1="0" x2="1" y2="0"/>')

    assert result.document_info.namespace == svg_semantic_parser.SVG_NAMESPACE
    assert result.metrics.elements_total == 2


@pytest.mark.parametrize(
    ("body", "primitive_type"),
    [
        ('<line x1="0" y1="1" x2="10" y2="1" stroke="black"/>', LinePrimitive),
        ('<polyline points="0,0 5,5 10,0" stroke="black" fill="none"/>', PolylinePrimitive),
        ('<polygon points="0,0 10,0 10,10" fill="red"/>', ClosedShapePrimitive),
        ('<rect x="1" y="2" width="8" height="6" fill="blue"/>', ClosedShapePrimitive),
        ('<circle cx="5" cy="6" r="3" fill="none" stroke="black"/>', CirclePrimitive),
        ('<ellipse cx="5" cy="6" rx="4" ry="2" fill="none" stroke="black"/>', EllipsePrimitive),
    ],
)
def test_basic_svg_elements_create_semantic_primitives(body: str, primitive_type: type) -> None:
    result = _parse(body)

    assert isinstance(result.primitives[0], primitive_type)
    assert result.metrics.elements_parsed == 1


@pytest.mark.parametrize("corner_attr", ['rx="2"', 'ry="2"', 'rx="3" ry="2"'])
def test_rounded_rects_are_preserved_as_groups_with_beziers(corner_attr: str) -> None:
    result = _parse(f'<rect x="0" y="0" width="20" height="10" {corner_attr} fill="black"/>')

    assert isinstance(result.primitives[0], PrimitiveGroup)
    assert sum(isinstance(item, BezierPrimitive) for item in result.primitives[0].flatten()) == 4


@pytest.mark.parametrize(
    ("path_data", "expected_type", "bezier_count"),
    [
        ("M0 0 L10 0", LinePrimitive, 0),
        ("m0 0 l10 0 l0 10", PolylinePrimitive, 0),
        ("M0 0 H10 V10", PolylinePrimitive, 0),
        ("M10 10 h5 v5", PolylinePrimitive, 0),
        ("M0 0 10 0 10 10", PolylinePrimitive, 0),
        ("M0 0 L10-5", LinePrimitive, 0),
        ("M0 0 C1 2 3 4 5 6", BezierPrimitive, 1),
        ("M0 0 c1 2 3 4 5 6", BezierPrimitive, 1),
        ("M0 0 C1 0 2 0 3 0 S5 0 6 0", PrimitiveGroup, 2),
        ("M0 0 c1 0 2 0 3 0 s2 0 3 0", PrimitiveGroup, 2),
        ("M0 0 Q3 3 6 0", BezierPrimitive, 1),
        ("M0 0 q3 3 6 0", BezierPrimitive, 1),
        ("M0 0 Q3 3 6 0 T12 0", PrimitiveGroup, 2),
        ("M0 0 q3 3 6 0 t6 0", PrimitiveGroup, 2),
        ("M0 0 L10 0 L10 10 Z", ClosedShapePrimitive, 0),
        ("M0 0 L10 0 L10 10 z", ClosedShapePrimitive, 0),
        ("M0 0 L10 0 M20 0 L30 0", PrimitiveGroup, 0),
        ("M0 0 A10 10 0 0 1 10 10", BezierPrimitive, 1),
        ("M0 0 a10 10 0 0 1 10 10", BezierPrimitive, 1),
        ("M0 0 A10 10 0 1 1 10 10", PrimitiveGroup, 3),
        ("M0 0 A10 10 0 0 0 10 10", BezierPrimitive, 1),
        ("M0 0 A0 10 0 0 1 10 10", LinePrimitive, 0),
    ],
)
def test_path_commands_are_converted(path_data: str, expected_type: type, bezier_count: int) -> None:
    result = _parse(f'<path d="{path_data}" stroke="black" fill="none"/>')

    assert isinstance(result.primitives[0], expected_type)
    assert sum(isinstance(item, BezierPrimitive) for item in _flat(result)) == bezier_count


def test_quadratic_conversion_matches_exact_cubic_formula() -> None:
    result = _parse('<path d="M0 0 Q3 3 6 0" stroke="black" fill="none"/>')
    bezier = result.primitives[0]

    assert isinstance(bezier, BezierPrimitive)
    assert bezier.control1 == Point2D(2.0, 2.0)
    assert bezier.control2 == Point2D(4.0, 2.0)


def test_arc_conversion_uses_few_cubic_segments_and_preserves_endpoints() -> None:
    result = _parse('<path d="M0 0 A10 10 0 1 1 10 10" stroke="black" fill="none"/>')
    beziers = [item for item in _flat(result) if isinstance(item, BezierPrimitive)]

    assert 1 < len(beziers) <= 4
    assert beziers[0].start == Point2D(0.0, 0.0)
    assert beziers[-1].end == Point2D(10.0, 10.0)


@pytest.mark.parametrize(
    ("transform", "expected_end"),
    [
        ("translate(5,2)", (15.0, 2.0)),
        ("scale(2)", (20.0, 0.0)),
        ("scale(2,3)", (20.0, 0.0)),
        ("rotate(90)", (0.0, 10.0)),
        ("rotate(90,5,0)", (5.0, 5.0)),
        ("matrix(1,0,0,1,7,8)", (17.0, 8.0)),
        ("skewX(45)", (10.0, 0.0)),
        ("skewY(45)", (10.0, 10.0)),
        ("translate(10,0) scale(2)", (30.0, 0.0)),
    ],
)
def test_line_transforms_are_applied(transform: str, expected_end: tuple[float, float]) -> None:
    result = _parse(f'<line x1="0" y1="0" x2="10" y2="0" transform="{transform}" stroke="black"/>')
    line = result.primitives[0]

    assert isinstance(line, LinePrimitive)
    assert line.end.x == pytest.approx(expected_end[0])
    assert line.end.y == pytest.approx(expected_end[1])
    assert result.metrics.transform_count == 1


def test_transform_inheritance_from_group() -> None:
    result = _parse('<g transform="translate(3,4)"><line x1="0" y1="0" x2="1" y2="0"/></g>')
    line = _flat(result)[0]

    assert isinstance(line, LinePrimitive)
    assert line.start == Point2D(3.0, 4.0)
    assert isinstance(result.primitives[0], PrimitiveGroup)


def test_circle_scale_transform_preserves_or_changes_simple_shape_safely() -> None:
    uniform = _parse('<circle cx="1" cy="1" r="2" transform="scale(2)" stroke="black" fill="none"/>')
    nonuniform = _parse('<circle cx="1" cy="1" r="2" transform="scale(2,3)" stroke="black" fill="none"/>')
    skewed = _parse('<circle cx="1" cy="1" r="2" transform="skewX(20)" stroke="black" fill="none"/>')

    assert isinstance(uniform.primitives[0], CirclePrimitive)
    assert uniform.primitives[0].radius == pytest.approx(4.0)
    assert isinstance(nonuniform.primitives[0], EllipsePrimitive)
    assert isinstance(skewed.primitives[0], PrimitiveGroup)
    assert sum(isinstance(item, BezierPrimitive) for item in skewed.primitives[0].flatten()) == 4


@pytest.mark.parametrize(
    ("style", "assertion"),
    [
        ("stroke:#123; fill:none; stroke-width:2", lambda p: p.stroke.color == RGBColor(17, 34, 51) and p.stroke.width == 2),
        ("stroke:#abcd; fill:none", lambda p: p.stroke.color == RGBColor(170, 187, 204) and p.stroke.opacity == pytest.approx(221 / 255)),
        ("stroke:#11223344; fill:none", lambda p: p.stroke.color == RGBColor(17, 34, 51) and p.stroke.opacity == pytest.approx(68 / 255)),
        ("stroke:rgb(10,20,30); fill:none", lambda p: p.stroke.color == RGBColor(10, 20, 30)),
        ("stroke:rgb(100%,0%,0%); fill:none", lambda p: p.stroke.color == RGBColor(255, 0, 0)),
        ("stroke:blue; fill:none", lambda p: p.stroke.color == RGBColor(0, 0, 255)),
        ("color:red; stroke:currentColor; fill:none", lambda p: p.stroke.color == RGBColor(255, 0, 0)),
        ("stroke:black; stroke-opacity:.5; fill:none", lambda p: p.stroke.opacity == pytest.approx(0.5)),
        ("stroke:black; opacity:.5; fill:none", lambda p: p.stroke.opacity == pytest.approx(0.5) and p.opacity == pytest.approx(0.5)),
        ("stroke:black; fill:green; fill-opacity:.25", lambda p: p.fill.color == RGBColor(0, 128, 0) and p.fill.opacity == pytest.approx(0.25)),
        ("stroke:black; stroke-linecap:round; stroke-linejoin:bevel", lambda p: p.stroke.line_cap == "round" and p.stroke.line_join == "bevel"),
        ("stroke:black; stroke-dasharray:1,2 3", lambda p: p.stroke.dash_pattern == (1.0, 2.0, 3.0)),
    ],
)
def test_style_and_color_values_are_resolved(style: str, assertion) -> None:
    result = _parse(f'<line x1="0" y1="0" x2="10" y2="0" style="{style}"/>')

    assert assertion(result.primitives[0])


def test_direct_attributes_are_overridden_by_inline_style() -> None:
    result = _parse('<line x1="0" y1="0" x2="10" y2="0" stroke="red" style="stroke:blue"/>')

    assert result.primitives[0].stroke.color == RGBColor(0, 0, 255)


def test_inherited_style_and_inherit_keyword() -> None:
    result = _parse('<g stroke="red"><line x1="0" y1="0" x2="10" y2="0" stroke="inherit"/></g>')
    line = _flat(result)[0]

    assert line.stroke.color == RGBColor(255, 0, 0)


@pytest.mark.parametrize(
    "body",
    [
        '<line x1="0" y1="0" x2="10" y2="0" display="none"/>',
        '<line x1="0" y1="0" x2="10" y2="0" visibility="hidden"/>',
        '<line x1="0" y1="0" x2="10" y2="0" opacity="0"/>',
        '<line x1="0" y1="0" x2="10" y2="0" stroke="none" fill="none"/>',
    ],
)
def test_invisible_elements_are_ignored(body: str) -> None:
    result = _parse(body)

    assert result.primitives == ()
    assert result.metrics.elements_ignored == 1


@pytest.mark.parametrize(
    ("attrs", "expected_end"),
    [
        ('width="100px" height="100px" viewBox="0 0 100 100"', (10.0, 10.0)),
        ('width="72pt" height="72pt" viewBox="0 0 72 72"', (13.3333333333, 13.3333333333)),
        ('width="25.4mm" height="25.4mm" viewBox="0 0 10 10"', (96.0, 96.0)),
        ('width="2.54cm" height="2.54cm" viewBox="0 0 10 10"', (96.0, 96.0)),
        ('width="1in" height="1in" viewBox="0 0 10 10"', (96.0, 96.0)),
        ('viewBox="0 0 10 10"', (10.0, 10.0)),
        ('width="200" height="100" viewBox="0 0 100 100" preserveAspectRatio="none"', (20.0, 10.0)),
        ('width="200" height="100" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet"', (60.0, 10.0)),
    ],
)
def test_viewbox_units_and_preserve_aspect_ratio(attrs: str, expected_end: tuple[float, float]) -> None:
    result = parse_svg_to_primitives(_svg('<line x1="0" y1="0" x2="10" y2="10"/>', attrs=attrs))
    line = result.primitives[0]

    assert isinstance(line, LinePrimitive)
    assert line.end.x == pytest.approx(expected_end[0])
    assert line.end.y == pytest.approx(expected_end[1])


def test_preserve_aspect_ratio_slice_is_supported() -> None:
    result = parse_svg_to_primitives(
        _svg('<line x1="0" y1="0" x2="100" y2="100"/>', attrs='width="200" height="100" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid slice"')
    )
    line = result.primitives[0]

    assert isinstance(line, LinePrimitive)
    assert line.end.x == pytest.approx(200.0)
    assert line.end.y == pytest.approx(150.0)


def test_groups_can_be_preserved_or_flattened() -> None:
    preserved = _parse('<g id="grp"><line x1="0" y1="0" x2="1" y2="0"/></g>')
    flattened = _parse(
        '<g id="grp"><line x1="0" y1="0" x2="1" y2="0"/></g>',
        SvgParserConfig(preserve_groups=False),
    )

    assert isinstance(preserved.primitives[0], PrimitiveGroup)
    assert isinstance(flattened.primitives[0], LinePrimitive)


def test_defs_and_use_href_are_resolved_with_position_and_transform() -> None:
    body = (
        '<defs><line id="seed" x1="0" y1="0" x2="10" y2="0" stroke="black"/></defs>'
        '<use href="#seed" x="5" y="2" transform="translate(1,1)"/>'
    )
    result = _parse(body)
    line = _flat(result)[0]

    assert isinstance(line, LinePrimitive)
    assert line.start == Point2D(6.0, 3.0)
    assert result.metrics.use_elements == 1
    assert result.metrics.references_resolved == 1


def test_xlink_href_is_resolved() -> None:
    body = (
        '<defs><circle id="dot" cx="1" cy="1" r="1"/></defs>'
        '<use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#dot" x="2" y="3"/>'
    )
    result = _parse(body)

    assert isinstance(_flat(result)[0], CirclePrimitive)


@pytest.mark.parametrize(
    "body",
    [
        '<use href="#missing"/>',
        '<use href="other.svg#shape"/>',
        '<unknownThing/>',
    ],
)
def test_tolerant_mode_warns_for_unsupported_references_and_elements(body: str) -> None:
    result = _parse(body)

    assert result.primitives == ()
    assert result.warnings


def test_strict_mode_raises_on_unknown_element() -> None:
    with pytest.raises(SvgParseError):
        _parse("<unknownThing/>", SvgParserConfig(strict=True))


def test_circular_reference_and_reference_depth_are_rejected_in_strict_mode() -> None:
    body = '<defs><g id="a"><use href="#a"/></g></defs><use href="#a"/>'
    with pytest.raises(SvgParseError):
        _parse(body, SvgParserConfig(strict=True))
    with pytest.raises(SvgParseError):
        _parse(body, SvgParserConfig(strict=True, maximum_reference_depth=0))


def test_multi_ring_path_is_preserved_as_group_without_boolean_operations() -> None:
    body = '<path fill="black" fill-rule="evenodd" d="M0 0 L20 0 L20 20 L0 20 Z M5 5 L15 5 L15 15 L5 15 Z"/>'
    result = _parse(body)

    assert isinstance(result.primitives[0], PrimitiveGroup)
    assert result.metrics.subpaths == 2
    assert sum(isinstance(item, ClosedShapePrimitive) for item in result.primitives[0].flatten()) == 2


def test_input_string_bytes_path_and_pathlib_are_supported(tmp_path: Path) -> None:
    text = _svg('<line x1="0" y1="0" x2="1" y2="0"/>')
    path = tmp_path / "input.svg"
    path.write_text(text, encoding="utf-8")

    assert parse_svg_to_primitives(text).source_type == "string"
    assert parse_svg_to_primitives(text.encode("utf-8")).source_type == "bytes"
    assert parse_svg_to_primitives(str(path)).source_type == "path"
    assert parse_svg_to_primitives(path).source_type == "path"


def test_tracer_result_with_svg_text_and_path_is_supported(tmp_path: Path) -> None:
    text = _svg('<path d="M0 0 L10 0 L10 10 Z" fill="black"/>')
    path = tmp_path / "trace.svg"
    path.write_text(text, encoding="utf-8")
    result = TracerResult(
        requested_tracer=TracerKind.POTRACE,
        effective_tracer=TracerKind.POTRACE,
        success=True,
        backend="cli",
        svg_text=text,
        svg_path=str(path),
        config=TracerConfig(),
        version="fake",
        svg_sha256=sha256(text.encode("utf-8")).hexdigest(),
    )

    parsed = parse_svg_to_primitives(result)

    assert parsed.source_type == "tracer_result"
    assert parsed.tracer_metadata["effective_tracer"] == "potrace"
    assert isinstance(parsed.primitives[0], ClosedShapePrimitive)


def test_invalid_tracer_result_is_rejected(tmp_path: Path) -> None:
    text = _svg("<line x2='1'/>")
    path = tmp_path / "trace.svg"
    path.write_text(text.replace("line", "circle"), encoding="utf-8")

    failed = TracerResult(TracerKind.POTRACE, TracerKind.POTRACE, False, "cli", svg_text=text)
    inconsistent = TracerResult(TracerKind.POTRACE, TracerKind.POTRACE, True, "cli", svg_text=text, svg_path=str(path))

    with pytest.raises(SvgParseError):
        parse_svg_to_primitives(failed)
    with pytest.raises(SvgParseError):
        parse_svg_to_primitives(inconsistent)


def test_determinism_and_to_dict() -> None:
    text = _svg('<g stroke="red"><path id="p" d="M0 0 L10 0 L10 10 Z" fill="none"/></g>')

    first = parse_svg_to_primitives(text).to_dict()
    second = parse_svg_to_primitives(text).to_dict()

    assert first == second
    assert first["input_hash"] == sha256(text.encode("utf-8")).hexdigest()
    assert "<svg" not in repr(first)
    assert first["metrics"]["closed_shape_primitives"] == 1


def test_metadata_contains_source_details_without_xml_objects() -> None:
    result = _parse('<line id="l1" class="guide" x1="0" y1="0" x2="10" y2="0" stroke="black"/>')
    metadata = result.primitives[0].metadata

    assert metadata["id"] == "l1"
    assert metadata["class"] == "guide"
    assert metadata["tag"] == "line"
    assert "Element" not in repr(metadata)


def test_style_and_transform_helpers_are_serializable() -> None:
    transform = parse_transform_list("translate(1,2) scale(3)")
    style = style_from_attributes({"style": "stroke:#fff; fill:rgb(0,0,0); opacity:.5"})
    color = parse_color("currentColor", RGBColor(1, 2, 3))

    assert transform.to_dict()["e"] == 1.0
    assert style.to_dict()["opacity"] == 0.5
    assert color.color == RGBColor(1, 2, 3)


@pytest.mark.parametrize(
    ("color_text", "expected"),
    [
        ("black", RGBColor(0, 0, 0)),
        ("white", RGBColor(255, 255, 255)),
        ("red", RGBColor(255, 0, 0)),
        ("green", RGBColor(0, 128, 0)),
        ("lime", RGBColor(0, 255, 0)),
        ("yellow", RGBColor(255, 255, 0)),
        ("cyan", RGBColor(0, 255, 255)),
        ("magenta", RGBColor(255, 0, 255)),
        ("gray", RGBColor(128, 128, 128)),
        ("orange", RGBColor(255, 165, 0)),
    ],
)
def test_basic_named_colors(color_text: str, expected: RGBColor) -> None:
    assert parse_color(color_text).color == expected


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SvgParserConfig(target_unit="meter"),
        lambda: SvgParserConfig(decimal_precision=-1),
        lambda: SvgParserConfig(default_dpi=0),
        lambda: SvgParserConfig(maximum_reference_depth=-1),
    ],
)
def test_invalid_parser_configuration_is_rejected(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_parser_import_does_not_start_gui() -> None:
    code = (
        "import sys; "
        "import fikzpy.core.svg_semantic_parser; "
        "assert 'PySide6' not in sys.modules; "
        "assert 'fikzpy.gui.main_window' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr


def test_parser_does_not_generate_tikz_or_call_svg_bridge_or_tracers() -> None:
    source = Path(svg_semantic_parser.__file__).read_text(encoding="utf-8").lower()
    result = _parse('<path d="M0 0 L10 0" stroke="black" fill="none"/>')

    assert "\\draw" not in repr(result.to_dict())
    assert "tikzpicture" not in repr(result.to_dict()).lower()
    assert "svg2tikz" not in source
    assert "subprocess" not in source
    assert "trace_image(" not in source


def test_visual_mode_is_preserved_and_parser_is_not_connected_to_classic(monkeypatch: pytest.MonkeyPatch) -> None:
    assert config_for_mode("visual").mode == "visual"
    assert config_for_mode("classic").mode == "classic"
    assert config_for_mode("contours").mode == "contours"

    called = {"visual": False}

    class DummyVisualResult:
        contours = []
        ink_mask = np.zeros((4, 4), dtype=np.uint8)

    def fake_trace_visual_contours(image, settings):
        called["visual"] = True
        return DummyVisualResult()

    monkeypatch.setattr(image_processor, "trace_visual_contours", fake_trace_visual_contours)
    process_image(np.full((4, 4, 3), 255, dtype=np.uint8), ProcessingSettings(vectorization_mode="visual"))

    tikz_pipeline_source = Path("fikzpy/core/tikz_pipeline.py").read_text(encoding="utf-8")
    image_processor_source = Path("fikzpy/core/image_processor.py").read_text(encoding="utf-8")
    assert called["visual"]
    assert "svg_semantic_parser" not in tikz_pipeline_source
    assert "svg_semantic_parser" not in image_processor_source


@pytest.mark.parametrize(
    ("svg_kind", "body"),
    [
        ("potrace", '<g transform="translate(1,2)" fill="black"><path d="M0 0L10 0L10 10L0 10z"/></g>'),
        ("autotrace", '<g stroke="black" fill="none"><path d="M0 0 L10 0"/><path d="M0 5 C3 1 7 1 10 5"/></g>'),
        ("vtracer", '<g opacity=".8"><path fill="red" d="M0 0 L10 0 L10 10 Z"/><path fill="blue" d="M20 0 L30 0 L30 10 Z"/></g>'),
    ],
)
def test_synthetic_tracer_svg_structures(svg_kind: str, body: str) -> None:
    result = _parse(body)

    assert result.primitives
    assert result.backend == "stdlib_elementtree"
    assert result.metrics.elements_parsed >= 1
    assert svg_kind in {"potrace", "autotrace", "vtracer"}
