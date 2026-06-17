from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import fikzpy.core.centerline_pipeline as centerline_pipeline
import fikzpy.core.skeleton_graph as skeleton_graph
from fikzpy.core.adaptive_preprocessing import preprocess_image
from fikzpy.core.centerline_pipeline import CenterlineConfig, PathClosureType, extract_centerlines
from fikzpy.core.centerline_pipeline import centerline_paths_to_polylines
from fikzpy.core.image_classifier import ImageCategory, classify_image
from fikzpy.core.semantic_geometry import PolylinePrimitive
from fikzpy.core.skeleton_graph import SkeletonEdge, build_skeleton_graph


def _mask(width: int = 80, height: int = 60) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def _horizontal_line() -> np.ndarray:
    mask = _mask()
    mask[30, 8:72] = 255
    return mask


def _square_loop() -> np.ndarray:
    mask = _mask()
    mask[15, 20:60] = 255
    mask[44, 20:60] = 255
    mask[15:45, 20] = 255
    mask[15:45, 59] = 255
    return mask


def test_empty_mask_returns_valid_empty_result() -> None:
    result = extract_centerlines(_mask())

    assert result.paths == ()
    assert result.metrics.skeleton_pixel_count == 0
    assert result.metrics.connected_component_count == 0
    assert "empty mask" in result.warnings


def test_isolated_pixel_is_reported_without_empty_path() -> None:
    mask = _mask()
    mask[20, 20] = 255

    result = extract_centerlines(mask)

    assert result.paths == ()
    assert result.metrics.isolated_pixel_count == 1
    assert "isolated skeleton pixel" in result.warnings


def test_horizontal_line_extracts_one_open_centerline() -> None:
    result = extract_centerlines(_horizontal_line())

    assert result.metrics.open_path_count == 1
    assert result.paths[0].closure is PathClosureType.OPEN
    assert result.paths[0].points[0].y == result.paths[0].points[-1].y


def test_vertical_line_extracts_one_open_centerline() -> None:
    mask = _mask()
    mask[8:52, 40] = 255

    result = extract_centerlines(mask)

    assert result.metrics.open_path_count == 1
    assert result.paths[0].points[0].x == result.paths[0].points[-1].x


def test_diagonal_line_extracts_ordered_path() -> None:
    mask = _mask()
    for offset in range(10, 45):
        mask[offset, offset] = 255

    result = extract_centerlines(mask)

    assert result.metrics.open_path_count == 1
    assert result.paths[0].points[0] != result.paths[0].points[-1]


def test_simple_curve_extracts_path() -> None:
    mask = _mask()
    cv2.ellipse(mask, (40, 35), (24, 14), 0, 180, 350, 255, 1)

    result = extract_centerlines(mask, CenterlineConfig(simplify_pixel_paths=False))

    assert result.metrics.open_path_count >= 1
    assert sum(len(path.points) for path in result.paths) > 20


def test_t_junction_reports_endpoints_and_junctions() -> None:
    mask = _mask()
    mask[30, 10:70] = 255
    mask[10:31, 40] = 255

    result = extract_centerlines(mask, CenterlineConfig(simplify_pixel_paths=False))

    assert result.metrics.endpoint_count >= 3
    assert result.metrics.junction_count >= 1
    assert result.metrics.open_path_count >= 3


def test_x_junction_reports_four_endpoints_and_junctions() -> None:
    mask = _mask()
    for offset in range(12, 48):
        mask[offset, offset] = 255
        mask[offset, 79 - offset] = 255

    result = extract_centerlines(mask, CenterlineConfig(simplify_pixel_paths=False))

    assert result.metrics.endpoint_count >= 4
    assert result.metrics.junction_count >= 1


def test_square_loop_is_preserved_as_closed_path() -> None:
    result = extract_centerlines(_square_loop())

    assert result.metrics.closed_path_count == 1
    assert result.metrics.cycle_count >= 1
    assert result.paths[0].closure is PathClosureType.CLOSED


def test_approximate_circular_loop_is_preserved() -> None:
    mask = _mask()
    cv2.circle(mask, (40, 30), 16, 255, 1)

    result = extract_centerlines(mask)

    assert result.metrics.closed_path_count >= 1
    assert result.metrics.cycle_count >= 1


def test_two_independent_components_are_kept() -> None:
    mask = _mask()
    mask[15, 5:30] = 255
    mask[45, 45:75] = 255

    result = extract_centerlines(mask)

    assert result.metrics.connected_component_count == 2
    assert result.metrics.open_path_count == 2


def test_cycle_with_branch_keeps_closed_and_open_paths() -> None:
    mask = _square_loop()
    mask[5:15, 40] = 255

    result = extract_centerlines(mask, CenterlineConfig(simplify_pixel_paths=False))

    assert result.metrics.closed_path_count >= 1
    assert result.metrics.open_path_count >= 1


def test_short_spur_can_be_pruned_conservatively() -> None:
    mask = _horizontal_line()
    mask[26:30, 38] = 255
    config = CenterlineConfig(
        preserve_small_details=False,
        maximum_pruning_ratio=0.25,
        simplify_pixel_paths=False,
    )

    result = extract_centerlines(mask, config)

    assert result.metrics.spurs_removed >= 1
    assert result.metrics.removed_length > 0


def test_spur_is_preserved_when_small_details_are_preserved() -> None:
    mask = _horizontal_line()
    mask[26:30, 38] = 255

    result = extract_centerlines(mask, CenterlineConfig(preserve_small_details=True))

    assert result.metrics.spurs_removed == 0


def test_pruning_can_be_disabled() -> None:
    mask = _horizontal_line()
    mask[26:30, 38] = 255
    config = CenterlineConfig(enable_spur_pruning=False, preserve_small_details=False)

    result = extract_centerlines(mask, config)

    assert result.metrics.spurs_removed == 0


def test_pruning_respects_maximum_ratio() -> None:
    mask = _horizontal_line()
    mask[26:30, 38] = 255
    config = CenterlineConfig(
        preserve_small_details=False,
        maximum_pruning_ratio=0.001,
        simplify_pixel_paths=False,
    )

    result = extract_centerlines(mask, config)

    assert result.metrics.spurs_removed == 0
    assert "maximum pruning ratio reached" in result.warnings


def test_endpoint_merge_connects_compatible_paths() -> None:
    mask = _mask()
    mask[30, 8:30] = 255
    mask[30, 33:58] = 255
    config = CenterlineConfig(merge_nearby_endpoints=True, maximum_endpoint_distance=4.0)

    result = extract_centerlines(mask, config)

    assert result.metrics.paths_after_merging == 1
    assert result.paths[0].metadata["merged_from"]


def test_endpoint_merge_rejects_incompatible_angle() -> None:
    mask = _mask()
    mask[30, 8:30] = 255
    mask[10:28, 32] = 255
    config = CenterlineConfig(
        merge_nearby_endpoints=True,
        maximum_endpoint_distance=6.0,
        maximum_merge_angle=10.0,
    )

    result = extract_centerlines(mask, config)

    assert result.metrics.paths_after_merging == result.metrics.paths_before_merging


def test_endpoint_merge_rejects_distant_endpoints() -> None:
    mask = _mask()
    mask[30, 8:25] = 255
    mask[30, 45:70] = 255
    config = CenterlineConfig(merge_nearby_endpoints=True, maximum_endpoint_distance=3.0)

    result = extract_centerlines(mask, config)

    assert result.metrics.paths_after_merging == 2


def test_simplification_preserves_open_path_endpoints() -> None:
    mask = _mask()
    for x in range(10, 70):
        y = 30 + int(3 * np.sin(x / 6.0))
        mask[y, x] = 255
    unsimplified = extract_centerlines(mask, CenterlineConfig(simplify_pixel_paths=False))
    simplified = extract_centerlines(mask, CenterlineConfig(simplify_pixel_paths=True, simplification_tolerance=1.0))

    assert simplified.paths[0].points[0] == unsimplified.paths[0].points[0]
    assert simplified.paths[0].points[-1] == unsimplified.paths[0].points[-1]
    assert simplified.metrics.points_after_simplification <= simplified.metrics.points_before_simplification


def test_simplification_preserves_closed_cycle() -> None:
    result = extract_centerlines(_square_loop(), CenterlineConfig(simplification_tolerance=1.0))

    assert result.paths[0].closure is PathClosureType.CLOSED
    assert len(result.paths[0].points) >= 3


def test_centerline_result_is_deterministic() -> None:
    config = CenterlineConfig(merge_nearby_endpoints=True, maximum_endpoint_distance=4.0)
    first = extract_centerlines(_horizontal_line(), config)
    second = extract_centerlines(_horizontal_line(), config)

    assert first.to_dict() == second.to_dict()
    assert np.array_equal(first.skeleton, second.skeleton)


def test_to_dict_uses_summaries_not_full_arrays() -> None:
    result = extract_centerlines(_horizontal_line())
    serialized = result.to_dict()

    assert "sha256" in serialized["skeleton"]
    assert "array(" not in repr(serialized)
    assert serialized["paths"][0]["point_count"] >= 2


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CenterlineConfig(connectivity=6),
        lambda: CenterlineConfig(skeleton_method="unknown"),
        lambda: CenterlineConfig(maximum_merge_angle=181.0),
        lambda: CenterlineConfig(minimum_path_length=0),
        lambda: CenterlineConfig(maximum_pruning_ratio=-0.1),
    ],
)
def test_invalid_centerline_config_is_rejected(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_topology_validation_rejects_invalid_edge() -> None:
    with pytest.raises(ValueError):
        SkeletonEdge(
            id="edge",
            start_node_id="",
            end_node_id="n0001",
            pixels=((1, 1),),
            component_id=0,
        )


def test_strict_topology_validation_accepts_valid_graph() -> None:
    graph = build_skeleton_graph(_horizontal_line(), strict=True)

    assert graph.endpoint_count == 2
    assert graph.warnings == ()


def test_optional_polyline_conversion_returns_semantic_primitives() -> None:
    result = extract_centerlines(_horizontal_line())
    polylines = centerline_paths_to_polylines(result)

    assert len(polylines) == 1
    assert isinstance(polylines[0], PolylinePrimitive)
    assert polylines[0].metadata["source"] == "centerline_path"


def test_modules_do_not_generate_output_documents_or_call_external_tracers() -> None:
    sources = [
        Path(centerline_pipeline.__file__).read_text(encoding="utf-8").lower(),
        Path(skeleton_graph.__file__).read_text(encoding="utf-8").lower(),
    ]
    forbidden = ("\\\\draw", "tikzpicture", "potrace", "autotrace", "vtracer", "svg2tikz")

    for source in sources:
        for token in forbidden:
            assert token not in source


def test_modules_do_not_perform_primitive_fitting() -> None:
    source = Path(centerline_pipeline.__file__).read_text(encoding="utf-8").lower()

    for token in ("fitline", "houghlines", "houghcircles", "fitellipse", "bezierprimitive"):
        assert token not in source


def test_importing_centerline_modules_does_not_start_gui() -> None:
    code = (
        "import sys; "
        "import fikzpy.core.centerline_pipeline; "
        "import fikzpy.core.skeleton_graph; "
        "assert 'PySide6' not in sys.modules; "
        "assert 'fikzpy.gui.main_window' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr


def test_pipeline_uses_preprocessing_result_without_reprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    preprocessing_result = preprocess_image(_horizontal_line(), category=ImageCategory.LINE_ART)

    def fail_preprocess(*_args, **_kwargs):
        raise AssertionError("preprocess_image should not be called for PreprocessingResult input")

    monkeypatch.setattr(centerline_pipeline, "preprocess_image", fail_preprocess)
    result = extract_centerlines(preprocessing_result)

    assert result.metrics.skeleton_pixel_count > 0
    assert result.selected_threshold_method == preprocessing_result.method


def test_raw_image_input_reuses_preprocessing() -> None:
    image = np.full((60, 80, 3), 255, dtype=np.uint8)
    cv2.line(image, (8, 30), (72, 30), (0, 0, 0), 1)

    result = extract_centerlines(image)

    assert result.preprocessing_summary is not None
    assert result.selected_threshold_method is not None


def test_invalid_masks_are_rejected() -> None:
    with pytest.raises(ValueError):
        extract_centerlines(np.array([[np.nan]]))


def test_full_mask_does_not_crash() -> None:
    result = extract_centerlines(np.full((20, 20), 255, dtype=np.uint8))

    assert "full mask" in result.warnings
    assert result.metrics.skeleton_pixel_count >= 0


def test_fallback_operates_without_sknw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skeleton_graph, "_is_sknw_available", lambda: False)
    config = CenterlineConfig(use_sknw_if_available=True)

    result = extract_centerlines(_horizontal_line(), config)

    assert result.graph.backend == "fallback"
    assert result.graph.sknw_available is False


def test_sknw_availability_is_diagnostic_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skeleton_graph, "_is_sknw_available", lambda: True)
    config = CenterlineConfig(use_sknw_if_available=True)

    result = extract_centerlines(_horizontal_line(), config)

    assert result.graph.backend == "fallback"
    assert result.graph.sknw_available is True


def test_baseline_line_art_images_can_be_processed_read_only() -> None:
    baseline = Path("examples/classic_semantic_baseline")
    selected = [
        baseline / "line_art_bw.png",
        baseline / "geometric_diagram.png",
        baseline / "noisy_grayscale.png",
    ]
    diagnostics: dict[str, dict[str, int | str]] = {}

    for image_path in selected:
        classification = classify_image(image_path)
        preprocessing = preprocess_image(image_path, category=classification.category)
        result = extract_centerlines(preprocessing)
        diagnostics[image_path.name] = {
            "category": classification.category.value,
            "method": preprocessing.method,
            "paths": len(result.paths),
            "skeleton_pixels": result.metrics.skeleton_pixel_count,
        }

    assert set(diagnostics) == {"line_art_bw.png", "geometric_diagram.png", "noisy_grayscale.png"}
    assert all(value["method"] for value in diagnostics.values())
