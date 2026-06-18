"""Generate the Issue 11.6 Classic line-art balanced refinement report.

The script exercises the balanced LINE_ART strategy (centerline plus
conservative outline-stroke recovery) and the overfilled/underdrawn line-art
validator added in Issue 11.6, alongside the existing filled-region and
mixed-monochrome regression cases. It uses only the public Classic semantic
pipeline and the new ``fikzpy.core.lineart_continuity`` validator; it does not
call svg2tikz, Visual, GUI code, or external tracers.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fikzpy.core.classic_pipeline_config import ClassicSemanticConfig
from fikzpy.core.classic_semantic_pipeline import ClassicSemanticResult, run_classic_semantic_pipeline
from fikzpy.core.lineart_continuity import LineArtContinuityMetrics, validate_lineart_balance
from fikzpy.core.semantic_geometry import ClosedShapePrimitive, FillStyle, LinePrimitive, Point2D, RGBColor
from fikzpy.core.semantic_tikz_exporter import export_primitives_to_tikz
from fikzpy.core.visual_validation import VisualValidationConfig, validate_semantic_output

OUTPUT_PATH = REPO_ROOT / "examples" / "classic_semantic_baseline" / "classic_lineart_balanced_refinement_report.json"


def p(x: float, y: float) -> Point2D:
    return Point2D(float(x), float(y))


def dinosaur_lineart_image() -> np.ndarray:
    image = np.full((160, 220, 3), 255, dtype=np.uint8)
    body = np.array(
        [
            [30, 120], [40, 90], [60, 70], [90, 55], [120, 50], [150, 55],
            [170, 50], [185, 60], [195, 75], [190, 95], [175, 110], [160, 120],
            [150, 140], [140, 150], [120, 150], [110, 140], [95, 150],
            [85, 150], [70, 140], [55, 130], [40, 130], [30, 120],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [body], True, (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (185, 60), (205, 55), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (205, 55), (195, 75), (0, 0, 0), 1, cv2.LINE_8)
    for x in range(190, 204, 4):
        cv2.line(image, (x, 58), (x, 66), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (60, 140), (55, 158), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (100, 148), (98, 158), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (130, 150), (128, 158), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (60, 110), (140, 120), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (80, 90), (90, 110), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (110, 80), (115, 105), (0, 0, 0), 1, cv2.LINE_8)
    cv2.circle(image, (175, 68), 2, (0, 0, 0), -1, cv2.LINE_8)
    return image


def line_art_simple_image() -> np.ndarray:
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    cv2.line(image, (12, 20), (84, 20), (0, 0, 0), 1)
    cv2.line(image, (20, 40), (78, 70), (0, 0, 0), 1)
    cv2.circle(image, (48, 56), 18, (0, 0, 0), 1)
    return image


def closed_contour_lineart_image() -> np.ndarray:
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (76, 76), (0, 0, 0), 1)
    cv2.line(image, (20, 48), (76, 48), (0, 0, 0), 1)
    return image


def filled_rectangle_image() -> np.ndarray:
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (24, 18), (72, 76), (0, 0, 0), -1)
    cv2.rectangle(image, (40, 34), (56, 52), (255, 255, 255), -1)
    return image


def silhouette_image() -> np.ndarray:
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    cv2.circle(image, (48, 30), 16, (0, 0, 0), -1)
    cv2.rectangle(image, (34, 46), (62, 80), (0, 0, 0), -1)
    return image


def mixed_monochrome_image() -> np.ndarray:
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    cv2.circle(image, (42, 34), 18, (0, 0, 0), -1)
    cv2.circle(image, (86, 36), 16, (0, 0, 0), -1)
    cv2.rectangle(image, (24, 70), (52, 104), (0, 0, 0), -1)
    cv2.rectangle(image, (76, 68), (105, 105), (0, 0, 0), -1)
    cv2.circle(image, (42, 39), 10, (255, 255, 255), -1)
    cv2.circle(image, (86, 41), 9, (255, 255, 255), -1)
    cv2.line(image, (30, 40), (37, 40), (0, 0, 0), 1)
    cv2.line(image, (46, 40), (52, 40), (0, 0, 0), 1)
    cv2.line(image, (38, 48), (47, 50), (0, 0, 0), 1)
    cv2.line(image, (79, 41), (84, 41), (0, 0, 0), 1)
    cv2.line(image, (90, 41), (96, 41), (0, 0, 0), 1)
    cv2.line(image, (20, 62), (58, 62), (0, 0, 0), 1)
    cv2.line(image, (72, 61), (110, 61), (0, 0, 0), 1)
    cv2.line(image, (60, 76), (72, 90), (0, 0, 0), 1)
    cv2.line(image, (64, 88), (70, 96), (0, 0, 0), 1)
    return image


def _record_from_pipeline_result(result: ClassicSemanticResult) -> dict:
    metrics = result.metrics
    return {
        "strategy_used": result.strategy_used.value,
        "filled_region_count": metrics.filled_region_primitives,
        "thin_stroke_count": metrics.thin_stroke_primitives,
        "outline_recovery_count": metrics.outline_recovery_count,
        "white_cutout_count": metrics.white_cutout_count,
        "edge_recall": metrics.edge_recall,
        "foreground_recall": result.validation_result.fidelity_score.raster_metrics.foreground_recall
        if result.validation_result is not None
        else 1.0,
        "contour_coverage": metrics.contour_coverage,
        "fragmentation_ratio": metrics.fragmentation_ratio,
        "dark_mass_preservation": metrics.dark_mass_preservation_ratio,
        "validation_score": metrics.validation_score,
        "regression_flags": list(metrics.lineart_regression_flags),
        "accepted": result.accepted,
        "rejection_reasons": list(result.rejection_reasons),
        "tikz_draw_commands": metrics.tikz_draw_commands,
        "tikz_fill_commands": metrics.tikz_fill_commands,
        "tikz_code_characters": metrics.tikz_code_characters,
        "deterministic_hash": result.deterministic_hash,
    }


def _record_from_synthetic_balance(
    *,
    strategy_label: str,
    mask: np.ndarray,
    primitives: list,
    continuity: LineArtContinuityMetrics,
) -> dict:
    balance = validate_lineart_balance(
        mask,
        primitives,
        continuity,
        max_filled_area_ratio_for_lineart=0.06,
        max_white_cutout_ratio_for_lineart=0.03,
        lineart_min_edge_recall=0.35,
        lineart_min_foreground_recall=0.55,
        lineart_min_contour_coverage=0.55,
        lineart_max_fragmentation_ratio=0.6,
        lineart_preserve_external_contour=True,
        reject_overfilled_lineart=True,
        reject_underdrawn_lineart=True,
    )
    tikz_result = export_primitives_to_tikz(primitives)
    return {
        "strategy_used": strategy_label,
        "filled_region_count": balance.fill_metrics.filled_region_count,
        "thin_stroke_count": sum(1 for primitive in primitives if getattr(primitive, "fill", None) is None),
        "outline_recovery_count": 0,
        "white_cutout_count": balance.fill_metrics.white_cutout_count,
        "edge_recall": continuity.edge_recall,
        "foreground_recall": continuity.foreground_recall,
        "contour_coverage": continuity.contour_coverage,
        "fragmentation_ratio": continuity.skeleton_fragmentation,
        "dark_mass_preservation": 1.0 - balance.fill_metrics.filled_area_ratio,
        "validation_score": 1.0 if balance.accepted else 0.0,
        "regression_flags": list(balance.flags),
        "accepted": balance.accepted,
        "rejection_reasons": list(balance.rejection_reasons),
        "tikz_draw_commands": tikz_result.metrics.draw_commands,
        "tikz_fill_commands": tikz_result.metrics.filled_paths_written,
        "tikz_code_characters": len(tikz_result.code),
        "deterministic_hash": tikz_result.deterministic_hash,
    }


def build_report() -> dict:
    cases: dict[str, dict] = {}

    cases["dinosaur_lineart_synthetic_good"] = _record_from_pipeline_result(
        run_classic_semantic_pipeline(dinosaur_lineart_image())
    )
    cases["line_art_simple"] = _record_from_pipeline_result(run_classic_semantic_pipeline(line_art_simple_image()))
    cases["closed_contour_lineart"] = _record_from_pipeline_result(
        run_classic_semantic_pipeline(closed_contour_lineart_image())
    )
    cases["filled_rectangle_real"] = _record_from_pipeline_result(run_classic_semantic_pipeline(filled_rectangle_image()))
    cases["silhouette_real"] = _record_from_pipeline_result(run_classic_semantic_pipeline(silhouette_image()))
    cases["mixed_monochrome_synthetic"] = _record_from_pipeline_result(
        run_classic_semantic_pipeline(mixed_monochrome_image())
    )

    flat_continuity = LineArtContinuityMetrics(
        components_before=8,
        components_after=8,
        lost_component_count=0,
        endpoint_count=4,
        junction_count=2,
        path_count=8,
        broken_path_count=0,
        average_path_length=40.0,
        contour_coverage=0.95,
        edge_recall=0.95,
        foreground_recall=0.95,
        skeleton_fragmentation=0.0,
        contour_bbox_coverage=0.95,
        external_contour_preservation=0.95,
    )
    overfilled_mask = cv2.cvtColor(dinosaur_lineart_image(), cv2.COLOR_RGB2GRAY) < 200
    cases["dinosaur_lineart_bad_overfilled"] = _record_from_synthetic_balance(
        strategy_label="line_art_overfilled_regression",
        mask=overfilled_mask,
        primitives=[
            ClosedShapePrimitive(
                (p(10, 10), p(200, 10), p(200, 150), p(10, 150)),
                fill=FillStyle(RGBColor(0, 0, 0)),
            )
        ],
        continuity=flat_continuity,
    )

    underdrawn_continuity = LineArtContinuityMetrics(
        components_before=8,
        components_after=2,
        lost_component_count=5,
        endpoint_count=12,
        junction_count=2,
        path_count=2,
        broken_path_count=4,
        average_path_length=8.0,
        contour_coverage=0.12,
        edge_recall=0.08,
        foreground_recall=0.10,
        skeleton_fragmentation=0.9,
        contour_bbox_coverage=0.20,
        external_contour_preservation=0.10,
    )
    underdrawn_mask = cv2.cvtColor(dinosaur_lineart_image(), cv2.COLOR_RGB2GRAY) < 200
    cases["dinosaur_lineart_bad_underdrawn"] = _record_from_synthetic_balance(
        strategy_label="line_art_underdrawn_regression",
        mask=underdrawn_mask,
        primitives=[LinePrimitive(p(30, 40), p(37, 40))],
        continuity=underdrawn_continuity,
    )

    bad_thin_only = [
        LinePrimitive(p(30, 40), p(37, 40)),
        LinePrimitive(p(46, 40), p(52, 40)),
        LinePrimitive(p(38, 48), p(47, 50)),
        LinePrimitive(p(20, 62), p(58, 62)),
        LinePrimitive(p(72, 61), p(110, 61)),
    ]
    bad_validation = validate_semantic_output(
        mixed_monochrome_image(),
        bad_thin_only,
        config=VisualValidationConfig(minimum_acceptable_score=0.45, minimum_fidelity_score=0.35),
    )
    bad_tikz = export_primitives_to_tikz(bad_thin_only)
    raster = bad_validation.fidelity_score.raster_metrics
    cases["mixed_monochrome_bad_regression"] = {
        "strategy_used": "mixed_monochrome_regression",
        "filled_region_count": 0,
        "thin_stroke_count": len(bad_thin_only),
        "outline_recovery_count": 0,
        "white_cutout_count": 0,
        "edge_recall": raster.edge_recall,
        "foreground_recall": raster.foreground_recall,
        "contour_coverage": raster.foreground_recall,
        "fragmentation_ratio": 0.0,
        "dark_mass_preservation": raster.dark_mass_preservation_ratio,
        "validation_score": bad_validation.fidelity_score.overall_score,
        "regression_flags": list(bad_validation.regression_flags),
        "accepted": bad_validation.accepted,
        "rejection_reasons": list(bad_validation.rejection_reasons),
        "tikz_draw_commands": bad_tikz.metrics.draw_commands,
        "tikz_fill_commands": bad_tikz.metrics.filled_paths_written,
        "tikz_code_characters": len(bad_tikz.code),
        "deterministic_hash": bad_validation.deterministic_hash,
    }

    return {"issue": "Issue 11.6", "cases": cases}


def main() -> None:
    report = build_report()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
