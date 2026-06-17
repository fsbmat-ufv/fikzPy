"""High-level TikZ build pipeline with classic and vector modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fikzpy.core.diagnostics import log_event
from fikzpy.core.image_processor import ProcessingResult, ProcessingSettings, process_image
from fikzpy.core.image_processor import visual_settings_from_processing
from fikzpy.core.tikz_generator import TikzOptions, generate_tikz_picture
from fikzpy.core.vector_exporter import VectorObjectStats, count_vector_objects
from fikzpy.core.vector_exporter import generate_tikz_from_vector_objects
from fikzpy.core.vector_objects import VectorObject
from fikzpy.core.vector_pipeline import fit_contours_to_vector_objects
from fikzpy.core.vectorization_config import config_for_mode
from fikzpy.core.visual_pipeline import VisualTikzStats, generate_visual_tikz_picture


@dataclass(frozen=True)
class TikzBuildResult:
    """Result of building TikZ from an image and mode selection."""

    requested_mode: str
    effective_mode: str
    processing_result: ProcessingResult
    tikz_code: str
    vector_objects: tuple[VectorObject, ...] = ()
    vector_stats: VectorObjectStats = VectorObjectStats()
    visual_stats: VisualTikzStats = VisualTikzStats()


def build_tikz_from_image(
    image: np.ndarray,
    settings: ProcessingSettings,
    options: TikzOptions,
) -> TikzBuildResult:
    """Build TikZ code from an image using the selected public mode."""
    requested_mode = settings.vectorization_mode
    effective_mode = config_for_mode(requested_mode).mode
    log_event("Vectorization", f"requested_mode={requested_mode}")
    log_event("Vectorization", f"effective_mode={effective_mode}")

    processing_result = process_image(image, settings)
    contours = processing_result.contours
    log_event("Vectorization", f"contours={len(contours)}")

    if effective_mode == "visual":
        log_event("Vectorization", "pipeline=visual_svg_trace")
        visual_result = generate_visual_tikz_picture(
            contours,
            processing_result.original_bgr.shape,
            options,
            visual_settings_from_processing(settings),
        )
        log_event("Visual", f"paths={visual_result.stats.paths}")
        log_event("Visual", f"svg_bytes={visual_result.stats.svg_bytes}")
        log_event("Visual", f"tikz_bytes={visual_result.stats.tikz_bytes}")
        log_event("Visual", f"used_svg2tikz={visual_result.stats.used_svg2tikz}")
        stats = VectorObjectStats()
        _log_vector_stats(stats)
        return TikzBuildResult(
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            processing_result=processing_result,
            tikz_code=visual_result.tikz_picture,
            vector_stats=stats,
            visual_stats=visual_result.stats,
        )

    if effective_mode in {"vector", "fidelity"}:
        log_event("Vectorization", "pipeline=contours_to_vector_objects")
        fit_result = fit_contours_to_vector_objects(
            contours,
            processing_result.original_bgr.shape,
            options,
            high_fidelity=effective_mode == "fidelity",
        )
        vector_objects = fit_result.objects
        stats = count_vector_objects(vector_objects)
        _log_bezier_fit_stats(fit_result.input_points, fit_result.simplified_points, stats, fit_result.geometric_reduction)
        _log_vector_stats(stats)
        tikz_code = generate_tikz_from_vector_objects(vector_objects, options=options, diagnostic_marker=False)
        return TikzBuildResult(
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            processing_result=processing_result,
            tikz_code=tikz_code,
            vector_objects=vector_objects,
            vector_stats=stats,
        )

    log_event("Vectorization", "pipeline=contours_to_tikz")
    stats = VectorObjectStats()
    _log_vector_stats(stats)
    tikz_code = generate_tikz_picture(contours, processing_result.original_bgr.shape, options)
    return TikzBuildResult(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        processing_result=processing_result,
        tikz_code=tikz_code,
        vector_stats=stats,
    )


def _log_vector_stats(stats: VectorObjectStats) -> None:
    log_event("Vectorization", f"vector_objects={stats.total}")
    log_event("Vectorization", f"lines={stats.lines}")
    log_event("Vectorization", f"polylines={stats.polylines}")
    log_event("Vectorization", f"bezier_curves={stats.bezier_curves}")
    log_event("Vectorization", f"circles={stats.circles}")
    log_event("Vectorization", f"ellipses={stats.ellipses}")
    log_event("Vectorization", f"rectangles={stats.rectangles}")


def _log_bezier_fit_stats(
    input_points: int,
    simplified_points: int,
    stats: VectorObjectStats,
    reduction: float,
) -> None:
    log_event("BezierFit", f"input_points={input_points}")
    log_event("BezierFit", f"simplified_points={simplified_points}")
    log_event("BezierFit", f"lines={stats.lines}")
    log_event("BezierFit", f"beziers={stats.bezier_curves}")
    log_event("BezierFit", f"polylines={stats.polylines}")
    log_event("BezierFit", f"geometric_commands={stats.total}")
    log_event("BezierFit", f"geometric_reduction={reduction:.1f}%")
