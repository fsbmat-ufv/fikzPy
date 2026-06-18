# Classic Line-Art Balanced Refinement (Issue 11.6)

## Purpose

Issue 11 connected the semantic pipeline to Classic mode. A first refinement
attempt (Issue 11.5, never merged into this branch) tried to stop the
`LINE_ART` strategy from producing artificial black mass by hardening the
filled-region rejection. That attempt overcorrected: it recovered cleaner
backgrounds but fragmented and lost real contour structure for complex
line-art drawings such as a dinosaur sketch, dropping legs, teeth, and
internal detail lines.

Issue 11.6 replaces that approach with a balanced policy: avoid artificial
fill **and** preserve contour continuity, instead of trading one failure mode
for the other.

## Why Issue 11.5 Was Not Enough

Issue 11.5's likely failure mode, as observed on the dinosaur fixture:

- centerline/skeleton output was treated as final; no fallback existed when
  it was fragmented or incomplete;
- components classified as ambiguous were dropped instead of being kept as a
  conservative outline;
- validation only penalized excess fill, never penalized lost contour
  structure, fragmented paths, or low edge/foreground recall;
- there was no distinction between "good line art" and "underdrawn line art"
  — both looked like "not overfilled" to the old validator.

This module treats the 11.5 regression as a concrete test case
(`dinosaur_lineart_bad_underdrawn` in the report below) instead of repeating
its approach.

## Overfilled vs. Underdrawn

Two distinct failure modes are now detected separately:

- **Overfilled line art**: artificial black mass, large filled regions, or
  white cutouts where the source is mostly thin strokes. Flags:
  `excessive_filled_area`, `artificial_black_mass`, `overfilled_lineart`,
  `lineart_converted_to_silhouette`, `excessive_white_cutouts`.
- **Underdrawn line art**: lost contour structure, fragmented paths, or low
  foreground/edge coverage. Flags: `underdrawn_lineart`,
  `lost_contour_structure`, `low_edge_recall`, `excessive_line_fragmentation`,
  `lost_external_contour`, `missing_internal_details`.

Both are configurable independently (`reject_overfilled_lineart`,
`reject_underdrawn_lineart`), so a caller can disable either guard without
disabling the other.

## New Module: `fikzpy.core.lineart_continuity`

This isolated module adds:

- `compute_lineart_continuity_metrics(mask, centerline_result, ...)` —
  compares a binary foreground mask against the centerline's rendered output
  and reports `contour_coverage`, `edge_recall`, `foreground_recall`,
  `skeleton_fragmentation`, `components_before`/`components_after`,
  `lost_component_count`, `endpoint_count`, `junction_count`, `path_count`,
  `broken_path_count`, `average_path_length`, `contour_bbox_coverage`, and
  `external_contour_preservation`.
- `decide_lineart_outline_recovery(metrics, ...)` — decides whether the
  centerline output is weak enough to need a conservative stroke-only
  fallback, and records which continuity flags triggered that decision.
- `extract_outline_strokes(mask, rendered_thin_mask, ...)` — for components
  whose centerline coverage is below a recall threshold, traces the external
  contour (`cv2.findContours` + `approxPolyDP`, the same conservative
  approach used by `filled_region_extraction.py`) and emits a closed
  `PolylinePrimitive` with a stroke only — **never a fill**. This guarantees
  outline recovery cannot introduce black mass or white cutouts by
  construction, regardless of configuration.
- `compute_lineart_fill_metrics(primitives, image_area)` — measures
  `filled_area_ratio` and `white_cutout_ratio` directly from primitive
  polygon areas (shoelace formula), without rendering a raster.
- `validate_lineart_balance(mask, primitives, continuity, ...)` — combines
  the fill metrics and continuity metrics into the overfilled/underdrawn
  flags above and a final `accepted` decision.

None of these functions import svg2tikz, call an external tracer process, or
touch Visual/GUI code.

## Contour Preservation Policy

When a `LINE_ART` component's centerline is fragmented or missing pixels, the
pipeline does **not** discard the component and does **not** fall back to a
filled shape. Instead:

1. `extract_thin_strokes()` still produces every centerline path it can.
2. `compute_lineart_continuity_metrics()` measures how much of the source
   mask that centerline output actually covers.
3. If coverage is weak, `extract_outline_strokes()` adds a stroke-only
   closed outline for the affected components only (components that already
   have good centerline recall are skipped).
4. Continuity metrics are recomputed against the *combined* primitive set
   (thin strokes + recovered outlines) before the overfilled/underdrawn
   validation runs, so recovery is credited correctly instead of being judged
   against the pre-recovery numbers.

Recovered outline primitives are tagged with
`metadata["source_layer"] = "outline_stroke"` so they remain distinguishable
from `thin_stroke` and `filled_region` primitives in diagnostics.

## Stroke Width

For `LINE_ART`, the default stroke width changed from `1.0pt` (effectively
invisible in the exporter because it matched the `default_line_width` and was
therefore omitted) to `line_art_stroke_width = 0.45pt`, inside the requested
`0.35pt`-`0.55pt` range. Outline recovery uses its own
`lineart_recovery_stroke_width` (default `0.45pt`) so a recovered outline does
not look heavier than the rest of the drawing. `BINARY_OUTLINE` and
`MIXED_MONOCHROME` filled-region/closed-shape stroke widths are unchanged.

## Configuration

New fields on `ClassicSemanticConfig` (`fikzpy/core/classic_pipeline_config.py`):

| Field | Default | Purpose |
| --- | --- | --- |
| `line_art_stroke_width` | `0.45` | Stroke width for `LINE_ART` thin strokes. |
| `lineart_min_edge_recall` | `0.35` | Minimum edge recall before flagging `low_edge_recall`. |
| `lineart_min_foreground_recall` | `0.55` | Minimum foreground recall before flagging `underdrawn_lineart`. |
| `lineart_min_contour_coverage` | `0.55` | Minimum contour coverage before flagging `lost_contour_structure`. |
| `lineart_max_fragmentation_ratio` | `0.6` | Maximum tolerated `skeleton_fragmentation` before flagging `excessive_line_fragmentation`. |
| `enable_lineart_outline_recovery` | `True` | Master switch for the outline-stroke fallback. |
| `lineart_outline_recovery_when_centerline_fails` | `True` | Whether continuity flags actually trigger recovery (vs. diagnostics only). |
| `lineart_recovery_stroke_width` | `0.45` | Stroke width used for recovered outlines. |
| `lineart_preserve_external_contour` | `True` | Whether to check/require external contour preservation. |
| `reject_underdrawn_lineart` | `True` | Whether underdrawn flags cause rejection. |
| `reject_overfilled_lineart` | `True` | Whether overfilled flags cause rejection. |
| `max_filled_area_ratio_for_lineart` | `0.06` | Maximum black-fill area ratio tolerated in `LINE_ART` output. |
| `max_white_cutout_ratio_for_lineart` | `0.03` | Maximum white-cutout area ratio tolerated in `LINE_ART` output. |
| `outline_recovery_max_components` | `24` | Cap on how many components outline recovery processes per run. |
| `outline_recovery_simplification_tolerance` | `0.01` | `approxPolyDP` epsilon as a fraction of contour perimeter. |

Defaults were chosen so that:

- the existing simple `LINE_ART` test fixtures remain accepted;
- `BINARY_OUTLINE`, `MIXED_MONOCHROME`, and Visual behavior are unaffected;
- the synthetic dinosaur fixture (see below) is accepted by default without
  needing outline recovery, while artificially raised thresholds can still
  exercise the recovery path deterministically in tests.

## Synthetic Dinosaur Fixture

`tests/test_classic_lineart_balanced_refinement.py` adds
`dinosaur_lineart_image()`: a closed external body contour, head, mouth with
teeth, legs, a belly line, internal body lines, an eye, all drawn as 1px
black strokes on white, with no filled regions. Running it through
`run_classic_semantic_pipeline()` yields `strategy_used == LINE_ART`,
`filled_region_primitives == 0`, `white_cutout_count == 0`, and `accepted ==
True` with the default configuration.

The "bad" overfilled and underdrawn dinosaur cases are exercised directly
against `validate_lineart_balance()` with constructed primitives and
continuity metrics, the same pattern already used by
`tests/test_visual_validation_score.py` for the mixed-monochrome regression
case. This keeps the regression deterministic without requiring the pipeline
to literally reproduce a previous bad configuration.

## Filled Regions Remain Preserved

`BINARY_OUTLINE` (filled rectangles, silhouettes) and `MIXED_MONOCHROME`
(drawings that mix filled black regions with thin strokes) are untouched by
this issue — `filled_region_extraction.py` and `mixed_monochrome_pipeline.py`
keep producing closed filled shapes for components with real fill evidence.
Only the `LINE_ART` path's stroke width and continuity policy changed; the
`extract_thin_strokes()` stroke-width parameter defaults to `1.0` for
backward compatibility, and only the `LINE_ART` branch passes
`line_art_stroke_width` explicitly.

## Report

`examples/classic_semantic_baseline/classic_lineart_balanced_refinement_report.json`
(generated by `scripts/generate_classic_lineart_balanced_refinement_report.py`)
records, for each of `dinosaur_lineart_synthetic_good`,
`dinosaur_lineart_bad_overfilled`, `dinosaur_lineart_bad_underdrawn`,
`line_art_simple`, `closed_contour_lineart`, `filled_rectangle_real`,
`silhouette_real`, `mixed_monochrome_synthetic`, and
`mixed_monochrome_bad_regression`: strategy, filled/thin/outline-recovery/
white-cutout counts, edge/foreground/contour-coverage metrics, fragmentation
ratio, dark-mass preservation, validation score, regression flags, acceptance,
rejection reasons, TikZ command/fill/character counts, and a deterministic
hash.

## Visual and Contornos Remain Separate

This issue does not modify `visual_pipeline.py`, `visual_postprocessor.py`,
the Visual svg2tikz flow, the Contornos contour-to-TikZ flow, or the GUI.
`tests/test_classic_lineart_balanced_refinement.py` re-checks that selecting
`visual` and `contours` modes still bypasses the Classic semantic pipeline
entirely, and that none of the new modules import svg2tikz, subprocess-based
tracers, or PySide6.

## Limitations

- Continuity metrics are computed on a rendered raster approximation of the
  centerline/outline output, not a topological diff of the skeleton graph;
  they are tuned for deterministic regression tests, not a perceptual
  guarantee.
- Outline recovery is per-connected-component and conservative; it does not
  reconstruct internal detail lines inside a recovered component, so heavily
  fragmented components still lose internal detail even after recovery
  (tracked by the `missing_internal_details` flag).
- The balance validator only runs for the `LINE_ART` strategy; `MIXED_MONOCHROME`
  and `BINARY_OUTLINE` keep using the Issue 10 validator unchanged.
- Default thresholds are calibrated against the synthetic fixtures in this
  repository, not a broad image corpus.

## Relationship to Issue 12

Issue 12 (benchmark and final documentation) is not implemented here. The
next manual step is testing this refinement against the real dinosaur image
used to report the original Issue 11 regression; only if that manual check is
satisfactory should Issue 12 begin.
