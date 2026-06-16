# Adaptive Preprocessing

## Purpose

Adaptive preprocessing prepares raster images for future semantic Classic
pipelines. It is isolated infrastructure: it does not change Classic, Vector,
Visual, TikZ generation, PDF compilation, preview, or GUI behavior.

## Available Steps

The central API is:

```python
from fikzpy.core.adaptive_preprocessing import preprocess_image

result = preprocess_image(image, category=None)
```

The pipeline performs:

1. input normalization to an internal RGB array;
2. deterministic grayscale luminance conversion;
3. optional percentile-clipped autocontrast;
4. optional illumination correction;
5. optional conservative denoising;
6. multi-candidate threshold selection;
7. optional conservative morphology and component cleanup.

Threshold selection is also available directly:

```python
from fikzpy.core.threshold_selector import select_best_threshold

selection = select_best_threshold(grayscale_image)
```

## Configuration

`PreprocessingConfig` controls preprocessing:

- `enable_autocontrast`;
- `enable_illumination_correction`;
- `denoise_method`;
- `median_kernel_size`;
- `bilateral_diameter`;
- `bilateral_sigma_color`;
- `bilateral_sigma_space`;
- `gaussian_kernel_size`;
- `morphological_close_iterations`;
- `morphological_open_iterations`;
- `minimum_component_area`;
- `preserve_small_details`;
- `maximum_noise_removal_ratio`;
- `connectivity`;
- `foreground_is_dark`;
- `alpha_background`;
- nested `threshold_selection`.

`ThresholdSelectionConfig` controls threshold candidates, foreground polarity,
plausible foreground range, component metrics, ambiguity margin, and score
weights.

## Threshold Methods

The selector can evaluate:

- fixed global threshold;
- Otsu threshold;
- adaptive mean threshold;
- adaptive Gaussian threshold;
- global threshold sweep.

Adaptive candidates do not have a single scalar threshold value, so their
diagnostic `threshold` is `null`.

## Score Logic

Each threshold candidate receives scalar metrics:

- foreground ratio;
- connected component count;
- tiny component count and ratio;
- largest component ratio;
- edge coverage;
- continuity score;
- noise score;
- background consistency;
- foreground plausibility;
- fragmentation penalty;
- extreme foreground penalty.

The deterministic score is:

```text
continuity_weight * continuity_score
+ coverage_weight * edge_coverage
+ plausibility_weight * foreground_plausibility
+ background_weight * background_consistency
+ noise_weight * noise_score
- fragmentation_weight * fragmentation_penalty
- tiny_component_weight * tiny_component_ratio
- extreme_mask_weight * extreme_foreground_penalty
```

Candidates are ranked by score, method name, and threshold for stable
tie-breaking. If the two best scores are close, the result is marked ambiguous.

## Category Handling

`ImageCategory.LINE_ART` narrows the plausible foreground range, raises the
relative importance of edge coverage and continuity, and keeps cleanup minimal.

`ImageCategory.BINARY_OUTLINE` accepts larger foreground regions and weights
foreground plausibility more strongly.

`ImageCategory.COLOR_REGIONS` currently produces only a grayscale diagnostic
mask and records a warning. Color segmentation and tracer selection belong to
later issues.

## Uniform Images

Uniform bright images produce an empty foreground mask. Uniform dark images
produce a full foreground mask when `foreground_is_dark=True`. Uniform gray
images remain deterministic, with warnings and finite scores.

## Detail Preservation

When `preserve_small_details=True`, component cleanup does not remove small
components automatically. Stronger cleanup must be requested explicitly with
`preserve_small_details=False`, and removal is capped by
`maximum_noise_removal_ratio`.

## Diagnostics

`PreprocessingResult.to_dict()` returns array summaries, hashes, selected
method, threshold, ranking, scalar metrics, warnings, effective configuration,
and category. Full matrices are intentionally omitted.

## Limitations

- No centerline extraction is implemented.
- No graph construction is implemented.
- No external tracer is executed.
- No SVG parsing or TikZ export is performed.
- No semantic primitive fitting is performed.
- Color-region handling is luminance-only in this issue.

## Future Relationship To Issue 4

Issue 4 can consume the selected binary mask for line-art routing. This issue
only prepares and scores masks; it does not trace centerlines or build graph
topology.

## Dependencies

The implementation uses existing project dependencies:

- `numpy`;
- `opencv-python`.
