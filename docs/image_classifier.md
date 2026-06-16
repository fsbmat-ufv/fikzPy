# Image Classifier

## Purpose

The image classifier is a deterministic diagnostic component for the future
semantic Classic pipeline. It inspects an input image and returns a structured
category recommendation without running tracing, primitive fitting, TikZ
generation, PDF compilation, or GUI code.

The classifier is currently isolated and is not connected to Classic, Vector,
or Visual mode.

## Categories

- `LINE_ART`: sparse dark strokes on a mostly uniform background.
- `BINARY_OUTLINE`: black-and-white or grayscale filled regions and silhouettes.
- `COLOR_REGIONS`: images with meaningful saturated color regions.

## Metrics Used

The first implementation intentionally uses a small set of explainable metrics:

- effective quantized color count;
- quantized tonal level count;
- mean saturation;
- chromatic variation;
- colored-pixel ratio;
- dominant-background ratio;
- dark-pixel ratio;
- foreground ratio;
- simple edge density;
- edge-to-foreground ratio;
- estimated stroke thickness;
- contrast;
- uniform-image flag.

These metrics are calculated from an internal RGB-like array. For RGB and BGR
arrays the channel order does not affect the current metrics because they use
channel spread, saturation, and intensity averages rather than hue names.

## Heuristics

`LINE_ART` receives higher scores when the image has a dominant background,
low foreground occupancy, little color, and relatively many edge transitions
compared with foreground area.

`BINARY_OUTLINE` receives higher scores when foreground occupancy is larger,
color content is low, tonal levels are few, contrast is high, and the edge area
is small relative to the filled foreground.

`COLOR_REGIONS` receives higher scores when colored-pixel ratio, mean
saturation, effective color count, chromatic variation, and foreground
occupancy are high.

## Configuration

Thresholds live in `ImageClassifierConfig`, including:

- `color_quantization_levels`;
- `tonal_quantization_levels`;
- `saturation_threshold`;
- `colored_pixel_ratio_threshold`;
- `dominant_background_threshold`;
- `foreground_dark_threshold`;
- `background_distance_threshold`;
- `line_art_foreground_max_ratio`;
- `binary_outline_foreground_min_ratio`;
- `ambiguity_margin`;
- `minimum_confidence`.

Defaults are conservative and intended for reproducible routing diagnostics,
not final production scoring.

## Manual Override

Use `classify_image(image, override=ImageCategory.LINE_ART)` to force a
category programmatically. The classifier still computes metrics, marks the
result as manual, returns confidence `1.0`, and records a short override reason.

No GUI control is added by this issue.

## Ambiguity

The classifier always returns a primary category. If the two highest scores are
within `ambiguity_margin`, the result is marked as ambiguous, confidence is
reduced, and `alternative_category` records the runner-up.

Ambiguity is diagnostic only. It does not raise an exception.

## Uniform Images

Uniform bright images are treated as blank `LINE_ART` with low-detail metrics.
Uniform dark images are treated as filled `BINARY_OUTLINE`. Uniform mid-tone
images remain deterministic and use the same scoring machinery with the
uniform flag set.

## Programmatic Example

```python
from fikzpy.core.image_classifier import ImageCategory, classify_image

result = classify_image("examples/classic_semantic_baseline/line_art_bw.png")
print(result.category)
print(result.metrics.foreground_ratio)

manual = classify_image("input.png", override=ImageCategory.COLOR_REGIONS)
print(manual.manual_override)
```

## Limitations

- The classifier does not run the current Classic pipeline.
- It does not choose or execute a tracer.
- It does not implement adaptive preprocessing.
- It does not skeletonize, fit primitives, simplify geometry, or export TikZ.
- The heuristics are intentionally simple and will need calibration against
  broader examples.

## Future Roadmap Relationship

Issue 3 can use this classifier's metrics to choose conservative preprocessing
defaults. Issue 4 can use the `LINE_ART` category for future centerline routing.
Issue 5 can use `BINARY_OUTLINE` and `COLOR_REGIONS` to select optional tracing
adapters, while keeping fallback behavior explicit.
