# Experimental SVG → TikZ comparison

Raw SVG bridge baseline; official pipelines unchanged.

| Case | Coordinates before → after | Reduction | TikZ bytes before → after | PDF IoU | PDF MAE | Geometric bound | Optimize seconds | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [redundant_line](redundant_line/comparison.png) | 180 → 2 | 98.9% | 3124 → 129 | 1.000000 | 0.0000 | 0.0513 | 0.095 | accepted |
| [circle](circle/comparison.png) | 240 → 240 | 0.0% | 4825 → 4825 | 1.000000 | 0.0000 | 0.0000 | 0.736 | baseline_retained |
| [ellipse](ellipse/comparison.png) | 240 → 240 | 0.0% | 4809 → 4809 | 1.000000 | 0.0000 | 0.0000 | 2.166 | baseline_retained |
| [nested_holes_small_detail](nested_holes_small_detail/comparison.png) | 720 → 3 | 99.6% | 14262 → 367 | 0.999511 | 0.0195 | 0.0564 | 2.913 | accepted |
| [smooth_curve](smooth_curve/comparison.png) | 401 → 401 | 0.0% | 7679 → 7679 | 1.000000 | 0.0000 | 0.0000 | 0.417 | baseline_retained |
| [junction_color](junction_color/comparison.png) | 185 → 7 | 96.2% | 2881 → 245 | 1.000000 | 0.0000 | 0.0513 | 0.263 | accepted |
| [visual_line_art_bw](visual_line_art_bw/comparison.png) | 130 → 127 | 2.3% | 2747 → 2687 | 1.000000 | 0.0000 | 0.0000 | 0.115 | accepted |
| [visual_geometric_diagram](visual_geometric_diagram/comparison.png) | 185 → 181 | 2.2% | 3509 → 3437 | 1.000000 | 0.0000 | 0.0000 | 0.119 | accepted |
| [visual_silhouette_bw](visual_silhouette_bw/comparison.png) | 70 → 69 | 1.4% | 1521 → 1501 | 1.000000 | 0.0000 | 0.0000 | 0.099 | accepted |
| [visual_simple_color](visual_simple_color/comparison.png) | 148 → 143 | 3.4% | 3102 → 3002 | 1.000000 | 0.0000 | 0.0000 | 0.117 | accepted |
| [visual_noisy_grayscale](visual_noisy_grayscale/comparison.png) | 5790 → 5237 | 9.6% | 119301 → 108354 | 1.000000 | 0.0000 | 0.0000 | 0.945 | accepted |
| [example_input_real](example_input_real/comparison.png) | 412 → 407 | 1.2% | 8290 → 8192 | 1.000000 | 0.0000 | 0.0000 | 0.264 | accepted |
| [original_real](original_real/comparison.png) | 2010 → 1926 | 4.2% | 38121 → 36582 | 1.000000 | 0.0000 | 0.0000 | 1.221 | accepted |
| [cara_existing_svg](cara_existing_svg/comparison.png) | 812 → 762 | 6.2% | 16705 → 15715 | 1.000000 | 0.0000 | 0.0000 | 0.302 | accepted |

All numerical errors and rejected candidate measurements: results.json.
Topology checks are finite-resolution (SVG at 4x, PDFs at 180 dpi), not a proof of isotopy.
