# Phase 2 reproducible diagnostics

Original phase-one criteria retained; no official pipeline changes.

| Case / candidate | Status | Coordinates | Reduction | TikZ bytes | Qt IoU | PDF IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [circle/subdivide_1](circle/subdivide_1/comparison.png) | rejected | 240 → 25 | 89.58% | 4825 → 623 | 0.989524 | 0.994032 |
| [circle/subdivide_2](circle/subdivide_2/comparison.png) | accepted | 240 → 49 | 79.58% | 4825 → 1159 | 0.994573 | 0.994204 |
| [ellipse/subdivide_1](ellipse/subdivide_1/comparison.png) | rejected | 240 → 25 | 89.58% | 4809 → 615 | 0.986110 | 0.994619 |
| [ellipse/subdivide_2](ellipse/subdivide_2/comparison.png) | accepted | 240 → 49 | 79.58% | 4809 → 1171 | 0.992786 | 0.994855 |
| [smooth_curve/subdivide_3](smooth_curve/subdivide_3/comparison.png) | rejected | 401 → 409 | -2.00% | 7679 → 9423 | 0.990623 | 0.993471 |
| [smooth_curve/tight_rdp_0.02](smooth_curve/tight_rdp_0.02/comparison.png) | rejected | 401 → 109 | 72.82% | 7679 → 2148 | 0.993573 | 0.992822 |
| [smooth_curve/tight_rdp_0.01](smooth_curve/tight_rdp_0.01/comparison.png) | rejected | 401 → 158 | 60.60% | 7679 → 3072 | 0.996534 | 0.996947 |
| [original_real/tracer_minimum4](original_real/tracer_minimum4/comparison.png) | rejected | 2010 → 1938 | 3.58% | 38121 → 39115 | 0.953058 | 0.960233 |
| [original_real/mixed_refit_0.02](original_real/mixed_refit_0.02/comparison.png) | accepted | 2010 → 1926 | 4.18% | 38121 → 36582 | 1.000000 | 1.000000 |
| [original_real/mixed_refit_0.05](original_real/mixed_refit_0.05/comparison.png) | accepted | 2010 → 1926 | 4.18% | 38121 → 36582 | 1.000000 | 1.000000 |
| [visual_noisy_grayscale/tracer_minimum4](visual_noisy_grayscale/tracer_minimum4/comparison.png) | rejected | 5790 → 5601 | 3.26% | 119301 → 121094 | 0.939763 | 0.938706 |
| [visual_noisy_grayscale/mixed_refit_0.02](visual_noisy_grayscale/mixed_refit_0.02/comparison.png) | accepted | 5790 → 5237 | 9.55% | 119301 → 108354 | 1.000000 | 1.000000 |
| [visual_noisy_grayscale/mixed_refit_0.05](visual_noisy_grayscale/mixed_refit_0.05/comparison.png) | accepted | 5790 → 5237 | 9.55% | 119301 → 108354 | 1.000000 | 1.000000 |
| [cara_existing_svg/tracer_minimum4](cara_existing_svg/tracer_minimum4/comparison.png) | rejected | 812 → 794 | 2.22% | 16705 → 17067 | 0.957673 | 0.958047 |
| [cara_existing_svg/mixed_refit_0.02](cara_existing_svg/mixed_refit_0.02/comparison.png) | accepted | 812 → 762 | 6.16% | 16705 → 15715 | 1.000000 | 1.000000 |
| [cara_existing_svg/mixed_refit_0.05](cara_existing_svg/mixed_refit_0.05/comparison.png) | accepted | 812 → 762 | 6.16% | 16705 → 15715 | 1.000000 | 1.000000 |
| [visual_line_art_bw/tracer_minimum4](visual_line_art_bw/tracer_minimum4/comparison.png) | rejected | 130 → 124 | 4.62% | 2747 → 2783 | 0.950055 | 0.950164 |
| [visual_line_art_bw/mixed_refit_0.02](visual_line_art_bw/mixed_refit_0.02/comparison.png) | accepted | 130 → 127 | 2.31% | 2747 → 2687 | 1.000000 | 1.000000 |
| [visual_line_art_bw/mixed_refit_0.05](visual_line_art_bw/mixed_refit_0.05/comparison.png) | accepted | 130 → 127 | 2.31% | 2747 → 2687 | 1.000000 | 1.000000 |
| [visual_geometric_diagram/tracer_minimum4](visual_geometric_diagram/tracer_minimum4/comparison.png) | rejected | 185 → 177 | 4.32% | 3509 → 3642 | 0.769453 | 0.771640 |
| [visual_geometric_diagram/mixed_refit_0.02](visual_geometric_diagram/mixed_refit_0.02/comparison.png) | accepted | 185 → 181 | 2.16% | 3509 → 3437 | 1.000000 | 1.000000 |
| [visual_geometric_diagram/mixed_refit_0.05](visual_geometric_diagram/mixed_refit_0.05/comparison.png) | accepted | 185 → 181 | 2.16% | 3509 → 3437 | 1.000000 | 1.000000 |
| [example_input_real/tracer_minimum4](example_input_real/tracer_minimum4/comparison.png) | rejected | 412 → 378 | 8.25% | 8290 → 8005 | 0.985067 | 0.985186 |
| [example_input_real/mixed_refit_0.02](example_input_real/mixed_refit_0.02/comparison.png) | accepted | 412 → 407 | 1.21% | 8290 → 8192 | 1.000000 | 1.000000 |
| [example_input_real/mixed_refit_0.05](example_input_real/mixed_refit_0.05/comparison.png) | accepted | 412 → 407 | 1.21% | 8290 → 8192 | 1.000000 | 1.000000 |
