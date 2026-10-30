# Evidências da fase 2

- [Relatório técnico e recomendações](../PHASE2_REPORT.md).
- [Tabela dos 25 candidatos principais](run-v2/README.md).
- [Inventário, instrumentação e critérios por candidato](run-v2/results.json).
- [Controles de referência física e três candidatos RDP mais estritos](reference-v2/results.json).

Melhorias novas aprovadas sob os mesmos critérios da fase 1:

| Caso | Coordenadas antes → depois | Bytes antes → depois | IoU Qt / PDF |
| --- | ---: | ---: | ---: |
| Círculo | 240 → 49 | 4.825 → 1.159 | 0,99457 / 0,99420 |
| Elipse | 240 → 49 | 4.809 → 1.171 | 0,99279 / 0,99486 |
| Curva suave | 401 → 223 | 7.679 → 4.299 | 0,99814 / 0,99836 |

Os desenhos reais não tiveram ganho geométrico adicional sobre a fase 1.
Todos os seis candidatos que mudaram apenas o mínimo do fitter para quatro
pontos foram rejeitados. Nenhuma regra de fidelidade foi relaxada.

## Exemplos ampliados e mapas

- [QtSvg: círculo em polilinha versus arcos](run-v2/circle/qt_scale_4.png).
- [Mesmo círculo cúbico com precisão diferente do stroker Qt](run-v2/circle/stroker_threshold.png).
- [Círculo aprovado: comparação PDF](run-v2/circle/subdivide_2/comparison.png).
- [Elipse aprovada: comparação PDF](run-v2/ellipse/subdivide_2/comparison.png).
- [Curva suave aprovada: comparação PDF](reference-v2/curve_sweep/epsilon_0.005/comparison.png).
- [Candidato mínimo quatro no diagrama, rejeitado](run-v2/visual_geometric_diagram/tracer_minimum4/comparison.png).
- [Ruído: máscara original](run-v2/visual_noisy_grayscale/tracing_mask.png).
- [Controle sem ruído: imagem](run-v2/visual_noisy_grayscale/noiseless_control.png).
- [Controle sem ruído: máscara](run-v2/visual_noisy_grayscale/noiseless_tracing_mask.png).
- [Efeito de reamostragem da referência](reference-v2/smooth_curve/legacy_reference_vs_direct.png).
- [Comparação física pelo MediaBox](reference-v2/smooth_curve/physical_frame_qt_vs_pdf.png).

Os mapas de diferenças assinados usam vermelho para baseline mais escuro e
azul para candidato mais escuro. Há heatmaps com ganho 8× e ROIs idênticas
ampliadas 4×. Não se faz alinhamento por features.

As pastas são novas e recusam sobrescrita. PDFs baseline nesta pasta são
cópias imutáveis para facilitar revisão; os PDFs candidatos foram compilados
especificamente para esta fase. As aprovações não substituem nenhum resultado
do aplicativo oficial.
