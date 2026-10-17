# Experimento isolado SVG → TikZ

Esta pasta é uma prova de conceito opt-in. Nenhum módulo da aplicação importa
este pacote. Os arquivos oficiais de Classic, Visual, Contornos, Ponto a Ponto,
GUI e compilação não foram modificados. Não há integração em produção.

## Executar

Na raiz do worktree experimental, usando as dependências já instaladas:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -B -m experiments.svg_tikz.run
```

O comando cria uma pasta com timestamp UTC em `experiments/svg_tikz/runs/`.
É possível indicar uma pasta NOVA, mas uma pasta existente causa erro antes
de qualquer escrita. Os exemplos históricos não são sobrescritos.

```powershell
python -B -m experiments.svg_tikz.run --output C:/temp/fikzpy-svg-experiment-new
python -B -m experiments.svg_tikz.run --input C:/temp/figure.svg --output C:/temp/fikzpy-one-svg-new
```

O SVG deve ter `viewBox` explícito e largura física em `cm`, como os SVGs
produzidos pelo aplicativo. A API de otimização isolada só exige `viewBox`:

```python
from experiments.svg_tikz.optimizer import OptimizationConfig, optimize_svg

result = optimize_svg(svg_text, OptimizationConfig(tolerance=0.25))
accepted_svg = result.svg
candidate_for_diagnostics = result.candidate_svg
print(result.changed, result.diagnostics)
```

O runner também verifica o PDF e a redução efetiva de código; a aceitação pela
API de SVG não equivale à aceitação final do benchmark. `--no-compile` deixa
explícito que o resultado não foi validado pelo renderizador LaTeX.

## Algoritmo e reaproveitamento

- Parser de comandos: `svg_semantic_parser._parse_path_data`.
- Douglas–Peucker: `geometry_error.simplify_polyline_rdp`.
- Ajuste de vários pontos por cúbica: `bezier_fit.fit_cubic_beziers`.
- Reconhecimento de círculo/elipse: `primitive_fitting.fit_primitive`.
- Ponte SVG → TikZ: `visual_pipeline._svg_to_tikz_picture`.
- Compilação: `latex_compiler.compile_latex_document` existente.

Os helpers privados são dependências explícitas deste protótipo. Antes de
uma integração seria necessário decidir uma API pública estável, sem copiar
seus algoritmos para outro pipeline.

O experimento altera somente o atributo `d` dos paths elegíveis. Mantém
elementos, estilos, ordem de pintura, subcaminhos e suas regras de
preenchimento. Não transforma anéis em objetos pintados independentes.
Retas consecutivas são divididas em cantos de pelo menos 35° e vértices
compartilhados. Extremos são fixos. Cúbicas preexistentes são preservadas.
Anéis completos reconhecidos podem virar arcos SVG dentro do MESMO path.
Um fechamento explícito redundante pode ser substituído pelo `Z` equivalente.

O parser semântico completo não é enviado ao exportador semântico, pois essa
combinação ainda separa subcaminhos preenchidos. Também não se usa o
pós-processador Visual de camadas brancas neste experimento. A comparação
usa a mesma ponte de conversão SVG antes/depois, mantendo paths compostos.

## Critérios de aceitação

- Teto geométrico padrão: 0,25 unidade do viewBox, não centímetro nem pixel do PDF.
- Erro bidirecional limitado com amostras densas e margem de Lipschitz;
  cúbicas são achatadas por De Casteljau com limite de desvio do casco de controle.
- Operações de fitting usam um orçamento local mais estrito que esse teto.
- QtSvg a 4×: IoU de primeiro plano ≥ 0,99 e MAE RGB ≤ 0,5 na escala 0–255.
- Componentes e furos iguais; comparação adicional da cobertura alpha.
- Com múltiplos paths, repetir as verificações por objeto pintado isolado.
- PDF a 180 dpi: mesmos limites de IoU/MAE e mesma topologia raster.
- Comparação à origem: queda de IoU ≤ 0,001 e aumento de MAE ≤ 0,1.
- Coordenadas emitidas E bytes TikZ devem diminuir.

Em rejeição, o runner escreve `after.*` com o baseline e registra o motivo.
O candidato rejeitado é preservado em `candidate.*`; nunca é apresentado
como uma otimização aprovada.

## Escopo e limitações

Não é um otimizador SVG geral. Documentos com transformações, CSS externo,
referências, clipping, máscaras, filtros, markers, nested SVG ou recursos
sem interpretação segura retornam intactos com diagnóstico. Comandos fora
de M/L/H/V/C/Z no SVG de entrada ficam intactos; arcos podem ser gerados na
saída após reconhecimento. Não há remoção de ruído, componentes ou detalhes.

A topologia é verificada em resoluções finitas. Contar componentes/furos,
preservar vértices compartilhados e impor erro geométrico não é uma prova
matemática de isotopia ou preservação de toda interseção subpixel. Isto é
um motivo para manter o experimento fora da produção.

O wrapper experimental recupera definições de cores customizadas que a
extração de `tikzpicture` da ponte atual omite. O enquadramento LaTeX é fixado
em um scope próprio, sem tocar no subsistema oficial de compilação.
Não há alinhamento por features para favorecer as métricas. O raster original
é reamostrado para o mesmo frame físico; essa medida absoluta também depende
de antialiasing e do tracing preexistente.

Nenhuma dependência foi adicionada. QtSvg vem de PySide6; NumPy, OpenCV,
Pillow e svg2tikz já fazem parte do ambiente do aplicativo. MiKTeX e Poppler
são ferramentas locais opcionais para a validação PDF. Falta de ferramentas
é registrada e não certifica o resultado renderizado.

## Artefatos e testes

A execução revisada está em [runs/final-v2/README.md](runs/final-v2/README.md).
O [relatório técnico](REPORT.md) interpreta também resultados negativos.
Cada caso contém SVGs, TeX, PDFs, PNGs, montagem comparativa, logs de
compilação e `metrics.json`. `results.json` agrega todos os casos;
`environment.json` registra versões, configuração, commit-base e hashes
dos fontes do experimento. Tempos são medições exploratórias de uma execução,
não distribuições estatísticas de desempenho.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:QT_QPA_PLATFORM = 'offscreen'
python -B -m pytest -p no:cacheprovider tests/test_experimental_svg_optimization.py
python -B -m pytest -p no:cacheprovider
```

Os testes verificam redução de redundância, preservação de extremos, cantos,
vértices compartilhados, estilos, regras de preenchimento e orientação;
rejeição de atalhos e alterações visuais; passagem intacta de recursos não
suportados; proteção de diretórios anteriores; cores e furo no PDF compilado.
