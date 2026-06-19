# Classic: estrategias explicitas

O modo Classic deixou de se apresentar como um vetorizador semantico
universal de alta fidelidade. Essa promessa pertence ao **Visual** (SVG +
`svg2tikz`), que continua inalterado e e o modo recomendado para imagens
fotograficas, complexas ou de regioes coloridas/mistas.

Classic prioriza **simplicidade, editabilidade e previsibilidade** do
codigo TikZ gerado, e agora expoe tres estrategias explicitas em vez de
depender apenas de heuristicas automaticas.

## Estrategias

### Classic Auto (`vectorization_strategy="auto"`, padrao)

Comportamento padrao, compativel com versoes anteriores. Usa as
heuristicas existentes (`ImageClassifier` + analise de componentes) para
escolher entre os caminhos internos (`LINE_ART`, `BINARY_OUTLINE`,
`COLOR_REGIONS`, `MIXED_MONOCHROME`):

- Em caso de classificacao `LINE_ART` sem evidencia forte de regioes
  preenchidas, prefere `LINE_ART`.
- So usa `MIXED_MONOCHROME`/`BINARY_OUTLINE` quando ha evidencia forte de
  regioes preenchidas.
- Se o candidato extrai dezenas de buracos pequenos demais para preservar
  (`skipped_small_hole`), trata isso como sinal de baixa confianca: o
  resultado e marcado como nao aceito (`auto_conservative_dropped_hole_detail`)
  em vez de ser apresentado como bom.
- Sempre que o resultado final nao for aceito, emite o aviso
  `classic_auto_recommend_visual` com a mensagem:
  > "Esta imagem parece complexa para o Classic. Para maior fidelidade, use
  > o modo Visual. Para codigo mais editavel, tente Classic Line Art ou
  > Classic Filled."
- **Nunca** faz fallback silencioso para Visual. Apenas recomenda.

### Classic Line Art (`vectorization_strategy="line_art"`)

Forca o caminho interno `LINE_ART` (centerline/traçado de linhas),
independente da classificacao automatica:

- Gera apenas `\draw` de traços (sem `fill`); `tikz_fill_commands` e
  sempre `0`.
- Pode usar centerline e recuperacao de contorno (`outline recovery`)
  quando a continuidade do traço esta fraca.
- Usa `line_art_stroke_width` / `lineart_recovery_stroke_width` como
  espessura configuravel.
- Nao tenta preservar massas pretas preenchidas; se a imagem for melhor
  representada por silhuetas solidas, o resultado pode ficar
  `accepted=False` (underdrawn), mas nunca produz uma silhueta preta
  grande para compensar.
- Ideal para: desenhos de linha, contornos, diagramas, figuras
  geometricas, material didatico.

### Classic Filled (`vectorization_strategy="filled"`)

Forca o caminho interno `BINARY_OUTLINE` (regioes preenchidas via
contornos), sem usar centerline como extracao principal:

- Gera `ClosedShapePrimitive` com `fill=`.
- Adequado para silhuetas, icones solidos, formas geometricas
  preenchidas.
- Se a imagem for classificada como `LINE_ART`, ou se houver muitos
  buracos/recortes brancos em relacao ao numero de regioes
  (`hole_count >= 3 * region_count`), emite o aviso
  `classic_filled_image_may_be_lineart` / `classic_filled_many_holes`
  recomendando Classic Line Art ou Visual.

## Como escolher (config / GUI)

Configuracao interna (`ClassicSemanticConfig.vectorization_strategy`,
aceita `"auto"`, `"line_art"`, `"filled"` ou o enum
`ClassicVectorizationStrategy`):

```python
from fikzpy.core.classic_pipeline_config import ClassicSemanticConfig
from fikzpy.core.classic_semantic_pipeline import run_classic_semantic_pipeline

config = ClassicSemanticConfig(vectorization_strategy="line_art")
result = run_classic_semantic_pipeline(image, config)
```

Via `TikzOptions.classic_strategy` (usado por `build_tikz_from_image`):

```python
options = TikzOptions(classic_strategy="filled")
```

Na GUI (`fikzpy/gui/main_window.py`), o painel de Parametros tem um
combo "Estrategia Classic" com as opcoes Auto / Line Art / Filled, ao
lado do seletor de Modo (Classic / Visual / Contornos). Visual e
Contornos nao foram alterados e o combo de estrategia e ignorado fora do
modo Classic.

## Visual e Contornos

Inalterados. Visual continua sendo o modo de alta fidelidade (SVG +
`svg2tikz`); Contornos mantem seu comportamento original.

## Limitacoes conhecidas

- Classic Auto ainda pode aceitar resultados imperfeitos quando nenhuma
  das heuristicas de baixa confianca dispara (este trabalho nao tenta
  resolver universalmente imagens mistas complexas, ex.: `tests/8.jpg`).
- A deteccao de "muitos buracos" para Classic Filled e um limiar simples
  (`hole_count >= 3 * region_count`), nao uma classificacao robusta.
- Nenhum fallback automatico para Visual existe ou deve ser adicionado;
  a decisao final e sempre do usuario.
