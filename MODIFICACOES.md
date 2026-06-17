# Modificações — Modo Classic repensado (traços completos a lápis)

Objetivo: melhorar drasticamente a **qualidade** e o **tamanho** do código do
modo **Classic**, repensando como o desenho é entendido. Em vez de polilinhas
longas e fragmentadas, o Classic agora produz poucos traços completos —
`\draw` com retas e curvas de Bézier, **sem preenchimento** — como alguém
desenhando a lápis por cima da imagem.

O modo **Visual** e a interface **não foram alterados**.

## O problema anterior

A esqueletização (centerline) de traços de 3 px com bordas serrilhadas deixa
"degraus" (staircases) de 8-conexão. Lidos como grafo, cada degrau vira uma
falsa junção de grau 3. Num desenho simples de teste isso gerava **291 falsas
junções** e **58 contornos** — a maioria com perímetro < 8 px (ruído) — e um
código enorme e feio (214 linhas).

## A solução (novo módulo `fikzpy/core/skeleton_strokes.py`)

`trace_complete_strokes(...)` lê o esqueleto como um grafo e o trata como um
desenhista leria os traços:

1. **Limpeza de degraus** (`clean_skeleton`): remove os cantos em escada da
   8-conexão usando o critério de *ponto simples* (preserva a topologia, via uma
   tabela de 256 vizinhanças pré-calculada). No teste, as 291 falsas junções
   caem para **2 junções reais**.
2. **Segmentação**: divide o esqueleto limpo em segmentos entre nós reais
   (extremidades e junções) e detecta laços fechados.
3. **Poda de esporas**: remove barbas curtas que pendem de junções
   (artefatos de esqueletização).
4. **Religação por continuidade**: em cada junção, une os dois segmentos que
   seguem mais retos através dela — assim um traço que cruza outro permanece
   inteiro.
5. **Suavização que preserva cantos** (Chaikin seletivo): cantos suaves são
   arredondados (curvas deixam de parecer facetadas/poligonais), mas cantos
   agudos — pontas de estrela, quinas de caixa — são mantidos. Isso elimina o
   "facetamento" das curvas observado no resultado da imagem do Piglet.
6. **Ajuste**: cada traço completo vira poucas curvas de Bézier (com tolerância
   de alta fidelidade, para curvas suaves), e formas fechadas viram
   `circle`/`ellipse`/`rectangle`.

### Resultado medido (mesmo desenho de teste)

| | Antes | Depois |
|---|---|---|
| Contornos | 58 | 4 |
| Linhas de TikZ | 214 | 15 |
| Caracteres | 5782 | 641 (~89% menor) |
| Elipse | fragmentada | 1 comando `ellipse` |
| Preenchimento | não | não |

Um círculo isolado vira **um** comando `circle`; um retângulo vira `rectangle`.
Imagens em branco e ruído são tratados sem erro.

## Integração (mudanças mínimas, sem afetar outros modos)

- `fikzpy/core/stroke_tracer.py`: novos campos opcionais em
  `StrokeTracingSettings` (`complete_strokes`, `spur_length`,
  `link_max_angle_deg`, `min_stroke_length`, padrão desligado) e um ramo em
  `trace_line_art_strokes` que usa o novo tracer quando ativado.
- `fikzpy/core/image_processor.py`: o ramo do Classic ativa `complete_strokes`,
  usa esqueleto `skimage` e calcula a tolerância de espora pelo tamanho da
  imagem. Outros modos seguem inalterados.
- `fikzpy/core/tikz_pipeline.py`: o Classic continua passando pelo pipeline
  compacto de objetos vetoriais (retas + Bézier, sem preenchimento).

## Testes

- `tests/test_skeleton_strokes.py` (novo, 7 testes): remoção de degraus,
  preservação de conectividade, recuperação de laço fechado (círculo),
  vantagem sobre o tracer antigo numa linha, religação de traços cruzados,
  esqueleto vazio e poda de esporas.
- `tests/test_tikz_pipeline.py`: testes do Classic atualizados (traços compactos
  `\draw` + Bézier, sem `fill`, menores que as polilinhas antigas).
- Suíte completa: todos os testes passam (exceto os que dependem do PySide6,
  ausente no ambiente de teste — sem relação com esta mudança).

## Histórico

A tentativa anterior (menu "Simplificar" sobre o código Visual) foi removida a
pedido; o foco passou a ser o Classic. O Visual permanece como estava.
