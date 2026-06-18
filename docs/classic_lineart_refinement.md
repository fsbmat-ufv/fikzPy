# Classic Line-Art Refinement

## Contexto

A Issue 11 integrou o pipeline semantico ao modo Classic. Depois disso, um caso manual de dinossauro em line art mostrou uma regressao visual importante: o desenho original tinha fundo branco e tracos pretos finos, mas a saida Classic podia virar uma grande massa preta preenchida com recortes brancos.

Esse padrao era semanticamente ruim mesmo quando a massa escura parecia razoavel no raster:

- um corpo inteiro virava `ClosedShapePrimitive` com `fill=black`;
- detalhes internos eram representados por formas brancas de recorte;
- o TikZ ficava pesado e menos editavel;
- a largura `1pt` deixava line art simples mais grossa que o original.

A Issue 11.5 refina somente o Classic semantico. Visual e Contornos continuam separados.

## Categorias

O Classic semantico agora separa tres casos monocromaticos com mais cautela:

- `LINE_ART`: desenho predominantemente vazio, explicado por skeleton/centerline e tracos finos.
- `BINARY_OUTLINE`: silhueta ou regiao realmente preenchida, com massa escura solida.
- `MIXED_MONOCHROME`: mistura de massas pretas reais e detalhes finos.

O objetivo nao e tratar tudo como `LINE_ART`. O caso de personagens com cabelo, roupa ou blocos pretos reais continua usando filled regions quando ha evidencia forte.

## Diagnosticos

O modulo `fikzpy.core.lineart_diagnostics` calcula evidencias globais e por componente:

- `foreground_ratio`;
- `skeleton_area_ratio`;
- espessura mediana e percentil 90;
- `fill_ratio` por componente;
- `compactness`;
- `skeleton_ratio`;
- area relativa do componente;
- contagem de componentes finos, preenchidos e ambiguos;
- confiancas para line art, mixed monochrome e binary outline.

Cada componente recebe uma decisao:

- `thin_stroke`;
- `filled_region`;
- `ambiguous`.

Tambem recebe um motivo, como `skeleton_explains_component`, `low_component_fill_ratio`, `component_too_thin` ou `solid_component_evidence`.

## Thin Stroke vs Filled Region

Uma regiao so vira filled region quando combina varias evidencias:

- area suficiente;
- `fill_ratio` alto dentro do bbox;
- `compactness` suficiente;
- `skeleton_ratio` baixo;
- espessura compativel com preenchimento.

Componentes grandes, ocos ou lineares passam a tender para `thin_stroke`, especialmente quando a confianca de line art e alta. Isso evita transformar contornos fechados em silhuetas pretas.

Quando a imagem parece mixed monochrome, a politica muda: massas pretas compactas continuam sendo preservadas como filled regions, e os detalhes finos sao processados via centerline.

## Filled Regions

`extract_filled_regions()` ficou mais conservador. Alem de emitir primitivas, ele preserva metadados de candidatos:

- `component_id`;
- area;
- bbox;
- `fill_ratio`;
- `compactness`;
- espessura estimada;
- `skeleton_ratio`;
- decisao;
- motivo.

Holes continuam suportados. A diferenca e que muitos holes em uma fonte line-art passam a ser um sinal de estrategia errada, nao uma solucao desejavel.

## Line Width

O Classic semantico agora usa larguras mais leves por estrategia:

- line art: `0.4pt`;
- mixed thin strokes: `0.45pt`;
- binary/filled outlines: configuraveis separadamente;
- filled regions reais usam `fill` e, por padrao, nao precisam de contorno desenhado por cima.

Esses valores vivem em `ClassicSemanticConfig` para evitar alterar o exportador semantico de forma ampla.

## Validacao De Overfill

O validador semantico agora mede uso de preenchimento em fontes line-art:

- `filled_area_ratio`;
- `white_cutout_count`;
- `white_cutout_area_ratio`;
- `white_cutout_to_black_fill_ratio`;
- crescimento artificial da massa escura renderizada.

Saidas ruins recebem flags como:

- `excessive_filled_area`;
- `artificial_black_mass`;
- `overfilled_lineart`;
- `lineart_converted_to_silhouette`;
- `excessive_white_cutouts`.

Uma saida line-art boa nao e reprovada apenas porque o metricador generico ve baixo recall de filled regions. Esse relaxamento so se aplica quando a fonte e line-art e nao ha overfill.

## Regressao Sintetica

Os testes adicionam um dinossauro sintetico com:

- fundo branco;
- corpo vazio com contorno fechado;
- olho, dentes, patas, garras e linhas internas;
- alguns tracos levemente mais grossos;
- nenhuma massa preta solida real.

O comportamento esperado e:

- estrategia `LINE_ART`;
- predominancia de `thin_stroke`;
- zero filled regions;
- zero white cutouts;
- `line width=0.4pt`;
- varias linhas/polylines editaveis;
- rejeicao da versao ruim `black fill + white cutouts`.

## Preservacao Do Mixed Monochrome

O refinamento mantem o avanco das Issues 10 e 11:

- `mixed_monochrome_synthetic` continua aceito;
- regioes pretas reais continuam preservadas;
- a regressao ruim com saida thin-only continua rejeitada;
- Visual e Contornos nao sao chamados pelo Classic semantico.

## Relatorio

`examples/classic_semantic_baseline/classic_lineart_refinement_report.json` registra os casos:

- `dinosaur_lineart_synthetic_good`;
- `dinosaur_lineart_bad_overfilled`;
- `line_art_simple`;
- `closed_contour_lineart`;
- `filled_rectangle_real`;
- `silhouette_real`;
- `mixed_monochrome_synthetic`;
- `mixed_monochrome_bad_regression`.

O relatorio inclui confiancas, contagens de stroke/fill, massa escura, white cutouts, score de validacao, flags, motivos de rejeicao, comandos TikZ e hash deterministico.

## Limitacoes

O refinamento ainda usa heuristicas leves sobre mascara binaria, skeleton e componentes conectados. Ele nao tenta resolver todos os casos de vetorizacao artistica, nem substitui um tracer visual dedicado.

Desenhos com contornos muito grossos e preenchimentos pequenos podem permanecer ambiguos. A politica atual favorece line art quando a evidencia de preenchimento real nao e forte.

## Relacao Com A Issue 12

Esta etapa nao implementa benchmark final amplo. A proxima etapa recomendada e:

Issue 12 - Benchmark e documentacao final.
