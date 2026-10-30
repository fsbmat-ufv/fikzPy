# Fase 2 — fragmentação geométrica e divergência entre renderizadores

## Resultado e escopo

A investigação demonstrou dois problemas distintos:

1. O Qt 6.11.1 deste ambiente produz diferenças relevantes no **contorno do
   traço de curvas**, mesmo quando a geometria central muda muito pouco. O
   limite de aproximação do stroker e a representação da curva explicam parte
   importante da rejeição de círculos e elipses.
2. A maioria da complexidade dos desenhos reais **já está no SVG recebido**.
   As reduções de 1% a 10% da fase anterior eliminavam principalmente uma
   coordenada de fechamento explícito redundante por anel. Não demonstravam
   redução do número de arestas geométricas necessárias para esses desenhos.

Foram avaliados 28 candidatos novos, todos compilados com pdflatex e
renderizados com Poppler. Há 17 aprovações e 11 rejeições, contando também
perfis que produziram resultados idênticos. Nenhum candidato ficou sem
verificação PDF nesta execução. Aprovações são experimentais sob os critérios
medidos, não autorização de integração nem prova de topologia contínua.

Dois candidatos de círculo/elipse e um de curva suave melhoraram os resultados
sintéticos com os MESMOS limites de fidelidade. Os novos testes em desenhos
reais não superaram os ganhos da fase 1. Não se recomenda integração ao app.

## Isolamento e reprodutibilidade

- Base: `98b57fa`, prova de conceito da fase 1.
- Branch desta fase: `codex/svg-tikz-phase2-diagnostics`.
- Worktree experimental reutilizado; checkout original e seus arquivos locais preservados.
- Nenhum arquivo de `fikzpy/`, GUI, modos ou compilador oficial alterado.
- Nenhuma dependência instalada/adicionada; nenhum merge/push.
- Baselines da fase 1 foram lidos e seus hashes comparados antes/depois.
- `run-v2` e `reference-v2` são pastas novas. Rodadas locais anteriores não
  foram sobrescritas e não são as evidências selecionadas para este relatório.

Código: `phase2.py`, `phase2_geometry.py` e `phase2_reference.py`, nesta pasta.
Algoritmos reutilizados: parser SVG, Douglas–Peucker, fitter cúbico,
reconhecimento de primitivas, serializer Visual, divisão de De Casteljau do
módulo Ponto a Ponto, ponte svg2tikz e compilador existentes.

Os probes do fitter mudam temporariamente um argumento ou adicionam contagem
em memória, com restauração automática e testes específicos. Nenhum pipeline
oficial recebe esse comportamento. A prova não altera o otimizador da fase 1.

Reprodução na raiz deste worktree, sempre indicando pastas NOVAS:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B -m experiments.svg_tikz.phase2 --output C:/temp/fikzpy-phase2-new
python -B -m experiments.svg_tikz.phase2_reference --output C:/temp/fikzpy-reference-new
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -B -m pytest -p no:cacheprovider
```

Resultados completos: [tabela principal](phase2_results/run-v2/README.md),
[JSON principal](phase2_results/run-v2/results.json),
[controles de referência e sweep](phase2_results/reference-v2/results.json).
Configuração, hashes e fontes da execução principal:
[environment.json](phase2_results/run-v2/environment.json).

## 1. Círculo e elipse: causa demonstrada no traço Qt

As formas de referência têm 240 arestas, largura de traço 2 e coordenadas
no mesmo viewBox de 200 × 140. O candidato circular tem raio 45; o elíptico,
semieixos 65 e 30. As diferenças geométricas não são exatamente zero: o
polígono original aproxima uma curva analítica, e a conversão de arcos para
cúbicas também é aproximada.

O desvio máximo do polígono para a elipse analítica tem limite de cerca de
0,00386 unidade para o círculo e 0,00557 para a elipse. Comparando a geometria
cúbica interpretada pelo parser com o original, os limites bidirecionais
conservadores medidos foram 0,01957 e 0,02339, incluindo margens de amostragem.
Isso é muito menor que a largura do traço e que o teto geométrico de 0,25.

Controles experimentais a 8×:

| Representação do círculo | Área coberta integrada, unidades² |
| --- | ---: |
| Polilinha original | 566,081 |
| Arcos SVG, QtSvg | 544,316 |
| Elemento SVG `<ellipse>`, QtSvg | 544,316 |
| `QPainterPathStroker`, threshold 0,25 | 544,316 |
| Mesmo stroker, threshold 0,1 | 560,079 |
| Mesmo stroker, threshold 0,01 | 565,446 |
| Mesmo stroker, threshold 0,001 | 565,658 |
| Anel preenchido analítico, sem construção de stroke | 565,776 |
| Área analítica esperada, `2π × 45 × 2` | 565,487 |

A área do traço em arcos fica aproximadamente 3,84% abaixo da polilinha e
3,74% abaixo do valor analítico. No caso elíptico, a cobertura da polilinha
é 617,948, contra 589,734 dos arcos, cerca de 4,57% menor.

As bounding boxes permaneceram iguais. Alterar cap entre butt/round e join
entre miter/round/bevel não resolveu a discrepância. Arcos SVG e o elemento
`ellipse` renderizaram identicamente no Qt, afastando, para estes casos, uma
falha específica de flags/orientação do comando `A`.

Um controle adicional manteve a MESMA geometria cúbica do candidato e apenas
a achatou com limite de 0,001 unidade. A mudança de aparência do traço continuou
grande, embora o erro central introduzido por esse achatamento fosse pequeno.
Variar as unidades numéricas antes de pintar, compensando a transformação,
também mudou a cobertura. São evidências de sensibilidade da construção do
stroke à aproximação/representação, não de um simples erro de enquadramento.

A documentação oficial explica o papel de `curveThreshold` e seu padrão 0,25:
[QPainterPathStroker](https://doc.qt.io/qt-6/qpainterpathstroker.html#setCurveThreshold).
O [fonte Qt 6.11.1](https://github.com/qt/qtbase/blob/v6.11.1/src/gui/painting/qstroker.cpp)
também registra esse padrão. O teste reproduzido confirma o efeito neste
build; não se presume que todos os renderizadores/versões tenham a mesma falha.

**Conclusão:** existe uma pequena alteração geométrica real, mas ela não
explica sozinha a perda de cobertura. A construção do traço curvo no Qt é
uma causa demonstrada e dominante nestes controles. O detalhe interno exato
do offset/tessellation exigiria instrumentar o Qt para uma explicação completa.

Exemplos: [círculo a 4× com ROI ampliada](phase2_results/run-v2/circle/qt_scale_4.png),
[stroker 0,25 versus 0,001](phase2_results/run-v2/circle/stroker_threshold.png),
[elipse a 16×](phase2_results/run-v2/ellipse/qt_scale_16.png).
Mapas assinados e heatmaps com ganho 8× ficam junto de cada montagem.

## 2. Curva suave, antialiasing e enquadramento

Na curva suave, o limite geométrico entre o candidato de 17 cúbicas da fase 1
e a referência é aproximadamente 0,02246 unidade. A IoU no Qt passa de
0,98378 a 4× para 0,99031 a 8× e 0,99214 a 16×. Diferentemente das formas
circulares, aqui a resolução/antialiasing explica uma parcela importante da
discrepância. Ainda há alteração geométrica real; ela não foi desconsiderada.

Na mesma grade de 3200 × 2240 pixels, Qt e PDF concordam melhor no baseline:
IoU de cerca de 0,9972. No candidato, a IoU cruzada é aproximadamente 0,9954.
Para círculo/elipse em arcos, a divergência Qt/PDF persiste em 16×, coerente
com o controle do stroker, em vez de desaparecer com a resolução.

Foi demonstrado também um problema de comparação de referências: um PDF
de 340,157 × 238,110 pontos ocupa 850,3925 × 595,275 pixels a 180 dpi, mas
seu PNG tem canvas inteiro 851 × 596. Esticar a referência SVG até todo esse
canvas desloca a geometria em frações de pixel. Reamostrar o PNG Qt de 4×
introduz outro efeito de interpolação.

Ao usar o MediaBox/DPI conhecido, sem alinhamento por features, a IoU entre
baseline Qt e PDF da curva sobe de aproximadamente 0,9137, com a referência
legada reamostrada, para 0,9915. Para o círculo, sobe de 0,9143 para 0,9879.
Logo, esses scores absolutos antigos não devem ser interpretados como medidas
puras de erro geométrico. Isso não invalida as comparações antes/depois dentro
do mesmo backend, mas pode influenciar diferenças marginais contra a origem.

Esses controles NÃO substituíram a referência de aceitação nesta fase.
Não houve relaxamento de IoU, MAE ou tolerâncias. Os candidatos continuam
classificados pelo protocolo da fase 1, e o resultado calibrado é diagnóstico.

Comparações: [Qt/PDF da curva no mesmo grid](phase2_results/reference-v2/smooth_curve/qt_vs_pdf_candidate.png),
[referência legada versus renderização direta](phase2_results/reference-v2/smooth_curve/legacy_reference_vs_direct.png),
[MediaBox físico controlado](phase2_results/reference-v2/smooth_curve/physical_frame_qt_vs_pdf.png).

## 3. Onde nasce a fragmentação dos desenhos reais

O SVG do Visual contém contornos de regiões preenchidas, não necessariamente
as linhas centrais ou objetos geométricos do desenho original. Um círculo
que cruza outros traços deixa de ser um anel circular independente. Tentar
reconhecer cada contorno completo como círculo não reconstrói esse objeto.

| Imagem | Contornos brutos / pontos | Pontos após RDP do tracing | Segmentos no SVG | Retas / cúbicas | Cantos ≥35° |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dinossauro | 89 / 25.484 | 2.090 | 1.811 | 1.756 / 55 | 708 |
| Ruidoso | 584 / 26.350 | 5.444 | 5.014 | 4.918 / 96 | 4.038 |
| Cara | 55 / 8.794 | 813 | 715 | 694 / 21 | 347 |
| Linear PB | 4 / 2.254 | 157 | 112 | 105 / 7 | 37 |
| Diagrama | 4 / 4.494 | 188 | 177 | 175 / 2 | 90 |
| Imagem exemplo | 5 / 6.195 | 438 | 391 | 383 / 8 | 316 |

Esses pontos já passaram por simplificação antes da otimização experimental.
Nesta configuração, o RDP do tracing usa 0,36 pixel e o fitting inicial usa
0,72 pixel, com upsampling 3×. O RDP já retirou cerca de 91,8% dos pontos
brutos do dinossauro. Repeti-lo depois com ε=0,02 ou 0,05, muito mais estrito,
não tem a mesma oportunidade de redução. Isso explica parte do resultado
negativo sem implicar que a representação recebida seja ótima.
Nos perfis de refit global, o dinossauro gerou 797 trechos delimitados por
cantos, sendo 503 com somente um segmento. Apenas 294 trechos tinham mais
de um segmento para testar. Nos candidatos RDP/fitting examinados, nenhum
produziu representação mais compacta dentro dessas restrições.

No ruidoso, foram 4.622 trechos, 4.299 unitários e 323 com oportunidade de
fitting. Na cara, 402 trechos, 290 unitários e 112 com oportunidade. Esses
números são limites impostos pela política de preservação de cantos do
experimento, não prova de que todo canto seja semanticamente importante.
Parte pode ser serrilhado de raster; removê-la requer evidência multiescala
e testes de detalhes, não elevar automaticamente o limite angular.

Instrumentação do fitter existente, sem mudar seu resultado:

| Imagem | Retas de fallback por poucos pontos | Retas de split instável | Cúbicas preservadas |
| --- | ---: | ---: | ---: |
| Dinossauro | 1.150 | 606 | 55 |
| Ruidoso | 4.136 | 782 | 96 |
| Cara | 440 | 254 | 21 |
| Linear PB | 100 | 5 | 7 |
| Diagrama | 120 | 55 | 2 |

O código Visual exige pelo menos seis pontos por trecho cúbico. A subdivisão
recursiva termina frequentemente em grupos menores e devolve polilinhas.
Isso demonstra um mecanismo de fragmentação no tracing/fitting original.
Também se verificou que o vetor `center_tangent` calculado no fitter legado
não é propagado às chamadas recursivas; portanto C0 pode existir sem uma
garantia de continuidade tangencial. O fitter semântico tem reparametrização,
mas também recompõe tangentes locais e pode rejeitar um conjunto por um
subtrecho reto. Essas observações são causas candidatas para trabalho futuro,
não correções já implementadas nos módulos oficiais.

Não há muitas cadeias de Béziers prontas para uma fusão simples: no dinossauro,
43 cúbicas estão isoladas e há seis pares consecutivos. Na cara, 19 isoladas
e um par. No ruidoso, 90 isoladas e três pares. Somente duas junções adjacentes
a cúbicas no dinossauro e uma na cara têm mudança tangencial abaixo de 8°;
no ruidoso, nenhuma. A principal oportunidade não é fundir centenas de cúbicas
suaves já existentes: elas não estão nessa representação.

O reconhecimento completo não aceitou círculos/elipses em nenhum anel desses
casos reais. No dinossauro, 61 candidatos circulares falharam por cobertura
angular; 25 tinham poucos pontos. Para elipse, 51 tinham poucos pontos e 34
falharam por cobertura. O baixo número de amostras por anel e a natureza dos
contornos preenchidos ajudam a explicar a falha de reconhecimento. Isso não
prova ausência de objetos circulares na imagem raster.
Essas estatísticas usam a configuração experimental de reconhecimento com
tolerância absoluta 0,075, herdada do experimento anterior. Não demonstram
que outras configurações ou pipelines do aplicativo nunca reconheçam formas.

## 4. Ruído: ablação controlada e detalhes pequenos

O gerador sintético ruidoso permitiu zerar SOMENTE a amostra de ruído usando
o mesmo gerador de desenho e o mesmo tracing. O controle não substitui a
origem nem é tratado como otimização aprovada.

| Medida | Com ruído | Controle sem ruído |
| --- | ---: | ---: |
| Componentes da máscara | 562 | 1 |
| Contornos brutos | 584 | 5 |
| Pontos dos contornos brutos | 26.350 | 2.494 |
| Pontos após simplificação do tracing | 5.444 | 126 |
| Componentes de área ≤16 pixels² | 483 | 0 |

Esse experimento demonstra que o ruído introduz grande parte da complexidade.
No SVG ruidoso, 334 anéis têm área de até quatro unidades² e contêm 1.514
dos 5.014 segmentos, cerca de 30,2%. São microrregiões para inspeção, não uma
licença para eliminá-las: em imagens reais elas podem ser olhos, dentes ou
vazados. No dinossauro, apenas nove anéis/38 segmentos caem nessa faixa; na
cara, dez anéis/45 segmentos. O mecanismo dominante não é igual em toda imagem.

[Controle sem ruído](phase2_results/run-v2/visual_noisy_grayscale/noiseless_control.png),
[máscara sem ruído](phase2_results/run-v2/visual_noisy_grayscale/noiseless_tracing_mask.png),
[máscara ruidosa](phase2_results/run-v2/visual_noisy_grayscale/tracing_mask.png).

A origem `tests/Cara.png` foi localizada e reproduziu exatamente o atributo
`d` do SVG histórico. O arquivo contém marcações vermelhas que o Visual
deliberadamente tenta ignorar; sua IoU absoluta contra o raster não deve ser
interpretada isoladamente como perda acidental. `tests/25.jpg` e
`examples/comparison/original.jpg` têm o mesmo SHA-256.

## 5. Novos candidatos: resultados aceitos e rejeitados

Foram mantidos IoU ≥0,99, MAE RGB ≤0,5 na escala 0–255, igualdade de
componentes/furos e os limites relativos à origem de 0,001 em IoU e 0,1 em MAE.
Teto geométrico: 0,25 unidade. Extremos, vértices compartilhados, cantos e
subcaminhos preenchidos continuam preservados pelos adaptadores.

| Melhor candidato sintético | Coordenadas | Bytes TikZ | Primitivas geométricas | IoU Qt / PDF | Resultado |
| --- | ---: | ---: | ---: | ---: | --- |
| Círculo, 16 cúbicas | 240 → 49 | 4.825 → 1.159 | 240 → 16 | 0,99457 / 0,99420 | Aceito |
| Elipse, 16 cúbicas | 240 → 49 | 4.809 → 1.171 | 240 → 16 | 0,99279 / 0,99486 | Aceito |
| Curva suave, RDP ε=0,005 | 401 → 223 | 7.679 → 4.299 | 400 → 222 | 0,99814 / 0,99836 | Aceito |

Círculo/elipse reduziram coordenadas em 79,58% e bytes em aproximadamente
76%. A curva suave reduziu coordenadas em 44,39% e bytes em 44,02%. Seus
limites geométricos foram aproximadamente 0,01957, 0,02339 e 0,02150.

Nas formas circulares, as quatro cúbicas da aproximação de arcos foram
subdivididas exatamente com De Casteljau. Oito cúbicas ainda falharam no Qt;
16 passaram. A subdivisão altera a representação, não a geometria cúbica
intermediária. A aproximação inicial de arcos por cúbicas continua sendo uma
alteração geométrica pequena e medida. Isso é um controle útil, mas 16 cúbicas
são menos editáveis que um círculo nativo; não se recomenda usar a subdivisão
como solução canônica para contornar um validador impreciso.

Na curva suave, subdividir as 17 cúbicas em 136 passou nas métricas visuais,
mas produziu 409 coordenadas e 9.423 bytes: foi rejeitado por complexidade.
RDP com ε=0,02 e 0,01 passou em Qt/PDF, mas falhou no critério relativo à
origem reamostrada. O sweep mais estrito encontrou ε=0,005, que passou sem
alterar os limites. O fitting mais compacto não era automaticamente a melhor
representação admissível. Não houve busca indiscriminada por menos bytes.

Comparações aceitas: [círculo](phase2_results/run-v2/circle/subdivide_2/comparison.png),
[elipse](phase2_results/run-v2/ellipse/subdivide_2/comparison.png),
[curva suave](phase2_results/reference-v2/curve_sweep/epsilon_0.005/comparison.png).

### Reduzir o mínimo do fitter para quatro pontos: rejeitado

| Imagem | Coordenadas | Bytes | Limite geométrico medido | IoU PDF | Decisão |
| --- | ---: | ---: | ---: | ---: | --- |
| Dinossauro | 2.010 → 1.938 | 38.121 → 39.115 | 1,632 | 0,9602 | Rejeitado |
| Ruidoso | 5.790 → 5.601 | 119.301 → 121.094 | 1,118 | 0,9387 | Rejeitado |
| Cara | 812 → 794 | 16.705 → 17.067 | 1,162 | 0,9580 | Rejeitado |
| Linear PB | 130 → 124 | 2.747 → 2.783 | 1,287 | 0,9502 | Rejeitado |
| Diagrama | 185 → 177 | 3.509 → 3.642 | 7,853 | 0,7716 | Rejeitado |
| Imagem exemplo | 412 → 378 | 8.290 → 8.005 | 1,839 | 0,9852 | Rejeitado |

A hipótese tinha fundamento nas contagens de fallback, mas a alteração de
um parâmetro sozinha é insegura. Além da maior quantidade de controles,
apareceram distorções geométricas importantes. O teste de erro usado pelo
fitter em amostras não substitui uma verificação contínua/bidirecional de
todo o contorno. Esses resultados impedem recomendar simplesmente min=4.

Exemplo negativo: [diagrama com mínimo quatro](phase2_results/run-v2/visual_geometric_diagram/tracer_minimum4/comparison.png).

## 6. Quantos segmentos são necessários nos desenhos reais?

Não foi demonstrado um mínimo global. As contagens abaixo são a representação
recebida e a redundância lexical certificada neste experimento:

| Imagem | Segmentos geométricos, incluindo fechamento | Fechamentos explícitos redundantes | Cúbicas antes/depois aceito | Ganho adicional sobre fase 1 |
| --- | ---: | ---: | ---: | ---: |
| Dinossauro | 1.811 | 84 | 55 / 55 | 0 |
| Ruidoso | 5.014 | 553 | 96 / 96 | 0 |
| Cara | 715 | 50 | 21 / 21 | 0 |
| Linear PB | 112 | 3 | 7 / 7 | 0 |
| Diagrama | 177 | 4 | 2 / 2 | 0 |
| Imagem exemplo | 391 | 5 | 8 / 8 | 0 |

Retirar `L ponto_inicial` antes de `Z` elimina uma coordenada escrita;
`Z` passa a representar a mesma aresta. Portanto o número de arestas geométricas
continua igual. O refit misto por trechos, permitindo atravessar fronteiras
L/C suaves, não encontrou redução adicional nos perfis ε=0,02 e 0,05.
Não se deve declarar que os 1.811 segmentos do dinossauro sejam indispensáveis:
somente que não foram reduzidos com estes candidatos e restrições.

Todos os resultados aceitos desses desenhos tiveram PDF idêntico ao baseline
a 180 dpi, componentes/furos iguais e mesmas medidas à origem. As IoUs absolutas
à origem permaneceram aproximadamente 0,7452 no dinossauro, 0,2739 no ruidoso
e 0,7113 na cara, sujeitas às limitações de referência já descritas.

## 7. Recomendações priorizadas

1. **Calibrar o validador experimental antes de escolher representações.**
   Usar o MediaBox e escalas físicas conhecidas, manter métricas de cobertura
   e distância de borda, estudar um stroker de precisão controlada e conservar
   renderizadores independentes. Isso corrige a medição; não autoriza reduzir
   o limite de fidelidade nem aceitar silenciosamente os candidatos antigos.
2. **Instrumentar o fitter nos contornos originais.** Investigar splits
   instáveis, uso real da tangente central, continuidade tangencial e erro
   entre amostras. Reutilizar o fitter semântico/reparametrização já existente
   em um adaptador separado antes de propor mudanças no Visual.
3. **Distinguir cantos do desenho de serrilhado de raster.** Os milhares de
   cantos protegidos bloqueiam a otimização tardia, mas elevar o ângulo
   indiscriminadamente pode apagar dentes, garras e junções. Exigir evidência
   multiescala e preservação explícita de detalhes, com limites de erro.
4. **Tratar ruído na seleção de máscara/classificação, em tarefa própria.**
   A ablação mostrou que a otimização SVG sozinha não resolve centenas de
   componentes. Não remover componentes pequenos automaticamente. Reutilizar
   os módulos de pré-processamento/classificação/centerline existentes quando
   apropriado, preservando o Visual como referência independente.
5. **Depois ampliar reconhecimento de objetos completos.** Contornos de tinta
   unidos por cruzamentos não equivalem ao objeto círculo ou reta original.
   Centerline/grafo ou decomposição por regiões deve preceder a tentativa de
   inferir primitivas completas nesses casos.

Critérios para uma próxima tarefa: explicar/corrigir o mecanismo específico
de erro, provar melhoria em imagens reais com os baselines preservados,
validar detalhes e contatos além da contagem de componentes, passar nos testes
e manter uma decisão humana antes de integração. Não se recomenda criar um
novo vetorizar-do-zero nem duplicar os algoritmos já presentes.

## Validação final e limites

A suíte completa contém 690 testes, incluindo 13 novos testes desta fase.
Os novos testes cobrem subdivisão exata, composição de preenchimentos,
extremos/cantos/junções compartilhadas, restauração dos probes, limite
geométrico, hashes imutáveis, frame fracionário e controles do stroker.

Componentes e furos são verificados em resolução finita. Junções geométricas
compartilhadas são mantidas pelos adaptadores e testadas; não há prova geral
de preservação de todo contato subpixel, identidade de cada furo ou grafo de
juntas numa imagem arbitrária. Os candidatos reais aceitos são equivalências
de serialização, o que evita atribuir a esses testes uma garantia maior.

Tempos por candidato estão nos JSONs. São execuções únicas; não se afirma
aceleração de processamento. Todos os comparativos são artefatos experimentais.
A fase termina sem merge ou integração, aguardando avaliação humana.
