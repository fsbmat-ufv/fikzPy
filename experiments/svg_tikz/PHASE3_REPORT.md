# Fase 3 — reconstrução antes do SVG

## Resultado para avaliação humana

O experimento comprovou reduções geométricas pequenas, sem demonstrar uma
reconstrução substancialmente mais compacta dos desenhos reais. Não se recomenda
integrar estas estratégias ao Visual. A meta exploratória de 30% de redução
de coordenadas não foi alcançada em nenhum dos quatro casos.

No dinossauro, A substituiu quatro retas por uma cúbica: 1.811 → 1.808
segmentos, redução de **0,166%**, com IoU PDF 0,999884. Na cara, B reuniu
duas retas em uma: 715 → 714 segmentos, redução de **0,140%**, com IoU PDF
0,999609. São alterações geométricas efetivas, mas seu benefício prático é
limitado. Não representam uma solução para o excesso de coordenadas.

O candidato B conservador do dinossauro foi **rejeitado**, apesar da IoU PDF
0,999692: duas regiões brancas se uniram na grade de 180 dpi. Todas as oito
propostas de ajuste bruto sem fallback também foram rejeitadas. Menos segmentos
não significou menos controles, código menor ou fidelidade suficiente.

[Tabela completa e galeria](phase3_results/README.md),
[resultados estruturados](phase3_results/run-v1/results.json) e
[ambiente, critérios e hashes](phase3_results/run-v1/environment.json).

## Isolamento e proveniência

Base: `db45ba8`, branch `codex/svg-tikz-phase2-diagnostics`. Trabalho realizado
na branch **`codex/svg-tikz-phase3-reconstruction`**, no worktree experimental
já existente. O checkout original permanece em
`claude/refine-dot-to-dot-output-ui`, com `.Rhistory` e `.claude/` preservados.

Foram lidos AGENTS, roadmap e os relatórios anteriores. Somente arquivos novos
de experimento, evidências e testes foram acrescentados. Não houve mudanças em
`fikzpy/`, GUI, modos, compilador, configurações ou manifests de dependências.
Não houve merge nem push. Nenhuma outra issue do roadmap foi iniciada.

Os hashes dos arquivos das Fases 1 e 2 foram conferidos antes/depois da rodada.
Os probes locais anteriores a `run-v1` permanecem separados e não qualificam
resultados finais. O runner recusa diretórios de saída existentes.

## Onde a fragmentação acontece no código atual

O fluxo demonstrado em
[visual_pipeline.py](../../fikzpy/core/visual_pipeline.py) é:

1. `trace_visual_contours`: resize 3× e `_build_high_resolution_mask`.
2. `_extract_visual_contours`: `findContours(RETR_LIST, CHAIN_APPROX_NONE)`.
3. Mesma função: filtros de área/perímetro e `approxPolyDP`, ε=0,36 pixel
   original nesta configuração; depois ordenação por área/perímetro.
4. `_contour_to_svg_path`: envio dos pontos JÁ reduzidos ao fitter legado,
   erro amostral 0,72, mínimo seis pontos, comprimento mínimo 0,9,
   straightness 0,35 e profundidade 18.
5. [bezier_fit.py](../../fikzpy/core/bezier_fit.py): tentativa de reta, ajuste
   LS com parâmetros por comprimento de corda, erro nas amostras e split
   recursivo. Grupos curtos ou splits próximos das extremidades retornam
   retas/polilinhas.
6. O serializador escreve M/L/C/Z; `contours_to_svg` reúne os anéis numa única
   pintura preta `evenodd`, sem stroke. A ponte existente usa svg2tikz.

**Não há uma etapa explícita de detecção semântica de cantos nesse fluxo
Visual.** A segmentação original decorre dos splits do fitter. A proteção
angular do otimizador tardio das fases anteriores é uma etapa experimental
distinta. Não se deve atribuir sua política ao tracing oficial.

O fitter legado calcula `center_tangent` mas não a passa aos filhos. Isso é
comprovado no código. O fitter semântico já possui Newton e LS reutilizáveis;
o adaptador usa esses kernels sem copiar outra biblioteca de Bézier.

O ponto de observação escolhido é a chamada `approxPolyDP`. Um patch de escopo
local captura o array bruto e devolve exatamente o resultado original. Assim,
filtragem, ordenação e RDP são executados pelo código oficial. Não se
reimplementou a vetorização. O patch é para este CLI isolado; não é adequado a
uma chamada concorrente dentro da GUI.

## Duas estratégias, mesmos dados iniciais

[reconstruction.py](reconstruction.py) contém os adaptadores.

**A — ajuste dos contornos brutos.** O fitter legado recebe amostras antes do
RDP. Cantos são âncoras presentes em duas aproximações espaciais, ε=0,5 e 1,0,
com giro ≥60°. O RDP serve aqui para localizar âncoras; os pontos usados no
ajuste continuam sendo os brutos. Repetições e coincidências entre contornos
também são âncoras. Microanéis de área ≤4 são conservados, sem presumir ruído.

**B — fitting com tangentes propagadas.** Mantém mínimo seis pontos e
comprimento 0,9. Usa as funções existentes de LS, avaliação e fallback;
propaga a tangente inicial/final e a tangente central, com sinal invertido
na extremidade esquerda do split. A direção é estimada em janela de cerca
de um pixel, evitando usar apenas um degrau do raster. Reutiliza até três
iterações Newton/LS do fitter semântico, conservando os parâmetros refinados
ao medir erro. Não é a alteração mínimo seis → quatro já rejeitada.

B garante C0 e G1 nos splits recursivos entre duas cúbicas aceitas antes do
arredondamento. Não promete G1 em fallbacks retos/poligonais, nas junções com
geometria original retida ou através de cantos. A serialização oficial a três
decimais também introduz pequeno erro, incluído na validação posterior.

Cada estratégia produz uma **proposta bruta** e uma **variante conservadora**.
São dois níveis de admissão das mesmas estratégias, não quatro algoritmos.
A variante conservadora localiza extremos do baseline nas amostras brutas,
preserva âncoras e só admite substituições com custo menor — reta=1 par,
cúbica=3 pares — e limite bidirecional ≤0,25 pixel original. Quando falha,
divide o trecho em fronteiras existentes; se necessário retém o baseline.
Não remove anéis, não muda a ordem de pintura nem separa furos em objetos.

O erro amostral de fitting é 0,72 como no Visual; straightness dos adaptadores
é 0,1. Esse parâmetro interno não substitui o teto final contínuo de 0,25.
Os probes com fitting 0,18 e 0,36 produziram quase somente conservação do
baseline. A configuração final usou o orçamento de fitting já existente,
mantendo todos os limites finais de fidelidade.

As mudanças de B incluem tangentes, janela espacial e reparametrização.
O experimento não isola a contribuição causal de cada uma. O teste de G1
comprova a propagação correta, não que ela sozinha explique a compactação.

## Dados de entrada e complexidade

| Caso | Pontos brutos | Após RDP oficial | Subcaminhos | Retas / cúbicas iniciais | Âncoras angulares multiescala |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dinossauro `tests/25.jpg` | 25.484 | 2.090 | 89 | 1.756 / 55 | 307 |
| Cara `tests/Cara.png` | 8.794 | 813 | 55 | 694 / 21 | 156 |
| Linear PB do baseline | 2.254 | 157 | 4 | 105 / 7 | 13 |
| Ruidoso do baseline | 26.350 | 5.444 | 584 | 4.918 / 96 | 1.758 |

As coordenadas dos paths regenerados coincidiram com o baseline histórico.
Máscaras e arrays brutos/RDP foram salvos em cada pasta, com hashes. Nenhum
filtro novo, remoção de componentes ou alteração de cor precedeu o ajuste.
As âncoras multiescala são uma definição operacional, não prova de que todos
os seus cantos sejam estruturalmente relevantes. Detalhes pequenos podem
existir fora dessa classificação; por isso há âncoras de contato, conservação
de microanéis e verificações de regiões.

## Resultados conservadores

Coordenadas são ocorrências de pares no TikZ; segmentos incluem a aresta de
fechamento implícita de Z. Uma cúbica custa três pares escritos. O número de
comandos de pintura permanece **um** em todos os casos.

| Caso / estratégia | Coordenadas antes → candidato | Bytes TikZ antes → candidato | Segmentos antes → candidato | IoU SVG / PDF | Limite geométrico | Decisão |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Dinossauro A | 2.010 → 1.926 | 38.121 → 36.597 | 1.811 → 1.808 | 0,999928 / 0,999884 | 0,19436 | Ganho geométrico aceito |
| Dinossauro B | 2.010 → 1.923 | 38.121 → 36.559 | 1.811 → 1.803 | 0,999522 / 0,999692 | 0,23286 | Rejeitado: furo PDF |
| Cara A | 812 → 762 | 16.705 → 15.715 | 715 → 715 | 1 / 1 | 0 | Somente serialização |
| Cara B | 812 → 761 | 16.705 → 15.695 | 715 → 714 | 0,999630 / 0,999609 | 0,22014 | Ganho geométrico aceito |
| Linear A/B | 130 → 127 | 2.747 → 2.687 | 112 → 112 | 1 / 1 | 0 | Somente serialização |
| Ruidoso A | 5.790 → 5.236 | 119.301 → 108.343 | 5.014 → 5.011 | 0,999931 / 0,999957 | 0,22028 | Ganho geométrico aceito |
| Ruidoso B | 5.790 → 5.237 | 119.301 → 108.354 | 5.014 → 5.014 | 1 / 1 | 0 | Somente serialização |

**Separação do ganho lexical:** as fases anteriores já chegavam a 1.926 pares
e 36.582 bytes no dinossauro, somente retirando fechamentos redundantes.
A elimina três arestas, mas usa os mesmos 1.926 pares e **15 bytes a mais**
que esse controle lexical. Não há ganho adicional de coordenadas nesse caso.
Na cara, B reduz o controle lexical 762 → 761 pares e 15.715 → 15.695 bytes:
um par e 20 bytes de ganho adicional. No ruidoso A, o ganho adicional é um
par, 11 bytes e três segmentos. O linear não melhora geometricamente.

As reduções totais de pares são 4,179% no dinossauro A, 6,281% na cara B e
9,568% no ruidoso A. Elas incluem quase integralmente a serialização já
conhecida. A redução do ruidoso em segmentos é apenas 0,060%: a complexidade
da máscara permanece, como esperado para o controle negativo.

## Propostas brutas rejeitadas e causas

| Caso / B sem fallback | Segmentos | Coordenadas | Bytes TikZ | IoU PDF | Limite ao baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dinossauro | 1.811 → 862 | 2.010 → 2.306 | 38.121 → 51.128 | 0,86967 | 1,71628 |
| Cara | 715 → 434 | 812 → 995 | 16.705 → 22.509 | 0,85724 | 0,86314 |
| Linear | 112 → 46 | 130 → 136 | 2.747 → 3.210 | 0,90364 | 0,85552 |
| Ruidoso | 5.014 → 4.282 | 5.790 → 6.677 | 119.301 → 146.143 | 0,90393 | 0,80270 |

B reduz 52,4% dos segmentos do dinossauro bruto, porém aumenta os pares em
14,7% e os bytes em 34,1%. A gera ainda mais cúbicas curtas: 1.231 contra
691 de B nesse caso. A substituição de muitas retas por controles explica
por que uma contagem menor de primitivas não fornece TikZ mais editável.

No dinossauro A, 1.445 tentativas de trechos produziram apenas 47 candidatos
de custo menor e uma substituição admitida. Em B foram 1.443 tentativas,
228 candidatos de custo menor e quatro substituições geometricamente
admitidas; o conjunto final falhou na topologia PDF. Na cara, B teve 75
candidatos de custo menor em 511 tentativas e uma admissão. São contagens
da busca adotada, não limites inferiores do problema geométrico.

O limite máximo bruto→baseline é 1,70123 pixel no dinossauro, 0,85289 na
cara e 0,71108 no linear. B bruto leva esses máximos a 0,73470, 0,73443
e 0,72597 respectivamente: melhora no dinossauro e na cara, sem melhora no
linear. Aproximar melhor o contorno extraído, isoladamente,
não garantiu maior semelhança visual ao raster nem ao SVG recebido. O RDP
e o fitting oficiais já deslocaram a referência; reconstruir os dados brutos
sob orçamento amostral de 0,72 move contornos finos e regiões estreitas.
Não se pode usar esse deslocamento para aumentar o teto final de 0,25.

**Demonstrado:** a combinação serrilhado/fallback/custo de controles e a
diferença entre o contorno bruto e a referência limita estas duas estratégias.
**Hipótese ainda não isolada:** uma parametrização/segmentação que minimize
diretamente o erro geométrico com menos controles pode superar o ajuste atual.
Não foi demonstrado um mínimo global nem que outras estratégias necessariamente
falhariam.

## Fidelidade, topologia e enquadramento

Mantidos IoU ≥0,99 e MAE RGB ≤0,5, na escala 0–255, em SVG Qt 4× e PDF
Poppler 180 dpi; componentes/furos devem coincidir. Também se exige identidade
bijetiva por sobreposição das regiões, incluindo furos pequenos, orientação e
fechamento dos anéis. O limite bidirecional de 0,25 inclui amostragem por
comprimento de arco, margens de Lipschitz, achatamento cúbico e conversão
float32 do kernel de distância OpenCV. Não é apenas erro nos vértices.

A referência raster é mapeada pelas bordas dos pixels ao frame fracionário
MediaBox×180/72, com interpolação linear e fundo branco. Não há registro por
features, deslocamento ajustado aos desenhos ou deformação para subir scores.
As comparações diretas Qt/PDF também usam esse frame, sem esticar o SVG ao
canvas inteiro arredondado.

| Candidato aceito | IoU raster→baseline / candidato | MAE raster→baseline / candidato |
| --- | --- | --- |
| Dinossauro A | 0,818994 / 0,818897 | 5,92517 / 5,92407 |
| Cara B | 0,682296 / 0,682537 | 17,46662 / 17,46696 |
| Ruidoso A | 0,269978 / 0,269972 | 59,22083 / 59,22166 |

Mantidos os limites relativos de queda IoU ≤0,001 e aumento MAE ≤0,1.
Os valores absolutos usam o protocolo calibrado desta fase e não são
comparáveis diretamente aos scores antigos com outra reamostragem. Nenhum
candidato anterior foi reclassificado. A cara contém anotações vermelhas
ignoradas pelo Visual; a máscara ruidosa tem centenas de componentes. Esses
fatos limitam a interpretação de fidelidade absoluta à origem.

Mesmo com frame calibrado, o baseline Qt/PDF tem IoU 0,96608 no dinossauro e
0,98593 na cara a 180 dpi. Logo, o frame não elimina todas as diferenças de
antialiasing/tessellation. A admissão compara antes/depois no mesmo backend;
a concordância cruzada é diagnóstico adicional, não um novo critério oculto.

No dinossauro B conservador, o PDF passa de 28 componentes/81 furos para
28/80. Duas regiões brancas de 416 e 24 pixels se unem na região dos dentes.
O Qt a 4× conserva 33 componentes/55 furos nos dois. No controle PDF de
2836×2192 pixels, ambos têm 33 componentes/56 furos. Isso demonstra
sensibilidade topológica à grade/limiar, juntamente com uma alteração
geométrica real medida. Não demonstra equivalência topológica contínua e
**não revoga a rejeição a 180 dpi**.

[Recorte dos dentes](phase3_results/run-v1/original_real/B_guarded/teeth_180_control.png)
e [parâmetros do controle](phase3_results/run-v1/original_real/B_guarded/topology_control.json).
O controle usa `phase2_reference.pdf_at_grid` para os dois PDFs na mesma grade;
o recorte tem origem (205,70), tamanho 96×96 e ampliação de exibição 4×.

Nos aceitos, as identidades de componentes/furos passaram nas duas grades,
microanéis ficaram intactos e os pontos de contato/cantos protegidos foram
retidos. A inspeção dos mapas e ROIs confirmou mudanças locais pequenas.
Isso não prova preservação de todos os contatos subpixel, do grafo semântico
de junções ou do significado de cada detalhe numa imagem arbitrária.

## Tempo, compilação e testes

| Caso | Máscara + extração observada | A: reconstrução | B: reconstrução |
| --- | ---: | ---: | ---: |
| Dinossauro | 0,262 s | 11,67 s | 42,16 s |
| Cara | 0,153 s | 3,47 s | 12,94 s |
| Linear | 0,245 s | 1,41 s | 4,11 s |
| Ruidoso | 0,212 s | 4,46 s | 6,24 s |

Tempos A/B incluem a proposta bruta e a variante conservadora, não o custo
de validação integral, compilação e montagem. A primeira coluna não inclui
o fitting/exportação oficial. Portanto a tabela **não é um benchmark de
aceleração antes/depois**. Conversão e compilação/renderização têm seus próprios
tempos nos JSONs; são execuções únicas, sem intervalos estatísticos.

Os quatro baselines e 16 candidatos compilaram com MiKTeX/pdflatex e foram
renderizados pelo Poppler. SVG, TEX, PDF, PNG, logs, matrizes de contornos,
mapas e referências foram preservados. Nenhum resultado ficou sem verificação
PDF nesta rodada. Compilar não implicou aceitação.

A suíte completa passou: **700 testes** (690 anteriores + 10 novos).
Os novos testes verificam captura sem mutação, âncoras multiescala,
propagação/sinal das tangentes, continuidade, overshoot entre amostras,
identidade de detalhes, frame físico, preenchimento composto, microfuros,
contato compartilhado e rejeição explícita de entradas inválidas. Os testes
analíticos anteriores de círculo, elipse, curva suave e anéis continuam
disponíveis e foram executados com os demais testes de regressão.

## Reprodução e próximos passos

Na raiz do worktree experimental, PowerShell, com os pacotes existentes,
pdflatex, pdftoppm e pdfinfo disponíveis:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:QT_QPA_PLATFORM='offscreen'
python -B -m experiments.svg_tikz.phase3 --output experiments/svg_tikz/phase3_results/review-new
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -B -m pytest -p no:cacheprovider
```

Use sempre um diretório novo. O comando recria os comparativos e métricas
primários; o controle adicional de grade/recorte está descrito acima. Não há
seleção automática para produção nem arquivos `after` apresentados como saída
oficial: cada candidato conserva seu status explícito.

Prioridades recomendadas, sujeitas a uma nova autorização:

1. Manter a validação física e a identidade de regiões como ferramentas de
   pesquisa. Investigar a estabilidade dos contatos finos por resolução,
   conservando a regra atual até estudo específico.
2. Se continuar o fitting, estudar um objetivo que considere erro geométrico
   contínuo e custo de controles desde a escolha do split. Usar estes casos
   como contraexemplos, em vez de promover a propagação de tangentes como
   solução suficiente. Um ensaio posterior deve separar tangentes, janela
   espacial e reparametrização para atribuir os efeitos.
3. Para ganhos grandes em desenhos lineares, avaliar em tarefa própria o
   centerline/grafo já existente: bordas de tinta preenchida não são os objetos
   semânticos originais. Para o ruidoso, a seleção de máscara continua sendo
   o problema dominante; não apagar componentes como parte deste otimizador.

Esta fase termina aqui, sem integração. As melhorias reais obtidas são
pequenas demais para justificar mudança do aplicativo; os resultados negativos
e os casos de topologia ficam disponíveis para a avaliação humana.
