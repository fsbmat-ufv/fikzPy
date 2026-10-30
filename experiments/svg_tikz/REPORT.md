# Relatório técnico — prova de conceito SVG → TikZ

## Conclusão

É viável reduzir fortemente entradas com redundância geométrica demonstrada,
mas este experimento NÃO comprovou redução de 30% nos desenhos reais com os
critérios de fidelidade adotados. Não se recomenda integrar o otimizador ao
funcionamento oficial nesta etapa.

No exemplo ruidoso que motivou a investigação, as coordenadas passaram de
5.790 para 5.237 (9,55%) e o TikZ de 119.301 para 108.354 bytes (9,18%). A
renderização PDF antes/depois foi idêntica a 180 dpi. O ganho veio de retirar
fechamentos redundantes; não foi demonstrada uma reconstrução geométrica
substancialmente melhor desse desenho.

Há resultados positivos fortes em casos sintéticos: reta redundante, junção
colorida e anéis preenchidos com um detalhe pequeno. Há também resultados
negativos registrados: círculos/elipses de contorno e uma curva suave geraram
candidatos compactos que foram rejeitados por divergência visual no QtSvg.
Esses desenhos mantiveram o baseline como saída aceita.

## Isolamento e estado inicial

- Checkout original: branch `claude/refine-dot-to-dot-output-ui`.
- Base: `cc032f3b92bd145d827f11e3e7ed4bb617678d2d`.
- Arquivos locais não rastreados `.Rhistory` e `.claude/` preservados.
- Worktree independente: `svg-tikz-experiment/fikzPy` no diretório de worktrees do Codex.
- Branch experimental: `codex/experimental-svg-tikz-optimization`.
- Alterações limitadas a `experiments/svg_tikz/` e um arquivo novo de testes.
- Nenhuma alteração em `fikzpy/`, modos, GUI, settings ou compilador oficial.
- Nenhum merge, push ou integração em produção.

Foram lidos `AGENTS.md`, o roadmap semântico e documentos de parser, ajuste,
otimização, validação e integração Visual. A missão foi tratada como um
experimento isolado, sem iniciar a implementação de issues posteriores.

## Recursos reutilizados e decisões técnicas

O fluxo Visual atual traça uma máscara, extrai contornos, ajusta caminhos SVG
e chama svg2tikz. Já há Douglas–Peucker, fitting recursivo e reconhecimento de
formas no repositório; esses algoritmos não foram reimplementados.

O protótipo reaproveita o parser de comandos mantendo a identidade do path.
Isso evita a falha conhecida em que anéis `evenodd` viram preenchimentos
independentes no exportador semântico. A composição permanece no SVG e na
mesma operação de pintura TikZ, inclusive com cores e opacidade.

Somente sequências de retas podem receber RDP/fitting; cúbicas preexistentes
ficam intactas. Cantos e vértices compartilhados são fixos. O reconhecimento
opera sobre um anel inteiro e escreve arcos SVG no mesmo path. Não há fusão
de objetos, alteração da ordem de pintura nem remoção de componentes.

Para tornar a prova funcional sem alterar o aplicativo, o wrapper recupera
definições de cores customizadas do documento svg2tikz. A ponte oficial que
extrai somente `tikzpicture` não conserva essas definições. O wrapper também
converte `path` para `draw` com `draw=none` inicial, permitindo que uma opção
de stroke explícita do conversor continue prevalecendo.

O baseline é a conversão do SVG pela ponte existente, com esse mesmo wrapper
aplicado antes/depois. Ele não é byte a byte a saída Visual oficial, que inclui
o pós-processador de camadas. Por isso o tamanho de 119 KB difere dos cerca
de 122 KB observados na auditoria. O objetivo é isolar a mudança na geometria
SVG; os modos oficiais continuam inalterados.

## Corpus e medições

A execução final inclui 14 casos:

- seis SVGs sintéticos: reta, círculo, elipse, anéis e detalhe pequeno,
  curva suave e junção com cores;
- cinco imagens dos geradores de baseline existentes, processadas pelo tracing Visual;
- `examples/example_input.png` e `examples/comparison/original.jpg`;
- `examples/comparison/visual_cara_output.svg`, sem repetir rasterização inicial.

O ícone colorido dos geradores percorre o Visual monocromático existente.
Ele não comprova vetorização colorida do raster. O suporte de preservação de
cores é testado nos SVGs sintéticos coloridos.

As tabelas completas estão em [runs/final-v2/README.md](runs/final-v2/README.md),
e os números detalhados em [results.json](runs/final-v2/results.json).
As coordenadas são ocorrências de pares numéricos no TikZ, não amostras nem
pontos únicos. Um arco pode precisar de um par inicial e parâmetros angulares;
por isso a redução dessa contagem deve ser lida junto com bytes e operadores.

| Caso | Coordenadas antes → saída aceita | Redução | Bytes TikZ antes → saída aceita | IoU PDF final/baseline |
| --- | ---: | ---: | ---: | ---: |
| Reta redundante | 180 → 2 | 98,89% | 3.124 → 129 | 1,000000 |
| Círculo de contorno | 240 → 240 | 0% | 4.825 → 4.825 | 1,000000 |
| Elipse de contorno | 240 → 240 | 0% | 4.809 → 4.809 | 1,000000 |
| Anéis preenchidos + detalhe | 720 → 3 | 99,58% | 14.262 → 367 | 0,999511 |
| Curva suave | 401 → 401 | 0% | 7.679 → 7.679 | 1,000000 |
| Junção colorida | 185 → 7 | 96,22% | 2.881 → 245 | 1,000000 |
| Visual linear PB | 130 → 127 | 2,31% | 2.747 → 2.687 | 1,000000 |
| Visual diagrama | 185 → 181 | 2,16% | 3.509 → 3.437 | 1,000000 |
| Visual silhueta | 70 → 69 | 1,43% | 1.521 → 1.501 | 1,000000 |
| Visual ícone | 148 → 143 | 3,38% | 3.102 → 3.002 | 1,000000 |
| Visual com ruído | 5.790 → 5.237 | 9,55% | 119.301 → 108.354 | 1,000000 |
| Imagem exemplo | 412 → 407 | 1,21% | 8.290 → 8.192 | 1,000000 |
| Dinossauro real | 2.010 → 1.926 | 4,18% | 38.121 → 36.582 | 1,000000 |
| SVG cara existente | 812 → 762 | 6,16% | 16.705 → 15.715 | 1,000000 |

Os 14 baselines e 14 candidatos foram compilados com MiKTeX/pdflatex e
renderizados com Poppler. O bundle contém também PDFs aceitos, sem recompilar
um arquivo já aprovado: quando um candidato falha, `after.*` é uma cópia do
baseline. Logs de compilação ficam junto dos exemplos.

No desenho ruidoso, operadores de reta passaram de 4.918 para 4.365 e
Béziers permaneceram em 96. No dinossauro, as retas passaram de 1.756 para
1.672 e as Béziers permaneceram em 55. No exemplo de anéis, 717 operadores
de reta foram substituídos por seis arcos TikZ; os três subcaminhos continuaram
na mesma operação de preenchimento, com dois componentes e um furo no raster.

Tempos de tracing, otimização, conversão, compilação/renderização e total por
caso estão no JSON. São execuções únicas, sujeitas a aquecimento de imports e
carga da máquina. O objetivo desta rodada é viabilidade, não alegar aceleração.

## Rejeições e fidelidade à origem

| Candidato rejeitado | Coordenadas propostas | IoU QtSvg | IoU PDF candidato/baseline | Motivo final |
| --- | ---: | ---: | ---: | --- |
| Círculo | 1 | 0,966594 | 0,994204 | QtSvg abaixo de 0,99 |
| Elipse | 1 | 0,957326 | 0,994776 | QtSvg abaixo de 0,99 |
| Curva suave | 52 | 0,983778 | 0,993037 | QtSvg abaixo de 0,99 |

A curva proposta usa 17 Béziers em lugar de 400 segmentos. Ela não é
declarada sucesso: passou na comparação PDF, mas falhou no renderizador
independente SVG. A discrepância entre renderizadores é comprovada pelos
números; sua causa exata (aproximação de arcos, joins, antialiasing ou outros
detalhes de rasterização) continua hipótese. Não se afrouxou o limite para
aceitá-la. `candidate.svg`, `candidate.tex`, `candidate.pdf` e `candidate.png`
permitem revisão humana desses resultados.

As saídas aceitas não pioraram as medidas de origem além das tolerâncias
declaradas. A maioria teve PDF idêntico; os anéis tiveram MAE RGB final/baseline
de 0,0195 na escala 0–255 e limite geométrico conservador de aproximadamente
0,0564 unidade do viewBox, abaixo do teto 0,25.

Preservar o baseline NÃO significa que ele seja fiel ao raster de entrada.
No exemplo ruidoso, a IoU da origem com o PDF é cerca de 0,2739 antes e
depois; o tracing preexistente conserva 562 componentes e 22 furos no PDF.
A otimização SVG não elimina essa tinta pois isso alteraria a topologia de
sua entrada. O dinossauro tem IoU origem/PDF de aproximadamente 0,7452,
também inalterada. São medidas dependentes do limiar e antialiasing, mas
evidenciam que compactar SVG não corrige perdas/criações do tracing inicial.

Nenhuma saída final piorou em relação ao baseline segundo os critérios
medidos. Três candidatos pioraram a concordância visual e foram retidos
somente como diagnóstico; o resultado oficial do experimento nesses casos
é o baseline, com redução zero.

## Validação e limites das conclusões

O teste geométrico usa distância em ambas as direções, amostragem com margem
de Lipschitz e achatamento adaptativo de cúbicas. Não se confunde erro nos
vértices com erro ao longo de toda a curva. O teste final é de PDF renderizado,
em frame físico comum, sem registro ou deformação para melhorar os scores.

As verificações topológicas contam componentes/furos em 4× no SVG e 180 dpi
no PDF, além de comparar cobertura alpha e cada path em documentos com mais
de um. Não são uma prova de topologia contínua: contatos subpixel, identidade
de furos e cruzamentos não amostrados exigem validação adicional. SVGs com
transformações e recursos não suportados são devolvidos intactos.

Durante a preparação apareceram dois problemas do wrapper experimental:
definições de cores fora de `tikzpicture` e enquadramento PDF degenerado por
clipping ativo. Ambos foram corrigidos apenas no experimento. Os resultados
finais usam frames não degenerados e incluem teste automatizado de compilação
de região colorida com furo. Rodadas preliminares não qualificantes ficaram
em pastas locais separadas e não substituem esta execução revisada.

A suíte completa passou: **677 testes**, incluindo 19 casos novos do
experimento e os testes anteriores de todos os modos. Nenhuma nova
dependência foi instalada ou acrescentada aos manifests.

## Recomendação para avaliação humana

Manter a branch experimental sem merge. Os resultados justificam continuar
investigando a eliminação exata de redundâncias e o reconhecimento de regiões
analíticas, mas não sustentam habilitar otimização geral no Visual.

Antes de qualquer integração, uma próxima tarefa separada deveria explicar
as divergências de rasterização dos candidatos simples e ampliar a validação
de contatos/junções e hierarquia de regiões. A meta de ≥30% precisa ser
reavaliada num corpus maior de desenhos reais; nesta rodada ela foi alcançada
somente em três exemplos sintéticos.

Os arquivos oficiais e referências históricas permanecem preservados. A
prova de conceito termina aqui, aguardando avaliação humana.
