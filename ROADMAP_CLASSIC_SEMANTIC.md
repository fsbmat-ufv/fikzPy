# fikzPy — Roadmap técnico do modo Classic semântico

## Objetivo

Substituir a lógica atual do modo **Classic** por um pipeline semântico de vetorização que:

- preserve o layout atual da interface;
- mantenha o modo Visual intacto;
- gere TikZ limpo, curto e editável;
- reconheça retas, círculos, elipses, polilinhas e curvas de Bézier;
- use Bézier apenas quando uma primitiva mais simples não representar bem a geometria;
- trate imagens em preto e branco, tons de cinza e coloridas;
- mantenha fallback seguro para o comportamento anterior.

## Regra de execução

Este roadmap **não deve ser executado inteiro de uma vez**.

O Codex deve:

1. ler `AGENTS.md`;
2. ler este arquivo;
3. analisar o repositório;
4. executar uma issue por vez;
5. fazer testes;
6. criar um commit pequeno;
7. esperar revisão antes de seguir para a próxima issue.

---

# Visão geral do pipeline

```text
Upload da imagem
        ↓
Classificação da imagem
        ↓
┌─────────────────────────────────────────────┐
│ Line art     → centerline / skeleton        │
│ PB/silhueta  → Potrace/AutoTrace outline    │
│ Colorida     → VTracer                       │
└─────────────────────────────────────────────┘
        ↓
Representação geométrica interna
        ↓
Reconhecimento de primitivas
        ↓
Simplificação controlada
        ↓
Geração TikZ semântica
        ↓
Compilação e comparação visual
        ↓
Score de fidelidade e complexidade
```

---

# Critérios globais de sucesso

O projeto será considerado bem-sucedido quando:

- uma reta simples resultar em um único `\draw ... -- ...;`;
- um círculo simples resultar em `circle`;
- uma elipse simples resultar em `ellipse`;
- curvas livres forem representadas por poucas Béziers;
- imagens lineares não gerarem bordas duplas;
- o código Classic ficar substancialmente menor que o SVG convertido diretamente;
- o modo Visual continuar funcionando;
- o layout da GUI permanecer inalterado;
- o último resultado válido seja preservado em caso de erro;
- todos os testes antigos continuem passando.

---

# Issue 0 — Auditoria e baseline

## Título sugerido

`Audit current Classic pipeline and create reproducible baseline`

## Objetivo

Mapear o fluxo atual do modo Classic e criar uma referência reproduzível antes de qualquer alteração.

## Tarefas

- localizar o evento da GUI que chama o modo Classic;
- identificar pré-processamento, detecção de contornos, simplificação, geração de TikZ, compilação e preview;
- registrar os nomes reais dos arquivos, classes e funções;
- criar `examples/classic_semantic_baseline/`;
- adicionar pelo menos cinco imagens de teste:
  - desenho linear PB;
  - diagrama geométrico;
  - silhueta PB;
  - imagem colorida simples;
  - desenho com ruído e tons de cinza;
- registrar para cada exemplo:
  - tamanho da imagem;
  - quantidade de comandos `\draw`;
  - quantidade de `--`;
  - quantidade de `.. controls`;
  - tamanho do `.tex`;
  - tempo de processamento;
  - PDF resultante.

## Entregáveis

- `docs/classic_pipeline_audit.md`;
- arquivos de baseline;
- script ou teste reproduzível.

## Critérios de aceitação

- nenhuma alteração funcional;
- nenhuma alteração visual;
- baseline executável em um único comando;
- resultados antigos preservados.

## Commit sugerido

```bash
git commit -m "Add Classic pipeline audit and reproducible baseline"
```

---

# Issue 1 — Modelo geométrico interno

## Título sugerido

`Add semantic geometry intermediate representation`

## Objetivo

Separar análise da imagem e geração de TikZ.

## Tarefas

Criar dataclasses equivalentes a:

```python
PointPrimitive
LinePrimitive
PolylinePrimitive
CirclePrimitive
EllipsePrimitive
BezierPrimitive
ClosedShapePrimitive
PrimitiveGroup
```

Cada primitiva deve conter:

- geometria;
- estilo de traço;
- cor;
- espessura;
- preenchimento opcional;
- confiança do ajuste;
- erro estimado;
- metadados de origem.

## Regras

- nenhuma classe deve gerar texto TikZ diretamente;
- o exporter será responsável pela serialização;
- usar tipos simples;
- evitar abstrações excessivas.

## Critérios de aceitação

- o modo Classic antigo continua funcionando;
- o novo modelo existe de forma isolada;
- nenhuma mudança na GUI.

## Commit sugerido

```bash
git commit -m "Add semantic geometry intermediate representation"
```

---

# Issue 2 — Classificador de imagem

## Título sugerido

`Classify input images for semantic vectorization`

## Objetivo

Decidir qual pipeline deve ser usado.

## Categorias

```text
LINE_ART
BINARY_OUTLINE
COLOR_REGIONS
```

## Heurísticas sugeridas

### LINE_ART

- fundo predominantemente uniforme;
- baixa proporção de pixels escuros;
- traços finos;
- poucas cores;
- grande quantidade de componentes alongados.

### BINARY_OUTLINE

- duas classes tonais dominantes;
- regiões preenchidas;
- formas fechadas;
- maior proporção de pixels escuros.

### COLOR_REGIONS

- múltiplas cores;
- regiões preenchidas;
- gradientes ou variação tonal relevante.

## Regras

- permitir configuração manual;
- guardar confiança da classificação;
- fallback conservador;
- não alterar layout;
- usar controle existente ou configuração interna até autorização para GUI.

## Critérios de aceitação

- classificação reproduzível;
- possibilidade de sobrescrever a decisão;
- logs claros.

## Commit sugerido

```bash
git commit -m "Add input image classifier for Classic vectorization"
```

---

# Issue 3 — Pré-processamento adaptativo

## Título sugerido

`Add adaptive preprocessing and threshold selection`

## Objetivo

Melhorar a recuperação de traços fracos, escuros e claros.

## Tarefas

Implementar funções opcionais para:

- autocontraste;
- correção de iluminação;
- conversão para tons de cinza;
- filtro mediano leve;
- filtro bilateral leve;
- threshold de Otsu;
- threshold adaptativo;
- varredura de thresholds;
- fechamento morfológico conservador;
- remoção de ruído muito pequeno.

## Seleção automática de threshold

Testar vários thresholds e calcular score baseado em:

- continuidade;
- número de componentes;
- comprimento total dos caminhos;
- número de segmentos;
- cobertura dos pixels relevantes;
- penalidade por ruído.

## Regras

- preservar detalhes pequenos;
- não usar filtro agressivo por padrão;
- não destruir olhos, dentes, garras e linhas internas;
- manter os parâmetros centralizados.

## Critérios de aceitação

- melhor recuperação de linhas fracas;
- ausência de aumento excessivo de ruído;
- parâmetros reproduzíveis.

## Commit sugerido

```bash
git commit -m "Add adaptive preprocessing and threshold selection"
```

---

# Issue 4 — Pipeline Line Art por linha central

## Título sugerido

`Add centerline tracing pipeline for line drawings`

## Objetivo

Gerar um único caminho central para cada traço.

## Estratégia principal

```text
imagem binária
→ skeletonize
→ grafo
→ poda de espinhos
→ segmentos ordenados
→ primitivas
```

## Tarefas

- usar `skimage.morphology.skeletonize`;
- usar `sknw` se disponível;
- criar fallback próprio se `sknw` não estiver instalado;
- identificar endpoints, junctions e cycles;
- extrair sequências ordenadas de pontos;
- podar ramificações microscópicas;
- unir segmentos compatíveis;
- preservar ciclos fechados.

## Regras

- não converter cada pixel em comando TikZ;
- não gerar borda dupla;
- manter topologia;
- tratar laços e auto-conexões.

## Critérios de aceitação

- traço central único;
- número de caminhos muito menor que no Classic antigo;
- detalhes principais preservados.

## Commit sugerido

```bash
git commit -m "Add centerline tracing pipeline for line art"
```

---

# Issue 5 — Adaptadores Potrace, AutoTrace e VTracer

## Título sugerido

`Add optional raster tracing adapters`

## Objetivo

Usar motores especializados sem acoplar o projeto a uma única dependência.

## Adaptadores

```text
PotraceAdapter
AutoTraceAdapter
VTracerAdapter
```

## Comportamento

### Potrace

Usar para preto e branco, silhuetas, contornos fechados e logotipos.

### AutoTrace

Usar para centerline opcional, outline, redução de cores e fallback.

### VTracer

Usar para imagens coloridas, regiões preenchidas, tons variados e SVG compacto.

## Regras

- detecção automática de disponibilidade;
- erro claro quando ferramenta estiver ausente;
- fallback sem travar o programa;
- não alterar o modo Visual;
- registrar comando executado;
- usar pasta temporária exclusiva.

## Critérios de aceitação

- adaptadores independentes;
- nenhuma dependência obrigatória desnecessária;
- comportamento degradável.

## Commit sugerido

```bash
git commit -m "Add optional Potrace AutoTrace and VTracer adapters"
```

---

# Issue 6 — Parser SVG semântico

## Título sugerido

`Parse traced SVG into semantic primitives`

## Objetivo

Não usar o SVG como código TikZ final.

## Tarefas

- usar `svgelements` para parsing robusto;
- usar `svgpathtools` quando necessário;
- interpretar line, polyline, polygon, rect, circle, ellipse, path, groups, transforms, stroke, fill e opacity;
- converter tudo para o modelo geométrico interno;
- aplicar transformações antes da análise;
- preservar cores e preenchimentos.

## Regras

- não chamar `svg2tikz` como caminho principal;
- permitir `svg2tikz` apenas como fallback;
- normalizar coordenadas;
- detectar paths fechados.

## Critérios de aceitação

- SVG convertido para primitivas internas;
- estilos preservados;
- sem geração de TikZ nesta etapa.

## Commit sugerido

```bash
git commit -m "Parse traced SVG into semantic primitives"
```

---

# Issue 7 — Ajuste de primitivas geométricas

## Título sugerido

`Fit semantic geometric primitives to traced paths`

## Objetivo

Substituir caminhos complexos por formas simples quando possível.

## Ordem obrigatória de tentativa

```text
Line
Circle
Ellipse
Polyline
Bezier
Closed freeform shape
```

## Ajustes

### Reta

- `cv2.fitLine`;
- distância ortogonal;
- comprimento mínimo;
- penalidade por curvatura.

### Círculo

- ajuste por mínimos quadrados;
- `HoughCircles` apenas como apoio;
- erro radial;
- cobertura angular mínima.

### Elipse

- `cv2.fitEllipse`;
- erro normalizado;
- cobertura do contorno;
- orientação.

### Polilinha

- simplificação controlada;
- preservação de cantos relevantes.

### Bézier

- ajuste cúbico global;
- divisão recursiva no ponto de maior erro;
- continuidade de posição;
- continuidade tangencial quando possível.

## Regra de seleção

Escolher a primitiva mais simples cujo erro esteja abaixo da tolerância.

## Critérios de aceitação

- círculos e elipses não devem virar dezenas de Béziers;
- retas não devem virar polilinhas longas;
- erro medido e registrado.

## Commit sugerido

```bash
git commit -m "Fit semantic primitives to traced geometry"
```

---

# Issue 8 — Simplificação e otimização

## Título sugerido

`Simplify and merge semantic geometry`

## Objetivo

Reduzir pontos e objetos sem perder fidelidade.

## Tarefas

- remover pontos duplicados;
- remover segmentos microscópicos;
- juntar retas colineares;
- juntar caminhos com endpoints próximos;
- preservar junções reais;
- simplificar polilinhas com tolerância;
- preservar topologia de formas fechadas;
- combinar Béziers compatíveis;
- remover curvas degeneradas;
- normalizar casas decimais.

## Ferramentas possíveis

- Shapely;
- vpype;
- implementação própria pequena.

## Regras

- nunca simplificar só para reduzir tamanho;
- toda simplificação deve respeitar erro máximo;
- usar tolerância proporcional ao tamanho da imagem.

## Critérios de aceitação

- redução relevante de pontos;
- nenhuma ruptura visível importante;
- detalhes pequenos preservados.

## Commit sugerido

```bash
git commit -m "Simplify and merge semantic geometry"
```

---

# Issue 9 — Exportador TikZ semântico

## Título sugerido

`Generate compact human-readable TikZ from semantic primitives`

## Objetivo

Gerar código organizado e fácil de editar.

## Saídas esperadas

### Ponto

```latex
\draw (x,y) circle[radius=0.5pt];
```

### Reta

```latex
\draw (x_1,y_1) -- (x_2,y_2);
```

### Polilinha

```latex
\draw (x_1,y_1) -- (x_2,y_2) -- (x_3,y_3);
```

### Círculo

```latex
\draw (c_x,c_y) circle[radius=r];
```

### Elipse

```latex
\draw[rotate around={a:(c_x,c_y)}]
  (c_x,c_y) ellipse[x radius=r_x, y radius=r_y];
```

### Bézier

```latex
\draw (p_0)
  .. controls (c_1) and (c_2) .. (p_1);
```

### Forma fechada colorida

```latex
\draw[fill={rgb,255:red,R;green,G;blue,B}]
  ... -- cycle;
```

## Regras

- usar `\draw` de forma consistente;
- separar estilos comuns;
- evitar repetir opções idênticas;
- agrupar elementos por cor/estilo quando seguro;
- manter indentação;
- arredondar coordenadas;
- não escrever pontos desnecessários.

## Critérios de aceitação

- código compilável;
- menor que a conversão SVG direta;
- legível;
- semanticamente organizado.

## Commit sugerido

```bash
git commit -m "Generate compact semantic TikZ output"
```

---

# Issue 10 — Validação visual e score

## Título sugerido

`Add fidelity and complexity scoring`

## Objetivo

Medir equilíbrio entre fidelidade e minimalismo.

## Métricas de fidelidade

- diferença binária de pixels;
- IoU das regiões relevantes;
- distância média de borda;
- Chamfer distance opcional;
- preservação de componentes;
- cores médias por região.

## Métricas de complexidade

- número de primitivas;
- número de pontos;
- número de Béziers;
- tamanho do `.tex`;
- número de comandos `\draw`.

## Score combinado

```text
score =
  fidelity_weight × fidelity
  - complexity_weight × complexity
  - topology_penalty
```

## Regras

- fidelidade tem prioridade;
- complexidade é critério secundário;
- nunca escolher solução muito menor se perder elementos importantes;
- registrar score no log.

## Critérios de aceitação

- comparação reproduzível;
- logs objetivos;
- não alterar GUI nesta etapa.

## Commit sugerido

```bash
git commit -m "Add fidelity and complexity scoring"
```

---

# Issue 11 — Integração no modo Classic

## Título sugerido

`Integrate semantic vectorization into Classic mode`

## Objetivo

Conectar o novo pipeline à interface atual sem alterar o layout.

## Regras de integração

- preservar o Classic antigo como fallback interno;
- o novo pipeline deve ser padrão somente após passar nos testes;
- em falha:
  - registrar erro;
  - preservar preview anterior;
  - usar fallback explicitamente identificado;
- não alterar o modo Visual;
- não alterar menus ou botões;
- não alterar o sistema de compilação sem necessidade.

## Fluxo esperado

```text
Classic button
→ classify
→ preprocess
→ choose tracer
→ semantic primitives
→ simplify
→ TikZ exporter
→ compile
→ preview
```

## Critérios de aceitação

- fluxo completo funcional;
- layout idêntico;
- Visual intacto;
- fallback seguro.

## Commit sugerido

```bash
git commit -m "Integrate semantic pipeline into Classic mode"
```

---

# Issue 12 — Benchmark e documentação

## Título sugerido

`Benchmark semantic Classic mode and document usage`

## Objetivo

Comprovar ganho e documentar limites.

## Comparações obrigatórias

```text
Classic antigo
Classic semântico
Visual SVG
```

## Registrar

- fidelidade;
- número de comandos;
- número de pontos;
- tamanho do arquivo;
- tempo;
- facilidade de edição;
- observações visuais.

## Documentação

Criar:

```text
docs/classic_semantic_mode.md
docs/external_tracers.md
docs/benchmark_classic_semantic.md
```

## Critérios de aceitação

- documentação completa;
- instalação opcional clara;
- limitações honestas;
- exemplos antes/depois.

## Commit sugerido

```bash
git commit -m "Document and benchmark semantic Classic mode"
```

---

# Dependências sugeridas

## Núcleo

```text
numpy
opencv-python
Pillow
scikit-image
```

## Geometria

```text
shapely
svgelements
svgpathtools
```

## Opcionais

```text
sknw
vtracer
vpype
```

## Executáveis opcionais

```text
potrace
autotrace
```

Toda dependência opcional deve ter detecção automática, mensagem clara, fallback e documentação.

---

# Estratégia de branches

Branch principal sugerida:

```bash
git checkout -b feature/classic-semantic-vectorization
```

Para cada issue grande, pode ser criada uma sub-branch:

```text
feature/classic-audit
feature/classic-image-classifier
feature/classic-centerline
feature/classic-tracer-adapters
feature/classic-primitive-fitting
feature/classic-tikz-exporter
```

Não usar `git push --force`.

---

# Estratégia de revisão

Após cada issue:

1. executar testes;
2. gerar exemplo;
3. comparar PDF;
4. revisar `git diff`;
5. criar commit;
6. parar;
7. solicitar aprovação antes da próxima issue.

---

# Prompt de execução de uma issue

```text
Read AGENTS.md and ROADMAP_CLASSIC_SEMANTIC.md.

Work only on Issue X — <issue title>.

Do not start any later issue.

Before editing:
1. run git status;
2. inspect the existing implementation;
3. present a short plan;
4. preserve the current GUI layout;
5. preserve Visual mode;
6. preserve rollback.

Implement only the scope of Issue X.
Add or update tests.
Run the relevant tests.
Show the changed files.
Create one small commit with the suggested message.
Stop after the commit and report:
- what changed;
- tests executed;
- known limitations;
- exact next issue, without implementing it.
```

---

# Prompt inicial de auditoria

```text
Read AGENTS.md and ROADMAP_CLASSIC_SEMANTIC.md.

Do not implement the full roadmap.

Start only with Issue 0 — Audit current Classic pipeline and create reproducible baseline.

Before editing:
1. run git status;
2. identify the active branch;
3. preserve all uncommitted changes;
4. inspect the current Classic and Visual pipelines;
5. do not change the GUI;
6. do not change current output.

Implement Issue 0 only.
Run the tests.
Create the baseline artifacts.
Create one commit:
"Add Classic pipeline audit and reproducible baseline"

Stop after the commit.
```

---

# Gate para tornar o novo Classic padrão

O pipeline semântico só deve substituir o antigo quando:

- todos os testes passarem;
- a fidelidade for igual ou superior na maioria dos exemplos;
- o TikZ for menor ou mais legível;
- não houver perda sistemática de detalhes;
- Visual continuar intacto;
- rollback estiver validado.

Até esse momento, o novo pipeline deve permanecer experimental ou protegido por configuração interna.
