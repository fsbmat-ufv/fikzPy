# Retomar o fikzPy em outro computador

## Ponto de entrada

A branch que reúne o aplicativo atual e os experimentos das Fases 1, 2 e 3 é:

```text
codex/svg-tikz-phase3-reconstruction
```

A branch padrão do GitHub continua sendo `master`. Para recuperar este
trabalho, clone explicitamente a branch experimental:

```powershell
git clone --branch codex/svg-tikz-phase3-reconstruction https://github.com/fsbmat-ufv/fikzPy.git
cd fikzPy
git status
git branch --show-current
```

Se já houver um clone no outro computador, confira e preserve alterações locais
antes de trocar de branch:

```powershell
git fetch origin
git switch codex/svg-tikz-phase3-reconstruction
git pull --ff-only
```

As demais branches locais de desenvolvimento também foram publicadas, para
preservar os trabalhos anteriores e seus históricos sem fazer merge.

## Leitura obrigatória e estado do trabalho

Leia primeiro [AGENTS.md](AGENTS.md) e
[ROADMAP_CLASSIC_SEMANTIC.md](ROADMAP_CLASSIC_SEMANTIC.md).

Não alterar GUI, modos ou compilador oficial. Os experimentos ficam isolados
em `experiments/svg_tikz/`; não foram integrados ao aplicativo. A Fase 3 foi
encerrada para avaliação humana. Não iniciar automaticamente outra fase nem
fazer merge por causa desta transferência.

| Etapa | Branch | Commit experimental |
| --- | --- | --- |
| Base do aplicativo, com refinamento Ponto a Ponto | `claude/refine-dot-to-dot-output-ui` | `cc032f3` |
| Fase 1: otimização SVG/TikZ | `codex/experimental-svg-tikz-optimization` | `98b57fa` |
| Fase 2: renderizadores e tracing | `codex/svg-tikz-phase2-diagnostics` | `db45ba8` |
| Fase 3: reconstrução pré-SVG | `codex/svg-tikz-phase3-reconstruction` | `d99e54c` |

Este guia é um commit posterior ao experimento, sem alterações funcionais.

Relatórios e evidências:

- [Fase 1](experiments/svg_tikz/REPORT.md), com baselines em `runs/final-v2/`.
- [Fase 2](experiments/svg_tikz/PHASE2_REPORT.md), com evidências em
  `phase2_results/run-v2/` e `phase2_results/reference-v2/`.
- [Fase 3](experiments/svg_tikz/PHASE3_REPORT.md),
  [tabela e galeria](experiments/svg_tikz/phase3_results/README.md),
  [métricas](experiments/svg_tikz/phase3_results/run-v1/results.json) e
  [validação](experiments/svg_tikz/phase3_results/run-v1/validation.json).

Os SVGs, TEXs, PDFs, PNGs, logs e métricas das execuções qualificadas estão
versionados. Não sobrescrever esses diretórios. Os caminhos absolutos de
executáveis nos JSONs documentam a máquina de origem; não devem ser copiados
como configuração para a nova máquina.

## Conclusões que devem acompanhar a transferência

- Reduções grandes nos sintéticos não se reproduziram nos desenhos reais.
- Na Fase 3, o dinossauro A reduziu 1.811 → 1.808 segmentos e a cara B reduziu
  715 → 714. Os ganhos adicionais são pequenos; a meta de 30% não foi atingida.
- Grande parte da redução de coordenadas vem de retirar fechamentos explícitos
  antes de Z, sem reduzir arestas geométricas.
- Dinossauro B conservador foi rejeitado por alteração de furos no PDF a
  180 dpi, embora passasse em IoU global. Não aceitar somente por código menor.
- O frame físico foi calibrado na Fase 2. A construção de stroke no Qt e a
  sensibilidade à resolução ainda importam; não afrouxar limites nem
  reclassificar candidatos históricos automaticamente.
- Diminuir apenas o mínimo de pontos do fitter de seis para quatro já falhou.
- A Fase 3 mantém duas estratégias: fitting bruto e fitting com propagação de
  tangentes/reparametrização, reutilizando o código existente.
- A suíte completa passou: **700 testes**; os **20 PDFs** da Fase 3 compilaram.
  O relatório não recomenda integração ao Visual neste estado.

## Ambiente e testes

Consulte também as instruções de instalação no [README](README.md).
Crie um ambiente Python próprio no outro computador:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:QT_QPA_PLATFORM='offscreen'
python -B -m pytest -p no:cacheprovider
```

LaTeX/TikZ e Poppler são ferramentas externas. Para repetir as comparações
PDF da Fase 3, `pdflatex`, `pdftoppm` e `pdfinfo` precisam estar disponíveis.
A máquina original usou MiKTeX. Não é necessário recompilar para consultar os
PDFs e comparativos já versionados.

O ambiente de origem usou Python 3.13.7, NumPy 2.4.6, OpenCV 4.13.0.92,
PySide6/Qt 6.11.1, svg2tikz 3.3.2 e Pillow 12.2.0. Esses números documentam a
reprodução; o projeto continua com seus requisitos existentes. Variações de
renderizador podem alterar os scores, portanto registre as versões ao repetir.

Para uma nova execução, use um nome de saída que ainda não exista:

```powershell
python -B -m experiments.svg_tikz.phase3 --output experiments/svg_tikz/phase3_results/new-pc-review
```

Ambientes virtuais, caches, produtos temporários e probes preliminares ignorados
pelo Git não fazem parte da transferência. Os resultados qualificados e o código
que os reproduz estão no repositório. `.Rhistory` e configurações locais de
`.claude/` foram preservados no computador original, sem publicação.

## Contexto para o próximo chat

As conversas do Codex não são transportadas pelo Git. No outro computador,
abra o clone e forneça este contexto ao novo chat:

> Leia AGENTS.md, CONTINUE_ON_ANOTHER_PC.md e os relatórios experimentais das
> Fases 1–3. Estou retomando o fikzPy na branch
> codex/svg-tikz-phase3-reconstruction. Preserve GUI, pipelines oficiais e
> resultados históricos. A Fase 3 terminou sem integração, com 700 testes
> passando e ganhos geométricos pequenos. Confira o estado do Git e aguarde
> minha orientação sobre a próxima etapa antes de modificar o projeto.
