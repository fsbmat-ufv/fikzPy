# Instalador do fikzPy para Windows

Esta pasta contém tudo o que é necessário para gerar o instalador
`fikzPy-Setup-<versão>.exe`, que qualquer pessoa pode executar para
instalar o fikzPy com um duplo clique.

## O que o instalador faz

1. **Apresenta as informações de criação** do software (autor:
   Fernando de Souza Bastos — fernando.bastos@ufv.br — criado para
   auxiliar na criação de imagens TikZ e de atividades educacionais),
   tanto na tela de boas-vindas quanto em uma página de informações
   dedicada ([info_criacao.txt](info_criacao.txt), também instalada
   como `SOBRE.txt`).
2. **Verifica o Python**: antes de instalar, confere se há Python 3.10+
   na máquina. Se não houver, oferece duas opções:
   - baixar e instalar o Python 3.13 automaticamente (python.org); ou
   - abrir a página de download para instalação manual, informando que
     o Python deve ser instalado antes de continuar.
3. **Copia o programa** para `%LOCALAPPDATA%\Programs\fikzPy` (instalação
   por usuário, sem exigir privilégios de administrador).
4. **Instala todos os pacotes necessários** (numpy, opencv-python,
   PySide6, scikit-image, svg2tikz) em um ambiente virtual isolado
   (`.venv` dentro da pasta de instalação), via
   [instalar_dependencias.bat](instalar_dependencias.bat).
5. **Cria atalhos** no Menu Iniciar (e opcionalmente na Área de
   Trabalho) que abrem a interface gráfica sem janela de console.

## Arquivos

| Arquivo | Papel |
| --- | --- |
| `fikzpy.iss` | Script do Inno Setup que define o instalador |
| `info_criacao.txt` | Página de informações exibida durante a instalação |
| `instalar_dependencias.bat` | Localiza o Python, cria o `.venv` e roda o `pip install` |
| `fikzPy.bat` | Launcher alternativo instalado junto ao programa |
| `requirements-runtime.txt` | Dependências de execução (sem pytest/pyinstaller) |
| `Output/` | Onde o `.exe` compilado é gerado (não versionado) |

## Como gerar o instalador

1. Instale o [Inno Setup 6](https://jrsoftware.org/isinfo.php)
   (por exemplo: `winget install JRSoftware.InnoSetup`).
2. Compile o script:

   ```powershell
   & "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" installer\fikzpy.iss
   ```

   (o ISCC também pode estar em `C:\Program Files (x86)\Inno Setup 6`.)
3. O instalador é gerado em `installer\Output\fikzPy-Setup-<versão>.exe`.

Ao atualizar a versão do fikzPy, ajuste `#define MyAppVersion` no topo
de `fikzpy.iss` (mantenha em sincronia com o `pyproject.toml`).

## Observações

- O instalador exige conexão com a internet na primeira instalação
  (download dos pacotes Python).
- Para compilar/visualizar o TikZ dentro do programa é recomendado ter
  uma distribuição LaTeX (MiKTeX ou TeX Live) com `pdflatex` — opcional.
- A desinstalação (Painel de Controle → Aplicativos) remove o programa
  e o ambiente virtual criados pelo instalador.
