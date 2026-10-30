# Instalador do fikzPy para Windows

## Instalador completo offline (versão atual)

O arquivo `fikzPy-Setup-0.1.1-Windows-x64.exe` instala o aplicativo, o runtime
Python, as bibliotecas de imagem/GUI/SVG e uma distribuição privada TinyTeX
(TeX Live 2026) com TikZ e `standalone`. O computador de destino não precisa
de Python, Git, LaTeX ou conexão à internet. Destina-se a Windows 10/11 x64.

Baixe o executável nos [Releases do projeto](https://github.com/fsbmat-ufv/fikzPy/releases),
execute-o e abra o atalho fikzPy. A instalação é por usuário em
`%LOCALAPPDATA%\Programs\fikzPy`, sem exigir administrador. O LaTeX é exposto
somente ao processo do aplicativo; o PATH global não é modificado. Os pacotes
LaTeX incluídos cobrem os documentos gerados pelo fikzPy; código personalizado
pode exigir pacotes adicionais. Os tracers externos opcionais Potrace,
AutoTrace e VTracer não fazem parte do pacote.

O código-fonte editável acompanha a instalação em `fikzPy-source.zip`. Licenças
de terceiros ficam em `licenses/` e `latex/`. O instalador não tem assinatura
digital; a versão publicada acompanha um arquivo SHA-256 para conferência.

### Gerar o pacote completo

Na raiz do repositório, com Python 3.13 x64 e Inno Setup 6 já instalados:

```powershell
python -m venv build/installer-venv
& build/installer-venv/Scripts/python.exe -m pip install --no-cache-dir --no-deps -r installer/requirements-build.lock.txt
& build/installer-venv/Scripts/python.exe scripts/build_windows_installer.py
```

O ambiente é separado do Python de desenvolvimento. As versões são fixadas
no lockfile. O uso de `--no-deps` é intencional: inkex 1.4 declara NumPy < 2,
mas é validado aqui com NumPy 2 e Python 3.13, evitando PyGObject no Windows.
Não usar `pip check` como substituto do teste funcional desse conjunto.

O script baixa o arquivo oficial TinyTeX v2026.10, confere seu SHA-256,
empacota o aplicativo com PyInstaller e roda `fikzPy.exe --self-test <relatório>`
com um PATH contendo somente diretórios do Windows. O teste inicializa a GUI,
exercita os cinco modos, exige svg2tikz no Visual e compila cinco PDFs com o
LaTeX privado. Uma falha interrompe a geração. Relatórios, PDFs e logs ficam em
`build/installer-validation/`; o executável, manifesto e SHA-256 são gerados
em `installer/Output/`, fora do Git. A pasta `dist/fikzPy/` contém a aplicação
portátil completa e pode ser copiada inteira para outro local.

Use `--validate-installation` para também instalar em uma pasta de teste,
validar os cinco PDFs instalados e desinstalar. Essa variante usa outro AppId
para preservar uma eventual instalação existente do usuário e não cria atalhos.
O instalador temporário de validação é removido após o teste.

Para repetir o empacotamento sem congelar novamente o aplicativo, use
`--skip-freeze`. Para especificar o compilador, use `--iscc <caminho>`.
Reconstrua o bundle sempre que houver alterações no aplicativo ou no launcher.

### Validação de instalação

Antes de publicar uma versão, instale o `.exe` em uma pasta de teste, execute
o mesmo `--self-test` sobre o `fikzPy.exe` instalado e confira a desinstalação.
Use um perfil de teste se já existir uma instalação: o AppId é preservado para
permitir atualizações. Os arquivos `.tex`/PDF do usuário não são removidos.

As instruções abaixo documentam o instalador antigo com downloads, mantido
como alternativa. Para a distribuição atual, use `fikzpy-offline.iss` através
do script acima.

## Instalador antigo com downloads

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
