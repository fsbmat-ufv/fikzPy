@echo off
setlocal
set "APP=%~dp0"
title fikzPy - Instalacao das dependencias Python

echo ============================================================
echo  fikzPy - Instalacao das dependencias
echo  Autor: Fernando de Souza Bastos ^<fernando.bastos@ufv.br^>
echo ============================================================
echo.

set "PYEXE="

rem 1) Tenta o launcher "py" do Windows
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%i in ('py -3 -c "import sys; print(sys.executable)"') do set "PYEXE=%%i"
    goto :found
)

rem 2) Tenta "python" no PATH
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)"') do set "PYEXE=%%i"
    goto :found
)

rem 3) Procura instalacoes por usuario (padrao do instalador do python.org)
for /d %%d in ("%LocalAppData%\Programs\Python\Python3*") do (
    "%%d\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYEXE=%%d\python.exe"
)
if defined PYEXE goto :found

echo [ERRO] O Python 3.10 ou superior nao foi encontrado neste computador.
echo.
echo Instale o Python (https://www.python.org/downloads/) marcando a opcao
echo "Add python.exe to PATH" e depois execute novamente este script:
echo   "%APP%instalar_dependencias.bat"
echo.
pause
exit /b 1

:found
echo Python encontrado: %PYEXE%
echo.
echo Criando o ambiente virtual em "%APP%.venv" ...
"%PYEXE%" -m venv "%APP%.venv"
if errorlevel 1 goto :fail

echo.
echo Atualizando o pip ...
"%APP%.venv\Scripts\python.exe" -m pip install --upgrade pip
echo.
echo Instalando os pacotes necessarios:
echo   numpy, opencv-python, PySide6, scikit-image, svg2tikz
echo (isso pode demorar alguns minutos, dependendo da internet) ...
echo.
rem --no-cache-dir evita falhas por caches do pip corrompidos ou com
rem permissoes invalidas na maquina do usuario.
"%APP%.venv\Scripts\python.exe" -m pip install --no-cache-dir -r "%APP%requirements-runtime.txt"
if errorlevel 1 goto :fail

echo.
echo Instalando svg2tikz e inkex (sem dependencias transitivas) ...
rem --no-deps evita que o pip tente instalar numpy^<2 (sem wheels no
rem Python 3.13) e pygobject (nao instala no Windows). As dependencias
rem reais do inkex ja foram instaladas pelo requirements-runtime.txt.
"%APP%.venv\Scripts\python.exe" -m pip install --no-cache-dir --no-deps inkex==1.4.0 svg2tikz==3.3.2
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  Instalacao concluida com sucesso!
echo  Use o atalho "fikzPy" do Menu Iniciar para abrir o programa.
echo ============================================================
echo.
timeout /t 10
exit /b 0

:fail
echo.
echo [ERRO] A instalacao dos pacotes falhou.
echo Verifique sua conexao com a internet e execute novamente:
echo   "%APP%instalar_dependencias.bat"
echo.
pause
exit /b 1
