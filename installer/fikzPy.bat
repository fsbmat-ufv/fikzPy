@echo off
rem Abre o fikzPy. Se as dependencias ainda nao tiverem sido instaladas,
rem executa primeiro o instalador de dependencias.
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo As dependencias ainda nao foram instaladas. Instalando agora...
    call "%~dp0instalar_dependencias.bat"
    if errorlevel 1 exit /b 1
)
cd /d "%~dp0"
start "" "%~dp0.venv\Scripts\pythonw.exe" -m fikzpy.main
