@echo off
setlocal

set "DASHBOARD_DIR=%~dp0"
for %%I in ("%DASHBOARD_DIR%..\..\..") do set "ROOT=%%~fI"
set "SCRIPT=%ROOT%\BOT_V2_DAYTIME_LAB\src\phase44_dashboard.py"
set "HTML=%DASHBOARD_DIR%dashboard.html"

echo ============================================================
echo MANIPULANTE - DASHBOARD OBSERVABILITY
echo ============================================================
echo.
echo Modo: SOLO LECTURA
echo No inicia el bot.
echo No detiene el bot.
echo No envia ordenes.
echo No modifica estrategia.
echo.

cd /d "%ROOT%"

python -c "import streamlit" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Abriendo dashboard Streamlit local...
    echo Cierre esta ventana solo cuando termine de mirar el dashboard.
    python -m streamlit run "%SCRIPT%"
) else (
    echo Streamlit no esta instalado. Usando HTML local.
    python "%SCRIPT%" --export-html
    if exist "%HTML%" (
        start "" "%HTML%"
    ) else (
        echo ERROR: No se pudo crear "%HTML%".
        pause
    )
)

endlocal
