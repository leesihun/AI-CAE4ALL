@echo off
setlocal

set "STUDIO_DIR=%~dp0"
set "STUDIO_PORT=%~1"
if "%STUDIO_PORT%"=="" set "STUDIO_PORT=8080"

if defined AI_CAE_STUDIO_PYTHON (
    set "STUDIO_PYTHON=%AI_CAE_STUDIO_PYTHON%"
    goto :start
)

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    set "STUDIO_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
    goto :start
)

python -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "STUDIO_PYTHON=python"
    goto :start
)

py -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "STUDIO_PYTHON=py"
    goto :start
)

python3 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "STUDIO_PYTHON=python3"
    goto :start
)

if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    set "STUDIO_PYTHON=%CONDA_PREFIX%\python.exe"
    goto :start
)

echo.
echo AI-CAE4ALL Studio could not find Python.
echo Install Python 3 or add it to PATH, then run this launcher again.
echo.
pause
exit /b 1

:start
echo.
echo Starting AI-CAE4ALL Studio
echo URL: http://127.0.0.1:%STUDIO_PORT%/index.html
echo Python: %STUDIO_PYTHON%
echo.
"%STUDIO_PYTHON%" "%STUDIO_DIR%start_studio.py" "%STUDIO_PORT%"

endlocal
