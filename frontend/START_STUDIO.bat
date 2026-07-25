@echo off
setlocal

set "STUDIO_DIR=%~dp0"
set "STUDIO_PORT=%~1"
if "%STUDIO_PORT%"=="" set "STUDIO_PORT=8080"

where py >nul 2>&1
if not errorlevel 1 (
    set "STUDIO_PYTHON=py"
    goto :start
)

where python >nul 2>&1
if not errorlevel 1 (
    set "STUDIO_PYTHON=python"
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
echo.
%STUDIO_PYTHON% "%STUDIO_DIR%start_studio.py" "%STUDIO_PORT%"

endlocal
