@echo off
REM =============================================================================
REM  Rio — Start Local Client (Windows)
REM  Activates local venv and runs local/main.py
REM =============================================================================

setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..") do set "RIO_ROOT=%%~fI"

set "LOCAL_DIR=%RIO_ROOT%\local"

if not exist "%LOCAL_DIR%\venv\Scripts\activate.bat" (
    echo ERROR: Local venv not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting Rio Local Client...
echo   Dir: %LOCAL_DIR%
echo   Press Ctrl+C to stop.
echo.

cd /d "%LOCAL_DIR%"
call venv\Scripts\activate.bat
python main.py

pause
