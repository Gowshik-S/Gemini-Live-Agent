@echo off
REM =============================================================================
REM  Rio — Start Cloud Server (Windows)
REM  Activates cloud venv and runs cloud/main.py
REM =============================================================================

setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..") do set "RIO_ROOT=%%~fI"

set "CLOUD_DIR=%RIO_ROOT%\cloud"

if not exist "%CLOUD_DIR%\venv\Scripts\activate.bat" (
    echo ERROR: Cloud venv not found. Run setup.bat first.
    pause
    exit /b 1
)

if not exist "%CLOUD_DIR%\.env" (
    echo WARNING: No .env file found. GEMINI_API_KEY may not be set.
    echo   Create %CLOUD_DIR%\.env with: GEMINI_API_KEY=your_key
    echo.
)

echo Starting Rio Cloud Server...
echo   Dir: %CLOUD_DIR%
echo   Press Ctrl+C to stop.
echo.

cd /d "%CLOUD_DIR%"
call venv\Scripts\activate.bat
python main.py

pause
