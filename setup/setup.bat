@echo off
REM =============================================================================
REM  Rio — One-Command Setup (Windows)
REM
REM  Usage:
REM    Double-click setup.bat   OR   run from cmd:  setup.bat
REM
REM  What it does:
REM    1. Checks Python 3.10+ is installed
REM    2. Creates cloud\venv and installs cloud dependencies
REM    3. Creates local\venv and installs local dependencies (incl. PyTorch CPU)
REM    4. Prompts for GEMINI_API_KEY and writes cloud\.env
REM    5. Prints summary
REM =============================================================================

setlocal enabledelayedexpansion

REM ---------------------------------------------------------------------------
REM  Resolve project root (rio/ folder — parent of setup/)
REM ---------------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
REM Go one level up from setup/ to get rio/
for %%I in ("%SCRIPT_DIR%\..") do set "RIO_ROOT=%%~fI"

set "CLOUD_DIR=%RIO_ROOT%\cloud"
set "LOCAL_DIR=%RIO_ROOT%\local"

echo.
echo ============================================================
echo   Rio — Automated Setup (Windows)
echo ============================================================
echo.
echo   Project root : %RIO_ROOT%
echo   Cloud dir    : %CLOUD_DIR%
echo   Local dir    : %LOCAL_DIR%
echo.

REM ---------------------------------------------------------------------------
REM  Step 0 — Check Python
REM ---------------------------------------------------------------------------
echo [1/5] Checking Python installation...

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Python not found in PATH.
    echo   Install Python 3.10+ from https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set "PY_VERSION=%%V"
echo   Found Python %PY_VERSION%

REM Basic version check (major.minor)
for /f "tokens=1,2 delims=." %%A in ("%PY_VERSION%") do (
    set "PY_MAJOR=%%A"
    set "PY_MINOR=%%B"
)

if !PY_MAJOR! lss 3 (
    echo ERROR: Python 3.10+ required. Found Python %PY_VERSION%.
    pause
    exit /b 1
)
if !PY_MAJOR! equ 3 if !PY_MINOR! lss 10 (
    echo ERROR: Python 3.10+ required. Found Python %PY_VERSION%.
    pause
    exit /b 1
)

echo   Python version OK.
echo.

REM ---------------------------------------------------------------------------
REM  Step 1 — Upgrade pip globally
REM ---------------------------------------------------------------------------
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo   pip upgraded.
echo.

REM ---------------------------------------------------------------------------
REM  Step 2 — Cloud virtual environment
REM ---------------------------------------------------------------------------
echo [3/5] Setting up Cloud environment...

if not exist "%CLOUD_DIR%" (
    echo ERROR: Cloud directory not found at %CLOUD_DIR%
    pause
    exit /b 1
)

if exist "%CLOUD_DIR%\venv" (
    echo   Cloud venv already exists, skipping creation.
) else (
    echo   Creating cloud\venv...
    python -m venv "%CLOUD_DIR%\venv"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create cloud virtual environment.
        pause
        exit /b 1
    )
)

echo   Installing cloud dependencies...
call "%CLOUD_DIR%\venv\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet
pip install -r "%CLOUD_DIR%\requirements.txt" --quiet
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install cloud dependencies.
    pause
    exit /b 1
)
call deactivate
echo   Cloud environment ready.
echo.

REM ---------------------------------------------------------------------------
REM  Step 3 — Local virtual environment
REM ---------------------------------------------------------------------------
echo [4/5] Setting up Local environment...

if not exist "%LOCAL_DIR%" (
    echo ERROR: Local directory not found at %LOCAL_DIR%
    pause
    exit /b 1
)

if exist "%LOCAL_DIR%\venv" (
    echo   Local venv already exists, skipping creation.
) else (
    echo   Creating local\venv...
    python -m venv "%LOCAL_DIR%\venv"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create local virtual environment.
        pause
        exit /b 1
    )
)

echo   Installing local dependencies...
call "%LOCAL_DIR%\venv\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet

REM Install PyTorch CPU separately (requires --index-url)
echo   Installing PyTorch (CPU-only)...
pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu --quiet
if %ERRORLEVEL% neq 0 (
    echo WARNING: PyTorch CPU install failed. VAD will be disabled.
    echo   You can retry manually: pip install torch --index-url https://download.pytorch.org/whl/cpu
)

REM Install remaining local deps (skip torch line since we already installed it)
REM Create a temp requirements file without the torch line
set "TEMP_REQ=%TEMP%\rio_local_req_temp.txt"
findstr /v /i "torch" "%LOCAL_DIR%\requirements.txt" > "%TEMP_REQ%"
pip install -r "%TEMP_REQ%" --quiet
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install local dependencies.
    call deactivate
    pause
    exit /b 1
)
del "%TEMP_REQ%" >nul 2>&1

call deactivate
echo   Local environment ready.
echo.

REM ---------------------------------------------------------------------------
REM  Step 4 — GEMINI_API_KEY
REM ---------------------------------------------------------------------------
echo [5/5] Configuring API key...

if exist "%CLOUD_DIR%\.env" (
    echo   .env file already exists in cloud\. Skipping.
    echo   Edit %CLOUD_DIR%\.env to change your API key.
) else (
    echo.
    echo   Rio needs a Google Gemini API key to function.
    echo   Get one at: https://aistudio.google.com/app/apikey
    echo.
    set /p "API_KEY=  Enter your GEMINI_API_KEY (or press Enter to skip): "
    if "!API_KEY!"=="" (
        echo   Skipped. Create cloud\.env manually later:
        echo     GEMINI_API_KEY=your_key_here
        REM Create a template .env anyway
        echo GEMINI_API_KEY=your_key_here> "%CLOUD_DIR%\.env"
    ) else (
        echo GEMINI_API_KEY=!API_KEY!> "%CLOUD_DIR%\.env"
        echo   API key saved to cloud\.env
    )
)

echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo   To start Rio:
echo.
echo   1. Start Cloud server (in one terminal):
echo        setup\run-cloud.bat
echo.
echo   2. Start Local client (in another terminal):
echo        setup\run-local.bat
echo.
echo   Config: rio\config.yaml
echo   API Key: rio\cloud\.env
echo ============================================================
echo.

pause
exit /b 0
