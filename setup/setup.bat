@echo off
REM =============================================================================
REM  Rio — One-Command Setup (Windows)
REM
REM  Usage:
REM    Double-click setup.bat   OR   run from cmd:  setup.bat
REM
REM  What it does:
REM    1. Ensures Python 3.11 is installed (auto-installs via winget when missing)
REM    2. Creates cloud\venv and installs cloud dependencies
REM    3. Creates local\venv and installs local dependencies (incl. PyTorch CPU)
REM    4. Prompts for GEMINI_API_KEY and writes cloud\.env
REM    5. Prompts dashboard/live-feed port and updates cloud\.env + config.yaml
REM    6. Asks consent for start-on-boot guidance
REM    7. Prints summary
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
set "CONFIG_PATH=%RIO_ROOT%\config.yaml"
set "RIO_PORT=8080"

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
echo [1/7] Checking Python 3.11 installation...

set "PYTHON_EXE="
for /f "usebackq delims=" %%P in (`py -3.11 -c "import sys; print(sys.executable)" 2^>nul`) do set "PYTHON_EXE=%%P"

if "%PYTHON_EXE%"=="" (
    where winget >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo   Python 3.11 not found. Installing via winget...
        winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
        for /f "usebackq delims=" %%P in (`py -3.11 -c "import sys; print(sys.executable)" 2^>nul`) do set "PYTHON_EXE=%%P"
    )
)

if "%PYTHON_EXE%"=="" (
    echo.
    echo ERROR: Python 3.11 not available.
    echo   Install Python 3.11 from https://www.python.org/downloads/release/python-3119/
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%V in ('"%PYTHON_EXE%" --version 2^>^&1') do set "PY_VERSION=%%V"
echo   Using Python %PY_VERSION% at %PYTHON_EXE%

for /f "tokens=1,2 delims=." %%A in ("%PY_VERSION%") do (
    set "PY_MAJOR=%%A"
    set "PY_MINOR=%%B"
)

if not "!PY_MAJOR!.!PY_MINOR!"=="3.11" (
    echo ERROR: Python 3.11.x required. Found Python %PY_VERSION%.
    pause
    exit /b 1
)

echo   Python 3.11 OK.
echo.

REM ---------------------------------------------------------------------------
REM  Step 1 — Upgrade pip globally
REM ---------------------------------------------------------------------------
echo [2/7] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip --quiet
echo   pip upgraded.
echo.

REM ---------------------------------------------------------------------------
REM  Step 2 — Cloud virtual environment
REM ---------------------------------------------------------------------------
echo [3/7] Setting up Cloud environment...

if not exist "%CLOUD_DIR%" (
    echo ERROR: Cloud directory not found at %CLOUD_DIR%
    pause
    exit /b 1
)

if exist "%CLOUD_DIR%\venv" (
    echo   Cloud venv already exists, skipping creation.
) else (
    echo   Creating cloud\venv...
    "%PYTHON_EXE%" -m venv "%CLOUD_DIR%\venv"
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
echo [4/7] Setting up Local environment...

if not exist "%LOCAL_DIR%" (
    echo ERROR: Local directory not found at %LOCAL_DIR%
    pause
    exit /b 1
)

if exist "%LOCAL_DIR%\venv" (
    echo   Local venv already exists, skipping creation.
) else (
    echo   Creating local\venv...
    "%PYTHON_EXE%" -m venv "%LOCAL_DIR%\venv"
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
echo [5/7] Configuring API key...

if exist "%CLOUD_DIR%\.env" (
    echo   .env file already exists in cloud\.
    set /p "UPDATE_KEY=  Update GEMINI_API_KEY now? [y/N]: "
    if /I "!UPDATE_KEY!"=="Y" (
        set /p "API_KEY=  Enter your GEMINI_API_KEY (or press Enter to keep current): "
        if not "!API_KEY!"=="" (
            call :upsert_env_var "%CLOUD_DIR%\.env" "GEMINI_API_KEY" "!API_KEY!"
            echo   API key updated in cloud\.env
        )
    )
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
REM ---------------------------------------------------------------------------
REM  Step 5 — Dashboard/live-feed port
REM ---------------------------------------------------------------------------
echo [6/7] Configuring dashboard/live-feed port...
set /p "PORT_INPUT=  Enter cloud/dashboard port [8080]: "
if not "!PORT_INPUT!"=="" set "RIO_PORT=!PORT_INPUT!"

for /f "delims=0123456789" %%A in ("!RIO_PORT!") do set "RIO_PORT=8080"
if !RIO_PORT! lss 1 set "RIO_PORT=8080"
if !RIO_PORT! gtr 65535 set "RIO_PORT=8080"

call :upsert_env_var "%CLOUD_DIR%\.env" "PORT" "!RIO_PORT!"
call :set_cloud_url_port "!CONFIG_PATH!" "!RIO_PORT!"
echo   Port set to !RIO_PORT!
echo   Updated cloud_url in config.yaml

echo.
REM ---------------------------------------------------------------------------
REM  Step 6 — Ask permission for start-on-boot
REM ---------------------------------------------------------------------------
echo [7/7] Start-on-boot configuration
set /p "BOOT_CONSENT=  Enable Rio on every boot? [y/N]: "
if /I "!BOOT_CONSENT!"=="Y" (
    echo   Windows auto-start helper is not generated by this script.
    echo   Use Task Scheduler to run setup\run-cloud.bat and setup\run-local.bat at logon.
) else (
    echo   Skipped start-on-boot setup.
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
echo      Dashboard: http://localhost:!RIO_PORT!/dashboard
echo      Live feed WS: ws://localhost:!RIO_PORT!/ws/dashboard
echo.
echo   2. Start Local client (in another terminal):
echo        setup\run-local.bat
echo.
echo   Config: rio\config.yaml
echo   API Key: rio\cloud\.env
echo.
echo   Launching mandatory onboarding now: rio configure
python "%RIO_ROOT%\cli.py" configure
echo.
echo   Rio is live at: http://localhost:!RIO_PORT!/dashboard
echo ============================================================
echo.

pause
exit /b 0

:upsert_env_var
set "ENV_FILE=%~1"
set "ENV_KEY=%~2"
set "ENV_VALUE=%~3"

if not exist "%ENV_FILE%" type nul > "%ENV_FILE%"
findstr /b /c:"%ENV_KEY%=" "%ENV_FILE%" >nul
if %ERRORLEVEL% equ 0 (
    powershell -NoProfile -Command "(Get-Content -Raw '%ENV_FILE%') -replace '(?m)^%ENV_KEY%=.*$', '%ENV_KEY%=%ENV_VALUE%' | Set-Content '%ENV_FILE%'"
) else (
    >> "%ENV_FILE%" echo %ENV_KEY%=%ENV_VALUE%
)
goto :eof

:set_cloud_url_port
set "CFG_FILE=%~1"
set "CFG_PORT=%~2"
powershell -NoProfile -Command "$p='%CFG_FILE%'; $port='%CFG_PORT%'; $t=Get-Content -Raw $p; $u=[regex]::Replace($t,'(cloud_url:\s*ws://localhost:)\d+(/ws/rio/live)','$1'+$port+'$2'); if($u -eq $t){ $u=[regex]::Replace($t,'cloud_url:\s*.*','cloud_url: ws://localhost:'+ $port +'/ws/rio/live',1)}; Set-Content -Path $p -Value $u"
goto :eof
