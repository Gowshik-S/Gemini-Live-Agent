# Rio Agent Installation Guide

This guide provides detailed installation instructions for Rio Agent.

## System Requirements

- **Python**: 3.11+ (3.10+ may work but 3.11+ is recommended)
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: ~2GB for dependencies and models
- **API Key**: Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Installation Methods

### Method 1: Quick Setup (Recommended)

Use the provided setup scripts for automated installation.

#### Windows

```cmd
cd Rio-Agent\rio\setup
setup.bat
```

#### Linux / macOS

```bash
cd Rio-Agent/rio/setup
chmod +x *.sh
./setup.sh
```

The setup script will:
1. Create a virtual environment
2. Install all required dependencies
3. Configure the environment
4. Set up the Gemini API key

### Method 2: Manual Installation

For more control over the installation process:

#### Step 1: Create Virtual Environment

```bash
cd Rio-Agent/rio

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

#### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

#### Step 3: Configure API Key

Create a `.env` file in `Rio-Agent/rio/cloud/`:

```bash
echo "GEMINI_API_KEY=your_api_key_here" > cloud/.env
```

Or set it as an environment variable:

```bash
# Windows (PowerShell):
$env:GEMINI_API_KEY="your_api_key_here"

# Linux/macOS:
export GEMINI_API_KEY="your_api_key_here"
```

#### Step 4: Configure Rio

Edit `Rio-Agent/rio/config.yaml` to customize settings, or use the CLI:

```bash
python -m rio.cli configure
```

## Dependency Structure

Rio Agent uses a modular dependency structure:

```
requirements.txt              # Main file (aggregates local + cloud)
├── local/requirements.txt    # Local client dependencies
│   ├── Audio: sounddevice, PyAudio, torch (VAD)
│   ├── Vision: Pillow, mss, rapidocr-onnxruntime
│   ├── Automation: pyautogui, pywinauto, playwright
│   ├── ML: scikit-learn, numpy, scipy
│   └── API: google-genai, websockets, structlog
└── cloud/requirements.txt    # Cloud relay dependencies
    ├── fastapi, uvicorn
    ├── websockets
    ├── google-genai
    └── structlog, python-dotenv, pyyaml

requirements-dev.txt          # Development tools (optional)
├── Testing: pytest, pytest-asyncio, pytest-cov
├── Linting: pylint, flake8, mypy
├── Formatting: black, isort
└── Debugging: ipython, ipdb
```

## Optional Dependencies

Some features require additional packages that are not installed by default.

### Hotkeys (F2-F6 controls)

```bash
pip install pynput>=1.7.0
```

### Vector Memory (Semantic recall)

```bash
pip install chromadb>=0.4.0 sentence-transformers>=2.2.0
```

### Offline Wake Word

Install Vosk model:

```bash
# Download model
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip -d ~/.rio/
mv ~/.rio/vosk-model-small-en-us-0.15 ~/.rio/vosk-model-small-en-us
```

## Platform-Specific Notes

### Windows

- **PyAudio**: Pre-built wheels are available, should install without issues
- **Desktop Automation**: Fully supported with DPI awareness
- **Hotkeys**: Global keyboard hooks work reliably

### Linux

- **PyAudio**: May require PortAudio development headers:
  ```bash
  # Ubuntu/Debian:
  sudo apt-get install portaudio19-dev python3-pyaudio
  
  # Fedora:
  sudo dnf install portaudio-devel
  ```

- **Desktop Automation**: 
  - X11: Fully supported
  - Wayland: Limited support (some automation features may not work)

- **Hotkeys**: Require X11 or appropriate Wayland permissions

### macOS

- **PyAudio**: May require Homebrew:
  ```bash
  brew install portaudio
  pip install pyaudio
  ```

- **Desktop Automation**: Requires accessibility permissions
- **Hotkeys**: Requires accessibility permissions in System Preferences

## Verification

After installation, verify your setup:

```bash
# Run diagnostics
python -m rio.cli doctor

# Test API connection
python -m rio.cli doctor --test-api

# View configuration
python -m rio.cli config show
```

## Running Rio Agent

### Start Cloud Relay

```bash
# Windows:
cd Rio-Agent\rio\setup
run-cloud.bat

# Linux/macOS:
cd Rio-Agent/rio/setup
./run-cloud.sh
```

### Start Local Client

```bash
# Windows:
cd Rio-Agent\rio\setup
run-local.bat

# Linux/macOS:
cd Rio-Agent/rio/setup
./run-local.sh
```

### Access Dashboard

Open your browser to:
- Health check: http://localhost:8080/health
- Dashboard: http://localhost:8080/dashboard
- Setup page: http://localhost:8080/dashboard/setup.html

## Troubleshooting

### Import Errors

If you encounter import errors, ensure your virtual environment is activated:

```bash
# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate
```

### PyTorch CPU Installation

Rio uses CPU-only PyTorch for VAD. If you encounter issues:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### API Connection Issues

1. Verify your API key is set correctly
2. Check network connectivity
3. Run diagnostics: `python -m rio.cli doctor --test-api`

### Audio Issues

- **Windows**: Ensure microphone permissions are granted
- **Linux**: Check PulseAudio/PipeWire configuration
- **macOS**: Grant microphone permissions in System Preferences

## Updating Dependencies

To update all dependencies to their latest compatible versions:

```bash
pip install --upgrade -r requirements.txt
```

## Uninstallation

To remove Rio Agent:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf .venv  # Linux/macOS
rmdir /s .venv  # Windows

# Remove generated data (optional)
rm -rf rio_chats.db rio_patterns.db rio_memory/ rio_tickets/ rio_progress/
```

## Getting Help

- **Documentation**: See [README.md](README.md) for feature documentation
- **Issues**: Report bugs on the project repository
- **Configuration**: Run `python -m rio.cli configure` for interactive setup

## Next Steps

After installation:
1. Configure your settings: `python -m rio.cli configure`
2. Run diagnostics: `python -m rio.cli doctor`
3. Start the cloud relay and local client
4. Access the dashboard to verify everything is working
5. Review [README.md](README.md) for usage instructions and hotkeys
