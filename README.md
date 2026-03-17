# Rio Agent

**Version:** 0.9.0

Rio is an autonomous AI assistant and pair programmer that combines voice interaction, computer vision, and intelligent tool orchestration. Built on Google's Gemini models, Rio can see your screen, execute commands, automate desktop tasks, and proactively assist when you're stuck—all through natural conversation.

**One-line pitch:** "The AI assistant that sees, acts, and helps before you ask."

## Core Capabilities

### 🎙️ Natural Interaction
- **Voice & Text Modes**: Live audio conversation with native audio models or text-based chat
- **Push-to-Talk**: F2 hotkey for instant voice input
- **Wake Word Detection**: Optional hands-free activation with Vosk models
- **Voice Activity Detection (VAD)**: Silero VAD for intelligent speech detection

### 👁️ Computer Vision
- **Screen Capture**: On-demand (F3) or continuous autonomous monitoring
- **Vision-Guided Actions**: Uses Gemini Computer Use model for precise UI element targeting
- **OCR Support**: RapidOCR for text extraction from screenshots
- **Multi-Monitor**: Full support for multi-display setups

### 🤖 Autonomous Execution
- **Multi-Agent Architecture**: Specialized agents for code, screen interaction, research, and creative tasks
- **Tool Orchestration**: 60+ tools for file operations, shell commands, screen control, and more
- **Smart Click**: Natural language UI targeting ("click the Save button")
- **Task Planning**: Automatic decomposition of complex goals into executable steps
- **Verification Loop**: Auto-captures screenshots after actions to verify success

### 🛠️ Development Tools
- **File Operations**: Read, write, patch with automatic `.rio.bak` backups
- **Shell Execution**: Safe command execution with dangerous pattern blocking
- **Browser Automation**: Playwright-based web automation via Chrome DevTools Protocol
- **Window Management**: Focus, minimize, maximize, close, resize windows
- **Process Control**: List, monitor, and manage system processes

### 🌐 Web & Cloud Integration
- **Web Search**: DuckDuckGo integration for research
- **Web Scraping**: Fetch and parse web content
- **Google Workspace**: Gmail, Drive, Calendar, Sheets, Docs integration
- **MCP Support**: Model Context Protocol for extensible tool integration

### 🧠 Memory & Learning
- **Persistent Memory**: Save and recall context across sessions
- **Vector Search**: Semantic memory with ChromaDB (optional)
- **Pattern Recognition**: ML-based user behavior modeling
- **Struggle Detection**: Proactive assistance when you're stuck
- **Task History**: Complete audit trail of all actions

### 🎨 Creative Capabilities
- **Image Generation**: Imagen 3 integration
- **Video Generation**: Veo 2 integration
- **Creative Agent**: Specialized agent for design and media tasks

### 📊 Monitoring & Control
- **Web Dashboard**: Real-time transcript, tool logs, and health monitoring
- **Telemetry**: Comprehensive logging with structlog
- **Rate Limiting**: Configurable per-tool and per-minute limits
- **Safety Controls**: Multi-layer validation and approval workflows

## Feature Status

| Area | Status | Notes |
| --- | --- | --- |
| Live audio + text fallback | Implemented | `session_mode: live` falls back to text mode if Live API connection fails |
| Screen capture | Implemented | F3 on-demand capture plus periodic autonomous capture |
| Autonomous computer-use loop | Implemented | Screen actions auto-capture a follow-up screenshot for the next step |
| Local code/file tools | Implemented | Safe path resolution, backups, command blocklist |
| Struggle detection | Implemented | Four screen-based signals plus cooldowns and demo mode |
| Dashboard UI | Implemented | Transcript, struggle gauge, tool log, health, setup page |
| Customer care + tutor skills | Implemented | Fixed-schema profiles and local persistence |
| Chat history + pattern tracking | Implemented | SQLite-backed stores |
| ML ensemble pipeline | Implemented/optional | Active when `scikit-learn` is available |
| Vector memory | Optional | Requires `chromadb` and `sentence-transformers` |
| Gemini Pro routing | Partial | Heuristic exists, but `ModelRouter.call_pro()` is still a stub |

## Architecture

Rio uses a distributed architecture with a local client and cloud relay:

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         LOCAL CLIENT                                 │
│  (rio/local/)                                                        │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   Audio I/O  │  │Screen Capture│  │ Tool Executor│             │
│  │  VAD + Wake  │  │  OCR + Vision│  │ File/Shell   │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │Screen Navigator│ │Browser Agent │  │Windows Agent │             │
│  │ Click/Type   │  │  Playwright  │  │  pywinauto   │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ Memory Store │  │  Chat Store  │  │Struggle Detect│             │
│  │ Vector/SQLite│  │   SQLite     │  │  ML Pipeline │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CLOUD RELAY                                  │
│  (rio/cloud/)                                                        │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  FastAPI     │  │Session Manager│ │ Tool Bridge  │             │
│  │  WebSocket   │  │ Live/Text    │  │ RPC Proxy    │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │Tool Orchestrator│ │Model Router │  │Rate Limiter  │             │
│  │ Multi-Agent  │  │Flash/Pro/CU  │  │ Quota Mgmt   │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐                                │
│  │  Dashboard   │  │ Skill Loader │                                │
│  │  UI Server   │  │ Profiles     │                                │
│  └──────────────┘  └──────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Gemini API
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         GEMINI MODELS                                │
│                                                                      │
│  • gemini-3-flash-preview (primary orchestrator)                    │
│  • gemini-3-pro-preview (complex reasoning)                         │
│  • gemini-2.5-flash-native-audio-latest (live audio)                │
│  • gemini-2.5-computer-use-preview (vision grounding)               │
│  • imagen-4 (image generation)                                      │
│  • veo-2 (video generation)                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Input** → Local client captures voice/text/screen
2. **WebSocket** → Encrypted connection to cloud relay
3. **Gemini Session** → Cloud manages model interactions
4. **Tool Calls** → Proxied back to local client for execution
5. **Results** → Returned to model for next iteration
6. **Response** → Delivered to user via voice/text
7. **Dashboard** → Real-time monitoring via WebSocket broadcast

## Project Structure

```text
rio/
├── __init__.py              # Package version (0.9.0)
├── cli.py                   # Command-line interface (rio config, doctor, run)
├── config.yaml              # Main configuration file
├── INSTALL.md               # Detailed installation guide
│
├── cloud/                   # Cloud relay service
│   ├── main.py             # FastAPI application entry point
│   ├── gemini_session.py   # Gemini Live/text session wrapper
│   ├── session_manager.py  # Multi-session lifecycle management
│   ├── rio_agent.py        # ADK agent definition & ToolBridge
│   ├── tool_orchestrator.py # Multi-agent tool execution engine (3330 lines)
│   ├── model_router.py     # Model selection logic (Flash/Pro/Computer Use)
│   ├── rate_limiter.py     # Request throttling & quota management
│   ├── mcp_client.py       # Model Context Protocol client
│   ├── mcp_server.py       # MCP server implementation
│   ├── skill_loader.py     # Dynamic skill loading
│   ├── voice_plugin.py     # Voice synthesis plugin system
│   ├── workspace_tools.py  # Workspace-aware tool wrappers
│   ├── Dockerfile          # Container image definition
│   ├── service.yaml        # Cloud Run deployment config
│   └── requirements.txt    # Cloud dependencies
│
├── local/                   # Local desktop client
│   ├── main.py             # Client entry point & event loop
│   ├── orchestrator.py     # Autonomous task execution engine (879 lines)
│   ├── tools.py            # Local tool implementations
│   ├── audio_io.py         # Audio input/output handling
│   ├── vad.py              # Voice activity detection (Silero)
│   ├── wake_word.py        # Wake word detection (Vosk)
│   ├── screen_capture.py   # Screenshot capture (mss)
│   ├── screen_navigator.py # Screen automation (pyautogui)
│   ├── windows_agent.py    # Windows-specific automation (pywinauto)
│   ├── browser_agent.py    # Browser automation (Playwright)
│   ├── browser_tools.py    # Browser tool implementations
│   ├── memory.py           # Persistent memory system
│   ├── unified_memory.py   # Unified memory interface
│   ├── chat_store.py       # Chat history persistence (SQLite)
│   ├── task_state.py       # Task state management
│   ├── struggle_detector.py # Proactive assistance detection
│   ├── user_pattern_model.py # User behavior modeling
│   ├── model_fallback.py   # Model failover logic
│   ├── config.py           # Configuration loader
│   ├── constants.py        # Shared constants
│   ├── platform_utils.py   # Cross-platform utilities
│   ├── profiles.py         # User profile management
│   ├── notifier.py         # Desktop notifications
│   ├── ocr.py              # OCR integration (RapidOCR)
│   ├── web_tools.py        # Web search & fetch
│   ├── ws_client.py        # WebSocket client
│   ├── telegram_bot.py     # Telegram integration
│   ├── whatsapp_channel.py # WhatsApp integration
│   ├── channel_manager.py  # Multi-channel messaging
│   ├── creative_agent.py   # Image/video generation
│   ├── push_to_talk.py     # PTT hotkey handler
│   ├── rio_logging.py      # Structured logging setup
│   ├── maintenance.py      # Cleanup & maintenance tasks
│   └── requirements.txt    # Local dependencies
│
├── ml/                      # Machine learning components
│   ├── feature_engine.py   # Feature extraction pipeline
│   ├── ensemble_model.py   # Model composition
│   ├── user_model_manager.py # Model lifecycle management
│   ├── train.py            # Training entry point
│   ├── models/             # Serialized model artifacts
│   └── datasets/           # Training data
│
├── data/                    # Runtime data (gitignored)
│   ├── conversations/      # Chat transcripts
│   ├── memory/             # Vector memory store
│   ├── sessions/           # Session state
│   ├── transcripts/        # Audio transcripts
│   ├── users/              # User profiles
│   ├── workspaces/         # Workspace metadata
│   ├── rio_chats.db        # Chat history database
│   ├── rio_patterns.db     # User pattern database
│   └── rio_tasks.db        # Task state database
│
├── ui/                      # Web dashboard
│   └── dashboard/          # Static HTML/CSS/JS
│
├── setup/                   # Installation scripts
│   ├── setup.sh            # Linux/macOS setup
│   ├── setup.bat           # Windows setup
│   ├── run-cloud.sh        # Start cloud relay (Unix)
│   ├── run-cloud.bat       # Start cloud relay (Windows)
│   ├── run-local.sh        # Start local client (Unix)
│   ├── run-local.bat       # Start local client (Windows)
│   └── deploy.sh           # Cloud Run deployment
│
├── tests/                   # Test suite
│   ├── test_all.py         # Integration tests
│   ├── eval_harness.py     # Evaluation framework
│   └── golden_trajectories.json # Test scenarios
│
├── modes/                   # Agent mode definitions
│   ├── automation.yaml     # Automation mode config
│   ├── developer.yaml      # Developer mode config
│   └── researcher.yaml     # Research mode config
│
├── skills/                  # Skill definitions
│   ├── customer_care.yaml  # Customer support skill
│   └── tutor.yaml          # Tutoring skill
│
├── logs/                    # Application logs (gitignored)
│   └── rio-YYYY-MM-DD.log  # Daily log files
│
└── output/                  # Generated content
    └── creative/           # Generated images/videos
```

## Quick Start

### Prerequisites

- **Python**: 3.11+ (3.10+ may work but 3.11+ is recommended)
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: ~2GB for dependencies and models
- **API Key**: Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Installation

#### Option 1: Automated Setup (Recommended)

**Windows:**
```cmd
cd Rio-Agent\rio\setup
setup.bat
```

**Linux / macOS:**
```bash
cd Rio-Agent/rio/setup
chmod +x *.sh
./setup.sh
```

The setup script will:
1. Create a Python virtual environment
2. Install all required dependencies
3. Configure the Gemini API key
4. Set up the configuration file

#### Option 2: Manual Installation

```bash
cd Rio-Agent/rio

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt

# Configure API key
echo "GEMINI_API_KEY=your_api_key_here" > cloud/.env
```

### Running Rio

#### Start the Cloud Relay

**Windows:**
```cmd
cd Rio-Agent\rio\setup
run-cloud.bat
```

**Linux / macOS:**
```bash
cd Rio-Agent/rio/setup
./run-cloud.sh
```

The cloud relay will start on `http://localhost:8080`

#### Start the Local Client

**Windows:**
```cmd
cd Rio-Agent\rio\setup
run-local.bat
```

**Linux / macOS:**
```bash
cd Rio-Agent/rio/setup
./run-local.sh
```

### Verify Installation

After starting both services:

- **Health Check**: http://localhost:8080/health
- **Dashboard**: http://localhost:8080/dashboard
- **Setup Page**: http://localhost:8080/dashboard/setup.html

### Using the CLI

Rio includes a powerful CLI for configuration and diagnostics:

```bash
# Interactive setup wizard
python -m rio.cli configure

# Run system diagnostics
python -m rio.cli doctor

# Test API connectivity
python -m rio.cli doctor --test-api

# View configuration
python -m rio.cli config show

# Get a config value
python -m rio.cli config get models.primary

# Set a config value
python -m rio.cli config set models.primary gemini-3-flash-preview

# Start Rio
python -m rio.cli run
```

## Runtime Controls

### Hotkeys

Global hotkeys work when `pynput` is installed and your OS allows keyboard hooks.

| Key | Function | Description |
|-----|----------|-------------|
| **F2** | Push to Talk | Hold to speak, release to send |
| **F3** | Screenshot | Force an immediate screenshot |
| **F4** | Demo Trigger | Force proactive assistance (demo mode) |
| **F5** | Screen Mode | Toggle between on_demand and autonomous |
| **F6** | Live Mode | Toggle continuous monitoring and wake word |

### Screen Modes

- **on_demand**: Screenshots only when requested (F3, tool call, or user request)
- **autonomous**: Periodic screenshots based on `vision.fps` setting
- **Live Mode**: Continuous screen monitoring + wake word listening for hands-free operation

### Voice Commands

When in Live Mode or after pressing F2:

- **"Hey Rio"** - Wake word to activate listening
- **"Open [application]"** - Launch applications
- **"Click [description]"** - Smart click on UI elements
- **"Search for [query]"** - Web search
- **"Read file [path]"** - Read file contents
- **"Run [command]"** - Execute shell commands
- **"Take a screenshot"** - Capture screen
- **"Resume"** / **"Continue"** - Resume paused tasks

### CLI Commands

```bash
# Configuration
rio config show                          # View all settings
rio config get models.primary            # Get specific value
rio config set models.primary gemini-3-flash-preview  # Set value

# Diagnostics
rio doctor                               # Run system checks
rio doctor --test-api                    # Test API connectivity

# Setup
rio configure                            # Interactive setup wizard

# Execution
rio run                                  # Start Rio agent
```

## Configuration

Rio's behavior is controlled by `rio/config.yaml`. Here are the key settings:

### Core Settings

```yaml
rio:
  agent_name: Rio                    # Agent display name
  session_mode: live                 # "live" (audio) or "text"
  backend: direct                    # Connection mode
  cloud_url: ws://localhost:8080/ws/rio/live  # WebSocket endpoint
```

### Model Configuration

```yaml
  models:
    primary: gemini-3-flash-preview           # Main orchestrator model
    secondary: gemini-3-pro-preview           # Fallback model
    live: gemini-2.5-flash-native-audio-latest # Live audio model
    computer_use: gemini-3-flash-preview      # Vision grounding model
    imagen: imagen-4                          # Image generation
    timeout_seconds: 30.0
    cooldown_seconds: 60.0
    pro_rpm_budget: 5                         # Pro model rate limit
```

### Multi-Agent Configuration

```yaml
  agents:
    task_executor:
      enabled: true
      max_iterations: 25
      model: gemini-3-flash-preview
      tools: all
      description: "General task execution, multi-step workflows"
    
    code_agent:
      enabled: true
      max_iterations: 15
      model: gemini-3-flash-preview
      tools: dev
      description: "Coding, debugging, file editing, git operations"
    
    computer_use_agent:
      enabled: true
      max_iterations: 25
      model: gemini-3-flash-preview
      tools: screen
      description: "Screen interaction, GUI automation, browser navigation"
    
    research_agent:
      enabled: true
      max_iterations: 10
      model: gemini-2.5-pro
      tools: memory
      description: "Deep research, complex analysis, reasoning"
    
    creative_agent:
      enabled: true
      max_iterations: 5
      tools: creative
      description: "Image generation, video creation, design"
```

### Vision & Screen Capture

```yaml
  vision:
    default_mode: on_demand  # "on_demand" or "autonomous"
    fps: 0.33                # Screenshots per second in autonomous mode
    quality: 85              # JPEG quality (0-100)
    resize_factor: 0.75      # Scale factor for screenshots
```

### Audio Settings

```yaml
  audio:
    sample_rate: 16000
    block_size: 320
    latency: low
    use_wasapi: true         # Windows-specific audio API
    input_device: null       # Auto-detect
    output_device: null      # Auto-detect
```

### Hotkeys

```yaml
  hotkeys:
    push_to_talk: f2         # Voice input
    screenshot: f3           # Force screenshot
    toggle_proactive: f4     # Demo mode trigger
    screen_mode: f5          # Toggle screen capture mode
    live_mode: f6            # Toggle Live Mode
```

### Memory & Persistence

```yaml
  memory:
    db_path: ./data/memory
    max_recall: 5            # Number of memories to recall
```

### Struggle Detection

```yaml
  struggle:
    enabled: true
    threshold: 0.85          # Confidence threshold
    cooldown_seconds: 300    # Time between proactive offers
    decline_cooldown: 600    # Cooldown after user declines
    demo_mode: false         # Lower thresholds for demos
```

### Tool Policy & Rate Limiting

```yaml
  orchestrator:
    tool_timeout_seconds: 120
    heartbeat_interval_seconds: 5
    
    tool_policy:
      default_per_minute: 20
      max_calls_per_task: 120
      max_cost_points_per_task: 200
      
      cost_points:
        generate_video: 6
        gmail_send: 3
        run_command: 4
        start_process: 5
      
      per_tool_per_minute:
        gmail_send: 6
        run_command: 8
        smart_click: 30
```

### Filesystem Access Control

```yaml
  filesystem:
    enabled: true
    read_paths:
      - .                    # Current directory
    write_paths:
      - .                    # Current directory
```

### Skills

```yaml
  skills:
    customer_care:
      enabled: true
      default_priority: medium
      auto_escalate_after: 300
      ticket_dir: ./rio_tickets
    
    tutor:
      enabled: true
      default_difficulty: intermediate
      socratic_mode: true
      quiz_num_questions: 5
      progress_dir: ./rio_progress
```

### Logging

```yaml
  logging:
    log_dir: ./logs
    max_files: 7             # Days of logs to keep
    verbose: false
```

### Environment Variables

Set these in `rio/cloud/.env` or your environment:

```bash
# Required
GEMINI_API_KEY=your_api_key_here

# Optional overrides
SESSION_MODE=live                    # Override session mode
TEXT_MODEL=gemini-3-flash-preview    # Override text model
LIVE_MODEL=gemini-2.5-flash-native-audio-latest
RIO_WS_TOKEN=secret_token            # WebSocket authentication
PRO_RPM_BUDGET=5                     # Pro model rate limit
ORCHESTRATOR_MODEL=gemini-3-flash-preview
RIO_TOOLBRIDGE_TIMEOUT_SECONDS=60
RIO_ORCHESTRATOR_USE_AGENTS_MD=true  # Load agents.md instructions
```

## Available Tools

Rio provides 60+ tools organized by category:

### File Operations
- `read_file(path)` - Read file contents
- `write_file(path, content)` - Write to file (creates .rio.bak backup)
- `patch_file(path, old_text, new_text)` - Find and replace in file

### Shell & Process Management
- `run_command(command)` - Execute shell command (30s timeout, dangerous patterns blocked)
- `start_process(command, label)` - Start long-running background process
- `check_process(pid)` - Check process status
- `stop_process(pid)` - Stop background process
- `list_processes(name_filter)` - List running processes
- `kill_process(name_or_pid)` - Terminate process

### Screen Capture & Vision
- `capture_screen()` - Take screenshot
- `get_screen_info()` - Get monitor information

### Screen Automation (Coordinate-Based)
- `screen_click(x, y, button, clicks)` - Click at coordinates
- `screen_type(text, interval)` - Type text
- `screen_scroll(x, y, clicks)` - Scroll at position
- `screen_hotkey(keys)` - Press keyboard shortcut
- `screen_move(x, y)` - Move mouse cursor
- `screen_drag(start_x, start_y, end_x, end_y)` - Click and drag

### Vision-Guided Automation
- `smart_click(target, action, clicks)` - Click UI element by description (uses Computer Use model)

### Window Management
- `open_application(name_or_path)` - Launch application
- `list_all_windows()` - List all visible windows
- `get_active_window()` - Get foreground window info
- `find_window(title)` - Search for window by title
- `focus_window(title)` - Bring window to foreground
- `minimize_window(title)` - Minimize window
- `maximize_window(title)` - Maximize window
- `close_window(title)` - Close window
- `resize_window(title, width, height)` - Resize window
- `move_window(title, x, y)` - Move window

### Clipboard
- `get_clipboard()` - Read clipboard text
- `set_clipboard(text)` - Set clipboard text

### Browser Automation (Playwright)
- `browser_connect(cdp_url, browser, profile)` - Connect to browser via CDP
- `browser_navigate(url)` - Navigate to URL
- `browser_click_element(selector)` - Click element by CSS selector
- `browser_fill_form(selector, value)` - Fill form field
- `browser_extract_text(selector)` - Extract text from element
- `browser_evaluate(javascript)` - Execute JavaScript
- `browser_wait_for(selector, timeout)` - Wait for element

### Web Tools
- `web_search(query, max_results)` - Search web with DuckDuckGo
- `web_fetch(url, max_chars)` - Fetch and parse web page
- `web_cache_get(url)` - Get cached web page

### Memory & Notes
- `save_note(key, value, media_paths)` - Save persistent note
- `get_notes()` - Retrieve all notes
- `search_notes(query, limit)` - Search notes by keyword
- `export_context()` - Export session memory to file
- `memory_stats()` - Get memory system statistics

### Google Workspace
- `gmail_search(query, max_results)` - Search Gmail
- `gmail_send(to, subject, body, cc, bcc)` - Send email
- `drive_list(folder_id, max_results)` - List Drive files
- `calendar_list_events(calendar, time_min, time_max)` - List calendar events
- `sheets_read(spreadsheet_id, range_a1)` - Read from Sheets
- `docs_create(title, content)` - Create Google Doc

### Creative Tools
- `generate_image(prompt, aspect_ratio, style)` - Generate image with Imagen 3
- `generate_video(prompt, duration_seconds, aspect_ratio)` - Generate video with Veo 2

### Customer Care (Skill)
- `create_ticket(title, category, priority, description)` - Create support ticket
- `update_ticket(ticket_id, status, priority, notes)` - Update ticket

### Tutoring (Skill)
- `generate_quiz(topic, difficulty, num_questions)` - Generate quiz
- `track_progress(action, subject, topic, score)` - Track learning progress
- `explain_concept(concept, level, context)` - Get concept explanation

### Tool Risk Classification

Tools are classified by risk level for safety controls:

- **SAFE**: Read-only operations (read_file, capture_screen, list_windows)
- **MODERATE**: Writes and UI interactions (write_file, screen_click, generate_image)
- **DANGEROUS**: Shell execution and process control (run_command, kill_process)
- **CRITICAL**: Reserved for future destructive operations

## Safety Model

Rio is designed to degrade gracefully and avoid destructive behavior by default.

- Dangerous shell patterns such as `rm -rf`, `mkfs`, `shutdown`, `curl | bash`, and similar commands are blocked.
- `write_file` and `patch_file` create `.rio.bak` backups before modifying files.
- File paths are resolved relative to the tool working directory and blocked from escaping it.
- Dangerous tool calls are confirmed locally before execution.
- Screen actions are rate-limited and logged, and `pyautogui.FAILSAFE` stays enabled.

## Working Directory Scope

Rio's local file tools are scoped to the process working directory. The provided `run-local` scripts start the client from `rio/local`, so by default Rio can only read or modify files in that directory tree.

If you want Rio to operate on a different project, launch `rio/local/main.py` from that project's root so the tool executor resolves paths there.

## Skills and Persistence

### Customer Care

- Deterministic profile schema in `rio/local/profiles.py`
- Setup UI saves `customer_care_profile.json`
- Ticket tools persist JSON tickets under `rio_tickets/`

### Tutor

- Deterministic student profile schema in `rio/local/profiles.py`
- Setup UI saves `tutor_profile.json`
- Tutor tools persist progress under `rio_progress/`

### Local state generated by Rio

- `rio/rio_chats.db` - chat history
- `rio/rio_patterns.db` - user behavior and struggle patterns
- `rio/rio_memory/` - optional vector memory store
- `rio/rio_profiles/` - saved customer care and tutor profiles
- `rio/local/*.rio.bak` or project-local `.rio.bak` files - backups from edits

## Testing

Rio includes a smoke/integration-style test script at [`rio/tests/test_all.py`](rio/tests/test_all.py).

Run it with:

```bash
cd Rio-Agent
python rio/tests/test_all.py
```

The suite validates imports, config loading, rate limiting, model routing, tool execution, dashboard files, and wire protocol constants. Optional features may be skipped if their dependencies are not installed.

## Cloud Run Deployment

The repo includes a deploy script and container spec:

- [`rio/deploy.sh`](rio/deploy.sh)
- [`rio/cloud/service.yaml`](rio/cloud/service.yaml)
- [`rio/cloud/Dockerfile`](rio/cloud/Dockerfile)

Deployment assumptions:

1. `gcloud` is installed and authenticated.
2. Cloud Run, Cloud Build, and Secret Manager are enabled.
3. A secret named `gemini-api-key` exists in Secret Manager.

Then run:

```bash
cd Rio-Agent/rio
chmod +x deploy.sh
./deploy.sh
```

The script prints the HTTP URL and the WebSocket URL to put back into `rio/config.yaml`.

## Known Limitations

- Gemini Pro routing is not complete yet. The router can decide that Pro should be used, but the actual Pro call path still returns `None`.
- The checked-in model settings and the cloud runtime are not perfectly aligned. `cloud/gemini_session.py` defaults to `gemini-2.5-flash` and `gemini-2.5-flash-native-audio-latest`, regardless of the older model notes elsewhere in the repo.
- Memory, some hotkeys, and some screen-navigation features are optional and may appear disabled until their dependencies are installed.
- Desktop automation on Wayland and elevated/admin windows is inherently less reliable than standard Windows desktop automation.
- The codebase contains version strings from multiple milestones (`v0.6.x`, `v0.7.x`, `v0.9.0`) because the project evolved layer by layer.

## Tech Stack

| Layer | Tech |
| --- | --- |
| Cloud relay | FastAPI, uvicorn, websockets, structlog |
| Gemini integration | `google-genai`, Gemini 2.5 Flash text mode, Gemini Live audio mode |
| Audio | `sounddevice`, CPU PyTorch, Silero VAD |
| Vision | `mss`, Pillow, RapidOCR |
| Desktop automation | `pyautogui`, `pygetwindow`, `pyperclip` |
| Memory and local state | SQLite, optional ChromaDB, sentence-transformers |
| ML | NumPy, scikit-learn ensemble pipeline |
| UI | Static HTML/CSS/JS dashboard served by FastAPI |

## Related Docs

- [`Rio-Plan.md`](Rio-Plan.md) - architecture and hackathon plan
- [`context.txt`](context.txt) - project-wide context snapshot
- [`rio/setup/README.md`](rio/setup/README.md) - setup-script-focused quick reference
