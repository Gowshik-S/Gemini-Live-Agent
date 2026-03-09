# Rio Agent — Master Implementation Plan

> Reference: OpenClaw architecture patterns  
> Date: March 8, 2026  
> Status: In Progress (~50% complete)

---

## Executive Summary

Rio Agent needs to evolve from a prototype into a production-grade autonomous AI agent.
Key gaps identified by analyzing OpenClaw's architecture:

1. **No model fallback** — Gemini Pro fails silently; no cascade to Flash
2. **No CLI config** — All config requires manual YAML editing
3. **No config dashboard** — Setup page exists but no config management UI
4. **Brittle browser automation** — Playwright/Selenium not degrading gracefully
5. **No structured error logging** — Errors printed but not categorized/actionable
6. **No multi-platform detection** — Hard-coded Windows assumptions
7. **No onboarding wizard** — No `rio configure` equivalent

---

## Architecture Comparison: OpenClaw vs Rio

| Feature | OpenClaw | Rio (Current) | Rio (Target) |
|---------|----------|---------------|--------------|
| Config storage | `~/.openclaw/openclaw.json` (JSON5) | `rio/config.yaml` (YAML) | `rio/config.yaml` + env vars |
| Config CLI | `openclaw config get/set/unset` | None | `rio config get/set` |
| Config validation | Zod schema | Basic dataclass validate() | Enhanced validation + suggestions |
| Model fallback | FailoverError chain with profile rotation | None — hard crash | Model cascade: Pro→Flash→offline |
| Error logging | Structured subsystem loggers + file + console | structlog only | structlog + file + diagnostic levels |
| Dashboard | Lit web components, settings UI | Basic transcript/health | Config UI + task monitor + model status |
| Onboarding | `openclaw onboard` wizard | Manual setup | `rio configure` interactive wizard |
| Multi-platform | macOS, Linux, Windows (WSL2) | Windows only | Windows, macOS, Linux |
| CLI commands | 20+ commands (models, agents, channels...) | None | `rio config`, `rio doctor`, `rio run` |
| Plugin system | Extensions under `extensions/` | Skills (hardcoded) | Skills with enable/disable |
| Task execution | Agent runner with model failover | Orchestrator (basic) | Orchestrator + fallback chain |

---

## Task Breakdown

### Phase 1: Foundation (DO NOW — 50% mark) ✅

#### 1.1 Model Fallback System
- [x] Create `rio/local/model_fallback.py` — cascade logic: Pro → Flash → offline error
- [x] Classify errors: `auth`, `billing`, `rate_limit`, `model_not_found`, `timeout`, `format`
- [x] Wire into orchestrator's `plan_task()` and browser_agent
- [x] Print detailed diagnostic logs when models fail
- [x] Config: `models.fallback_chain` in config.yaml

#### 1.2 CLI Config System  
- [x] Create `rio/cli.py` — entry point for `rio config get/set/doctor/run`
- [x] `rio config get <path>` — read config values (dot notation)
- [x] `rio config set <path> <value>` — write config values
- [x] `rio config show` — pretty-print full config
- [x] `rio doctor` — validate config, check deps, test API keys

#### 1.3 Structured Error Logging
- [x] Create `rio/local/rio_logging.py` — subsystem logger + file output
- [x] Error classification: categories, severity, actionable suggestions
- [x] Log file rotation: `rio/logs/rio-YYYY-MM-DD.log`
- [x] Console formatting: colored severity, subsystem tags

#### 1.4 Multi-Platform Detection
- [x] Create `rio/local/platform_utils.py` — OS detection, path resolution
- [x] Platform-specific browser launch, screen interaction, hotkeys
- [x] Graceful degradation when platform features unavailable

### Phase 2: Dashboard Config UI (NEXT)

#### 2.1 Config API Endpoints
- [x] `GET /api/config` — return full config as JSON
- [x] `POST /api/config` — update config fields  
- [ ] `GET /api/config/schema` — return config schema for form generation
- [x] `GET /api/doctor` — run diagnostics and return results
- [x] `GET /api/models/status` — check each model's availability

#### 2.2 Dashboard Settings Page
- [ ] New `settings.html` page in dashboard
- [ ] Config editor: models, audio, vision, skills sections
- [ ] API key input with validation feedback
- [ ] Model status indicators (green/yellow/red)
- [ ] Save/apply/reset buttons

#### 2.3 Task Monitor Panel
- [ ] Real-time task execution view in dashboard
- [ ] Step-by-step progress with status icons
- [ ] Task history with filtering
- [ ] Cancel/pause/resume from dashboard

### Phase 3: Browser & Screen Navigation Fixes

#### 3.1 Playwright Reliability
- [ ] Add proper page wait strategies (networkidle, domcontentloaded)
- [ ] Implement screenshot comparison for change detection
- [ ] Add viewport auto-sizing for different screen resolutions
- [ ] Timeout handling with partial result return

#### 3.2 Selenium Fallback
- [ ] Add selenium-wire as backup browser automation
- [ ] Auto-detect: Playwright → Selenium → system browser fallback
- [ ] Shared interface for both engines

#### 3.3 Screen Navigation Improvements
- [ ] OCR-assisted element detection (rapidocr integration)
- [ ] Set-of-Mark overlay for visual grounding
- [ ] Coordinate calibration tool in dashboard
- [ ] Action replay/undo capability

### Phase 4: Advanced Features

#### 4.1 Interactive Onboarding (`rio configure`)
- [ ] Step-by-step wizard: API key → model selection → audio → vision
- [ ] Validate each step before proceeding
- [ ] Generate config.yaml from wizard answers
- [ ] Test connection to cloud server

#### 4.2 Skill Management
- [ ] `rio skills list/enable/disable`
- [ ] Skill marketplace (future)
- [ ] Custom skill creation template

#### 4.3 Session Management
- [ ] Task history persistence
- [ ] Session export/import
- [ ] Crash recovery improvements

---

## Files Created/Modified (Phase 1)

### New Files
- `rio/local/model_fallback.py` — Model cascade + error classification
- `rio/local/rio_logging.py` — Structured logging system
- `rio/local/platform_utils.py` — Cross-platform utilities
- `rio/cli.py` — CLI entry point (`rio config/doctor/run/configure`)
- `rio/tasks/RIO_MASTER_PLAN.md` — This file

### Modified Files
- `rio/local/orchestrator.py` — Wired model fallback chain into `plan_task()`, added error diagnostics
- `rio/local/config.py` — Added `LoggingConfig`, `ModelConfig.fallback_chain/cooldown_seconds/timeout_seconds`
- `rio/config.yaml` — Added `models.fallback_chain`, `logging` section
- `rio/local/main.py` — Added rio_logging + platform_utils initialization at startup
- `rio/cloud/main.py` — Added `/api/config` (GET/POST), `/api/doctor`, `/api/models/status` endpoints

### Modified Files
- `rio/local/orchestrator.py` — Wire model fallback into plan_task()
- `rio/local/config.py` — Add fallback_chain to ModelConfig
- `rio/config.yaml` — Add fallback and logging config
- `rio/local/main.py` — Wire new logging system
- `rio/cloud/main.py` — Add config API endpoints

---

## Design Decisions

1. **YAML stays** — Don't migrate to JSON5. YAML is more familiar to Python users and Rio's audience.
2. **CLI uses Python** — No Node.js dependency. `rio` CLI is a Python script using argparse.
3. **Fallback is local-first** — Model cascade happens in local client, not cloud. Cloud relays.
4. **Logging to file** — Not just console. Structured logs in `rio/logs/` for debugging.
5. **Platform detection at startup** — Detect once, cache in config, use throughout session.
6. **Dashboard config reads/writes config.yaml** — Single source of truth. No separate DB for config.
