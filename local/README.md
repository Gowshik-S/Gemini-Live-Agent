# Local Client

This folder contains the desktop-side Rio runtime.

Primary files:
- `main.py`: local runtime entrypoint.
- `orchestrator.py`: event and action coordination.
- `audio_io.py`, `vad.py`, `wake_word.py`: voice input pipeline.
- `screen_capture.py`, `screen_navigator.py`, `windows_agent.py`: screen and desktop automation.
- `tools.py`: local tool execution and safety checks.
- `chat_store.py`, `memory.py`, `task_state.py`: local persistence/state.

Runtime environment:
- `venv/` is local-only and ignored by Git.

