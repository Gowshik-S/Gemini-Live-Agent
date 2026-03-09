# Cloud Service

This folder contains the FastAPI cloud relay that connects local Rio clients to Gemini sessions.

Primary files:
- `main.py`: FastAPI app, WebSocket relay, dashboard endpoints.
- `gemini_session.py`: Gemini live/text session handling.
- `session_manager.py`: multi-session lifecycle management.
- `rate_limiter.py`: request throttling.
- `model_router.py`: model selection logic.

Deployment artifacts:
- `Dockerfile`
- `service.yaml`

