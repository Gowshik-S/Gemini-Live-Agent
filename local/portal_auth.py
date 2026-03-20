"""Rio Local - Portal backend key validation helper.

This validates a local RIO API key against the portal backend before starting
an interactive session.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def _post_json(endpoint: str, payload: dict, timeout_seconds: float) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = response.getcode()
        data = json.loads(response.read().decode("utf-8"))
        return status, data if isinstance(data, dict) else {}


def _normalize_backend_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return ""
    if value.endswith("/"):
        return value[:-1]
    return value


def validate_rio_key(backend_url: str, rio_api_key: str, timeout_seconds: float = 8.0) -> dict:
    """Validate a RIO key via POST /api/session/validate.

    Returns a dict with keys:
      - ok: bool
      - valid: bool
      - status_code: int | None
      - tier: str | None
      - credits: dict | None
      - error: str | None
    """
    base = _normalize_backend_url(backend_url)
    key = (rio_api_key or "").strip()

    if not base:
        return {
            "ok": False,
            "valid": False,
            "status_code": None,
            "tier": None,
            "credits": None,
            "error": "Missing backend URL",
        }

    if not key:
        return {
            "ok": False,
            "valid": False,
            "status_code": None,
            "tier": None,
            "credits": None,
            "error": "Missing RIO API key",
        }

    endpoint = f"{base}/api/session/validate"

    try:
        status, payload = _post_json(endpoint, {"rio_api_key": key}, timeout_seconds)
        return {
            "ok": True,
            "valid": bool(payload.get("valid")),
            "status_code": status,
            "tier": payload.get("tier"),
            "credits": payload.get("credits"),
            "error": None,
        }
    except urllib.error.HTTPError as exc:
        error_payload = None
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            error_payload = None
        detail = error_payload.get("detail") if isinstance(error_payload, dict) else str(exc)
        return {
            "ok": False,
            "valid": False,
            "status_code": exc.code,
            "tier": None,
            "credits": None,
            "error": str(detail),
        }
    except Exception as exc:  # pragma: no cover - network path
        return {
            "ok": False,
            "valid": False,
            "status_code": None,
            "tier": None,
            "credits": None,
            "error": str(exc),
        }


def report_task_usage(
    backend_url: str,
    rio_api_key: str,
    credits_used: int,
    cu_used: int,
    task_description: str | None = None,
    timeout_seconds: float = 8.0,
) -> dict:
    """Report task usage via POST /api/session/task-complete.

    Returns a dict with keys:
      - ok: bool
      - status_code: int | None
      - new_balance: dict | None
      - error: str | None
    """
    base = _normalize_backend_url(backend_url)
    key = (rio_api_key or "").strip()
    if not base:
        return {
            "ok": False,
            "status_code": None,
            "new_balance": None,
            "error": "Missing backend URL",
        }
    if not key:
        return {
            "ok": False,
            "status_code": None,
            "new_balance": None,
            "error": "Missing RIO API key",
        }

    endpoint = f"{base}/api/session/task-complete"
    payload = {
        "rio_api_key": key,
        "credits_used": max(0, int(credits_used)),
        "cu_used": max(0, int(cu_used)),
        "task_description": (task_description or "autonomous-task")[:512],
    }

    try:
        status, data = _post_json(endpoint, payload, timeout_seconds)
        return {
            "ok": True,
            "status_code": status,
            "new_balance": data.get("new_balance") if isinstance(data, dict) else None,
            "error": None,
        }
    except urllib.error.HTTPError as exc:
        error_payload = None
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            error_payload = None
        detail = error_payload.get("detail") if isinstance(error_payload, dict) else str(exc)
        return {
            "ok": False,
            "status_code": exc.code,
            "new_balance": None,
            "error": str(detail),
        }
    except Exception as exc:  # pragma: no cover - network path
        return {
            "ok": False,
            "status_code": None,
            "new_balance": None,
            "error": str(exc),
        }
