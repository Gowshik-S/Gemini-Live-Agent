"""
Rio Local — Model Fallback System

Provides intelligent model cascade when the primary model (Gemini Pro) fails.
Inspired by OpenClaw's FailoverError + auth profile rotation pattern.

Fallback chain: Pro → Flash → offline error with detailed diagnostics.

Error categories (from OpenClaw):
  - auth:             Invalid/expired API key
  - auth_permanent:   403 — blocked account
  - billing:          402 — out of credits
  - rate_limit:       429 — too many requests
  - timeout:          408 — slow response
  - model_not_found:  404 — invalid model name
  - format:           400 — invalid request (not retried)
  - network:          Connection error
  - unknown:          Catch-all
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

import structlog

log = structlog.get_logger(__name__)


class FailoverReason(str, Enum):
    AUTH = "auth"
    AUTH_PERMANENT = "auth_permanent"
    BILLING = "billing"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    MODEL_NOT_FOUND = "model_not_found"
    FORMAT = "format"
    NETWORK = "network"
    UNKNOWN = "unknown"


# Reasons that should NOT trigger fallback (re-raise immediately)
_NO_FALLBACK_REASONS = {FailoverReason.FORMAT}

# Reasons where waiting + retry on SAME model makes sense
_RETRY_SAME_REASONS = {FailoverReason.RATE_LIMIT, FailoverReason.TIMEOUT}


class ModelFailoverError(Exception):
    """Raised when a model call fails with classifiable reason."""

    def __init__(
        self,
        reason: FailoverReason,
        message: str = "",
        model: str = "",
        status: int = 0,
        original: Optional[Exception] = None,
    ):
        self.reason = reason
        self.model = model
        self.status = status
        self.original = original
        super().__init__(message or f"Model {model} failed: {reason.value}")

    def __repr__(self) -> str:
        return (f"ModelFailoverError(reason={self.reason.value}, "
                f"model={self.model!r}, status={self.status})")


@dataclass
class FallbackAttempt:
    """Record of a single fallback attempt."""
    model: str
    reason: Optional[FailoverReason] = None
    success: bool = False
    latency_ms: float = 0.0
    error: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelCooldown:
    """Tracks cooldown state for a model after failure."""
    model: str
    reason: FailoverReason
    until: float  # Unix timestamp when cooldown expires
    failure_count: int = 1


def classify_error(exc: Exception, model: str = "") -> ModelFailoverError:
    """Classify an exception into a ModelFailoverError with actionable reason.

    Maps HTTP status codes and error message patterns to FailoverReason.
    """
    msg = str(exc).lower()
    status = 0

    # Extract HTTP status from common error patterns
    if hasattr(exc, "status_code"):
        status = exc.status_code
    elif hasattr(exc, "code"):
        try:
            status = int(exc.code)
        except (ValueError, TypeError):
            pass
    # google-genai wraps status in the message
    for code in (401, 403, 402, 429, 408, 400, 404, 410, 500, 503):
        if str(code) in msg:
            status = code
            break

    # Classify by status code first
    if status == 401:
        reason = FailoverReason.AUTH
    elif status == 403:
        reason = FailoverReason.AUTH_PERMANENT
    elif status == 402:
        reason = FailoverReason.BILLING
    elif status == 429:
        reason = FailoverReason.RATE_LIMIT
    elif status in (408, 504):
        reason = FailoverReason.TIMEOUT
    elif status == 404:
        reason = FailoverReason.MODEL_NOT_FOUND
    elif status == 400:
        reason = FailoverReason.FORMAT
    elif status in (500, 503):
        reason = FailoverReason.RATE_LIMIT  # Server overload → treat as rate limit
    # Classify by message patterns
    elif "api key" in msg or "api_key" in msg or "unauthorized" in msg:
        reason = FailoverReason.AUTH
    elif "permission" in msg or "forbidden" in msg:
        reason = FailoverReason.AUTH_PERMANENT
    elif "billing" in msg or "quota" in msg or "payment" in msg:
        reason = FailoverReason.BILLING
    elif "rate" in msg or "limit" in msg or "429" in msg or "resource exhausted" in msg:
        reason = FailoverReason.RATE_LIMIT
    elif "timeout" in msg or "timed out" in msg or "deadline" in msg:
        reason = FailoverReason.TIMEOUT
    elif "not found" in msg or "does not exist" in msg or "invalid model" in msg:
        reason = FailoverReason.MODEL_NOT_FOUND
    elif "connection" in msg or "network" in msg or "dns" in msg or "refused" in msg:
        reason = FailoverReason.NETWORK
    else:
        reason = FailoverReason.UNKNOWN

    return ModelFailoverError(
        reason=reason,
        message=str(exc),
        model=model,
        status=status,
        original=exc,
    )


def get_diagnostic_message(error: ModelFailoverError) -> str:
    """Return a human-readable diagnostic message with fix suggestions.

    Printed to logs so the user knows exactly what to configure.
    """
    diag = {
        FailoverReason.AUTH: (
            f"[AUTH ERROR] Model '{error.model}' — Invalid or expired API key.\n"
            "  Fix: Set a valid GEMINI_API_KEY in rio/cloud/.env\n"
            "  Or:  rio config set models.api_key <your-key>\n"
            "  Get a key at: https://aistudio.google.com/apikey"
        ),
        FailoverReason.AUTH_PERMANENT: (
            f"[AUTH BLOCKED] Model '{error.model}' — Account blocked or forbidden.\n"
            "  Fix: Check your Google Cloud console for account status.\n"
            "  This is permanent — the API key's account cannot access this model."
        ),
        FailoverReason.BILLING: (
            f"[BILLING ERROR] Model '{error.model}' — Out of credits or billing not enabled.\n"
            "  Fix: Enable billing in Google Cloud Console.\n"
            "  Or:  Use the free tier models (Flash) via: rio config set models.primary gemini-2.5-flash\n"
            "  Setup billing: https://console.cloud.google.com/billing"
        ),
        FailoverReason.RATE_LIMIT: (
            f"[RATE LIMITED] Model '{error.model}' — Too many requests.\n"
            "  Fix: Wait a moment, or reduce request frequency.\n"
            "  Or:  Lower pro_rpm_budget: rio config set models.pro_rpm_budget 2\n"
            "  The agent will automatically retry with backoff."
        ),
        FailoverReason.TIMEOUT: (
            f"[TIMEOUT] Model '{error.model}' — Response too slow.\n"
            "  Fix: Check internet connection.\n"
            "  Or:  Increase timeout: rio config set models.timeout_seconds 60\n"
            "  Falling back to faster model."
        ),
        FailoverReason.MODEL_NOT_FOUND: (
            f"[MODEL NOT FOUND] Model '{error.model}' — Does not exist or not available.\n"
            "  Fix: Check model name in config.yaml → models.primary / models.secondary\n"
            "  Available models: gemini-2.5-flash, gemini-2.5-pro-preview-03-25\n"
            "  Or:  rio config set models.primary gemini-2.5-flash"
        ),
        FailoverReason.FORMAT: (
            f"[FORMAT ERROR] Model '{error.model}' — Invalid request format.\n"
            "  This is a bug in the request construction. Check logs for details.\n"
            f"  Error: {str(error.original)[:200]}"
        ),
        FailoverReason.NETWORK: (
            f"[NETWORK ERROR] Model '{error.model}' — Cannot reach API server.\n"
            "  Fix: Check internet connection.\n"
            "  Fix: Check if firewall/proxy blocks googleapis.com\n"
            "  Fix: Verify cloud_url in config.yaml"
        ),
        FailoverReason.UNKNOWN: (
            f"[UNKNOWN ERROR] Model '{error.model}' — Unexpected failure.\n"
            f"  Error: {str(error.original)[:200]}\n"
            "  Check rio/logs/ for full stack trace."
        ),
    }
    return diag.get(error.reason, f"[ERROR] {error}")


class ModelFallbackChain:
    """Manages model fallback with cooldown tracking.

    Usage::

        chain = ModelFallbackChain(
            primary="gemini-2.5-pro-preview-03-25",
            fallbacks=["gemini-2.5-flash"],
        )
        result = await chain.call_with_fallback(my_async_func)
    """

    def __init__(
        self,
        primary: str,
        fallbacks: Optional[list[str]] = None,
        cooldown_seconds: float = 60.0,
        max_retries_per_model: int = 2,
    ):
        self.primary = primary
        self.fallbacks = fallbacks or []
        self.cooldown_seconds = cooldown_seconds
        self.max_retries_per_model = max_retries_per_model
        self._cooldowns: dict[str, ModelCooldown] = {}
        self._attempts: list[FallbackAttempt] = []
        self._log = log.bind(component="model_fallback")

    def _is_in_cooldown(self, model: str) -> bool:
        """Check if a model is currently in cooldown."""
        cd = self._cooldowns.get(model)
        if cd is None:
            return False
        if time.time() >= cd.until:
            del self._cooldowns[model]
            return False
        return True

    def _mark_cooldown(self, model: str, reason: FailoverReason) -> None:
        """Put a model in cooldown after failure."""
        # Permanent failures get longer cooldown
        if reason in (FailoverReason.AUTH_PERMANENT, FailoverReason.BILLING):
            duration = self.cooldown_seconds * 10  # 10x normal
        elif reason == FailoverReason.RATE_LIMIT:
            duration = self.cooldown_seconds
        elif reason == FailoverReason.MODEL_NOT_FOUND:
            duration = self.cooldown_seconds * 60  # Basically permanent for session
        else:
            duration = self.cooldown_seconds

        existing = self._cooldowns.get(model)
        if existing:
            existing.until = time.time() + duration
            existing.failure_count += 1
        else:
            self._cooldowns[model] = ModelCooldown(
                model=model, reason=reason,
                until=time.time() + duration,
            )

    def get_available_models(self) -> list[str]:
        """Return ordered list of models not in cooldown."""
        all_models = [self.primary] + self.fallbacks
        return [m for m in all_models if not self._is_in_cooldown(m)]

    async def call_with_fallback(
        self,
        func: Callable[..., Coroutine],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call func with model fallback.

        func must accept a `model` keyword argument.
        On failure, classifies error and cascades to next available model.

        Returns the result from the first successful call.
        Raises ModelFailoverError if all models exhausted.
        """
        models = self.get_available_models()
        if not models:
            # All models in cooldown — find the one that expires soonest
            all_models = [self.primary] + self.fallbacks
            soonest = min(
                (self._cooldowns[m].until for m in all_models if m in self._cooldowns),
                default=time.time(),
            )
            wait_time = max(0, soonest - time.time())
            self._log.warning("fallback.all_in_cooldown",
                              models=all_models,
                              wait_seconds=round(wait_time, 1))
            raise ModelFailoverError(
                reason=FailoverReason.RATE_LIMIT,
                message=f"All models in cooldown. Try again in {wait_time:.0f}s",
                model="all",
            )

        last_error: Optional[ModelFailoverError] = None

        for model in models:
            attempt = FallbackAttempt(model=model)
            start = time.time()

            for retry in range(self.max_retries_per_model):
                try:
                    result = await func(*args, model=model, **kwargs)
                    attempt.success = True
                    attempt.latency_ms = (time.time() - start) * 1000
                    self._attempts.append(attempt)

                    # Clear cooldown on success
                    self._cooldowns.pop(model, None)

                    if model != self.primary:
                        self._log.info("fallback.used_fallback",
                                       model=model,
                                       primary=self.primary)
                    return result

                except Exception as exc:
                    error = classify_error(exc, model=model)
                    attempt.reason = error.reason
                    attempt.error = str(exc)[:200]
                    last_error = error

                    # Log diagnostic
                    diag = get_diagnostic_message(error)
                    self._log.warning("fallback.model_failed",
                                      model=model,
                                      reason=error.reason.value,
                                      retry=retry + 1,
                                      diagnostic=diag)
                    print(f"\n  {diag}")

                    # Don't retry for certain reasons
                    if error.reason in _NO_FALLBACK_REASONS:
                        raise error

                    if error.reason not in _RETRY_SAME_REASONS:
                        break  # Move to next model

                    # Rate limit / timeout — wait before retry
                    wait = min(2 ** retry, 8)
                    self._log.info("fallback.retry_wait", seconds=wait)
                    await asyncio.sleep(wait)

            # Model failed all retries — mark cooldown
            attempt.latency_ms = (time.time() - start) * 1000
            self._attempts.append(attempt)
            self._mark_cooldown(model, last_error.reason if last_error else FailoverReason.UNKNOWN)

        # All models exhausted
        if last_error:
            raise last_error
        raise ModelFailoverError(
            reason=FailoverReason.UNKNOWN,
            message="All models failed",
            model="all",
        )

    def get_stats(self) -> dict:
        """Return fallback chain statistics for dashboard/diagnostics."""
        return {
            "primary": self.primary,
            "fallbacks": self.fallbacks,
            "cooldowns": {
                m: {
                    "reason": cd.reason.value,
                    "expires_in": max(0, round(cd.until - time.time())),
                    "failures": cd.failure_count,
                }
                for m, cd in self._cooldowns.items()
            },
            "recent_attempts": [
                {
                    "model": a.model,
                    "success": a.success,
                    "reason": a.reason.value if a.reason else None,
                    "latency_ms": round(a.latency_ms),
                }
                for a in self._attempts[-10:]
            ],
        }
