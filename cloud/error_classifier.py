"""
Error classification and recovery strategy selection for Rio.

Categorizes exceptions into known error types and provides
appropriate recovery strategies (retry, backoff, compact, fail_fast).
Used by the ToolOrchestrator to handle errors intelligently (C1).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorCategory(str, Enum):
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    CONTEXT_OVERFLOW = "context_overflow"
    TIMEOUT = "timeout"
    TOOL_FAILURE = "tool_failure"
    MODEL_ERROR = "model_error"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class RecoveryStrategy:
    category: ErrorCategory
    action: str          # "retry", "backoff", "compact", "fail_fast", "notify"
    delay_seconds: float  # Delay before retry (0 = immediate)
    max_retries: int      # Max retries for this error type
    message: str          # Human-readable description


_STRATEGIES: dict[ErrorCategory, RecoveryStrategy] = {
    ErrorCategory.RATE_LIMIT: RecoveryStrategy(
        ErrorCategory.RATE_LIMIT, "backoff", 5.0, 3,
        "Rate limited — backing off"),
    ErrorCategory.AUTH: RecoveryStrategy(
        ErrorCategory.AUTH, "fail_fast", 0, 0,
        "Authentication error — check API key"),
    ErrorCategory.CONTEXT_OVERFLOW: RecoveryStrategy(
        ErrorCategory.CONTEXT_OVERFLOW, "compact", 0, 1,
        "Context overflow — compacting and retrying"),
    ErrorCategory.TIMEOUT: RecoveryStrategy(
        ErrorCategory.TIMEOUT, "retry", 2.0, 2,
        "Timeout — retrying with extended deadline"),
    ErrorCategory.TOOL_FAILURE: RecoveryStrategy(
        ErrorCategory.TOOL_FAILURE, "retry", 0, 1,
        "Tool error — retrying once"),
    ErrorCategory.MODEL_ERROR: RecoveryStrategy(
        ErrorCategory.MODEL_ERROR, "backoff", 3.0, 2,
        "Model error — retrying"),
    ErrorCategory.NETWORK: RecoveryStrategy(
        ErrorCategory.NETWORK, "backoff", 2.0, 3,
        "Network error — retrying"),
    ErrorCategory.UNKNOWN: RecoveryStrategy(
        ErrorCategory.UNKNOWN, "retry", 1.0, 1,
        "Unknown error — retrying once"),
}


def classify_error(error: Exception) -> ErrorCategory:
    """Classify an exception into an error category."""
    msg = str(error).lower()
    err_type = type(error).__name__

    if "429" in msg or "rate" in msg or "quota" in msg or "resource_exhausted" in msg:
        return ErrorCategory.RATE_LIMIT
    if "401" in msg or "403" in msg or "api key" in msg or "unauthorized" in msg:
        return ErrorCategory.AUTH
    if "context" in msg and ("length" in msg or "overflow" in msg or "too long" in msg):
        return ErrorCategory.CONTEXT_OVERFLOW
    if "timeout" in err_type.lower() or "timeout" in msg:
        return ErrorCategory.TIMEOUT
    if "500" in msg or "internal" in msg or "server error" in msg:
        return ErrorCategory.MODEL_ERROR
    if "connect" in msg or "network" in msg or "dns" in msg:
        return ErrorCategory.NETWORK
    return ErrorCategory.UNKNOWN


def get_strategy(category: ErrorCategory) -> RecoveryStrategy:
    """Get the recovery strategy for an error category."""
    return _STRATEGIES.get(category, _STRATEGIES[ErrorCategory.UNKNOWN])
