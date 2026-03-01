"""
Rio ML — User Model Manager

Per-user model lifecycle management:
    - Creates/loads per-user ensemble models from pkl files
    - Fetches training data from SQLite pattern DB (rio_patterns.db)
    - Performs online learning after each session
    - Auto-labels training data using heuristic rules
    - Generates context strings for Gemini prompt injection

This is the top-level orchestrator that ties together:
    FeatureExtractor → RioEnsembleModel → pkl files → DB

Usage::

    manager = UserModelManager(db_path="rio_patterns.db")
    
    # Real-time prediction during session
    pred = manager.predict_from_message("fix this TypeError", time.time())
    context = manager.get_context_for_gemini()
    
    # Record new data (auto-learns)
    manager.record_interaction("user", "how do I fix this?", time.time())
    manager.record_struggle(0.9, ["repeated_error"])
    
    # End of session — train on accumulated data
    manager.train_on_session()
    manager.save()
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from .feature_engine import FeatureExtractor, RawInteractionData, FEATURE_DIM
from .ensemble_model import (
    RioEnsembleModel,
    EnsemblePrediction,
    ALL_TARGETS,
    STRUGGLE_CLASSES,
    STYLE_CLASSES,
    ENGAGEMENT_CLASSES,
    MOOD_CLASSES,
    SKLEARN_AVAILABLE,
)

log = structlog.get_logger(__name__)

# Paths
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # rio/
DEFAULT_DB_PATH = os.path.join(_BASE_DIR, "rio_patterns.db")
DEFAULT_MODELS_DIR = os.path.join(_BASE_DIR, "ml", "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODELS_DIR, "user_default.pkl")

# Minimum samples before training
MIN_SAMPLES_FOR_TRAINING = 5
# How often to re-train (seconds) — every 30 min
RETRAIN_INTERVAL = 1800


# ---------------------------------------------------------------------------
# Heuristic auto-labeling
# ---------------------------------------------------------------------------

def _auto_label_struggle(confidences: list[float]) -> str:
    """Label struggle risk from confidence scores."""
    if not confidences:
        return "low"
    avg = np.mean(confidences)
    if avg >= 0.7:
        return "high"
    elif avg >= 0.4:
        return "medium"
    return "low"


def _auto_label_style(messages: list[str]) -> str:
    """Label chat style from message lengths."""
    if not messages:
        return "moderate"
    avg_words = np.mean([len(m.split()) for m in messages])
    if avg_words < 5:
        return "concise"
    elif avg_words > 20:
        return "verbose"
    return "moderate"


def _auto_label_engagement(
    msg_count: int,
    session_duration_min: float,
    help_accepted: list[bool],
) -> str:
    """Label engagement level."""
    if msg_count < 3 or session_duration_min < 2:
        return "passive"
    rate = msg_count / max(1, session_duration_min)
    if rate > 1.0 or (help_accepted and sum(help_accepted) / len(help_accepted) > 0.7):
        return "power_user"
    return "active"


def _auto_label_mood(
    error_count: int,
    struggle_count: int,
    msg_count: int,
    messages: list[str],
) -> str:
    """Label mood based on error/struggle frequency and language."""
    if msg_count == 0:
        return "neutral"

    # Check for frustration signals in messages
    frustration_words = [
        "wtf", "broken", "doesn't work", "still not working", "ugh",
        "annoying", "help", "stuck", "confused", "why", "again",
    ]
    frustration_count = 0
    for m in messages:
        ml = m.lower()
        frustration_count += sum(1 for w in frustration_words if w in ml)

    error_ratio = error_count / max(1, msg_count)
    struggle_ratio = struggle_count / max(1, msg_count)

    if error_ratio > 0.3 or struggle_ratio > 0.2 or frustration_count > 3:
        return "frustrated"
    elif error_ratio < 0.05 and frustration_count == 0:
        return "calm"
    return "neutral"


class UserModelManager:
    """Manages per-user ML models with online learning.
    
    Each user gets their own pkl model file that evolves over time.
    The model starts with cold-start defaults and improves with each session.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        models_dir: str = DEFAULT_MODELS_DIR,
        user_id: str = "default",
    ) -> None:
        self.user_id = user_id
        self.db_path = os.path.abspath(db_path)
        self.models_dir = os.path.abspath(models_dir)
        self.model_path = os.path.join(self.models_dir, f"user_{user_id}.pkl")

        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)

        # Feature extractor
        self.extractor = FeatureExtractor()

        # Load or create model
        self.model: Optional[RioEnsembleModel] = None
        if SKLEARN_AVAILABLE:
            self._load_or_create_model()
        else:
            log.warning("ml.manager.no_sklearn", note="ML predictions disabled")

        # Session accumulator — collects data during current session
        self._session_messages: list[tuple[str, str, float]] = []  # (speaker, text, timestamp)
        self._session_errors: list[str] = []
        self._session_struggles: list[float] = []
        self._session_struggle_signals: list[list[str]] = []
        self._session_help_accepted: list[bool] = []
        self._session_languages: list[str] = []
        self._session_start = time.time()
        self._last_train_time = 0.0
        self._last_prediction: Optional[EnsemblePrediction] = None

        # DB connection for reading historical data
        self._db: Optional[sqlite3.Connection] = None
        if os.path.exists(self.db_path):
            try:
                self._db = sqlite3.connect(self.db_path, check_same_thread=False)
                self._db.execute("PRAGMA journal_mode=WAL")
            except Exception:
                log.warning("ml.manager.db_open_failed", path=self.db_path)

        log.info(
            "ml.manager.init",
            user_id=user_id,
            model_path=self.model_path,
            model_loaded=self.model is not None,
            db_connected=self._db is not None,
        )

    def _load_or_create_model(self) -> None:
        """Load existing model from pkl or create a new one."""
        if os.path.exists(self.model_path):
            try:
                self.model = RioEnsembleModel.load(self.model_path)
                log.info("ml.manager.model_loaded", path=self.model_path)
                return
            except Exception as e:
                log.warning("ml.manager.model_load_failed", error=str(e))

        # Create new model
        self.model = RioEnsembleModel()
        log.info("ml.manager.model_created", user_id=self.user_id)

    # ------------------------------------------------------------------
    # Recording methods (called during session)
    # ------------------------------------------------------------------

    def record_interaction(self, speaker: str, text: str, timestamp: float) -> None:
        """Record a message during the current session."""
        self._session_messages.append((speaker, text, timestamp))

        # Detect languages in user messages
        if speaker == "user":
            langs = self.extractor.detect_languages_in_text(text)
            self._session_languages.extend(langs)

        # Check if we should do incremental training
        if (
            self.model is not None
            and len(self._session_messages) % 10 == 0  # every 10 messages
            and time.time() - self._last_train_time > 60  # at most every 60s
        ):
            self._incremental_train()

    def record_error(self, error_text: str) -> None:
        """Record an error during the current session."""
        self._session_errors.append(error_text)

    def record_struggle(self, confidence: float, signals: list[str]) -> None:
        """Record a struggle detection event."""
        self._session_struggles.append(confidence)
        self._session_struggle_signals.append(signals)

    def record_help_response(self, accepted: bool) -> None:
        """Record whether the user accepted proactive help."""
        self._session_help_accepted.append(accepted)

    # ------------------------------------------------------------------
    # Prediction methods
    # ------------------------------------------------------------------

    def predict_current(self) -> Optional[EnsemblePrediction]:
        """Predict based on current session data accumulated so far."""
        if self.model is None or not self._session_messages:
            return None

        raw = self._build_raw_data_from_session()
        features = self.extractor.extract(raw)
        pred = self.model.predict(features)
        self._last_prediction = pred
        return pred

    def predict_from_message(self, message: str, timestamp: float) -> Optional[EnsemblePrediction]:
        """Quick prediction from a single message (combines with session context)."""
        if self.model is None:
            return None

        # Add to session first
        self.record_interaction("user", message, timestamp)

        # Predict using full session context
        return self.predict_current()

    def get_context_for_gemini(self) -> str:
        """Generate a context string for Gemini prompt injection.
        
        Combines ML predictions with behavioral patterns into a natural
        language string that Gemini can use to personalize responses.
        """
        parts = []

        # Get latest prediction
        pred = self._last_prediction or self.predict_current()
        if pred is not None:
            # Struggle risk
            if pred.struggle_risk == "high":
                parts.append(
                    f"[ML] User struggle risk: HIGH ({pred.struggle_risk_proba.get('high', 0):.0%}). "
                    "Be extra patient and proactive with help."
                )
            elif pred.struggle_risk == "medium":
                parts.append("[ML] User struggle risk: moderate. Watch for signs of frustration.")

            # Chat style adaptation
            if pred.chat_style == "concise":
                parts.append("[ML] User prefers concise responses. Keep answers short and direct.")
            elif pred.chat_style == "verbose":
                parts.append("[ML] User prefers detailed explanations. Provide thorough answers.")

            # Engagement level
            if pred.engagement == "power_user":
                parts.append("[ML] Power user detected. Skip basics, focus on advanced solutions.")
            elif pred.engagement == "passive":
                parts.append("[ML] User is relatively passive. Be more inviting and ask clarifying questions.")

            # Mood
            if pred.mood == "frustrated":
                parts.append("[ML] User seems frustrated. Be empathetic and solution-focused.")
            elif pred.mood == "calm":
                parts.append("[ML] User is calm and focused. Match their tone.")

        # Add language preferences from session
        if self._session_languages:
            from collections import Counter
            top_langs = Counter(self._session_languages).most_common(3)
            lang_str = ", ".join(f"{lang}" for lang, _ in top_langs)
            parts.append(f"[ML] User is working with: {lang_str}")

        # Error patterns
        if len(self._session_errors) > 2:
            parts.append(f"[ML] User has encountered {len(self._session_errors)} errors this session.")

        return "\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Training methods
    # ------------------------------------------------------------------

    def _incremental_train(self) -> None:
        """Perform incremental training on current session data."""
        if self.model is None:
            return

        raw = self._build_raw_data_from_session()
        features = self.extractor.extract(raw)

        # Auto-generate labels from heuristics
        user_msgs = [m[1] for m in self._session_messages if m[0] == "user"]
        duration_min = (time.time() - self._session_start) / 60.0

        labels = {
            "struggle_risk": np.array([_auto_label_struggle(self._session_struggles)]),
            "chat_style": np.array([_auto_label_style(user_msgs)]),
            "engagement": np.array([_auto_label_engagement(
                len(user_msgs), duration_min, self._session_help_accepted
            )]),
            "mood": np.array([_auto_label_mood(
                len(self._session_errors), len(self._session_struggles),
                len(user_msgs), user_msgs
            )]),
        }

        try:
            self.model.partial_fit(features.reshape(1, -1), labels)
            self._last_train_time = time.time()
            log.debug("ml.incremental_train", messages=len(self._session_messages))
        except Exception as e:
            log.warning("ml.incremental_train.failed", error=str(e))

    def train_on_session(self) -> Optional[dict]:
        """Train on the complete current session (called at session end).
        
        Returns training stats or None if training skipped.
        """
        if self.model is None:
            return None

        user_msgs = [m[1] for m in self._session_messages if m[0] == "user"]
        if len(user_msgs) < MIN_SAMPLES_FOR_TRAINING:
            log.info("ml.train_on_session.skipped", reason="too_few_messages", count=len(user_msgs))
            return None

        raw = self._build_raw_data_from_session()
        features = self.extractor.extract(raw)
        duration_min = (time.time() - self._session_start) / 60.0

        labels = {
            "struggle_risk": np.array([_auto_label_struggle(self._session_struggles)]),
            "chat_style": np.array([_auto_label_style(user_msgs)]),
            "engagement": np.array([_auto_label_engagement(
                len(user_msgs), duration_min, self._session_help_accepted
            )]),
            "mood": np.array([_auto_label_mood(
                len(self._session_errors), len(self._session_struggles),
                len(user_msgs), user_msgs
            )]),
        }

        try:
            self.model.partial_fit(features.reshape(1, -1), labels)
            log.info(
                "ml.train_on_session",
                messages=len(self._session_messages),
                user_msgs=len(user_msgs),
                labels={k: v[0] for k, v in labels.items()},
            )
            return {
                "messages": len(self._session_messages),
                "user_messages": len(user_msgs),
                "labels": {k: v[0] for k, v in labels.items()},
                "duration_min": duration_min,
            }
        except Exception as e:
            log.exception("ml.train_on_session.failed")
            return None

    def train_on_history(self, days: int = 30) -> Optional[dict]:
        """Train on historical data from the DB (batch training).
        
        Fetches past session data from rio_patterns.db, generates features
        and labels, then does a full fit or large partial_fit.
        
        Returns training stats or None.
        """
        if self.model is None or self._db is None:
            return None

        cutoff = time.time() - days * 86400
        try:
            # Fetch historical data in chunks by session
            rows = self._db.execute(
                "SELECT DISTINCT session_id FROM activities WHERE timestamp > ? AND event_type = 'session_start'",
                (cutoff,),
            ).fetchall()
        except Exception:
            log.warning("ml.train_history.no_activity_table")
            return None

        if not rows:
            return None

        all_features = []
        all_labels = {k: [] for k in ALL_TARGETS}

        for (session_id,) in rows:
            raw = self._fetch_session_data(session_id, cutoff)
            if not raw.user_messages or len(raw.user_messages) < 2:
                continue

            features = self.extractor.extract(raw)
            all_features.append(features)

            user_msgs = raw.user_messages
            duration_min = 0.0
            if raw.session_durations:
                duration_min = np.mean(raw.session_durations) / 60.0

            all_labels["struggle_risk"].append(_auto_label_struggle(raw.struggle_confidences))
            all_labels["chat_style"].append(_auto_label_style(user_msgs))
            all_labels["engagement"].append(_auto_label_engagement(
                len(user_msgs), duration_min, raw.help_accepted
            ))
            all_labels["mood"].append(_auto_label_mood(
                len(raw.error_texts), len(raw.struggle_confidences),
                len(user_msgs), user_msgs
            ))

        if len(all_features) < MIN_SAMPLES_FOR_TRAINING:
            log.info("ml.train_history.skipped", sessions=len(all_features))
            return None

        X = np.array(all_features)
        y = {k: np.array(v) for k, v in all_labels.items()}

        results = self.model.fit(X, y)
        log.info("ml.train_history.done", sessions=len(all_features), results=results)

        return {
            "sessions": len(all_features),
            "accuracies": results,
            "days": days,
        }

    # ------------------------------------------------------------------
    # DB data fetching
    # ------------------------------------------------------------------

    def _fetch_session_data(self, session_id: str, cutoff: float) -> RawInteractionData:
        """Fetch all interaction data for a session from the DB."""
        raw = RawInteractionData()
        if self._db is None:
            return raw

        try:
            # Messages (from chat store DB if available)
            chat_db_path = os.path.join(os.path.dirname(self.db_path), "rio_chats.db")
            if os.path.exists(chat_db_path):
                chat_conn = sqlite3.connect(chat_db_path, check_same_thread=False)
                try:
                    msgs = chat_conn.execute(
                        "SELECT speaker, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp",
                        (session_id,),
                    ).fetchall()
                    for speaker, content, ts in msgs:
                        raw.message_timestamps.append(ts)
                        if speaker == "user":
                            raw.user_messages.append(content)
                        elif speaker == "rio":
                            raw.rio_messages.append(content)
                finally:
                    chat_conn.close()

            # Errors
            errors = self._db.execute(
                "SELECT error_text, category, timestamp FROM errors WHERE timestamp > ? ORDER BY timestamp",
                (cutoff,),
            ).fetchall()
            for text, cat, ts in errors:
                raw.error_texts.append(text)
                if cat:
                    raw.error_categories.append(cat)

            # Struggles
            struggles = self._db.execute(
                "SELECT confidence, signals, timestamp FROM struggles WHERE timestamp > ? ORDER BY timestamp",
                (cutoff,),
            ).fetchall()
            for conf, sigs, ts in struggles:
                raw.struggle_confidences.append(conf)
                try:
                    raw.struggle_signals.append(json.loads(sigs) if sigs else [])
                except Exception:
                    raw.struggle_signals.append([])

            # Help responses
            responses = self._db.execute(
                "SELECT accepted FROM help_responses WHERE timestamp > ?",
                (cutoff,),
            ).fetchall()
            raw.help_accepted = [bool(r[0]) for r in responses]

            # Languages
            langs = self._db.execute(
                "SELECT language FROM language_detections WHERE timestamp > ?",
                (cutoff,),
            ).fetchall()
            raw.languages_detected = [r[0] for r in langs]

            # Session durations
            sessions = self._db.execute(
                "SELECT timestamp FROM activities WHERE event_type IN ('session_start', 'session_end') ORDER BY timestamp",
                (),
            ).fetchall()
            ts_list = [r[0] for r in sessions]
            for i in range(0, len(ts_list) - 1, 2):
                raw.session_durations.append(ts_list[i + 1] - ts_list[i])

        except Exception:
            log.debug("ml.fetch_session.partial_failure", session_id=session_id)

        return raw

    # ------------------------------------------------------------------
    # Session data builder
    # ------------------------------------------------------------------

    def _build_raw_data_from_session(self) -> RawInteractionData:
        """Build RawInteractionData from current session accumulator."""
        raw = RawInteractionData()

        for speaker, text, ts in self._session_messages:
            raw.message_timestamps.append(ts)
            if speaker == "user":
                raw.user_messages.append(text)
            elif speaker == "rio":
                raw.rio_messages.append(text)

        raw.error_texts = list(self._session_errors)
        raw.struggle_confidences = list(self._session_struggles)
        raw.struggle_signals = list(self._session_struggle_signals)
        raw.help_accepted = list(self._session_help_accepted)
        raw.languages_detected = list(self._session_languages)
        raw.session_durations = [time.time() - self._session_start]
        raw.session_timestamps = [self._session_start]

        return raw

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save the model to its pkl file."""
        if self.model is not None:
            try:
                self.model.save(self.model_path)
            except Exception:
                log.exception("ml.manager.save_failed")

    def close(self) -> None:
        """End-of-session cleanup: train, save, close DB."""
        # Final training on session data
        stats = self.train_on_session()
        if stats:
            log.info("ml.manager.session_trained", **stats)

        # Save model
        self.save()

        # Close DB
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass

        log.info("ml.manager.closed", user_id=self.user_id)

    # ------------------------------------------------------------------
    # Info / Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get comprehensive ML pipeline stats."""
        stats = {
            "user_id": self.user_id,
            "model_path": self.model_path,
            "model_exists": os.path.exists(self.model_path),
            "session_messages": len(self._session_messages),
            "session_errors": len(self._session_errors),
            "session_struggles": len(self._session_struggles),
            "session_languages": list(set(self._session_languages)),
            "session_duration_min": (time.time() - self._session_start) / 60.0,
            "sklearn_available": SKLEARN_AVAILABLE,
        }
        if self.model is not None:
            stats["model"] = self.model.get_stats()
        if self._last_prediction is not None:
            stats["last_prediction"] = {
                "struggle_risk": self._last_prediction.struggle_risk,
                "chat_style": self._last_prediction.chat_style,
                "engagement": self._last_prediction.engagement,
                "mood": self._last_prediction.mood,
                "confidence": self._last_prediction.confidence,
            }
        return stats
