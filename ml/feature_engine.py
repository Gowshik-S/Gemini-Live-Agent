"""
Rio ML — Feature Engineering Pipeline

Extracts numerical feature vectors from raw user interaction data.
All features are designed for incremental/online learning compatibility.

Feature Groups:
    1. Temporal features  — time-of-day, day-of-week, session duration patterns
    2. Interaction features — message length, frequency, response patterns
    3. Language features   — programming language distribution, complexity
    4. Behavioral features — error rates, struggle patterns, help acceptance
    5. Chat style features — vocabulary diversity, formality, verbosity
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Temporal buckets
HOUR_BUCKETS = 6  # 4-hour windows: [0-3, 4-7, 8-11, 12-15, 16-19, 20-23]
DAY_BUCKETS = 7   # Mon-Sun

# Chat style markers
FORMAL_MARKERS = [
    "please", "could you", "would you", "kindly", "thank you", "thanks",
    "appreciate", "sorry", "excuse me", "pardon",
]
CASUAL_MARKERS = [
    "yo", "hey", "sup", "lol", "haha", "bruh", "dude", "gonna", "wanna",
    "idk", "nvm", "tbh", "imo", "btw", "omg", "wtf", "smh",
]
TECHNICAL_MARKERS = [
    "function", "class", "method", "variable", "bug", "error", "exception",
    "stack trace", "debug", "compile", "runtime", "api", "endpoint",
    "database", "query", "index", "algorithm", "git", "commit", "branch",
    "deploy", "docker", "kubernetes", "async", "thread", "mutex",
]

# Programming language detection patterns
LANG_PATTERNS = {
    "python":     [r"\.py\b", r"\bdef\s", r"\bimport\s", r"\bclass\s", r"\basync\s", r"\bawait\s"],
    "javascript": [r"\.js\b", r"\bconst\s", r"\blet\s", r"\bfunction\s", r"=>", r"\brequire\("],
    "typescript": [r"\.tsx?\b", r"\binterface\s", r"\btype\s", r":\s*string\b", r":\s*number\b"],
    "rust":       [r"\.rs\b", r"\bfn\s", r"\blet\s+mut\b", r"\bimpl\s", r"\bmatch\s"],
    "go":         [r"\.go\b", r"\bfunc\s", r"\bpackage\s", r"\bgo\s+func\b"],
    "java":       [r"\.java\b", r"\bpublic\s+class\b", r"\bprivate\s", r"\bvoid\s"],
    "c_cpp":      [r"\.[ch]pp?\b", r"#include", r"\bint\s+main\b", r"\bprintf\b"],
    "shell":      [r"\.sh\b", r"#!/bin/bash", r"\becho\s", r"\bexport\s"],
    "sql":        [r"\.sql\b", r"\bSELECT\s", r"\bINSERT\s", r"\bCREATE\s+TABLE\b"],
    "html_css":   [r"\.html?\b", r"\.css\b", r"<div", r"<span", r"\bdisplay:"],
}
NUM_LANGUAGES = len(LANG_PATTERNS)

# Error categories
ERROR_CATS = [
    "syntax", "type", "import", "runtime", "network",
    "permission", "memory", "build", "test", "git", "other",
]
NUM_ERROR_CATS = len(ERROR_CATS)

# Total feature vector length
FEATURE_DIM = (
    HOUR_BUCKETS          # 6  — time-of-day distribution
    + DAY_BUCKETS         # 7  — day-of-week distribution
    + 5                   # session_duration_min, msg_count, avg_msg_len, msg_frequency, input_speed
    + 3                   # chat style: formality, technicality, verbosity
    + NUM_LANGUAGES       # 10 — language distribution
    + NUM_ERROR_CATS      # 11 — error category distribution
    + 4                   # struggle_rate, help_accept_rate, error_rate, active_hours_ratio
    + 2                   # vocabulary_diversity (type-token ratio), avg_word_length
)
# Total: 6+7+5+3+10+11+4+2 = 48 features


@dataclass
class RawInteractionData:
    """Container for raw user interaction data from DB."""
    # Temporal
    session_timestamps: list[float] = field(default_factory=list)
    message_timestamps: list[float] = field(default_factory=list)
    session_durations: list[float] = field(default_factory=list)

    # Messages
    user_messages: list[str] = field(default_factory=list)
    rio_messages: list[str] = field(default_factory=list)

    # Errors & struggles
    error_texts: list[str] = field(default_factory=list)
    error_categories: list[str] = field(default_factory=list)
    struggle_confidences: list[float] = field(default_factory=list)
    struggle_signals: list[list[str]] = field(default_factory=list)

    # Help responses
    help_accepted: list[bool] = field(default_factory=list)

    # Languages detected
    languages_detected: list[str] = field(default_factory=list)


class FeatureExtractor:
    """Extracts feature vectors from raw user interaction data.
    
    Usage::
    
        extractor = FeatureExtractor()
        raw = RawInteractionData(
            user_messages=["fix this bug", "what's wrong?"],
            message_timestamps=[time.time(), time.time()],
            ...
        )
        features = extractor.extract(raw)  # np.ndarray of shape (48,)
        names = extractor.feature_names()  # list of feature names
    """

    def extract(self, data: RawInteractionData) -> np.ndarray:
        """Extract a complete feature vector from raw interaction data.
        
        Returns:
            numpy array of shape (FEATURE_DIM,), all values normalized to [0, 1] range.
        """
        features = np.zeros(FEATURE_DIM, dtype=np.float64)
        idx = 0

        # --- 1. Temporal features (6 + 7 = 13) ---
        hour_dist = self._hour_distribution(data.message_timestamps)
        features[idx:idx + HOUR_BUCKETS] = hour_dist
        idx += HOUR_BUCKETS

        day_dist = self._day_distribution(data.message_timestamps)
        features[idx:idx + DAY_BUCKETS] = day_dist
        idx += DAY_BUCKETS

        # --- 2. Interaction features (5) ---
        # Average session duration in minutes (capped at 1.0 = 120 min)
        if data.session_durations:
            avg_dur = np.mean(data.session_durations) / 60.0
            features[idx] = min(1.0, avg_dur / 120.0)
        idx += 1

        # Message count (normalized, cap at 500)
        msg_count = len(data.user_messages)
        features[idx] = min(1.0, msg_count / 500.0)
        idx += 1

        # Average message length (normalized, cap at 500 chars)
        if data.user_messages:
            avg_len = np.mean([len(m) for m in data.user_messages])
            features[idx] = min(1.0, avg_len / 500.0)
        idx += 1

        # Message frequency (msgs/min, capped at 5)
        if len(data.message_timestamps) >= 2:
            ts = sorted(data.message_timestamps)
            span_min = max(1.0, (ts[-1] - ts[0]) / 60.0)
            freq = len(ts) / span_min
            features[idx] = min(1.0, freq / 5.0)
        idx += 1

        # Typing speed proxy: avg chars/second between messages
        if len(data.user_messages) >= 2 and len(data.message_timestamps) >= 2:
            total_chars = sum(len(m) for m in data.user_messages)
            ts = sorted(data.message_timestamps)
            total_sec = max(1.0, ts[-1] - ts[0])
            speed = total_chars / total_sec
            features[idx] = min(1.0, speed / 10.0)
        idx += 1

        # --- 3. Chat style features (3) ---
        all_text = " ".join(data.user_messages).lower()
        features[idx] = self._formality_score(all_text)
        idx += 1
        features[idx] = self._technicality_score(all_text)
        idx += 1
        features[idx] = self._verbosity_score(data.user_messages)
        idx += 1

        # --- 4. Language features (10) ---
        lang_dist = self._language_distribution(data.languages_detected)
        features[idx:idx + NUM_LANGUAGES] = lang_dist
        idx += NUM_LANGUAGES

        # --- 5. Error category features (11) ---
        err_dist = self._error_distribution(data.error_categories)
        features[idx:idx + NUM_ERROR_CATS] = err_dist
        idx += NUM_ERROR_CATS

        # --- 6. Behavioral features (4) ---
        # Struggle rate (struggles per 100 messages)
        if len(data.user_messages) > 0:
            features[idx] = min(1.0, len(data.struggle_confidences) / max(1, len(data.user_messages)) * 10)
        idx += 1

        # Help acceptance rate
        if data.help_accepted:
            features[idx] = sum(data.help_accepted) / len(data.help_accepted)
        else:
            features[idx] = 0.5  # neutral default
        idx += 1

        # Error rate (errors per 100 messages)
        if len(data.user_messages) > 0:
            features[idx] = min(1.0, len(data.error_texts) / max(1, len(data.user_messages)) * 10)
        idx += 1

        # Active hours ratio (how many hour-buckets have activity)
        active_buckets = np.count_nonzero(hour_dist)
        features[idx] = active_buckets / HOUR_BUCKETS
        idx += 1

        # --- 7. Vocabulary features (2) ---
        features[idx] = self._vocabulary_diversity(data.user_messages)
        idx += 1
        features[idx] = self._avg_word_length(data.user_messages)
        idx += 1

        assert idx == FEATURE_DIM, f"Feature index mismatch: {idx} != {FEATURE_DIM}"
        return features

    def extract_single_message(self, message: str, timestamp: float) -> np.ndarray:
        """Extract features from a single message (for real-time prediction).
        
        Returns a simplified feature vector using only available data.
        """
        data = RawInteractionData(
            user_messages=[message],
            message_timestamps=[timestamp],
        )
        return self.extract(data)

    def feature_names(self) -> list[str]:
        """Return human-readable names for each feature dimension."""
        names = []
        for i in range(HOUR_BUCKETS):
            h_start = i * 4
            names.append(f"hour_{h_start:02d}_{h_start+3:02d}")
        for d in ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]:
            names.append(f"day_{d}")

        names.extend([
            "avg_session_duration", "message_count", "avg_message_length",
            "message_frequency", "input_speed",
        ])
        names.extend(["formality", "technicality", "verbosity"])

        for lang in LANG_PATTERNS:
            names.append(f"lang_{lang}")

        for cat in ERROR_CATS:
            names.append(f"err_{cat}")

        names.extend([
            "struggle_rate", "help_accept_rate", "error_rate", "active_hours_ratio",
        ])
        names.extend(["vocab_diversity", "avg_word_length"])

        assert len(names) == FEATURE_DIM
        return names

    # ------------------------------------------------------------------
    # Internal feature computation
    # ------------------------------------------------------------------

    def _hour_distribution(self, timestamps: list[float]) -> np.ndarray:
        """Compute 6-bucket hour distribution (normalized)."""
        dist = np.zeros(HOUR_BUCKETS, dtype=np.float64)
        if not timestamps:
            return dist
        import datetime
        for ts in timestamps:
            h = datetime.datetime.fromtimestamp(ts).hour
            bucket = h // 4
            dist[bucket] += 1
        total = dist.sum()
        if total > 0:
            dist /= total
        return dist

    def _day_distribution(self, timestamps: list[float]) -> np.ndarray:
        """Compute 7-day-of-week distribution (normalized)."""
        dist = np.zeros(DAY_BUCKETS, dtype=np.float64)
        if not timestamps:
            return dist
        import datetime
        for ts in timestamps:
            day = datetime.datetime.fromtimestamp(ts).weekday()
            dist[day] += 1
        total = dist.sum()
        if total > 0:
            dist /= total
        return dist

    def _formality_score(self, text: str) -> float:
        """Score formality of text [0=very casual, 1=very formal]."""
        if not text:
            return 0.5
        words = text.split()
        if not words:
            return 0.5
        formal_count = sum(1 for m in FORMAL_MARKERS if m in text)
        casual_count = sum(1 for m in CASUAL_MARKERS if m in text)
        total = formal_count + casual_count
        if total == 0:
            return 0.5
        return formal_count / total

    def _technicality_score(self, text: str) -> float:
        """Score how technical the text is [0=non-technical, 1=very technical]."""
        if not text:
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        tech_count = sum(1 for m in TECHNICAL_MARKERS if m in text)
        return min(1.0, tech_count / max(1, len(words)) * 20)

    def _verbosity_score(self, messages: list[str]) -> float:
        """Score verbosity [0=very concise, 1=very verbose]."""
        if not messages:
            return 0.5
        avg_words = np.mean([len(m.split()) for m in messages])
        # Normalize: 1-5 words = concise (0.0-0.3), 5-20 = medium (0.3-0.7), 20+ = verbose (0.7-1.0)
        return min(1.0, avg_words / 30.0)

    def _language_distribution(self, languages: list[str]) -> np.ndarray:
        """Compute programming language distribution (normalized)."""
        dist = np.zeros(NUM_LANGUAGES, dtype=np.float64)
        if not languages:
            return dist
        lang_keys = list(LANG_PATTERNS.keys())
        counter = Counter(languages)
        for lang, count in counter.items():
            if lang in lang_keys:
                dist[lang_keys.index(lang)] = count
        total = dist.sum()
        if total > 0:
            dist /= total
        return dist

    def _error_distribution(self, categories: list[str]) -> np.ndarray:
        """Compute error category distribution (normalized)."""
        dist = np.zeros(NUM_ERROR_CATS, dtype=np.float64)
        if not categories:
            return dist
        counter = Counter(categories)
        for cat, count in counter.items():
            if cat in ERROR_CATS:
                dist[ERROR_CATS.index(cat)] = count
            else:
                dist[-1] += count  # "other"
        total = dist.sum()
        if total > 0:
            dist /= total
        return dist

    def _vocabulary_diversity(self, messages: list[str]) -> float:
        """Type-Token Ratio — unique words / total words."""
        if not messages:
            return 0.5
        all_words = []
        for m in messages:
            all_words.extend(m.lower().split())
        if not all_words:
            return 0.5
        unique = len(set(all_words))
        return min(1.0, unique / len(all_words))

    def _avg_word_length(self, messages: list[str]) -> float:
        """Average word length (normalized, cap at 1.0 = avg 10 chars)."""
        if not messages:
            return 0.5
        all_words = []
        for m in messages:
            all_words.extend(m.split())
        if not all_words:
            return 0.5
        avg = np.mean([len(w) for w in all_words])
        return min(1.0, avg / 10.0)

    def detect_languages_in_text(self, text: str) -> list[str]:
        """Detect programming languages present in text."""
        if not text or len(text) < 5:
            return []
        detected = []
        for lang, patterns in LANG_PATTERNS.items():
            matches = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
            if matches >= 2:
                detected.append(lang)
        return detected
