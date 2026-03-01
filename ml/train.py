"""
Rio ML — Training Pipeline Script

Standalone script to train/retrain the ensemble model.
Can be run manually or as a scheduled job.

Usage:
    # Train from historical DB data
    python -m ml.train --from-db --days 30
    
    # Train from dataset files
    python -m ml.train --from-dataset sentiment140
    
    # Generate synthetic training data (cold start bootstrap)
    python -m ml.train --bootstrap
    
    # Show model stats
    python -m ml.train --stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Add parent dir so we can import ml package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.feature_engine import FeatureExtractor, RawInteractionData, FEATURE_DIM
from ml.ensemble_model import RioEnsembleModel, SKLEARN_AVAILABLE
from ml.user_model_manager import (
    UserModelManager,
    _auto_label_struggle,
    _auto_label_style,
    _auto_label_engagement,
    _auto_label_mood,
    DEFAULT_MODELS_DIR,
    DEFAULT_DB_PATH,
)


def generate_synthetic_data(n_samples: int = 200) -> tuple[np.ndarray, dict]:
    """Generate synthetic training data for cold-start bootstrap.
    
    Creates realistic feature vectors and labels covering all user archetypes:
    - Beginner: high struggle, verbose, frustrated
    - Intermediate: medium struggle, moderate, neutral  
    - Expert: low struggle, concise, calm, power_user
    - Casual: low struggle, concise, passive
    """
    rng = np.random.RandomState(42)
    extractor = FeatureExtractor()

    X = np.zeros((n_samples, FEATURE_DIM))
    labels = {
        "struggle_risk": [],
        "chat_style": [],
        "engagement": [],
        "mood": [],
    }

    archetypes = [
        # (weight, struggle, style, engagement, mood, feature_params)
        (0.25, "high", "verbose", "active", "frustrated", {
            "msg_len": (50, 150), "msg_count": (30, 100),
            "error_rate": (0.2, 0.5), "formality": (0.2, 0.4),
        }),
        (0.30, "medium", "moderate", "active", "neutral", {
            "msg_len": (15, 50), "msg_count": (10, 50),
            "error_rate": (0.05, 0.2), "formality": (0.4, 0.7),
        }),
        (0.25, "low", "concise", "power_user", "calm", {
            "msg_len": (5, 20), "msg_count": (20, 80),
            "error_rate": (0.0, 0.1), "formality": (0.5, 0.8),
        }),
        (0.20, "low", "concise", "passive", "calm", {
            "msg_len": (3, 15), "msg_count": (2, 10),
            "error_rate": (0.0, 0.05), "formality": (0.3, 0.6),
        }),
    ]

    idx = 0
    for weight, struggle, style, engagement, mood, params in archetypes:
        count = int(n_samples * weight)
        for _ in range(count):
            if idx >= n_samples:
                break

            # Generate feature vector with realistic distributions
            features = rng.uniform(0, 1, FEATURE_DIM)

            # Time distribution — normalize
            features[0:6] = rng.dirichlet(np.ones(6))  # hour buckets
            features[6:13] = rng.dirichlet(np.ones(7))  # day buckets

            # Interaction features
            msg_count = rng.randint(*params["msg_count"])
            avg_len = rng.randint(*params["msg_len"])
            features[13] = min(1.0, rng.uniform(0.1, 2.0) / 2.0)  # session duration
            features[14] = min(1.0, msg_count / 500.0)
            features[15] = min(1.0, avg_len / 500.0)
            features[16] = min(1.0, rng.uniform(0.1, 3.0) / 5.0)  # frequency
            features[17] = min(1.0, rng.uniform(0.5, 5.0) / 10.0)  # speed

            # Chat style
            features[18] = rng.uniform(*params["formality"])  # formality
            features[19] = rng.uniform(0.1, 0.8)  # technicality
            features[20] = min(1.0, avg_len / 30.0)  # verbosity

            # Language distribution
            features[21:31] = rng.dirichlet(np.ones(10))

            # Error distribution
            features[31:42] = rng.dirichlet(np.ones(11)) * rng.uniform(*params["error_rate"])

            # Behavioral features
            features[42] = rng.uniform(*params["error_rate"])  # struggle_rate
            features[43] = rng.uniform(0.3, 0.9)  # help_accept_rate
            features[44] = rng.uniform(*params["error_rate"])  # error_rate
            features[45] = rng.uniform(0.2, 0.8)  # active_hours_ratio

            # Vocabulary
            features[46] = rng.uniform(0.3, 0.9)  # diversity
            features[47] = rng.uniform(0.3, 0.7)  # avg word length

            X[idx] = features
            labels["struggle_risk"].append(struggle)
            labels["chat_style"].append(style)
            labels["engagement"].append(engagement)
            labels["mood"].append(mood)
            idx += 1

    # Trim to actual count
    X = X[:idx]
    labels = {k: np.array(v[:idx]) for k, v in labels.items()}

    return X, labels


def train_bootstrap(model_path: str = None) -> dict:
    """Bootstrap training with synthetic data.
    
    Creates a base model that works from day 1 without any real user data.
    Subsequent online learning will refine it per user.
    """
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        return {}

    print("Generating synthetic training data...")
    X, y = generate_synthetic_data(n_samples=500)
    print(f"  Generated {len(X)} samples with {FEATURE_DIM} features each")
    print(f"  Label distribution:")
    for target, labels in y.items():
        unique, counts = np.unique(labels, return_counts=True)
        dist = {str(u): int(c) for u, c in zip(unique, counts)}
        print(f"    {target}: {dist}")

    print("\nTraining ensemble model...")
    model = RioEnsembleModel()
    results = model.fit(X, y)

    print("\nTraining Results:")
    for target, acc in results.items():
        print(f"  {target}: accuracy = {acc:.3f}")

    # Save
    path = model_path or os.path.join(DEFAULT_MODELS_DIR, "user_default.pkl")
    model.save(path)
    print(f"\nModel saved to: {path}")
    print(f"  Size: {os.path.getsize(path) / 1024:.1f} KB")

    # Quick validation
    print("\nQuick validation (predicting on training data):")
    pred = model.predict(X[0])
    print(f"  Sample prediction:")
    print(f"    struggle_risk = {pred.struggle_risk} (confidence: {pred.struggle_risk_proba})")
    print(f"    chat_style    = {pred.chat_style} (confidence: {pred.chat_style_proba})")
    print(f"    engagement    = {pred.engagement} (confidence: {pred.engagement_proba})")
    print(f"    mood          = {pred.mood} (confidence: {pred.mood_proba})")
    print(f"    prediction_time = {pred.prediction_time_ms:.2f} ms")

    return results


def train_from_db(days: int = 30) -> dict:
    """Train from historical DB data."""
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn not installed")
        return {}

    if not os.path.exists(DEFAULT_DB_PATH):
        print(f"WARNING: DB not found at {DEFAULT_DB_PATH}")
        print("  No historical data available. Use --bootstrap first.")
        return {}

    print(f"Training from DB ({days} days of history)...")
    manager = UserModelManager(db_path=DEFAULT_DB_PATH)
    stats = manager.train_on_history(days=days)

    if stats:
        print(f"\nTrained on {stats['sessions']} sessions:")
        for target, acc in stats.get("accuracies", {}).items():
            print(f"  {target}: accuracy = {acc:.3f}")
        manager.save()
        print(f"\nModel saved to: {manager.model_path}")
    else:
        print("  Not enough data for training. Use the app first, or run --bootstrap.")

    manager.close()
    return stats or {}


def show_stats() -> None:
    """Show model and data stats."""
    print("=" * 60)
    print("Rio ML Pipeline Stats")
    print("=" * 60)

    # Check sklearn
    print(f"\nscikit-learn available: {SKLEARN_AVAILABLE}")

    # Check model files
    print(f"\nModels directory: {DEFAULT_MODELS_DIR}")
    if os.path.exists(DEFAULT_MODELS_DIR):
        files = [f for f in os.listdir(DEFAULT_MODELS_DIR) if f.endswith(".pkl")]
        if files:
            print(f"  Model files: {len(files)}")
            for f in files:
                path = os.path.join(DEFAULT_MODELS_DIR, f)
                size = os.path.getsize(path)
                print(f"    {f} ({size / 1024:.1f} KB)")
        else:
            print("  No model files found")
    else:
        print("  Directory does not exist")

    # Check DB
    print(f"\nPattern DB: {DEFAULT_DB_PATH}")
    if os.path.exists(DEFAULT_DB_PATH):
        import sqlite3
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        for table in ["activities", "errors", "struggles", "help_responses", "language_detections"]:
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                print(f"  {table}: {row[0]} records")
            except Exception:
                print(f"  {table}: not found")
        conn.close()
    else:
        print("  DB does not exist yet")

    # Check chat DB
    chat_db = os.path.join(os.path.dirname(DEFAULT_DB_PATH), "rio_chats.db")
    print(f"\nChat DB: {chat_db}")
    if os.path.exists(chat_db):
        import sqlite3
        conn = sqlite3.connect(chat_db)
        try:
            row = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
            print(f"  messages: {row[0]}")
            row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            print(f"  sessions: {row[0]}")
        except Exception:
            print("  tables not found")
        conn.close()
    else:
        print("  DB does not exist yet")

    # Feature info
    ext = FeatureExtractor()
    print(f"\nFeature vector dimension: {FEATURE_DIM}")
    print(f"  Feature names: {ext.feature_names()[:5]}... (+{FEATURE_DIM - 5} more)")


def main():
    parser = argparse.ArgumentParser(description="Rio ML Training Pipeline")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap with synthetic data")
    parser.add_argument("--from-db", action="store_true", help="Train from historical DB data")
    parser.add_argument("--days", type=int, default=30, help="Days of history to use (with --from-db)")
    parser.add_argument("--stats", action="store_true", help="Show ML pipeline stats")
    parser.add_argument("--model-path", type=str, help="Custom model output path")

    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.bootstrap:
        train_bootstrap(args.model_path)
    elif args.from_db:
        train_from_db(days=args.days)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python -m ml.train --bootstrap    # Create initial model")
        print("  python -m ml.train --stats        # Show pipeline info")


if __name__ == "__main__":
    main()
