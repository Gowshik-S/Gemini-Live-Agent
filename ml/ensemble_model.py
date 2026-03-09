"""
Rio ML — Ensemble Learning Model

Three-model ensemble with soft voting for user behavior classification:

    1. SGDClassifier     — Linear model, supports partial_fit (online learning)
    2. MultinomialNB     — Naive Bayes, supports partial_fit (online learning)
    3. PassiveAggressiveClassifier — Online-learning linear model

All three support partial_fit() → incremental/online learning from new data.
Combined via soft-voting (probability averaging) for final predictions.

Classification Targets:
    - struggle_risk:  low / medium / high
    - chat_style:     concise / moderate / verbose
    - engagement:     passive / active / power_user
    - mood:           calm / neutral / frustrated

The ensemble predicts ALL targets simultaneously via multi-output strategy.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# Lazy imports for sklearn — guarded so Rio still starts without it
try:
    from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("ml.sklearn_not_installed", note="pip install scikit-learn")

from .feature_engine import FEATURE_DIM


# ---------------------------------------------------------------------------
# Classification labels
# ---------------------------------------------------------------------------

STRUGGLE_CLASSES = ["low", "medium", "high"]
STYLE_CLASSES = ["concise", "moderate", "verbose"]
ENGAGEMENT_CLASSES = ["passive", "active", "power_user"]
MOOD_CLASSES = ["calm", "neutral", "frustrated"]

ALL_TARGETS = {
    "struggle_risk": STRUGGLE_CLASSES,
    "chat_style": STYLE_CLASSES,
    "engagement": ENGAGEMENT_CLASSES,
    "mood": MOOD_CLASSES,
}


@dataclass
class EnsemblePrediction:
    """Prediction from the ensemble model."""
    struggle_risk: str = "low"
    struggle_risk_proba: dict[str, float] = field(default_factory=dict)
    chat_style: str = "moderate"
    chat_style_proba: dict[str, float] = field(default_factory=dict)
    engagement: str = "active"
    engagement_proba: dict[str, float] = field(default_factory=dict)
    mood: str = "neutral"
    mood_proba: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    model_version: str = ""
    prediction_time_ms: float = 0.0


class SingleTargetEnsemble:
    """Ensemble of 3 online-learnable classifiers for a single target.
    
    Uses soft voting: averages class probabilities from all 3 models.
    All models support partial_fit() for incremental learning.
    """

    def __init__(self, target_name: str, classes: list[str]) -> None:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for ML pipeline")

        self.target_name = target_name
        self.classes = np.array(classes)
        self.n_classes = len(classes)
        self._is_fitted = False
        self._train_count = 0

        # Model 1: SGD with log_loss → logistic regression (online)
        self.sgd = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            learning_rate="adaptive",
            eta0=0.01,
            max_iter=1,           # single pass for partial_fit
            warm_start=True,
            random_state=42,
        )

        # Model 2: Multinomial Naive Bayes (online)
        # Needs non-negative features — we ensure via MinMaxScaler
        self.mnb = MultinomialNB(alpha=1.0, fit_prior=True)

        # Model 3: Passive-Aggressive Classifier (online)
        self.pac = PassiveAggressiveClassifier(
            C=0.5,
            loss="hinge",
            max_iter=1,
            warm_start=True,
            random_state=42,
        )

        # Scaler for ensuring non-negative features (needed for MultinomialNB)
        self.scaler = MinMaxScaler(feature_range=(0.01, 1.0))
        self._scaler_fitted = False

    def ensure_online_learning_compatible(self) -> bool:
        """Normalize older pickled estimators for single-sample partial_fit()."""
        changed = False
        for estimator_name in ("sgd", "pac"):
            estimator = getattr(self, estimator_name, None)
            if estimator is None:
                continue
            if getattr(estimator, "class_weight", None) == "balanced":
                estimator.class_weight = None
                changed = True
                log.info(
                    "ml.ensemble.class_weight_normalized",
                    target=self.target_name,
                    estimator=estimator_name,
                )
        return changed

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Full batch fit (for initial training on datasets)."""
        if len(X) < 2:
            log.warning("ml.ensemble.fit.too_few_samples", target=self.target_name, n=len(X))
            return

        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        self._scaler_fitted = True

        # Fit all 3 models
        self.sgd.fit(X_scaled, y)
        self.mnb.fit(X_scaled, y)
        self.pac.fit(X_scaled, y)

        self._is_fitted = True
        self._train_count = len(X)
        log.info("ml.ensemble.fit", target=self.target_name, samples=len(X))

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Online/incremental learning — update models with new samples."""
        if len(X) == 0:
            return
        self.ensure_online_learning_compatible()
        y = np.asarray(y)
        invalid_labels = np.setdiff1d(np.unique(y), self.classes)
        if invalid_labels.size:
            raise ValueError(
                f"Invalid labels for target '{self.target_name}': "
                f"{invalid_labels.tolist()} not in {self.classes.tolist()}"
            )

        # Fit or transform scaler
        if not self._scaler_fitted:
            X_scaled = self.scaler.fit_transform(X)
            self._scaler_fitted = True
        else:
            # Partial update of scaler bounds
            self.scaler.partial_fit(X)
            X_scaled = self.scaler.transform(X)

        # Clip to ensure non-negative for MNB
        X_scaled = np.clip(X_scaled, 0.01, None)

        # Partial fit all models
        self.sgd.partial_fit(X_scaled, y, classes=self.classes)
        self.mnb.partial_fit(X_scaled, y, classes=self.classes)
        self.pac.partial_fit(X_scaled, y, classes=self.classes)

        self._is_fitted = True
        self._train_count += len(X)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict classes and probabilities via soft voting.
        
        Returns:
            (predictions, probabilities) where probabilities is (n_samples, n_classes)
        """
        if not self._is_fitted:
            # Cold start — return default prediction
            n = len(X)
            default_idx = len(self.classes) // 2  # middle class
            preds = np.full(n, self.classes[default_idx])
            probas = np.full((n, self.n_classes), 1.0 / self.n_classes)
            return preds, probas

        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, 0.01, None)

        # Get probabilities from each model
        probas = []

        # SGD supports predict_proba with log_loss
        try:
            probas.append(self.sgd.predict_proba(X_scaled))
        except Exception:
            probas.append(np.full((len(X), self.n_classes), 1.0 / self.n_classes))

        # MNB supports predict_proba natively
        try:
            probas.append(self.mnb.predict_proba(X_scaled))
        except Exception:
            probas.append(np.full((len(X), self.n_classes), 1.0 / self.n_classes))

        # PAC doesn't have predict_proba — use decision function as proxy
        try:
            dec = self.pac.decision_function(X_scaled)
            if dec.ndim == 1:
                # Binary: convert to 2-class probabilities via sigmoid
                sigmoid = 1.0 / (1.0 + np.exp(-dec))
                pac_proba = np.column_stack([1 - sigmoid, sigmoid])
            else:
                # Multi-class: softmax
                exp_dec = np.exp(dec - dec.max(axis=1, keepdims=True))
                pac_proba = exp_dec / exp_dec.sum(axis=1, keepdims=True)
            # Ensure shape matches
            if pac_proba.shape[1] != self.n_classes:
                pac_proba = np.full((len(X), self.n_classes), 1.0 / self.n_classes)
            probas.append(pac_proba)
        except Exception:
            probas.append(np.full((len(X), self.n_classes), 1.0 / self.n_classes))

        # Soft voting: average probabilities
        avg_proba = np.mean(probas, axis=0)

        # Predictions from highest probability
        pred_indices = np.argmax(avg_proba, axis=1)
        predictions = self.classes[pred_indices]

        return predictions, avg_proba

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def train_count(self) -> int:
        return self._train_count


class RioEnsembleModel:
    """Multi-target ensemble model for user behavior prediction.
    
    Manages one SingleTargetEnsemble per classification target.
    All models share the same feature space (48-dim vectors from FeatureExtractor).
    
    Usage::
    
        model = RioEnsembleModel()
        
        # Train on batch data
        model.fit(X_train, y_dict)  # y_dict = {"struggle_risk": [...], "chat_style": [...], ...}
        
        # Online learning from new data
        model.partial_fit(X_new, y_dict_new)
        
        # Predict
        pred = model.predict(X_test[0])  # EnsemblePrediction
        
        # Save/load
        model.save("models/user_123.pkl")
        model = RioEnsembleModel.load("models/user_123.pkl")
    """

    VERSION = "1.0.1"

    def __init__(self) -> None:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required. pip install scikit-learn")

        self.ensembles: dict[str, SingleTargetEnsemble] = {}
        for target_name, classes in ALL_TARGETS.items():
            self.ensembles[target_name] = SingleTargetEnsemble(target_name, classes)

        self._created_at = time.time()
        self._last_trained_at: Optional[float] = None
        self._total_samples = 0
        log.info("ml.ensemble_model.init", targets=list(ALL_TARGETS.keys()))

    def fit(
        self,
        X: np.ndarray,
        y: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Full batch fit on training data.
        
        Args:
            X: Feature matrix, shape (n_samples, FEATURE_DIM)
            y: Dict mapping target name → array of labels
            
        Returns:
            Dict of target_name → training accuracy
        """
        results = {}
        for target_name, ensemble in self.ensembles.items():
            if target_name in y:
                ensemble.fit(X, y[target_name])
                # Quick training accuracy
                preds, _ = ensemble.predict(X)
                acc = accuracy_score(y[target_name], preds)
                results[target_name] = acc
                log.info("ml.fit.target", target=target_name, accuracy=f"{acc:.3f}", samples=len(X))

        self._last_trained_at = time.time()
        self._total_samples = len(X)
        return results

    def partial_fit(
        self,
        X: np.ndarray,
        y: dict[str, np.ndarray],
    ) -> None:
        """Online/incremental fit with new samples.
        
        Args:
            X: Feature matrix, shape (n_samples, FEATURE_DIM)
            y: Dict mapping target name → array of labels
        """
        for target_name, ensemble in self.ensembles.items():
            if target_name in y:
                ensemble.partial_fit(X, y[target_name])

        self._last_trained_at = time.time()
        self._total_samples += len(X)

    def predict(self, features: np.ndarray) -> EnsemblePrediction:
        """Predict all targets for a single feature vector.
        
        Args:
            features: Feature vector of shape (FEATURE_DIM,)
            
        Returns:
            EnsemblePrediction with all classifications and probabilities
        """
        start = time.perf_counter()
        X = features.reshape(1, -1) if features.ndim == 1 else features[:1]

        pred = EnsemblePrediction(model_version=self.VERSION)

        for target_name, ensemble in self.ensembles.items():
            labels, probas = ensemble.predict(X)
            label = labels[0]
            proba_dict = {
                cls: float(probas[0, i])
                for i, cls in enumerate(ensemble.classes)
            }

            setattr(pred, target_name, label)
            setattr(pred, f"{target_name}_proba", proba_dict)

        # Overall confidence = average of max probabilities across targets
        max_probas = []
        for target_name in ALL_TARGETS:
            proba_dict = getattr(pred, f"{target_name}_proba", {})
            if proba_dict:
                max_probas.append(max(proba_dict.values()))
        pred.confidence = float(np.mean(max_probas)) if max_probas else 0.0

        pred.prediction_time_ms = (time.perf_counter() - start) * 1000
        return pred

    def predict_batch(self, X: np.ndarray) -> list[EnsemblePrediction]:
        """Predict for multiple feature vectors."""
        return [self.predict(X[i]) for i in range(len(X))]

    # ------------------------------------------------------------------
    # Serialization (pkl)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the complete model to a pkl file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "version": self.VERSION,
            "ensembles": self.ensembles,
            "created_at": self._created_at,
            "last_trained_at": self._last_trained_at,
            "total_samples": self._total_samples,
            "saved_at": time.time(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("ml.model.saved", path=path, samples=self._total_samples)

    @classmethod
    def load(cls, path: str) -> "RioEnsembleModel":
        """Load a model from a pkl file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        model = cls.__new__(cls)
        model.ensembles = state["ensembles"]
        normalized_targets = []
        for target_name, ensemble in model.ensembles.items():
            if hasattr(ensemble, "ensure_online_learning_compatible"):
                if ensemble.ensure_online_learning_compatible():
                    normalized_targets.append(target_name)
        model._created_at = state.get("created_at", time.time())
        model._last_trained_at = state.get("last_trained_at")
        model._total_samples = state.get("total_samples", 0)
        log.info(
            "ml.model.loaded",
            path=path,
            samples=model._total_samples,
            version=state.get("version"),
            normalized_targets=normalized_targets,
        )
        return model

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get model statistics."""
        return {
            "version": self.VERSION,
            "total_samples": self._total_samples,
            "created_at": self._created_at,
            "last_trained_at": self._last_trained_at,
            "targets": {
                name: {
                    "fitted": ens.is_fitted,
                    "train_count": ens.train_count,
                    "classes": list(ens.classes),
                }
                for name, ens in self.ensembles.items()
            },
            "sklearn_available": SKLEARN_AVAILABLE,
        }
