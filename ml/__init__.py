"""
Rio ML — User Pattern Learning Pipeline

Ensemble learning system that adapts to individual user patterns:
- Usage pattern recognition (when, how, what the user does)
- Chat style classification (verbose/concise, technical/casual)
- Struggle prediction (predict before it happens)
- Per-user model adaptation via online/incremental learning
- pkl-based model serialization for fast load/save

Architecture:
    DB (rio_patterns.db)
        ↓
    FeatureExtractor  →  raw user data → feature vectors
        ↓
    RioEnsembleModel  →  SGDClassifier + MultinomialNB + DecisionTree
        ↓                 + VotingClassifier (soft voting)
    UserModelManager  →  per-user pkl files, online partial_fit
        ↓
    Integration       →  feeds context strings into Gemini prompts
"""

__version__ = "1.0.0"
