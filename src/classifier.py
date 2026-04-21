"""
Tragedy classifier — loaded once at module import.

Exposes:
    classify_query(text: str) -> dict
        {"label": "tragedy"|"non-tragedy"|"unknown",
         "confidence": float,
         "raw_proba": [prob_class_0, prob_class_1]}
"""
import logging
import os
import pickle

logger = logging.getLogger(__name__)

_BASE_DIR  = os.path.join(os.path.dirname(__file__), '..')
_MODEL_PATH = os.path.join(_BASE_DIR, 'model', 'nb_tfidf.pkl')
_VEC_PATH   = os.path.join(_BASE_DIR, 'data', 'features', 'tfidf_vectorizer.pkl')

_FALLBACK = {"label": "unknown", "confidence": 0.0, "raw_proba": [0.5, 0.5]}

# ── Load at import time ───────────────────────────────────────────────────────

_model = None
_vectorizer = None

try:
    with open(_MODEL_PATH, 'rb') as f:
        _model = pickle.load(f)
    with open(_VEC_PATH, 'rb') as f:
        _vectorizer = pickle.load(f)
    logger.info("Tragedy classifier loaded successfully.")
except FileNotFoundError as exc:
    logger.warning(
        "Classifier model files not found (%s). "
        "classify_query() will return fallback values. "
        "Run `dvc repro` to train the models.",
        exc,
    )
except Exception as exc:
    logger.warning("Failed to load classifier: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def classify_query(text: str) -> dict:
    """Classify text as tragedy or non-tragedy using the trained NB+TF-IDF model.

    Returns a dict with keys: label, confidence, raw_proba.
    Falls back gracefully if the model was not loaded.
    """
    if _model is None or _vectorizer is None:
        return dict(_FALLBACK)

    try:
        X = _vectorizer.transform([text])
        proba = _model.predict_proba(X)[0].tolist()
        pred  = int(_model.predict(X)[0])
        label = "tragedy" if pred == 1 else "non-tragedy"
        return {
            "label":     label,
            "confidence": round(float(proba[pred]), 4),
            "raw_proba": [round(p, 4) for p in proba],
        }
    except Exception as exc:
        logger.warning("classify_query failed: %s", exc)
        return dict(_FALLBACK)
