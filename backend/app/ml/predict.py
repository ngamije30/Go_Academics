"""
Go Academics — ML inference.

Loads the trained XGBoost model and returns a risk score + top contributing
factors for a single student record. Called by Phase 3 and Phase 4.

Risk thresholds are loaded from model_meta.json (computed from the PR curve
during training) rather than being hardcoded arbitrary values.

SHAP: attempt TreeExplainer first; if the installed SHAP/XGBoost combination
is incompatible, fall back to feature_importances_ × per-value risk weights.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent / "saved_models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
META_PATH  = MODELS_DIR / "model_meta.json"

FEATURE_COLS = ["School", "Gender", "Stream", "Term",
                "Subject", "CA_Score", "CA_Trend", "Attendance_Pct"]

# Per-value risk weight: 1.0 = worst, 0.0 = best
# Used in fallback explanation when SHAP is unavailable
_RISK_SCORE_FN = {
    "CA_Score":       lambda v: max(0.0, (15 - float(v)) / 15),   # 0/30 → 1.0, 15+/30 → 0.0
    "CA_Trend":       lambda v: max(0.0, -float(v) / 15),          # negative trend → risk
    "Attendance_Pct": lambda v: max(0.0, (80 - float(v)) / 80),   # 0% → 1.0, 80%+ → 0.0
    "Term":           lambda v: (float(v) - 1) / 2,                # higher term → slightly more risk
    "Stream":         lambda v: {"S4": 0.2, "S5": 0.5, "S6": 0.8}.get(str(v), 0.5),
    "Subject":        lambda v: 0.5,
    "Gender":         lambda v: 0.5,
    "School":         lambda v: 0.5,
}

_model_cache = None
_meta_cache  = None


def _load_model():
    global _model_cache
    if _model_cache is None:
        with open(MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
    return _model_cache


def _load_meta() -> dict:
    global _meta_cache
    if _meta_cache is None:
        if META_PATH.exists():
            with open(META_PATH) as f:
                _meta_cache = json.load(f)
        else:
            _meta_cache = {"thresholds": {"medium": 0.35, "high": 0.60},
                           "shap_compatible": False}
    return _meta_cache


def _encode(features: dict, encodings: dict) -> pd.DataFrame:
    row = {}
    for col in FEATURE_COLS:
        val = features[col]
        if col in encodings:
            if val not in encodings[col]:
                known = sorted(encodings[col])
                raise ValueError(
                    f"Unknown {col} value {val!r} — model was trained on {known}."
                )
            row[col] = encodings[col][val]
        else:
            row[col] = float(val)
    return pd.DataFrame([row])


def _human_factor(col: str, original_val, shap_val: float | None = None) -> str:
    direction = ""
    if shap_val is not None:
        direction = " (+risk)" if shap_val > 0 else " (-risk)"

    if col == "CA_Score":
        v = float(original_val)
        status = "below threshold (< 15)" if v < 15 else "within range"
        return f"CA score {v}/30 — {status}{direction}"
    if col == "CA_Trend":
        v = float(original_val)
        trend = "declining" if v < 0 else ("improving" if v > 0 else "stable")
        return f"CA trend {v:+.1f} pts — {trend}{direction}"
    if col == "Attendance_Pct":
        v = float(original_val)
        status = "below threshold (< 80%)" if v < 80 else "adequate"
        return f"Attendance {v}% — {status}{direction}"
    if col == "Stream":
        return f"Class stream: {original_val}{direction}"
    if col == "Term":
        return f"Term {original_val}{direction}"
    if col == "Subject":
        return f"Subject: {original_val}{direction}"
    if col == "Gender":
        return f"Gender: {original_val}{direction}"
    if col == "School":
        return f"School: {original_val}{direction}"
    return f"{col}: {original_val}{direction}"


def _top_factors_shap(model, X_encoded: pd.DataFrame, features: dict) -> list[str]:
    """Use real SHAP values for instance-level explanations."""
    import shap
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_encoded)

    # TreeExplainer's return shape varies by SHAP version and model type:
    # older versions (and XGBoost) return a single ndarray (n_samples, n_features)
    # for binary classification; newer versions return either a list of
    # per-class arrays or one ndarray (n_samples, n_features, n_classes).
    # Normalize to this row's shap values for the "Pass" class (index 1).
    if isinstance(shap_vals, list):
        pass_shap = shap_vals[1][0]
    elif shap_vals.ndim == 3:
        pass_shap = shap_vals[0, :, 1]
    else:
        pass_shap = shap_vals[0]

    # positive shap value increases P(Pass), so negate for risk
    risk_shap = -np.asarray(pass_shap, dtype=float)
    top_idx   = np.argsort(np.abs(risk_shap))[::-1][:3]
    return [_human_factor(FEATURE_COLS[i], features[FEATURE_COLS[i]], risk_shap[i])
            for i in top_idx]


def _top_factors_fallback(model, features: dict) -> list[str]:
    """Weight global feature importances by per-value risk score (fallback if SHAP fails)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_          # tree-based models
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])               # linear models
    else:
        importances = np.ones(len(FEATURE_COLS))
    weighted    = np.array([
        importances[i] * _RISK_SCORE_FN[FEATURE_COLS[i]](features[FEATURE_COLS[i]])
        for i in range(len(FEATURE_COLS))
    ])
    top_idx = np.argsort(weighted)[::-1][:3]
    return [_human_factor(FEATURE_COLS[i], features[FEATURE_COLS[i]]) for i in top_idx]


def predict_risk(student_features: dict) -> dict:
    """
    Parameters
    ----------
    student_features : dict with keys:
        School, Gender, Stream, Term (int), Subject,
        CA_Score (float), CA_Trend (float), Attendance_Pct (float)

    Returns
    -------
    {
        "risk_score":  float,       # 0.0-1.0 (higher = more at risk)
        "risk_level":  str,         # "Low" | "Medium" | "High"
        "top_factors": list[str],   # top 3 instance-level risk reasons
        "explanation_method": str,  # "shap" | "feature_importance"
    }
    """
    model = _load_model()
    meta  = _load_meta()
    X     = _encode(student_features, meta.get("encodings", {}))

    prob_pass  = float(model.predict_proba(X)[0][1])
    risk_score = round(1.0 - prob_pass, 4)

    thresholds  = meta.get("thresholds", {"medium": 0.35, "high": 0.60})
    medium_thr  = thresholds.get("medium", 0.35)
    high_thr    = thresholds.get("high",   0.60)

    if risk_score >= high_thr:
        risk_level = "High"
    elif risk_score >= medium_thr:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Prefer real SHAP; fall back gracefully
    explanation_method = "feature_importance"
    if meta.get("shap_compatible", False):
        try:
            top_factors        = _top_factors_shap(model, X, student_features)
            explanation_method = "shap"
        except Exception:
            top_factors = _top_factors_fallback(model, student_features)
    else:
        top_factors = _top_factors_fallback(model, student_features)

    return {
        "risk_score":          risk_score,
        "risk_level":          risk_level,
        "top_factors":         top_factors,
        "explanation_method":  explanation_method,
    }
