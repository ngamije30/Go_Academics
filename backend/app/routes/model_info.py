"""
Go Academics — /model-info route.

Exposes the currently deployed model's identity and metrics so the dashboard
never has to hardcode numbers from whichever training run happened to be
current when the UI was last written.
"""
from fastapi import APIRouter, HTTPException

from app.ml.predict import _load_meta

router = APIRouter()


@router.get("/model-info")
def model_info():
    meta = _load_meta()
    if "best_model" not in meta:
        raise HTTPException(status_code=503, detail="Model metadata not available — run backend/app/ml/train.py.")

    metrics = meta.get("metrics", {})
    return {
        "model":       meta["best_model"],
        "n_samples":   meta.get("n_samples"),
        "accuracy":    metrics.get("accuracy"),
        "recall":      metrics.get("recall"),
        "precision":   metrics.get("precision"),
        "f1":          metrics.get("f1"),
        "auc_roc":     metrics.get("auc_roc"),
        "thresholds":  meta.get("thresholds"),
    }
