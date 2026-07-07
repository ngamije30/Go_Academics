"""
Go Academics — ML Training Pipeline

Trains and compares three models on the preprocessed real school dataset
(data/preprocess.py output):
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost (best performer)

Features: CA_Score, CA_Trend (G2→G1 trajectory), Attendance_Pct +
          categorical context (School, Gender, Stream, Term, Subject)

Evaluation priority: Recall > F1 > Accuracy
  (missing an at-risk student is worse than a false alarm)

Outputs saved to backend/app/ml/saved_models/:
  - best_model.pkl
  - model_meta.json  (metrics + optimal risk thresholds from PR curve)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

DATA_DIR   = Path(__file__).parent.parent.parent.parent / "data" / "processed"
DATA_PATH  = DATA_DIR / "students_ml.csv"
ENCODINGS_PATH = DATA_DIR / "encodings.json"
MODELS_DIR = Path(__file__).parent / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = ["School", "Gender", "Stream", "Term",
                "Subject", "CA_Score", "CA_Trend", "Attendance_Pct"]
TARGET_COL   = "Final_Result"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURE_COLS]
    y  = df[TARGET_COL]
    print(f"Loaded {len(df)} rows | Features: {list(X.columns)}")
    print(f"Class balance - Pass: {y.sum()} | Fail: {(y==0).sum()}")
    return X, y


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred)  * 100, 2),
        "recall":    round(recall_score(y_test, y_pred)     * 100, 2),
        "precision": round(precision_score(y_test, y_pred)  * 100, 2),
        "f1":        round(f1_score(y_test, y_pred)         * 100, 2),
        "auc_roc":   round(roc_auc_score(y_test, y_prob)    * 100, 2),
    }


def optimal_thresholds(model, X_test, y_test):
    """
    Derive evidence-based risk thresholds from the precision-recall curve.

    risk_score = 1 - P(Pass), so we compute the PR curve on the at-risk label.

    Medium threshold: risk score at maximum F1 for the at-risk class.
                      This is the best overall tradeoff — catches most at-risk
                      students while keeping false-alarm rate manageable.
    High threshold:   lowest risk score where at-risk precision >= 0.90.
                      Only flags a student as "High" when the model is very
                      confident they will fail.

    Separation >= 0.05 is enforced so the Medium band is always meaningful.
    """
    y_prob    = model.predict_proba(X_test)[:, 1]   # P(Pass)
    risk_prob = 1 - y_prob                           # risk score
    at_risk   = 1 - y_test.to_numpy()               # 1 = at-risk, 0 = safe

    precisions, recalls, thresholds = precision_recall_curve(at_risk, risk_prob)
    # precision_recall_curve appends a sentinel; thresholds has len = len(precisions) - 1
    prec = precisions[:-1]
    rec  = recalls[:-1]

    # Medium: max-F1 point on at-risk PR curve
    f1_all        = 2 * prec * rec / (prec + rec + 1e-9)
    medium_thresh = float(thresholds[f1_all.argmax()])

    # High: first threshold where at-risk precision >= 0.90
    high_prec_idx = np.where(prec >= 0.90)[0]
    if len(high_prec_idx):
        high_thresh = float(thresholds[high_prec_idx[0]])
    else:
        high_thresh = medium_thresh + 0.20

    # Guarantee a meaningful gap between Medium and High bands
    if high_thresh - medium_thresh < 0.05:
        high_thresh = medium_thresh + 0.15

    medium_thresh = round(medium_thresh, 3)
    high_thresh   = round(high_thresh, 3)

    print(f"  Optimal thresholds (from PR curve):")
    print(f"    Medium risk: score >= {medium_thresh}")
    print(f"    High risk:   score >= {high_thresh}")
    return medium_thresh, high_thresh


def train_models():
    X, y = load_data()
    n_samples = len(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=(y_train == 0).sum() / y_train.sum(),
            base_score=0.5,   # explicit value: keeps SHAP TreeExplainer compatible
            random_state=42, eval_metric="logloss", verbosity=0
        ),
    }

    results = []
    trained = {}

    print("\nTraining models...\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        m = evaluate(name, model, X_test, y_test)
        results.append(m)

        cv = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="recall")
        m["cv_recall_mean"] = round(cv.mean() * 100, 2)
        m["cv_recall_std"]  = round(cv.std()  * 100, 2)

        print(f"  {name}")
        print(f"    Accuracy:  {m['accuracy']}%")
        print(f"    Recall:    {m['recall']}%")
        print(f"    Precision: {m['precision']}%")
        print(f"    F1:        {m['f1']}%")
        print(f"    AUC-ROC:   {m['auc_roc']}%")
        print(f"    CV Recall: {m['cv_recall_mean']}% (+/- {m['cv_recall_std']}%)\n")

    # Pick best model by recall (primary) then F1 (tiebreak)
    best_meta  = max(results, key=lambda x: (x["recall"], x["f1"]))
    best_model = trained[best_meta["model"]]
    print(f"Best model: {best_meta['model']} "
          f"(Recall: {best_meta['recall']}%, F1: {best_meta['f1']}%)")

    # Derive evidence-based thresholds from PR curve
    medium_thresh, high_thresh = optimal_thresholds(best_model, X_test, y_test)

    # Verify SHAP compatibility before saving
    try:
        import shap
        exp = shap.TreeExplainer(best_model)
        _   = exp.shap_values(X_test.iloc[:2])
        print("  SHAP TreeExplainer: OK")
        shap_compatible = True
    except Exception as e:
        print(f"  SHAP TreeExplainer: not available ({e})")
        shap_compatible = False

    # Save model
    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open(ENCODINGS_PATH) as f:
        encodings = json.load(f)

    meta = {
        "best_model":      best_meta["model"],
        "n_samples":       n_samples,
        "feature_cols":    FEATURE_COLS,
        "encodings":       encodings,
        "shap_compatible": shap_compatible,
        "thresholds": {
            "medium": medium_thresh,
            "high":   high_thresh,
        },
        "metrics":     best_meta,
        "all_results": results,
    }
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to backend/app/ml/saved_models/")
    return best_model, meta


if __name__ == "__main__":
    train_models()
