# Plan B Dataset — UCI Student Performance

> **Status: inactive.** Real Kigali school data arrived (Excella Secondary
> School, School A, Term 1) and is now the active pipeline via
> `data/preprocess.py`. The Plan B logic described below has moved to
> `data/preprocess_plan_b.py` and is kept only for reference / reactivation
> if data collection for a future school or term is delayed. Performance
> numbers below are from the last Plan B training run — see
> `backend/app/ml/saved_models/model_meta.json` for current real-data metrics.

## Why Plan B exists

Go Academics was designed around primary data collected directly from two secondary schools in Kigali (referred to as School A and School B throughout all project outputs). If that data collection is delayed or the ethics clearance process extends beyond the available capstone timeline, the system falls back to a publicly available proxy dataset so that the ML pipeline can still be demonstrated and evaluated end-to-end.

**Plan B is the UCI Student Performance Dataset** (Cortez & Silva, 2008, Portugal):
> P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, April 2008.

Original source: [UCI ML Repository — Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/student+performance)

Files used: `student-mat.csv` (Mathematics, n=395) and `student-por.csv` (English/Portuguese, n=649).

---

## Column mapping to Go Academics schema

The UCI dataset uses different column names and value ranges. `data/preprocess.py` remaps every field to match the Go Academics internal schema before any ML pipeline step runs.

| UCI column | Go Academics column | Transformation |
|---|---|---|
| `school` ("GP"/"MS") | `School` | "GP" → "School A", "MS" → "School B" |
| `sex` ("F"/"M") | `Gender` | "F" → "Female", "M" → "Male" |
| `age` (15-22) | `Stream` | ≤16 → "S4", ≤17 → "S5", ≥18 → "S6" |
| `absences` (count) | `Attendance_Pct` | `min(max(0, (1 - absences/50) × 100), 100)` |
| `G1` (/20) | `CA_Score` | `(G1/20) × 30` — remapped to /30 range |
| `G2 - G1` | `CA_Trend` | `((G2-G1)/20) × 30` — learning trajectory |
| `G3` (/20) | `Final_Result` | G3 ≥ 10 → "Pass", G3 < 10 → "Fail" |

`G3` (the final exam grade) is **not** included as a training feature — it is the source of the target label, so using it would constitute direct data leakage. `G2` is used only to compute `CA_Trend` (the change from G1 to G2), not as a standalone feature.

---

## Class distribution (post-remapping)

| Label | Count | % |
|---|---|---|
| Pass | 814 | 78% |
| Fail | 230 | 22% |

The dataset is imbalanced. SMOTE is applied during preprocessing to create a balanced training set (814 Pass : 814 Fail = 1,628 rows total).

---

## Model performance on Plan B data

Results from the most recent training run (`backend/app/ml/train.py`):

| Model | Recall | Precision | F1 | AUC-ROC | CV Recall |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 88.96% | 96.03% | 92.36% | 98.10% | 84.54% ± 10.23% |
| Random Forest | 90.80% | 98.01% | 94.27% | 99.01% | 74.96% ± 25.85% |
| **XGBoost (selected)** | **93.87%** | **97.45%** | **95.62%** | **99.04%** | **83.56% ± 15.03%** |

XGBoost wins on both recall (priority metric) and F1. Results saved in full to `backend/app/ml/saved_models/model_meta.json`.

**CV instability note**: Random Forest shows high cross-validation variance (±25.85%). This is expected with 1,628 rows — the model is at the boundary of reliable generalisation. XGBoost is more stable (±15.03%) and is the chosen model. When real Kigali school data is available (~480 rows), CV variance will likely increase further due to fewer samples; regularisation parameters (`max_depth`, `learning_rate`) should be re-tuned at that point.

---

## Risk thresholds

Thresholds are derived from the precision-recall curve on the hold-out test set, not chosen arbitrarily. The derivation is in `train.py → optimal_thresholds()`:

| Level | Threshold | Method |
|---|---|---|
| Medium | score ≥ 0.621 | Maximum F1 point on at-risk PR curve |
| High | score ≥ 0.771 | First point where at-risk precision ≥ 90% |

Where `risk_score = 1 − P(Pass)`.

---

## Limitations vs. real Kigali data

| Dimension | Plan B (UCI/Portugal) | Plan A (Kigali schools) |
|---|---|---|
| Country context | Portugal, 2008 | Rwanda, current |
| Grading system | /20 (Portuguese) | /30 CA + /70 Exam (Rwanda) |
| Age → class mapping | Approximate | Exact (S4/S5/S6 enrollment) |
| Attendance proxy | Absence count → % formula | Direct teacher records |
| Subject coverage | Mathematics + Portuguese | Mathematics + English |
| Cultural/SES factors | Not Rwanda-specific | Captured via school/stream context |

The Plan B remapping preserves the structural relationships between features and the Pass/Fail outcome. However, a model trained on Portuguese data may not generalise perfectly to Rwandan students — particularly for features like attendance patterns (Kigali urban context) and socioeconomic correlates that differ between countries. All claims about model accuracy in the capstone report are explicitly scoped to the Plan B dataset.

---

## What changes when real Kigali data arrives

1. **Remove the school mapping** — real data will already have "School A" / "School B" anonymised codes.
2. **Remove the age→stream heuristic** — real data will have direct S4/S5/S6 enrollment.
3. **Recalculate CA_Trend** — use the actual two CA scores per term instead of G1/G2 proxy.
4. **Retrain from scratch** — run `python data/preprocess.py && python backend/app/ml/train.py`.
5. **Re-evaluate thresholds** — the PR-curve threshold computation in `train.py` will automatically derive new values from the new data.
6. **Update this file** — replace performance numbers above with real-data numbers.

No code changes are required in `predict.py`, `routes/predict.py`, or the dashboard — only the preprocessing and training steps change.
