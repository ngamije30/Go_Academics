# Plan B Dataset — UCI Student Performance

> **Status: merged into training data (not a standalone fallback anymore).**
> Real Kigali school data (Excella Secondary School, School A, Term 1)
> remains the only data ever seeded into the live demo database
> (`data/preprocess.py` → `data/processed/students.csv` → `data/seed.py`).
> But per supervisor instruction, this UCI dataset is now combined with
> Excella data for **model training only**, via `data/preprocess_merge.py`,
> to give the model more rows to learn from and calibrate risk thresholds
> against. The column-mapping logic below still lives in
> `data/preprocess_plan_b.py`, now imported as a module rather than run
> standalone. Performance numbers below are from the last **combined**
> training run — see `backend/app/ml/saved_models/model_meta.json` for the
> current live metrics.

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

## Model performance on Plan B data (historical, UCI-only — superseded)

Results below are from the last **UCI-only** training run, before Excella data
existed. Kept for historical comparison only — see CLAUDE.md's "Current Model
Performance" section (or live `/model-info`) for the current combined-dataset
metrics, which is what's actually deployed.

| Model | Recall | Precision | F1 | AUC-ROC | CV Recall |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 88.96% | 96.03% | 92.36% | 98.10% | 84.54% ± 10.23% |
| Random Forest | 90.80% | 98.01% | 94.27% | 99.01% | 74.96% ± 25.85% |
| **XGBoost (selected)** | **93.87%** | **97.45%** | **95.62%** | **99.04%** | **83.56% ± 15.03%** |

**CV instability note**: Random Forest showed high cross-validation variance
(±25.85%) on UCI-only data. The same instability shows up on the combined
Excella+UCI dataset too (±26.06%) — see CLAUDE.md — so this looks like a
structural sensitivity of Random Forest to this feature set at this sample
size, not something that resolves with more rows alone.

---

## Risk thresholds

Thresholds are derived from the precision-recall curve on the hold-out test set, not chosen arbitrarily. The derivation is in `train.py → optimal_thresholds()`.

Historical UCI-only thresholds: Medium ≥ 0.621, High ≥ 0.771. **Current
thresholds** (combined Excella+UCI dataset) are lower — Medium ≥ 0.414, High ≥
0.564 — see `backend/app/ml/saved_models/model_meta.json` for the live values.

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

## What happened when real Kigali data arrived

Real Excella data arrived and became the sole source for the live demo
database (`data/preprocess.py` → `students.csv` → `data/seed.py`) — UCI rows
never reach the database or any real-student-facing output.

The supervisor then asked for Excella + UCI to be **merged for training**
(more rows → better-calibrated thresholds, fixing the 0-Medium-risk-students
problem the 151-row Excella-only dataset produced). That merge is implemented
in `data/preprocess_merge.py`:

1. Excella rows keep their real "School A" / anonymised codes; UCI rows get
   "School A"/"School B" via the same `map_school()` mapping as before.
2. UCI's age→stream heuristic (`map_stream()`) is unchanged — Excella rows
   already carry their real "Grade 10" stream directly.
3. CA_Trend is calculated per-source: Excella uses actual two-CA-score
   history (`preprocess.py`), UCI still uses the G1→G2 proxy.
4. Retrain: `python data/preprocess.py && python data/preprocess_merge.py &&
   python backend/app/ml/train.py`.
5. Thresholds re-derived automatically from the combined PR curve — see above.
6. Performance numbers above are kept as historical UCI-only reference;
   current combined numbers live in CLAUDE.md and `model_meta.json`.

No code changes were required in `predict.py`, `routes/predict.py`, or the
dashboard — only the preprocessing and training steps changed.
