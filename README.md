# Go Academics

**Offline-first ML early warning system for student academic risk in Rwanda.**

Predicts at-risk secondary school students across four structured phases of the academic term and delivers actionable alerts to teachers — before end-of-term exams.

> Capstone project — ALU BSc Software Engineering · NGAMIJE RUHUMULIZA Davy

---

## What it does

| Phase | Weeks | Method | Trigger |
|-------|-------|--------|---------|
| 1 | 1–3 | Rule-based | Attendance < 80% |
| 2 | 4–6 | Rule-based | Attendance < 80% **or** CA < 15/30 |
| 3 | 7–9 | XGBoost ML | Risk score ≥ 0.35 |
| 4 | 10–12 | XGBoost ML | Risk score ≥ 0.35 + intervention recommendation |

---

## Prerequisites

- Python 3.10+
- pip (comes with Python)
- A modern browser (Chrome/Edge recommended)

---

## Setup (first time only)

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Prepare the dataset

Real school term files live in `data/raw/` already, named `<school>_term<N>.csv`
(e.g. `excella_school_a_term1.csv`), matching the Go Academics column schema.
To add a new school or term, drop the file in `data/raw/` and register the
real school name in `SCHOOL_CODES` in `data/preprocess.py` if it's new.

Then run preprocessing:
```bash
python data/preprocess.py
```

This creates:
- `data/processed/students.csv` — anonymized, human-readable
- `data/processed/students_ml.csv` — SMOTE-balanced for training
- `data/processed/encodings.json` — category encodings used by inference

(If real data collection for a school/term is delayed, the UCI-based Plan B
pipeline is still available via `python data/preprocess_plan_b.py` — see
`docs/plan_b.md`.)

### 3. Train the ML model

```bash
python backend/app/ml/train.py
```

Trains Logistic Regression, Random Forest, and XGBoost, and picks the best by recall (tiebreak F1). Saved to `backend/app/ml/saved_models/best_model.pkl`.

### 4. Seed the database

```bash
python data/seed.py
```

Creates `data/go_academics.db` and populates it with all 78 real Excella students from
`data/processed/students.csv` — assessments, attendance, and risk scores are computed by
the actual trained model, not fabricated.

---

## Running the system

### Start the backend API

```bash
cd backend
python -m uvicorn app.main:app --port 8000 --reload
```

API is now live at `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

### Open the dashboard

Open this file in your browser:
```
dashboard/Go Academics.html
```

The live prediction widget appears in the bottom-right corner. A green dot confirms the API is connected.

---

## Project structure

```
Go_Academics/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entrypoint
│   │   ├── db.py                # SQLite session factory
│   │   ├── models/database.py   # SQLAlchemy models
│   │   ├── ml/
│   │   │   ├── train.py         # Training pipeline
│   │   │   ├── predict.py       # Inference + risk factors
│   │   │   └── saved_models/    # best_model.pkl + model_meta.json
│   │   ├── routes/
│   │   │   ├── predict.py       # POST /predict (phase-aware)
│   │   │   └── students.py      # GET/POST /students
│   │   └── data/phase_logic.py  # Four-phase rule engine
│   └── requirements.txt
├── data/
│   ├── raw/                     # Real school term CSVs + UCI fallback files
│   ├── processed/               # Preprocessed CSVs + encodings.json
│   ├── preprocess.py            # Real data → Go Academics pipeline (active)
│   ├── preprocess_plan_b.py     # UCI fallback pipeline (inactive)
│   └── seed.py                  # Demo database seed
├── dashboard/
│   └── Go Academics.html        # Teacher dashboard (offline-capable)
├── docs/
│   └── ML_Track_Notebook.ipynb  # ML demo notebook (Initial Software Demo)
└── CLAUDE.md                    # Project memory and conventions
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| POST | `/predict` | Phase-aware risk prediction |
| GET | `/students` | List all students |
| POST | `/students` | Add a student |
| GET | `/students/{code}` | Student detail + latest risk |
| POST | `/students/{code}/assessment` | Record CA score + attendance |

---

## Data privacy

- No real student names or national IDs are stored anywhere
- Students are identified by anonymised codes only (S001, S002, …)
- Schools are referred to by anonymized codes ("School A", …) in all raw data, the
  database, CSVs, and model encodings. "School A" is the internal system code for
  Excella Secondary School Rwanda — the dashboard UI displays the real name to
  teachers (`SCHOOL_DISPLAY_NAMES` in `dashboard/index.html`), but nothing else
  (data files, API payloads, model training) ever uses it. Keep this in mind before
  screenshotting the dashboard into anything that should stay anonymized.
- Current data is one pilot school, Excella Secondary School Rwanda, located in Kigali (urban context)

---

## Tech stack

- **Backend**: FastAPI + SQLAlchemy + SQLite
- **ML**: scikit-learn · XGBoost · imbalanced-learn (SMOTE)
- **Frontend**: React (Claude Design export) — runs fully offline
- **Data**: real anonymized Kigali school records (Plan A); UCI Student Performance Dataset kept as an inactive Plan B fallback
