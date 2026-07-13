"""
Go Academics — Combined Dataset Preprocessing (Excella + UCI Plan B)

Per supervisor instruction: merge the real Excella data with the UCI Plan B
proxy dataset before splitting, to increase training data volume and improve
model calibration (particularly the 0-Medium-risk-students issue caused by
the small 151-row Excella-only dataset). See CLAUDE.md.

This imports build_dataframe() from both data/preprocess.py and
data/preprocess_plan_b.py (rather than shelling out to those scripts) because
both write to the same output filenames — running them back-to-back would
have the second silently clobber the first's output.

IMPORTANT: this does NOT touch data/processed/students.csv (Excella-only,
human-readable). That file is what data/seed.py uses to seed the live demo
database with the 78 real Excella students — synthetic UCI rows must never
reach the seeded database.

Run after data/preprocess.py, before backend/app/ml/train.py:
    python data/preprocess.py
    python data/preprocess_merge.py
    python backend/app/ml/train.py

Output:
    data/processed/students_combined.csv  (merged, human-readable, tagged by Source)
    data/processed/students_ml.csv        (merged + SMOTE-balanced, training-ready —
                                            overwrites the Excella-only version)
    data/processed/encodings.json         (single consistent encoding across both
                                            sources — overwrites the Excella-only version)
"""

import json
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

import preprocess as plan_a
import preprocess_plan_b as plan_b

OUT_DIR = Path(__file__).parent / "processed"
OUT_DIR.mkdir(exist_ok=True)

FEATURE_COLS = ["School", "Gender", "Stream", "Term",
                "Subject", "CA_Score", "CA_Trend", "Attendance_Pct"]


def merge():
    # ── 1 & 3. Build both dataframes and tag by source ──────────────────────
    df_a = plan_a.build_dataframe()
    df_a["Source"] = "Excella"

    df_b = plan_b.build_dataframe()
    df_b["Source"] = "UCI"
    # UCI StudentIDs are auto-generated S001... which collide with the real
    # anonymized Excella codes (S001-S078) — reprefix so they're never confused.
    df_b["StudentID"] = df_b["StudentID"].str.replace("^S", "U", regex=True)

    combined = pd.concat([df_a, df_b], ignore_index=True)
    print(f"\nCombined dataset: {len(df_a)} Excella + {len(df_b)} UCI = "
          f"{len(combined)} rows")

    combined.to_csv(OUT_DIR / "students_combined.csv", index=False)
    print(f"Saved {len(combined)} rows to data/processed/students_combined.csv")

    # ── 2. Encode categoricals on the FULL combined data ────────────────────
    # (source is documentation only — excluded from FEATURE_COLS below)
    ml = combined.copy()
    cat_cols = ["School", "Gender", "Stream", "Subject"]
    encodings = {}
    for col in cat_cols:
        le = LabelEncoder()
        ml[col] = le.fit_transform(ml[col])
        encodings[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

    ml["Final_Result"] = (ml["Final_Result"] == "Pass").astype(int)

    X = ml[FEATURE_COLS]
    y = ml["Final_Result"]

    with open(OUT_DIR / "encodings.json", "w") as f:
        json.dump(encodings, f, indent=2)
    print("Saved combined category encodings to data/processed/encodings.json")

    # ── 4. Apply SMOTE on the combined data ─────────────────────────────────
    print(f"\nBefore SMOTE - Pass: {y.sum()} | Fail: {(y==0).sum()}")
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    print(f"After  SMOTE - Pass: {y_bal.sum()} | Fail: {(y_bal==0).sum()}")

    ml_balanced = pd.DataFrame(X_bal, columns=FEATURE_COLS)
    ml_balanced["Final_Result"] = y_bal
    ml_balanced.to_csv(OUT_DIR / "students_ml.csv", index=False)
    print(f"\nSaved {len(ml_balanced)} rows to data/processed/students_ml.csv")
    print("\nCombined preprocessing complete. Run backend/app/ml/train.py next "
          "(train_test_split there does the 80/20 split, after this merge).")


if __name__ == "__main__":
    merge()
