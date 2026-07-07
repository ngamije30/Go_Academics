"""
Go Academics — Data Preprocessing (Plan B: UCI Student Performance Dataset)

INACTIVE FALLBACK — real Kigali school data (Excella Secondary School) has
arrived, so the active pipeline is data/preprocess.py. Keep this script only
in case future data collection is delayed and the UCI proxy is needed again.
See docs/plan_b.md.

Remaps UCI columns to the Go Academics feature structure and adds a
CA_Trend feature (change from G1 to G2) which captures performance
trajectory — more predictive than a single snapshot.

Output: data/processed/students.csv       (raw remapped, human-readable)
        data/processed/students_ml.csv    (encoded + SMOTE balanced, training-ready)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

RAW_DIR = Path(__file__).parent / "raw"
OUT_DIR = Path(__file__).parent / "processed"
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Load both CSVs ─────────────────────────────────────────────────────────

mat = pd.read_csv(RAW_DIR / "student-mat.csv", sep=";")
por = pd.read_csv(RAW_DIR / "student-por.csv", sep=";")
mat["Subject"] = "Mathematics"
por["Subject"] = "English"
raw = pd.concat([mat, por], ignore_index=True)
print(f"Loaded {len(mat)} Math + {len(por)} English = {len(raw)} total rows")

# ── 2. Column mapping functions ───────────────────────────────────────────────

def map_school(s):    return "School A" if s == "GP" else "School B"
def map_gender(s):    return "Female" if s == "F" else "Male"
def map_stream(age):
    if age <= 16: return "S4"
    if age <= 17: return "S5"
    return "S6"

def calc_attendance(absences):
    return round(min(max(0.0, (1 - absences / 50) * 100), 100), 1)

def scale_ca(g):
    return round((g / 20) * 30, 1)

def scale_exam(g3):
    return round((g3 / 20) * 70, 1)

def pass_fail(g3):
    return "Pass" if g3 >= 10 else "Fail"

# ── 3. Build feature table ────────────────────────────────────────────────────

terms = ([1] * (len(raw) // 3 + 1) +
         [2] * (len(raw) // 3 + 1) +
         [3] * (len(raw) // 3 + 1))[:len(raw)]

df = pd.DataFrame()
df["StudentID"]      = [f"S{str(i+1).zfill(3)}" for i in range(len(raw))]
df["School"]         = raw["school"].apply(map_school)
df["Gender"]         = raw["sex"].apply(map_gender)
df["Stream"]         = raw["age"].apply(map_stream)
df["Term"]           = terms
df["Subject"]        = raw["Subject"]
df["CA_Score"]       = raw["G1"].apply(scale_ca)          # first CA (/30)
df["CA_Trend"]       = (raw["G2"] - raw["G1"]).apply(     # change G1→G2, scaled to /30
                           lambda d: round((d / 20) * 30, 1))
df["Exam_Score"]     = raw["G3"].apply(scale_exam)
df["Attendance_Pct"] = raw["absences"].apply(calc_attendance)
df["Final_Result"]   = raw["G3"].apply(pass_fail)

print(f"\nClass distribution:\n{df['Final_Result'].value_counts()}")
print(f"Pass rate: {(df['Final_Result']=='Pass').mean()*100:.1f}%")
print(f"\nCA_Trend stats (negative = declining, positive = improving):")
print(df["CA_Trend"].describe().round(2))

df.to_csv(OUT_DIR / "students.csv", index=False)
print(f"\nSaved {len(df)} rows to data/processed/students.csv")

# ── 4. Encode for ML training ─────────────────────────────────────────────────

ml = df.copy()

cat_cols = ["School", "Gender", "Stream", "Subject"]
for col in cat_cols:
    le = LabelEncoder()
    ml[col] = le.fit_transform(ml[col])

ml["Final_Result"] = (ml["Final_Result"] == "Pass").astype(int)

# CA_Trend replaces Exam_Score (which leaks from G3, the same source as Final_Result)
FEATURE_COLS = ["School", "Gender", "Stream", "Term",
                "Subject", "CA_Score", "CA_Trend", "Attendance_Pct"]
X = ml[FEATURE_COLS]
y = ml["Final_Result"]

# ── 5. Apply SMOTE to balance classes ─────────────────────────────────────────

print(f"\nBefore SMOTE - Pass: {y.sum()} | Fail: {(y==0).sum()}")
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)
print(f"After  SMOTE - Pass: {y_bal.sum()} | Fail: {(y_bal==0).sum()}")

ml_balanced = pd.DataFrame(X_bal, columns=FEATURE_COLS)
ml_balanced["Final_Result"] = y_bal
ml_balanced.to_csv(OUT_DIR / "students_ml.csv", index=False)
print(f"\nSaved {len(ml_balanced)} rows to data/processed/students_ml.csv")
print("Preprocessing complete.")
