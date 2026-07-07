"""
Go Academics — Data Preprocessing (Plan A: real Kigali school data)

Loads real, already-anonymized-format term files collected from the pilot
schools (e.g. data/raw/excella_school_a_term1.csv) and prepares them for the
ML pipeline. Unlike the Plan B (UCI) pipeline, these files already match the
Go Academics column structure — only the real school name needs anonymizing
and CA_Trend needs deriving across terms.

Subjects and streams are NOT hardcoded anywhere in this script: whatever
values appear in the raw files (e.g. "English"/"French" for School A Term 1)
flow straight through, since the subject/stream mix will differ by school and
grow as more term files are collected.

Output: data/processed/students.csv       (anonymized, human-readable)
        data/processed/students_ml.csv    (encoded + SMOTE balanced, training-ready)
        data/processed/encodings.json     (LabelEncoder class mappings, consumed by
                                            backend/app/ml/train.py so inference never
                                            has to hardcode category values)
"""

import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
OUT_DIR = Path(__file__).parent / "processed"
OUT_DIR.mkdir(exist_ok=True)

# Real school name (as transcribed in raw files) -> anonymized Go Academics code.
# Add an entry here the first time a new school's term file is collected.
SCHOOL_CODES = {
    "Excella Secondary School Rwanda": "School A",
}

# ── 1. Load every real term file ────────────────────────────────────────────

RAW_FILES = sorted(RAW_DIR.glob("*_term*.csv"))
if not RAW_FILES:
    raise FileNotFoundError(f"No real school term files found in {RAW_DIR}")

frames = [pd.read_csv(f) for f in RAW_FILES]
raw = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(raw)} rows from {len(RAW_FILES)} file(s): "
      f"{[f.name for f in RAW_FILES]}")

# ── 2. Anonymize school name ────────────────────────────────────────────────

unknown = set(raw["School"]) - set(SCHOOL_CODES)
if unknown:
    raise ValueError(
        f"Unmapped school name(s) {unknown} — add them to SCHOOL_CODES in "
        f"data/preprocess.py before preprocessing."
    )

df = raw.copy()
df["School"] = df["School"].map(SCHOOL_CODES)

# ── 3. Derive CA_Trend (change vs. this student's previous term, same subject) ──
# First-term rows have no prior CA score, so trend is 0 (neutral) until a
# second term is collected for that student/subject.

df = df.sort_values(["StudentID", "Subject", "Term"]).reset_index(drop=True)
df["CA_Trend"] = (
    df.groupby(["StudentID", "Subject"])["CA_Score"]
    .diff()
    .fillna(0.0)
    .round(1)
)

print(f"\nClass distribution:\n{df['Final_Result'].value_counts()}")
print(f"Pass rate: {(df['Final_Result']=='Pass').mean()*100:.1f}%")
print(f"\nSubjects present: {sorted(df['Subject'].unique())}")
print(f"Streams present:  {sorted(df['Stream'].unique())}")
print(f"Terms present:    {sorted(df['Term'].unique())}")

df.to_csv(OUT_DIR / "students.csv", index=False)
print(f"\nSaved {len(df)} rows to data/processed/students.csv")

# ── 4. Encode for ML training ───────────────────────────────────────────────

ml = df.copy()

cat_cols = ["School", "Gender", "Stream", "Subject"]
encodings = {}
for col in cat_cols:
    le = LabelEncoder()
    ml[col] = le.fit_transform(ml[col])
    encodings[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

ml["Final_Result"] = (ml["Final_Result"] == "Pass").astype(int)

FEATURE_COLS = ["School", "Gender", "Stream", "Term",
                "Subject", "CA_Score", "CA_Trend", "Attendance_Pct"]
X = ml[FEATURE_COLS]
y = ml["Final_Result"]

with open(OUT_DIR / "encodings.json", "w") as f:
    json.dump(encodings, f, indent=2)
print("Saved category encodings to data/processed/encodings.json")

# ── 5. Apply SMOTE to balance classes ───────────────────────────────────────

print(f"\nBefore SMOTE - Pass: {y.sum()} | Fail: {(y==0).sum()}")
minority_count = y.value_counts().min()
smote = SMOTE(random_state=42, k_neighbors=min(5, minority_count - 1))
X_bal, y_bal = smote.fit_resample(X, y)
print(f"After  SMOTE - Pass: {y_bal.sum()} | Fail: {(y_bal==0).sum()}")

ml_balanced = pd.DataFrame(X_bal, columns=FEATURE_COLS)
ml_balanced["Final_Result"] = y_bal
ml_balanced.to_csv(OUT_DIR / "students_ml.csv", index=False)
print(f"\nSaved {len(ml_balanced)} rows to data/processed/students_ml.csv")
print("Preprocessing complete.")
