"""
Go Academics — Demo Database Seed

Populates the local SQLite database from the real Excella Secondary School
Rwanda dataset (data/processed/students.csv — see data/preprocess.py), rather
than hand-authored fixtures:
  - 1 school (School A), 78 real anonymised students, all Grade 10, Term 1
  - Assessment (per subject) + attendance (per student) records
  - Risk scores computed by the actual trained model (backend/app/ml/predict.py),
    not fabricated — every RiskScore here is a real Phase 3 prediction

Attendance_Pct is per student-term (not per subject), so each student gets one
Attendance row; days_absent isn't in the raw data, so it's estimated from
Attendance_Pct assuming a ~60-school-day term (12 weeks x 5 days).

Run from the project root:
    python data/seed.py
"""

import csv
import sys
from pathlib import Path

# Make backend importable
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.db import init_db, SessionLocal
from app.data.phase_logic import Phase as PhaseEnum
from app.ml.predict import predict_risk
from app.models.database import Assessment, Attendance, Phase, RiskScore, School, Student
from app.routes.students import _risk_recommendation

STUDENTS_CSV = Path(__file__).parent / "processed" / "students.csv"
TERM_SCHOOL_DAYS = 60  # ~12 weeks x 5 days, used only to estimate days_absent for display

SCHOOLS = [
    {"name": "School A", "location": "Kigali"},
]

PHASES = [
    {"phase_number": 1, "label": "Foundation",   "week_range": "Weeks 1-3"},
    {"phase_number": 2, "label": "Early Check",  "week_range": "Weeks 4-6"},
    {"phase_number": 3, "label": "Prediction",   "week_range": "Weeks 7-9"},
    {"phase_number": 4, "label": "Intervention", "week_range": "Weeks 10-12"},
]


def _load_real_students():
    """Group data/processed/students.csv rows by StudentID (order-preserving)."""
    with open(STUDENTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    by_student = {}
    for row in rows:
        by_student.setdefault(row["StudentID"], []).append(row)
    return by_student


def _estimate_days_absent(attendance_pct: float) -> int:
    return max(0, round((100 - attendance_pct) / 100 * TERM_SCHOOL_DAYS))


def seed():
    init_db()
    db = SessionLocal()

    try:
        # Skip if already seeded
        if db.query(Student).count() > 0:
            print("Database already contains student records — skipping seed.")
            print(f"  Students: {db.query(Student).count()}")
            return

        print("Seeding database from real Excella Secondary School data...")

        # Schools
        school_map = {}
        for s in SCHOOLS:
            school = School(**s)
            db.add(school)
            db.flush()
            school_map[s["name"]] = school
        print(f"  Added {len(SCHOOLS)} schools")

        # Phases
        phase_map = {}
        for p in PHASES:
            phase = Phase(**p)
            db.add(phase)
            db.flush()
            phase_map[p["phase_number"]] = phase
        print(f"  Added {len(PHASES)} phases")

        # Students, assessments, attendance, and real model-predicted risk scores
        by_student = _load_real_students()
        prediction_phase = phase_map[3]  # full CA + attendance data == Phase 3 (ML prediction)

        for i, (code, subject_rows) in enumerate(by_student.items()):
            # The dashboard shows each student's *last-inserted* assessment as
            # "current" (mirroring how a live system treats the most recent
            # submission). The raw CSV always lists English before French for
            # every student, so without this, French would always win and
            # English would never appear in the summary view. Alternate the
            # insertion order so both subjects are fairly represented.
            if i % 2 == 1:
                subject_rows = list(reversed(subject_rows))

            first = subject_rows[0]
            student = Student(
                student_code=code,
                school_id=school_map[first["School"]].id,
                gender=first["Gender"],
                stream=first["Stream"],
                class_year=first["Stream"],
            )
            db.add(student)
            db.flush()

            attendance_pct = float(first["Attendance_Pct"])
            db.add(Attendance(
                student_id=student.id,
                term=int(first["Term"]),
                attendance_pct=attendance_pct,
                days_absent=_estimate_days_absent(attendance_pct),
            ))

            for row in subject_rows:
                ca_score   = float(row["CA_Score"])
                exam_score = float(row["Exam_Score"])
                db.add(Assessment(
                    student_id=student.id,
                    term=int(row["Term"]),
                    subject=row["Subject"],
                    ca_score=ca_score,
                    exam_score=exam_score,
                    result=row["Final_Result"],
                ))

                pred = predict_risk({
                    "School":         row["School"],
                    "Gender":         row["Gender"],
                    "Stream":         row["Stream"],
                    "Term":           int(row["Term"]),
                    "Subject":        row["Subject"],
                    "CA_Score":       ca_score,
                    "CA_Trend":       float(row["CA_Trend"]),
                    "Attendance_Pct": attendance_pct,
                })
                db.add(RiskScore(
                    student_id=student.id,
                    phase_id=prediction_phase.id,
                    score=pred["risk_score"],
                    level=pred["risk_level"],
                    recommendation=_risk_recommendation(pred["risk_level"], PhaseEnum.PHASE_3),
                ))

        db.commit()
        print(f"  Added {len(by_student)} students (real data) with assessments, "
              f"attendance, and model-predicted risk scores")
        print("\nSeed complete. Database is ready for demo.")

    except Exception as e:
        db.rollback()
        print(f"Seed failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
