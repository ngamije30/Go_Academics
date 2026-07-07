"""
Go Academics — /students routes.

Covers the core CRUD operations the dashboard needs:
  GET  /students           — list all (paginated)
  POST /students           — add a new anonymized student record
  GET  /students/{code}    — fetch one student + their latest risk score
  POST /students/{code}/assessment  — record a new CA score / attendance entry
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.data.phase_logic import Phase as PhaseEnum, assess_risk_rule_based, get_phase
from app.ml.predict import predict_risk
from app.models.database import Assessment, Attendance, Phase, RiskScore, School, Student

router = APIRouter()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class StudentIn(BaseModel):
    student_code: str = Field(..., json_schema_extra={"example": "S001"})
    school_name: str = Field(..., json_schema_extra={"example": "School A"})
    gender: str = Field(..., json_schema_extra={"example": "Female"})
    stream: str = Field(..., json_schema_extra={"example": "Grade 10"})
    class_year: str = Field(..., json_schema_extra={"example": "Grade 10"})


class AssessmentIn(BaseModel):
    term: int = Field(..., ge=1, le=3)
    subject: str = Field(..., json_schema_extra={"example": "English"})
    ca_score: float = Field(..., ge=0, le=30)
    exam_score: float = Field(0.0, ge=0, le=70)
    result: str = Field(..., json_schema_extra={"example": "Pass"})
    attendance_pct: float = Field(..., ge=0, le=100)
    days_absent: int = Field(0, ge=0)


class StudentOut(BaseModel):
    student_code: str
    school: str
    gender: str
    stream: str
    class_year: str
    # Latest assessment
    subject: Optional[str] = None
    ca_score: Optional[float] = None
    term: Optional[int] = None
    result: Optional[str] = None
    # Latest attendance
    attendance_pct: Optional[float] = None
    days_absent: Optional[int] = None
    # Latest risk
    latest_risk_level: Optional[str] = None
    latest_risk_score: Optional[float] = None
    latest_recommendation: Optional[str] = None
    latest_phase: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_or_create_school(db: Session, name: str) -> School:
    school = db.query(School).filter(School.name == name).first()
    if not school:
        school = School(name=name, location="Kigali")
        db.add(school)
        db.flush()
    return school


def _student_out(student: Student) -> StudentOut:
    latest_risk  = student.risk_scores[-1]    if student.risk_scores          else None
    latest_asmt  = student.assessments[-1]    if student.assessments          else None
    latest_att   = student.attendance_records[-1] if student.attendance_records else None
    phase_label  = None
    if latest_risk and latest_risk.phase:
        phase_label = latest_risk.phase.label

    return StudentOut(
        student_code=student.student_code,
        school=student.school.name,
        gender=student.gender,
        stream=student.stream,
        class_year=student.class_year,
        subject=latest_asmt.subject         if latest_asmt else None,
        ca_score=latest_asmt.ca_score       if latest_asmt else None,
        term=latest_asmt.term               if latest_asmt else None,
        result=latest_asmt.result           if latest_asmt else None,
        attendance_pct=latest_att.attendance_pct if latest_att else None,
        days_absent=latest_att.days_absent   if latest_att else None,
        latest_risk_level=latest_risk.level  if latest_risk else None,
        latest_risk_score=latest_risk.score  if latest_risk else None,
        latest_recommendation=latest_risk.recommendation if latest_risk else None,
        latest_phase=phase_label,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/students")
def list_students(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    students = (
        db.query(Student)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [_student_out(s) for s in students]


@router.post("/students", status_code=201)
def create_student(data: StudentIn, db: Session = Depends(get_db)):
    existing = db.query(Student).filter(Student.student_code == data.student_code).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Student {data.student_code} already exists.")

    school = _get_or_create_school(db, data.school_name)
    student = Student(
        student_code=data.student_code,
        school_id=school.id,
        gender=data.gender,
        stream=data.stream,
        class_year=data.class_year,
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return {"message": f"Student {data.student_code} created.", "id": student.id}


@router.get("/students/{student_code}")
def get_student(student_code: str, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_code == student_code).first()
    if not student:
        raise HTTPException(status_code=404, detail=f"Student {student_code} not found.")
    return _student_out(student)


@router.post("/students/{student_code}/assessment", status_code=201)
def add_assessment(
    student_code: str,
    data: AssessmentIn,
    db: Session = Depends(get_db),
):
    student = db.query(Student).filter(Student.student_code == student_code).first()
    if not student:
        raise HTTPException(status_code=404, detail=f"Student {student_code} not found.")

    assessment = Assessment(
        student_id=student.id,
        term=data.term,
        subject=data.subject,
        ca_score=data.ca_score,
        exam_score=data.exam_score,
        result=data.result,
    )
    db.add(assessment)

    attendance = Attendance(
        student_id=student.id,
        term=data.term,
        attendance_pct=data.attendance_pct,
        days_absent=data.days_absent,
    )
    db.add(attendance)
    db.flush()  # flush so we can read previous assessments for CA_Trend

    # ── Auto risk prediction ───────────────────────────────────────────────
    # Determine phase from available data:
    # CA score provided and > 0 → Phase 3 (ML), else → Phase 1 (rule-based)
    has_ca    = data.ca_score > 0
    week_num  = 8 if has_ca else 2
    phase_val = get_phase(week_num)

    # Compute CA_Trend from previous assessment if one exists
    prev_assessments = [a for a in student.assessments if a.id != assessment.id]
    ca_trend = 0.0
    if prev_assessments and has_ca:
        prev_ca  = prev_assessments[-1].ca_score
        ca_trend = round(((data.ca_score - prev_ca) / 20) * 30, 1)

    if phase_val in (PhaseEnum.PHASE_1, PhaseEnum.PHASE_2):
        risk_level, recommendation = assess_risk_rule_based(
            phase_val,
            attendance_pct=data.attendance_pct,
            ca_score=data.ca_score if phase_val == PhaseEnum.PHASE_2 else None,
        )
        risk_score = None
    else:
        features = {
            "School":         student.school.name,
            "Gender":         student.gender,
            "Stream":         student.stream,
            "Term":           data.term,
            "Subject":        data.subject,
            "CA_Score":       data.ca_score,
            "CA_Trend":       ca_trend,
            "Attendance_Pct": data.attendance_pct,
        }
        try:
            pred = predict_risk(features)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        risk_level    = pred["risk_level"]
        risk_score    = pred["risk_score"]
        recommendation = _risk_recommendation(risk_level, phase_val)

    # Map phase enum to Phase DB record
    phase_num_map = {
        PhaseEnum.PHASE_1: 1, PhaseEnum.PHASE_2: 2,
        PhaseEnum.PHASE_3: 3, PhaseEnum.PHASE_4: 4,
    }
    phase_record = (
        db.query(Phase)
        .filter(Phase.phase_number == phase_num_map[phase_val])
        .first()
    )
    if not phase_record:
        # Create on the fly if phases were never seeded
        phase_record = Phase(
            phase_number=phase_num_map[phase_val],
            label=phase_val.value,
            week_range=f"Weeks {week_num}-{week_num+2}",
        )
        db.add(phase_record)
        db.flush()

    db.add(RiskScore(
        student_id=student.id,
        phase_id=phase_record.id,
        score=risk_score,
        level=risk_level,
        recommendation=recommendation,
    ))
    db.commit()

    return {
        "message":       "Assessment recorded and risk score updated.",
        "risk_level":    risk_level,
        "risk_score":    risk_score,
        "recommendation": recommendation,
    }


@router.delete("/students/{student_code}", status_code=200)
def delete_student(student_code: str, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_code == student_code).first()
    if not student:
        raise HTTPException(status_code=404, detail=f"Student {student_code} not found.")
    for a in student.assessments:      db.delete(a)
    for a in student.attendance_records: db.delete(a)
    for r in student.risk_scores:      db.delete(r)
    db.delete(student)
    db.commit()
    return {"message": f"Student {student_code} deleted."}


def _risk_recommendation(risk_level: str, phase: PhaseEnum) -> str:
    if risk_level == "High":
        if phase == PhaseEnum.PHASE_4:
            return "Urgent: schedule parent-teacher meeting before end-of-term exams."
        return "Schedule one-on-one teacher check-in this week."
    if risk_level == "Medium":
        return "Monitor closely - review CA scores and attendance next week."
    return "No action required. Continue regular monitoring."
