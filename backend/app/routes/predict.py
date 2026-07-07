"""
Go Academics — /predict route.

Phase-aware risk assessment:
  Phases 1-2: rule-based only (no ML score, just flag + recommendation)
  Phases 3-4: ML model + SHAP explanation
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.data.phase_logic import Phase, assess_risk_rule_based, get_phase
from app.ml.predict import predict_risk

router = APIRouter()


class PredictRequest(BaseModel):
    student_code: str = Field(..., json_schema_extra={"example": "S001"})
    week_number: int = Field(..., ge=1, le=12, description="Current week of term (1-12)")
    school: str = Field(..., json_schema_extra={"example": "School A"})
    gender: str = Field(..., json_schema_extra={"example": "Female"})
    stream: str = Field(..., json_schema_extra={"example": "Grade 10"})
    term: int = Field(..., ge=1, le=3)
    subject: str = Field(..., json_schema_extra={"example": "English"})
    ca_score: float = Field(..., ge=0, le=30)
    ca_trend: float = Field(default=0.0, description="Change from G1 to G2 scaled to /30 range")
    attendance_pct: float = Field(..., ge=0, le=100)


class PredictResponse(BaseModel):
    student_code: str
    phase: str
    risk_level: str
    risk_score: float | None        # null for phases 1-2
    recommendation: str
    top_factors: list[str]          # empty for phases 1-2
    explanation_method: str = ""    # "shap" | "feature_importance" | "" for rule-based


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    phase = get_phase(req.week_number)

    if phase in (Phase.PHASE_1, Phase.PHASE_2):
        risk_level, recommendation = assess_risk_rule_based(
            phase,
            attendance_pct=req.attendance_pct,
            ca_score=req.ca_score if phase == Phase.PHASE_2 else None,
        )
        return PredictResponse(
            student_code=req.student_code,
            phase=phase.value,
            risk_level=risk_level,
            risk_score=None,
            recommendation=recommendation,
            top_factors=[],
        )

    # Phases 3 & 4 — ML model
    features = {
        "School":         req.school,
        "Gender":         req.gender,
        "Stream":         req.stream,
        "Term":           req.term,
        "Subject":        req.subject,
        "CA_Score":       req.ca_score,
        "CA_Trend":       req.ca_trend,
        "Attendance_Pct": req.attendance_pct,
    }
    try:
        result = predict_risk(features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    recommendation = _recommendation(result["risk_level"], phase)

    return PredictResponse(
        student_code=req.student_code,
        phase=phase.value,
        risk_level=result["risk_level"],
        risk_score=result["risk_score"],
        recommendation=recommendation,
        top_factors=result["top_factors"],
        explanation_method=result.get("explanation_method", "feature_importance"),
    )


def _recommendation(risk_level: str, phase: Phase) -> str:
    if risk_level == "High":
        if phase == Phase.PHASE_4:
            return "Urgent: schedule parent-teacher meeting before end-of-term exams."
        return "Schedule one-on-one teacher check-in this week."
    if risk_level == "Medium":
        return "Monitor closely — review CA scores and attendance next week."
    return "No action required. Continue regular monitoring."
