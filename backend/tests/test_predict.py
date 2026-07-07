"""
Go Academics — /predict endpoint tests.

One test per phase to verify the phase-aware routing logic:
  Phase 1 (week ≤ 3):  rule-based, attendance only
  Phase 2 (week 4-6):  rule-based, attendance + CA
  Phase 3 (week 7-9):  ML model, high-risk inputs → risk score returned
  Phase 4 (week 10-12): ML model, low-risk inputs → "Low" level

Run from backend/:
    pytest tests/test_predict.py -v
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

BASE = {
    "student_code": "TEST-001",
    "school": "School A",
    "gender": "Female",
    "stream": "Grade 10",
    "term": 1,
    "subject": "English",
    "ca_score": 14.0,
    "ca_trend": -3.0,
    "attendance_pct": 70.0,
}


def post_predict(**overrides):
    payload = {**BASE, **overrides}
    return client.post("/predict", json=payload)


# ── Phase 1 (weeks 1-3): rule-based, attendance only ─────────────────────────

def test_phase1_at_risk_attendance():
    r = post_predict(week_number=2, attendance_pct=65.0)
    assert r.status_code == 200
    d = r.json()
    assert d["phase"] == "Phase 1"
    assert d["risk_score"] is None          # rule-based → no ML score
    assert d["risk_level"] == "At Risk"
    assert "attendance" in d["recommendation"].lower()


def test_phase1_safe_attendance():
    r = post_predict(week_number=1, attendance_pct=95.0)
    assert r.status_code == 200
    d = r.json()
    assert d["phase"] == "Phase 1"
    assert d["risk_score"] is None
    assert d["risk_level"] == "Low Risk"


# ── Phase 2 (weeks 4-6): rule-based, attendance + CA ─────────────────────────

def test_phase2_high_risk_both_metrics_fail():
    r = post_predict(week_number=5, ca_score=10.0, attendance_pct=70.0)
    assert r.status_code == 200
    d = r.json()
    assert d["phase"] == "Phase 2"
    assert d["risk_score"] is None
    assert d["risk_level"] == "At Risk"


def test_phase2_low_risk_both_metrics_pass():
    r = post_predict(week_number=6, ca_score=22.0, attendance_pct=92.0)
    assert r.status_code == 200
    d = r.json()
    assert d["phase"] == "Phase 2"
    assert d["risk_level"] == "Low Risk"


# ── Phase 3 (weeks 7-9): ML model ────────────────────────────────────────────

def test_phase3_ml_score_returned():
    r = post_predict(week_number=8)
    assert r.status_code == 200
    d = r.json()
    assert d["phase"] == "Phase 3"
    assert d["risk_score"] is not None
    assert 0.0 <= d["risk_score"] <= 1.0


def test_phase3_high_risk_inputs():
    r = post_predict(
        week_number=8,
        ca_score=8.0,
        ca_trend=-6.0,
        attendance_pct=60.0,
    )
    assert r.status_code == 200
    d = r.json()
    assert d["risk_score"] is not None
    assert d["risk_level"] in ("High", "Medium")   # poor inputs → elevated risk


def test_phase3_low_risk_inputs():
    r = post_predict(
        week_number=8,
        ca_score=27.0,
        ca_trend=3.0,
        attendance_pct=97.0,
    )
    assert r.status_code == 200
    d = r.json()
    assert d["risk_score"] is not None
    assert d["risk_level"] == "Low"


def test_phase3_top_factors_returned():
    r = post_predict(week_number=8)
    assert r.status_code == 200
    d = r.json()
    assert isinstance(d["top_factors"], list)
    assert len(d["top_factors"]) > 0


# ── Phase 4 (weeks 10-12): ML model + intervention recommendation ─────────────

def test_phase4_high_risk_recommendation():
    r = post_predict(
        week_number=11,
        ca_score=7.0,
        ca_trend=-8.0,
        attendance_pct=55.0,
    )
    assert r.status_code == 200
    d = r.json()
    assert d["phase"] == "Phase 4"
    assert d["risk_score"] is not None
    if d["risk_level"] == "High":
        assert "parent" in d["recommendation"].lower() or "exam" in d["recommendation"].lower()


# ── Validation ────────────────────────────────────────────────────────────────

def test_invalid_week_number():
    r = post_predict(week_number=0)
    assert r.status_code == 422


def test_invalid_ca_score_too_high():
    r = post_predict(week_number=8, ca_score=35.0)
    assert r.status_code == 422
