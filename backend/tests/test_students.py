"""
Go Academics — /students endpoint tests.

Uses an in-memory SQLite database so tests don't pollute the demo DB.

Run from backend/:
    pytest tests/test_students.py -v
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db import get_db
from app.models.database import Base

# ── In-memory test database ───────────────────────────────────────────────────
# StaticPool keeps a single connection so Base.metadata.create_all and
# all subsequent queries share the same in-memory SQLite instance.

TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


client = TestClient(app)

STUDENT_A = {
    "student_code": "S001",
    "school_name": "School A",
    "gender": "Female",
    "stream": "Grade 10",
    "class_year": "Grade 10",
}

STUDENT_B = {
    "student_code": "S002",
    "school_name": "School B",
    "gender": "Male",
    "stream": "S6",
    "class_year": "S6",
}


# ── Creation ──────────────────────────────────────────────────────────────────

def test_create_student():
    r = client.post("/students", json=STUDENT_A)
    assert r.status_code == 201
    d = r.json()
    assert "S001" in d["message"]


def test_create_duplicate_student():
    client.post("/students", json=STUDENT_A)
    r = client.post("/students", json=STUDENT_A)
    assert r.status_code == 409


# ── Listing ───────────────────────────────────────────────────────────────────

def test_list_students_empty():
    r = client.get("/students")
    assert r.status_code == 200
    assert r.json() == []


def test_list_students_after_create():
    client.post("/students", json=STUDENT_A)
    client.post("/students", json=STUDENT_B)
    r = client.get("/students")
    assert r.status_code == 200
    codes = [s["student_code"] for s in r.json()]
    assert "S001" in codes
    assert "S002" in codes


def test_list_students_pagination():
    client.post("/students", json=STUDENT_A)
    client.post("/students", json=STUDENT_B)
    r = client.get("/students?limit=1")
    assert r.status_code == 200
    assert len(r.json()) == 1


# ── Detail ────────────────────────────────────────────────────────────────────

def test_get_student_existing():
    client.post("/students", json=STUDENT_A)
    r = client.get("/students/S001")
    assert r.status_code == 200
    d = r.json()
    assert d["student_code"] == "S001"
    assert d["school"] == "School A"
    assert d["gender"] == "Female"
    assert d["latest_risk_level"] is None   # no risk score recorded yet


def test_get_student_not_found():
    r = client.get("/students/S999")
    assert r.status_code == 404


# ── Assessment recording ──────────────────────────────────────────────────────

ASSESSMENT = {
    "term": 1,
    "subject": "English",
    "ca_score": 18.5,
    "exam_score": 45.0,
    "result": "Pass",
    "attendance_pct": 88.0,
    "days_absent": 5,
}


def test_add_assessment():
    client.post("/students", json=STUDENT_A)
    r = client.post("/students/S001/assessment", json=ASSESSMENT)
    assert r.status_code == 201
    assert "recorded" in r.json()["message"].lower()


def test_add_assessment_unknown_student():
    r = client.post("/students/S999/assessment", json=ASSESSMENT)
    assert r.status_code == 404
