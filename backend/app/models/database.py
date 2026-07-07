"""
Go Academics — Database models.

These map directly to the ERD designed for the capstone proposal:
School -> Student -> {Assessment, Attendance, RiskScore}
RiskScore belongs to a Phase.

Kept deliberately simple (SQLite-friendly) per the offline-first design —
no advanced Postgres-only features here.
"""
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class School(Base):
    __tablename__ = "schools"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)       # "School A" / "School B" only — never real names
    location = Column(String, nullable=False)   # "Kigali"

    students = relationship("Student", back_populates="school")


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True)
    student_code = Column(String, unique=True, nullable=False)  # e.g. "S001" — anonymized ID only
    school_id = Column(Integer, ForeignKey("schools.id"), nullable=False)
    gender = Column(String, nullable=False)
    stream = Column(String, nullable=False)      # e.g. "S4 PCB"
    class_year = Column(String, nullable=False)  # e.g. "S4"

    school = relationship("School", back_populates="students")
    assessments = relationship("Assessment", back_populates="student")
    attendance_records = relationship("Attendance", back_populates="student")
    risk_scores = relationship("RiskScore", back_populates="student")


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    term = Column(Integer, nullable=False)        # 1, 2, or 3
    subject = Column(String, nullable=False)      # varies by school, e.g. "English", "French"
    ca_score = Column(Float, nullable=False)       # out of 30
    exam_score = Column(Float, nullable=False)     # out of 70
    result = Column(String, nullable=False)        # "Pass" / "Fail"

    student = relationship("Student", back_populates="assessments")


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    term = Column(Integer, nullable=False)
    attendance_pct = Column(Float, nullable=False)
    days_absent = Column(Integer, nullable=False)

    student = relationship("Student", back_populates="attendance_records")


class Phase(Base):
    __tablename__ = "phases"

    id = Column(Integer, primary_key=True)
    phase_number = Column(Integer, nullable=False)  # 1-4
    label = Column(String, nullable=False)           # e.g. "Foundation"
    week_range = Column(String, nullable=False)       # e.g. "Week 1-3"

    risk_scores = relationship("RiskScore", back_populates="phase")


class RiskScore(Base):
    __tablename__ = "risk_scores"

    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    phase_id = Column(Integer, ForeignKey("phases.id"), nullable=False)
    score = Column(Float, nullable=True)           # null in phases 1-2 (rule-based, no probability)
    level = Column(String, nullable=False)          # "Low" / "Medium" / "High"
    recommendation = Column(String, nullable=True)  # teacher-facing suggested action

    student = relationship("Student", back_populates="risk_scores")
    phase = relationship("Phase", back_populates="risk_scores")
