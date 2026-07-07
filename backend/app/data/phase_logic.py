"""
Go Academics — Four-phase early warning logic.

Phases 1-2 are rule-based (insufficient term data for ML yet).
Phases 3-4 hand off to the ML model in app/ml/predict.py.

See CLAUDE.md at the project root for the full phase definitions —
do not change the phase boundaries or rule thresholds without updating
that file too, since the research proposal cites these exact numbers.
"""
from enum import Enum


class Phase(str, Enum):
    PHASE_1 = "Phase 1"  # Weeks 1-3 — attendance only
    PHASE_2 = "Phase 2"  # Weeks 4-6 — attendance + first CA
    PHASE_3 = "Phase 3"  # Weeks 7-9 — ML model, first prediction
    PHASE_4 = "Phase 4"  # Weeks 10-12 — ML model, final prediction


def get_phase(week_number: int) -> Phase:
    """Map the current week of term (1-12) to one of the four phases."""
    if week_number <= 3:
        return Phase.PHASE_1
    elif week_number <= 6:
        return Phase.PHASE_2
    elif week_number <= 9:
        return Phase.PHASE_3
    else:
        return Phase.PHASE_4


def assess_risk_rule_based(phase: Phase, attendance_pct: float, ca_score: float | None = None):
    """
    Rule-based risk check for Phase 1 and Phase 2.

    TODO: replace placeholder thresholds with values validated against
    the real Kigali school dataset once collected (see data/raw/).
    """
    if phase == Phase.PHASE_1:
        if attendance_pct < 80:
            return "At Risk", "Monitor attendance"
        return "Low Risk", "No action needed yet"

    if phase == Phase.PHASE_2:
        if attendance_pct < 80 or (ca_score is not None and ca_score < 15):
            return "At Risk", "Schedule teacher check-in"
        return "Low Risk", "No action needed yet"

    raise ValueError(f"{phase} is not rule-based — use the ML model instead (app/ml/predict.py)")
