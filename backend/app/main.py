"""
Go Academics — FastAPI entrypoint.

Serves the dashboard's static HTML at "/" and the API alongside it, so the
whole app deploys as a single service behind one URL — no separate frontend
host, no cross-origin requests.

Routes:
  GET  /                                  — teacher dashboard (dashboard/index.html)
  GET  /health
  POST /predict                           — phase-aware risk assessment
  GET  /students                          — list students
  POST /students                          — add student
  GET  /students/{code}                   — student detail + latest risk
  POST /students/{code}/assessment        — record CA score + attendance
  GET  /model-info                        — currently deployed model's identity + metrics
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.db import init_db, SessionLocal
from app.models.database import Student
from app.routes import model_info, predict, students

DASHBOARD_DIR = Path(__file__).parent.parent.parent / "dashboard"


def _seed_if_empty() -> None:
    """
    Cloud hosts (e.g. Render's free tier) don't guarantee the SQLite file
    survives a redeploy, so auto-seed real demo data on startup if the
    students table is empty. No-ops locally once data/go_academics.db exists.
    """
    db = SessionLocal()
    try:
        if db.query(Student).count() > 0:
            return
    finally:
        db.close()

    data_dir = Path(__file__).parent.parent.parent / "data"
    sys.path.insert(0, str(data_dir))
    import seed
    seed.seed()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    _seed_if_empty()
    yield


app = FastAPI(
    title="Go Academics API",
    description="Offline-first early warning system for student academic risk in Rwanda.",
    version="0.2.0",
    lifespan=lifespan,
)

# Allow the dashboard HTML (served locally) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, tags=["prediction"])
app.include_router(students.router, tags=["students"])
app.include_router(model_info.router, tags=["model"])


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "go-academics-api", "version": "0.2.0"}


# Mounted last: a mount at "/" would otherwise shadow the API routes above,
# since Starlette matches routes in registration order.
app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
