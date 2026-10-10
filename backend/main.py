from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.health import router as health_router
from backend.api.telemetry import router as telemetry_router
from backend.core.config import settings


# =========================================================
# Vanguard-GWM FastAPI Application
# =========================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Backend API for Vanguard-GWM — "
        "AI-powered network attack forecasting "
        "and defensive decision support."
    ),
)


# =========================================================
# CORS
# =========================================================
#
# Development configuration.
# We will restrict this before production deployment.
#

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# SYSTEM ROUTES
# =========================================================

app.include_router(
    health_router,
    prefix="/api/v1",
    tags=["System"],
)


# =========================================================
# TELEMETRY ROUTES
# =========================================================

app.include_router(
    telemetry_router,
    prefix="/api/v1",
    tags=["Telemetry"],
)


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get(
    "/",
    summary="Vanguard-GWM API information",
)
async def root():
    return {
        "name": "Vanguard-GWM",
        "service": "Cybersecurity Intelligence API",
        "version": settings.app_version,
        "status": "online",
    }