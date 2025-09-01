"""
API v1 router configuration
"""

from fastapi import APIRouter
from .sites import router as sites_router
from .metrics import router as metrics_router
from .events import router as events_router
from .kpis import router as kpis_router
from .reports import router as reports_router
from .admin import router as admin_router

# Create main API router
api_router = APIRouter()

# Include all sub-routers
api_router.include_router(
    sites_router,
    prefix="/sites",
    tags=["sites"]
)

api_router.include_router(
    metrics_router,
    prefix="/metrics",
    tags=["metrics"]
)

api_router.include_router(
    events_router,
    prefix="/events",
    tags=["events"]
)

api_router.include_router(
    kpis_router,
    prefix="/kpis",
    tags=["kpis"]
)

api_router.include_router(
    reports_router,
    prefix="/reports",
    tags=["reports"]
)

api_router.include_router(
    admin_router,
    prefix="/admin",
    tags=["admin"]
)