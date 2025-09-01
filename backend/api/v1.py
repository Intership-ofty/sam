"""
API v1 Router - Main API routing configuration
"""

from fastapi import APIRouter
from routers import aiops, business_intelligence, kpi, monitoring, noc, notifications, sites, tenants

api_router = APIRouter()

# Include all the routers from the 'routers' directory
api_router.include_router(sites.router, prefix="/sites", tags=["sites"])
api_router.include_router(tenants.router, prefix="/tenants", tags=["tenants"])
api_router.include_router(noc.router, prefix="/noc", tags=["noc"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
api_router.include_router(notifications.router, prefix="/notifications", tags=["notifications"])
api_router.include_router(kpi.router, prefix="/kpi", tags=["kpi"])
api_router.include_router(business_intelligence.router, prefix="/bi", tags=["bi"])
api_router.include_router(aiops.router, prefix="/aiops", tags=["aiops"])
