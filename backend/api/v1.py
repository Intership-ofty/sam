"""
API v1 Router - Main API routing configuration
"""

from fastapi import APIRouter
from ..routers import aiops, business_intelligence, kpi, monitoring, noc, notifications, sites, tenants 

api_router = APIRouter()

# Include all sub-routers
api_router.include_router(aiops.router, tags=["AIOps"])
api_router.include_router(business_intelligence.router, tags=["Business Intelligence"])
api_router.include_router(kpi.router, tags=["KPI Management"])
api_router.include_router(monitoring.router, tags=["Monitoring"])
api_router.include_router(noc.router, tags=["Noc"])
api_router.include_router(notifications.router, tags=["Notifications"])
api_router.include_router(sites.router, tags=["Sites"])
api_router.include_router(tenants.router, tags=["Tenants"])