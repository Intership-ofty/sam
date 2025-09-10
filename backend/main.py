#!/usr/bin/env python3
"""
Towerco AIOps Platform - Main FastAPI Application
Enterprise-grade AIOps platform for telecom infrastructure management
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.config import settings
from core.database import init_db
from core.cache import init_redis
from core.messaging import init_kafka, start_consumer
from core.monitoring import init_monitoring, metrics
from core.auth import init_auth
from api.v1 import api_router
from api.health import router as health_router
from api.auth import router as auth_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Towerco AIOps Platform...")
    
    # Initialize monitoring first
    await init_monitoring()
    
    # Initialize core services
    await init_db()
    await init_redis()
    await init_kafka()
    await init_auth()
    
    # Start background tasks
    asyncio.create_task(background_tasks())
    
    # Start backend Kafka consumers to receive worker results
    await start_consumer(
        'backend_results',
        ['towerco.kpi.calculations', 'towerco.aiops.predictions', 'towerco.alerts'],
        'backend_group',
        backend_message_handler
    )
    
    logger.info("Towerco AIOps Platform started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Towerco AIOps Platform...")


# Create FastAPI application
app = FastAPI(
    title="Towerco AIOps Platform",
    description="""
    Enterprise AIOps platform specialized for telecom infrastructure management.
    
    ## Features
    
    - **Real-time Data Ingestion**: Universal connectors for OSS, ITSM, IoT systems
    - **KPI Intelligence**: Automated calculation of 50+ telecom KPIs
    - **AIOps & RCA**: Machine learning-powered root cause analysis
    - **Multi-tenant Portal**: Client self-service with SLA dashboards
    - **NOC Intelligence**: Unified operations center with AI assistance
    - **Business Intelligence**: Executive reporting and ROI tracking
    
    ## Authentication
    
    This API uses OAuth2 with OpenID Connect via Keycloak.
    All endpoints require valid JWT tokens except health checks.
    """,
    version="1.0.0",
    contact={
        "name": "Towerco AIOps Team",
        "email": "support@towerco-aiops.com",
    },
    license_info={
        "name": "Proprietary",
    },
    openapi_tags=[
        {
            "name": "health",
            "description": "System health and monitoring endpoints",
        },
        {
            "name": "sites",
            "description": "Telecom site management",
        },
        {
            "name": "metrics",
            "description": "Network and energy metrics ingestion/retrieval",
        },
        {
            "name": "kpis",
            "description": "KPI calculation and monitoring",
        },
        {
            "name": "events",
            "description": "Event and alarm management",
        },
        {
            "name": "aiops",
            "description": "AI operations and root cause analysis",
        },
        {
            "name": "reports",
            "description": "Business intelligence and reporting",
        },
        {
            "name": "admin",
            "description": "Administrative endpoints",
        },
    ],
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "path": str(request.url.path),
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": 422,
                "message": "Validation error",
                "details": exc.errors(),
                "path": str(request.url.path),
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "path": str(request.url.path),
            }
        },
    )


# Request middleware for monitoring
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """Add monitoring to all requests"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        response = await call_next(request)
        duration = asyncio.get_event_loop().time() - start_time
        
        # Record metrics
        metrics.http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        metrics.http_request_duration_seconds.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response
        
    except Exception as e:
        duration = asyncio.get_event_loop().time() - start_time
        
        metrics.http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=500
        ).inc()
        
        raise


# Include routers
app.include_router(health_router, prefix="/health", tags=["health"])

# Add direct health endpoint to avoid 307 redirects
@app.get("/health")
async def health_direct():
    """Direct health check endpoint"""
    from api.health import health_check
    return await health_check()
app.include_router(auth_router)  # No prefix, already has /api/v1/auth
app.include_router(api_router, prefix="/api/v1")


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Towerco AIOps Platform",
        "version": "1.0.0",
        "description": "Enterprise AIOps platform for telecom infrastructure",
        "status": "operational",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
        "health_url": "/health",
        "metrics_url": "/metrics",
    }


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from starlette.responses import Response
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def backend_message_handler(topic, key, value, headers, timestamp, partition, offset):
    """Handle messages from workers"""
    logger.info(f"Backend received message from {topic}: {key}")
    
    try:
        if topic == 'towerco.kpi.calculations':
            # KPI Worker sent results
            await handle_kpi_result(value)
            
        elif topic == 'towerco.aiops.predictions':
            # AIOps Workers sent predictions/analysis
            await handle_aiops_result(value)
            
        elif topic == 'towerco.alerts':
            # Workers detected alerts
            await handle_alert_result(value)
            
    except Exception as e:
        logger.error(f"Error handling worker message: {e}")

async def handle_kpi_result(data):
    """Handle KPI calculation results from workers"""
    # Store results in database, trigger notifications, etc.
    logger.info(f"Processing KPI result: {data.get('kpi_name', 'unknown')}")

async def handle_aiops_result(data):
    """Handle AIOps prediction results from workers"""
    # Store predictions, trigger alerts if needed
    logger.info(f"Processing AIOps result: {data.get('prediction_type', 'unknown')}")

async def handle_alert_result(data):
    """Handle alert results from workers"""
    # Process alerts, send notifications
    logger.info(f"Processing alert: {data.get('alert_type', 'unknown')}")

async def background_tasks():
    """Background tasks for system maintenance"""
    while True:
        try:
            # Update system metrics every 30 seconds
            await asyncio.sleep(30)
            await update_system_metrics()
            
        except Exception as e:
            logger.exception(f"Background task error: {e}")
            await asyncio.sleep(60)  # Wait longer on error


async def update_system_metrics():
    """Update system performance metrics"""
    try:
        # Database health check - test connectivity
        from core.database import get_db_pool
        pool = await get_db_pool()
        if pool:
            try:
                # Test database connectivity
                async with pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                # Database is healthy
                metrics.db_health.set(1)
            except Exception as e:
                # Database is unhealthy
                logger.warning(f"Database health check failed: {e}")
                metrics.db_health.set(0)
        else:
            metrics.db_health.set(0)
        
        # Update Redis connection metrics
        from core.cache import redis_client
        if redis_client:
            info = await redis_client.info('memory')
            metrics.redis_memory_used_bytes.set(info.get('used_memory', 0))
        
    except Exception as e:
        logger.exception(f"Failed to update system metrics: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )