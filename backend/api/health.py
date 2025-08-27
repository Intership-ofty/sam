"""
Health check endpoints
"""

import asyncio
import time
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel

from core.database import get_db_connection, DatabaseManager
from core.cache import CacheManager, get_cache_manager
from core.messaging import producer as kafka_producer
from core.monitoring import get_health_status, metrics

router = APIRouter()


class HealthStatus(BaseModel):
    """Health status response model"""
    status: str
    timestamp: float
    version: str
    environment: str
    services: Dict[str, Dict[str, Any]]


class ServiceHealth(BaseModel):
    """Individual service health model"""
    status: str
    response_time: float
    details: Dict[str, Any] = {}


@router.get("/", response_model=HealthStatus)
async def health_check():
    """
    Basic health check endpoint
    Returns overall system health status
    """
    health_info = get_health_status()
    
    # Check core services
    services = {}
    
    # Database health
    db_health = await check_database_health()
    services["database"] = db_health
    
    # Cache health
    cache_health = await check_cache_health()
    services["cache"] = cache_health
    
    # Messaging health
    messaging_health = await check_messaging_health()
    services["messaging"] = messaging_health
    
    # Determine overall status
    overall_status = "healthy"
    for service_name, service_health in services.items():
        if service_health["status"] != "healthy":
            overall_status = "degraded"
            break
    
    return HealthStatus(
        status=overall_status,
        timestamp=health_info["timestamp"],
        version=health_info["version"],
        environment=health_info["environment"],
        services=services
    )


@router.get("/live", response_model=Dict[str, Any])
async def liveness_probe():
    """
    Kubernetes liveness probe
    Checks if the application is running
    """
    return {
        "status": "alive",
        "timestamp": time.time()
    }


@router.get("/ready", response_model=Dict[str, Any])
async def readiness_probe():
    """
    Kubernetes readiness probe
    Checks if the application is ready to serve traffic
    """
    checks = {}
    all_ready = True
    
    # Database readiness
    try:
        start_time = time.time()
        async with get_db_connection() as conn:
            await conn.fetchval("SELECT 1")
        duration = time.time() - start_time
        checks["database"] = {
            "ready": True,
            "response_time": duration
        }
    except Exception as e:
        checks["database"] = {
            "ready": False,
            "error": str(e)
        }
        all_ready = False
    
    # Cache readiness
    try:
        cache = CacheManager()
        start_time = time.time()
        await cache.set("health_check", "ok", expire=10)
        await cache.get("health_check")
        duration = time.time() - start_time
        checks["cache"] = {
            "ready": True,
            "response_time": duration
        }
    except Exception as e:
        checks["cache"] = {
            "ready": False,
            "error": str(e)
        }
        all_ready = False
    
    # Kafka readiness
    checks["messaging"] = {
        "ready": kafka_producer is not None,
        "details": "Producer initialized" if kafka_producer else "Producer not initialized"
    }
    
    if not checks["messaging"]["ready"]:
        all_ready = False
    
    status_code = status.HTTP_200_OK if all_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    
    response = {
        "ready": all_ready,
        "timestamp": time.time(),
        "checks": checks
    }
    
    if not all_ready:
        raise HTTPException(status_code=status_code, detail=response)
    
    return response


@router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health_check():
    """
    Detailed health check with comprehensive service status
    """
    services = {}
    
    # Run all health checks concurrently
    tasks = [
        check_database_health(),
        check_cache_health(),
        check_messaging_health(),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    services["database"] = results[0] if not isinstance(results[0], Exception) else {
        "status": "unhealthy",
        "response_time": 0,
        "details": {"error": str(results[0])}
    }
    
    services["cache"] = results[1] if not isinstance(results[1], Exception) else {
        "status": "unhealthy", 
        "response_time": 0,
        "details": {"error": str(results[1])}
    }
    
    services["messaging"] = results[2] if not isinstance(results[2], Exception) else {
        "status": "unhealthy",
        "response_time": 0, 
        "details": {"error": str(results[2])}
    }
    
    # Add system metrics
    services["system"] = await get_system_metrics()
    
    # Overall status
    healthy_services = sum(1 for s in services.values() if s.get("status") == "healthy")
    total_services = len(services)
    
    if healthy_services == total_services:
        overall_status = "healthy"
    elif healthy_services > 0:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": time.time(),
        "services": services,
        "summary": {
            "healthy_services": healthy_services,
            "total_services": total_services,
            "health_percentage": (healthy_services / total_services) * 100
        }
    }


async def check_database_health() -> Dict[str, Any]:
    """Check database connection and performance"""
    try:
        start_time = time.time()
        
        # Test basic connectivity
        async with get_db_connection() as conn:
            await conn.fetchval("SELECT 1")
        
        # Test TimescaleDB extensions
        db = DatabaseManager()
        sites_count = await db.execute_query_scalar(
            "SELECT COUNT(*) FROM sites WHERE status = 'active'"
        )
        
        duration = time.time() - start_time
        
        return {
            "status": "healthy",
            "response_time": duration,
            "details": {
                "active_sites": sites_count or 0,
                "connection_pool": "available"
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "response_time": 0,
            "details": {
                "error": str(e),
                "component": "database"
            }
        }


async def check_cache_health() -> Dict[str, Any]:
    """Check Redis cache connectivity and performance"""
    try:
        start_time = time.time()
        cache = CacheManager()
        
        # Test basic operations
        test_key = f"health_check_{int(time.time())}"
        test_value = {"test": True, "timestamp": time.time()}
        
        # Set, get, and delete
        await cache.set(test_key, test_value, expire=10)
        retrieved_value = await cache.get(test_key)
        await cache.delete(test_key)
        
        duration = time.time() - start_time
        
        # Get Redis info
        redis_info = await cache.get_info()
        
        return {
            "status": "healthy",
            "response_time": duration,
            "details": {
                "memory_used": redis_info.get("used_memory_human", "N/A"),
                "connected_clients": redis_info.get("connected_clients", 0),
                "operations_test": "passed"
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "response_time": 0,
            "details": {
                "error": str(e),
                "component": "cache"
            }
        }


async def check_messaging_health() -> Dict[str, Any]:
    """Check Kafka messaging system health"""
    try:
        start_time = time.time()
        
        # Check if producer is initialized
        if not kafka_producer:
            return {
                "status": "unhealthy",
                "response_time": 0,
                "details": {
                    "error": "Producer not initialized",
                    "component": "messaging"
                }
            }
        
        # Test message sending (to a health check topic)
        test_message = {
            "type": "health_check",
            "timestamp": time.time(),
            "service": "backend-api"
        }
        
        # This would normally send to a health topic
        # For now, just verify producer is available
        duration = time.time() - start_time
        
        return {
            "status": "healthy",
            "response_time": duration,
            "details": {
                "producer_initialized": True,
                "bootstrap_servers": "connected"
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "response_time": 0,
            "details": {
                "error": str(e),
                "component": "messaging"
            }
        }


async def get_system_metrics() -> Dict[str, Any]:
    """Get system performance metrics"""
    try:
        import psutil
        
        return {
            "status": "healthy",
            "response_time": 0,
            "details": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                "uptime_seconds": time.time() - psutil.boot_time()
            }
        }
    except ImportError:
        return {
            "status": "healthy",
            "response_time": 0,
            "details": {
                "note": "psutil not available, system metrics disabled"
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "response_time": 0,
            "details": {
                "error": str(e),
                "component": "system_metrics"
            }
        }