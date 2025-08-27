"""
Monitoring and Metrics API Endpoints
Real-time monitoring and system metrics endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncpg

from ..core.database import get_connection
from ..core.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/monitoring",
    tags=["Monitoring"],
    dependencies=[Depends(get_current_user)]
)

@router.get("/metrics/latest")
async def get_latest_metrics(
    site_id: Optional[str] = Query(None),
    metric_name: Optional[str] = Query(None),
    technology: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get latest network metrics"""
    try:
        query = """
        SELECT DISTINCT ON (site_id, metric_name)
            site_id, metric_name, metric_value, unit, technology,
            quality_score, timestamp, metadata
        FROM network_metrics
        WHERE timestamp >= NOW() - INTERVAL '1 hour'
        """
        
        params = []
        
        if site_id:
            query += f" AND site_id = ${len(params) + 1}"
            params.append(site_id)
            
        if metric_name:
            query += f" AND metric_name = ${len(params) + 1}"
            params.append(metric_name)
            
        if technology:
            query += f" AND technology = ${len(params) + 1}"
            params.append(technology)
        
        query += f" ORDER BY site_id, metric_name, timestamp DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        rows = await conn.fetch(query, *params)
        
        metrics = []
        for row in rows:
            metric = {
                "site_id": row['site_id'],
                "metric_name": row['metric_name'],
                "value": float(row['metric_value']),
                "unit": row['unit'],
                "technology": row['technology'],
                "quality_score": float(row['quality_score']),
                "timestamp": row['timestamp'].isoformat(),
                "metadata": row['metadata'] or {}
            }
            metrics.append(metric)
        
        logger.info(f"Retrieved {len(metrics)} latest metrics")
        return {"metrics": metrics, "count": len(metrics)}
        
    except Exception as e:
        logger.error(f"Error retrieving latest metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@router.get("/events/recent")
async def get_recent_events(
    site_id: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    hours: int = Query(24, le=168),  # Max 1 week
    limit: int = Query(100, le=1000),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get recent events and alarms"""
    try:
        query = """
        SELECT timestamp, event_type, severity, title, description,
               site_id, source_system, event_id, acknowledged, metadata
        FROM events
        WHERE timestamp >= NOW() - INTERVAL %s
        """
        
        params = [f'{hours} hours']
        
        if site_id:
            query += f" AND site_id = ${len(params) + 1}"
            params.append(site_id)
            
        if severity:
            query += f" AND severity = ${len(params) + 1}"
            params.append(severity)
            
        if event_type:
            query += f" AND event_type = ${len(params) + 1}"
            params.append(event_type)
        
        query += f" ORDER BY timestamp DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        # Replace %s with proper parameter
        query = query.replace('%s', '$1')
        
        rows = await conn.fetch(query, *params)
        
        events = []
        for row in rows:
            event = {
                "timestamp": row['timestamp'].isoformat(),
                "event_type": row['event_type'],
                "severity": row['severity'],
                "title": row['title'],
                "description": row['description'],
                "site_id": row['site_id'],
                "source_system": row['source_system'],
                "event_id": row['event_id'],
                "acknowledged": row['acknowledged'],
                "metadata": row['metadata'] or {}
            }
            events.append(event)
        
        logger.info(f"Retrieved {len(events)} recent events")
        return {"events": events, "count": len(events)}
        
    except Exception as e:
        logger.error(f"Error retrieving recent events: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve events")

@router.get("/system/status")
async def get_system_status(
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get overall system status and statistics"""
    try:
        # Get basic statistics
        stats = await conn.fetchrow("""
            SELECT 
                (SELECT COUNT(*) FROM sites WHERE status = 'active') as active_sites,
                (SELECT COUNT(*) FROM network_metrics WHERE timestamp >= NOW() - INTERVAL '1 hour') as recent_metrics,
                (SELECT COUNT(*) FROM events WHERE timestamp >= NOW() - INTERVAL '1 hour') as recent_events,
                (SELECT COUNT(*) FROM kpi_alerts WHERE status = 'active') as active_alerts
        """)
        
        # Get data quality statistics
        quality_stats = await conn.fetchrow("""
            SELECT 
                AVG(quality_score) as avg_quality_score,
                MIN(quality_score) as min_quality_score,
                MAX(quality_score) as max_quality_score
            FROM network_metrics
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
        """)
        
        # Get top alert sites
        top_alert_sites = await conn.fetch("""
            SELECT site_id, COUNT(*) as alert_count
            FROM kpi_alerts
            WHERE status = 'active'
            GROUP BY site_id
            ORDER BY alert_count DESC
            LIMIT 5
        """)
        
        status_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": {
                "active_sites": stats['active_sites'],
                "recent_metrics": stats['recent_metrics'],
                "recent_events": stats['recent_events'], 
                "active_alerts": stats['active_alerts']
            },
            "data_quality": {
                "average_score": float(quality_stats['avg_quality_score']) if quality_stats['avg_quality_score'] else 0.0,
                "min_score": float(quality_stats['min_quality_score']) if quality_stats['min_quality_score'] else 0.0,
                "max_score": float(quality_stats['max_quality_score']) if quality_stats['max_quality_score'] else 0.0
            },
            "top_alert_sites": [
                {"site_id": row['site_id'], "alert_count": row['alert_count']}
                for row in top_alert_sites
            ]
        }
        
        logger.info("Retrieved system status")
        return status_info
        
    except Exception as e:
        logger.error(f"Error retrieving system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")