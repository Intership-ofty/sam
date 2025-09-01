"""
KPI Management API Endpoints
Business Intelligence et KPI Calculation API pour Towerco AIOps
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import json
import asyncio
import logging
from pydantic import BaseModel, Field
import asyncpg

from ..core.database import get_connection
from ..core.auth import get_current_user
from ..core.models import KPICalculationRequest, KPIAlert, KPIDashboard

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/kpi",
    tags=["KPI Management"],
    dependencies=[Depends(get_current_user)]
)

# Pydantic models for API
class KPIMetricResponse(BaseModel):
    """Response model for KPI metrics"""
    kpi_name: str
    current_value: float
    target_value: Optional[float] = None
    unit: str
    category: str
    trend: str = Field(..., description="up, down, stable")
    quality_score: float = Field(ge=0.0, le=1.0)
    last_calculated: datetime
    prediction_7d: Optional[float] = None
    metadata: Dict[str, Any] = {}

class KPITrendData(BaseModel):
    """KPI trend data points"""
    timestamp: datetime
    value: float
    predicted: bool = False
    
class KPITrendResponse(BaseModel):
    """KPI trend response with historical data"""
    kpi_name: str
    time_range: str
    data_points: List[KPITrendData]
    statistics: Dict[str, float]

class KPIAlertCreate(BaseModel):
    """Create KPI alert model"""
    kpi_name: str
    site_id: Optional[str] = None
    condition: str = Field(..., description="gt, lt, eq, change_rate")
    threshold: float
    severity: str = Field(..., pattern="^(critical|major|minor|warning)$")
    notification_channels: List[str] = Field(default=["email"])
    enabled: bool = True

class KPIDashboardCreate(BaseModel):
    """Create KPI dashboard model"""
    name: str
    description: Optional[str] = None
    kpi_list: List[str]
    layout_config: Dict[str, Any] = {}
    filters: Dict[str, Any] = {}
    refresh_interval: int = Field(default=30, description="Refresh interval in seconds")
    is_public: bool = False

@router.get("/metrics", response_model=List[KPIMetricResponse])
async def get_kpi_metrics(
    site_id: Optional[str] = Query(None, description="Filter by site ID"),
    category: Optional[str] = Query(None, description="KPI category: network, energy, operational, financial"),
    time_range: str = Query("1h", description="Time range: 15m, 1h, 4h, 24h, 7d"),
    limit: int = Query(50, le=500),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get current KPI metrics with trend analysis"""
    try:
        # Build base query
        query = """
        WITH latest_kpis AS (
            SELECT DISTINCT ON (site_id, kpi_name)
                site_id, kpi_name, kpi_value, unit, category, quality_score,
                calculated_at, target_value, metadata
            FROM kpi_calculations
            WHERE calculated_at >= NOW() - INTERVAL %s
        """
        
        params = [time_range.replace('m', ' minutes').replace('h', ' hours').replace('d', ' days')]
        
        # Add filters
        if site_id:
            query += " AND site_id = $%s"
            params.append(site_id)
            
        if category:
            query += " AND category = $%s"
            params.append(category)
        
        query += """
            ORDER BY site_id, kpi_name, calculated_at DESC
        ),
        kpi_trends AS (
            SELECT 
                kpi_name,
                site_id,
                AVG(kpi_value) as avg_value,
                CASE 
                    WHEN AVG(kpi_value) > LAG(AVG(kpi_value)) OVER (PARTITION BY kpi_name, site_id ORDER BY date_trunc('hour', calculated_at)) THEN 'up'
                    WHEN AVG(kpi_value) < LAG(AVG(kpi_value)) OVER (PARTITION BY kpi_name, site_id ORDER BY date_trunc('hour', calculated_at)) THEN 'down'
                    ELSE 'stable'
                END as trend_direction
            FROM kpi_calculations
            WHERE calculated_at >= NOW() - INTERVAL '2 days'
            GROUP BY kpi_name, site_id, date_trunc('hour', calculated_at)
        ),
        predictions AS (
            SELECT 
                kpi_name, site_id,
                AVG(predicted_value) as prediction_7d
            FROM kpi_predictions
            WHERE prediction_date >= NOW() AND prediction_date <= NOW() + INTERVAL '7 days'
            GROUP BY kpi_name, site_id
        )
        SELECT 
            k.kpi_name,
            k.kpi_value as current_value,
            k.target_value,
            k.unit,
            k.category,
            COALESCE(t.trend_direction, 'stable') as trend,
            k.quality_score,
            k.calculated_at as last_calculated,
            p.prediction_7d,
            k.metadata
        FROM latest_kpis k
        LEFT JOIN kpi_trends t ON k.kpi_name = t.kpi_name AND k.site_id = t.site_id
        LEFT JOIN predictions p ON k.kpi_name = p.kpi_name AND k.site_id = p.site_id
        ORDER BY k.category, k.kpi_name
        LIMIT $%s
        """
        
        params.append(limit)
        
        # Execute query
        query_params = [f"${i+1}" for i in range(len(params))]
        final_query = query % tuple(query_params[:-1]) + f" LIMIT ${len(params)}"
        
        rows = await conn.fetch(final_query, *params)
        
        # Transform results
        metrics = []
        for row in rows:
            metric = KPIMetricResponse(
                kpi_name=row['kpi_name'],
                current_value=float(row['current_value']),
                target_value=float(row['target_value']) if row['target_value'] else None,
                unit=row['unit'],
                category=row['category'],
                trend=row['trend'] or 'stable',
                quality_score=float(row['quality_score']),
                last_calculated=row['last_calculated'],
                prediction_7d=float(row['prediction_7d']) if row['prediction_7d'] else None,
                metadata=row['metadata'] or {}
            )
            metrics.append(metric)
        
        logger.info(f"Retrieved {len(metrics)} KPI metrics")
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving KPI metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve KPI metrics")

@router.get("/trends/{kpi_name}", response_model=KPITrendResponse)
async def get_kpi_trend(
    kpi_name: str,
    site_id: Optional[str] = Query(None),
    time_range: str = Query("24h"),
    resolution: str = Query("1h", description="Data resolution: 5m, 15m, 1h, 4h"),
    include_predictions: bool = Query(True),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get KPI trend data with historical values and predictions"""
    try:
        # Historical data query
        interval_map = {
            "5m": "5 minutes", "15m": "15 minutes", 
            "1h": "1 hour", "4h": "4 hours"
        }
        
        time_interval = time_range.replace('m', ' minutes').replace('h', ' hours').replace('d', ' days')
        data_resolution = interval_map.get(resolution, "1 hour")
        
        query = """
        WITH historical_data AS (
            SELECT 
                date_trunc(%s, calculated_at) as time_bucket,
                AVG(kpi_value) as avg_value,
                false as predicted
            FROM kpi_calculations
            WHERE kpi_name = $2
                AND calculated_at >= NOW() - INTERVAL %s
        """
        
        params = [resolution, kpi_name, time_interval]
        
        if site_id:
            query += " AND site_id = $4"
            params.append(site_id)
        
        query += """
            GROUP BY date_trunc(%s, calculated_at)
            ORDER BY time_bucket
        ),
        prediction_data AS (
            SELECT 
                prediction_date as time_bucket,
                predicted_value as avg_value,
                true as predicted
            FROM kpi_predictions
            WHERE kpi_name = $2
        """
        
        if site_id:
            query += " AND site_id = $4"
        
        query += " AND prediction_date >= NOW()"
        
        if include_predictions:
            query += """
        )
        SELECT time_bucket, avg_value, predicted
        FROM historical_data
        UNION ALL
        SELECT time_bucket, avg_value, predicted
        FROM prediction_data
        ORDER BY time_bucket
        """
        else:
            query += """
        )
        SELECT time_bucket, avg_value, predicted
        FROM historical_data
        ORDER BY time_bucket
        """
        
        # Execute with proper parameter substitution
        final_params = []
        param_placeholders = []
        
        final_params.extend([resolution, kpi_name, time_interval])
        param_placeholders.extend(['$1', '$2', '$3'])
        
        if site_id:
            final_params.append(site_id)
            param_placeholders.append('$4')
        
        # Replace %s with actual parameter placeholders
        query_parts = query.split('%s')
        final_query = query_parts[0]
        for i, part in enumerate(query_parts[1:], 1):
            if i <= 2:  # First two %s are for resolution parameters
                final_query += param_placeholders[0] + part  # Use resolution parameter
            else:
                final_query += part
        
        rows = await conn.fetch(final_query, *final_params)
        
        # Transform to response format
        data_points = []
        values = []
        
        for row in rows:
            point = KPITrendData(
                timestamp=row['time_bucket'],
                value=float(row['avg_value']),
                predicted=row['predicted']
            )
            data_points.append(point)
            if not row['predicted']:
                values.append(float(row['avg_value']))
        
        # Calculate statistics
        statistics = {}
        if values:
            statistics = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "last": values[-1] if values else 0,
                "data_points": len(values)
            }
        
        response = KPITrendResponse(
            kpi_name=kpi_name,
            time_range=time_range,
            data_points=data_points,
            statistics=statistics
        )
        
        logger.info(f"Retrieved trend data for {kpi_name}: {len(data_points)} points")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving KPI trend for {kpi_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve KPI trend")

@router.post("/calculate")
async def trigger_kpi_calculation(
    request: KPICalculationRequest,
    background_tasks: BackgroundTasks,
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Trigger manual KPI calculation for specific sites/KPIs"""
    try:
        # Add calculation request to queue
        calculation_id = await conn.fetchval("""
            INSERT INTO kpi_calculation_requests (
                site_ids, kpi_names, time_range, priority, status,
                requested_by, created_at
            ) VALUES ($1, $2, $3, $4, 'pending', $5, NOW())
            RETURNING id
        """, 
        request.site_ids or [],
        request.kpi_names or [],
        request.time_range,
        request.priority,
        request.requested_by
        )
        
        # Add background task to process calculation
        background_tasks.add_task(process_kpi_calculation, calculation_id, request)
        
        logger.info(f"KPI calculation request {calculation_id} queued")
        
        return {
            "calculation_id": calculation_id,
            "status": "queued",
            "message": "KPI calculation has been queued for processing"
        }
        
    except Exception as e:
        logger.error(f"Error triggering KPI calculation: {e}")
        raise HTTPException(status_code=500, detail="Failed to queue KPI calculation")

@router.get("/alerts", response_model=List[KPIAlert])
async def get_kpi_alerts(
    severity: Optional[str] = Query(None, pattern="^(critical|major|minor|warning)$"),
    status: Optional[str] = Query(None, pattern="^(active|acknowledged|resolved)$"),
    site_id: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get KPI alerts with filtering options"""
    try:
        query = """
        SELECT id, kpi_name, site_id, condition_type, threshold_value,
               current_value, severity, status, triggered_at, acknowledged_at,
               resolved_at, message, metadata
        FROM kpi_alerts
        WHERE 1=1
        """
        
        params = []
        
        if severity:
            query += f" AND severity = ${len(params) + 1}"
            params.append(severity)
            
        if status:
            query += f" AND status = ${len(params) + 1}"
            params.append(status)
            
        if site_id:
            query += f" AND site_id = ${len(params) + 1}"
            params.append(site_id)
        
        query += f" ORDER BY triggered_at DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        rows = await conn.fetch(query, *params)
        
        alerts = []
        for row in rows:
            alert = KPIAlert(
                id=row['id'],
                kpi_name=row['kpi_name'],
                site_id=row['site_id'],
                condition_type=row['condition_type'],
                threshold_value=row['threshold_value'],
                current_value=row['current_value'],
                severity=row['severity'],
                status=row['status'],
                triggered_at=row['triggered_at'],
                acknowledged_at=row['acknowledged_at'],
                resolved_at=row['resolved_at'],
                message=row['message'],
                metadata=row['metadata'] or {}
            )
            alerts.append(alert)
        
        logger.info(f"Retrieved {len(alerts)} KPI alerts")
        return alerts
        
    except Exception as e:
        logger.error(f"Error retrieving KPI alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve KPI alerts")

@router.post("/alerts", response_model=Dict[str, Any])
async def create_kpi_alert(
    alert: KPIAlertCreate,
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Create new KPI alert rule"""
    try:
        alert_id = await conn.fetchval("""
            INSERT INTO kpi_alert_rules (
                kpi_name, site_id, condition_type, threshold_value,
                severity, notification_channels, enabled, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            RETURNING id
        """,
        alert.kpi_name,
        alert.site_id,
        alert.condition,
        alert.threshold,
        alert.severity,
        alert.notification_channels,
        alert.enabled
        )
        
        logger.info(f"Created KPI alert rule {alert_id}")
        
        return {
            "alert_id": alert_id,
            "status": "created",
            "message": "KPI alert rule created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating KPI alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to create KPI alert")

@router.get("/dashboards", response_model=List[KPIDashboard])
async def get_dashboards(
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get available KPI dashboards"""
    try:
        rows = await conn.fetch("""
            SELECT id, name, description, kpi_list, layout_config,
                   filters, refresh_interval, is_public, created_at, updated_at
            FROM kpi_dashboards
            ORDER BY name
        """)
        
        dashboards = []
        for row in rows:
            dashboard = KPIDashboard(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                kpi_list=row['kpi_list'],
                layout_config=row['layout_config'] or {},
                filters=row['filters'] or {},
                refresh_interval=row['refresh_interval'],
                is_public=row['is_public'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            dashboards.append(dashboard)
        
        logger.info(f"Retrieved {len(dashboards)} KPI dashboards")
        return dashboards
        
    except Exception as e:
        logger.error(f"Error retrieving dashboards: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboards")

@router.post("/dashboards", response_model=Dict[str, Any])
async def create_dashboard(
    dashboard: KPIDashboardCreate,
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Create new KPI dashboard"""
    try:
        dashboard_id = await conn.fetchval("""
            INSERT INTO kpi_dashboards (
                name, description, kpi_list, layout_config, filters,
                refresh_interval, is_public, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
            RETURNING id
        """,
        dashboard.name,
        dashboard.description,
        dashboard.kpi_list,
        dashboard.layout_config,
        dashboard.filters,
        dashboard.refresh_interval,
        dashboard.is_public
        )
        
        logger.info(f"Created KPI dashboard {dashboard_id}")
        
        return {
            "dashboard_id": dashboard_id,
            "status": "created",
            "message": "KPI dashboard created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to create dashboard")

@router.get("/stream/{kpi_name}")
async def stream_kpi_data(
    kpi_name: str,
    site_id: Optional[str] = Query(None),
    interval: int = Query(30, description="Update interval in seconds")
):
    """Stream real-time KPI data via Server-Sent Events"""
    async def event_stream():
        try:
            while True:
                # Get latest KPI value
                async with get_connection() as conn:
                    query = """
                    SELECT kpi_value, unit, quality_score, calculated_at, metadata
                    FROM kpi_calculations
                    WHERE kpi_name = $1
                    """
                    params = [kpi_name]
                    
                    if site_id:
                        query += " AND site_id = $2"
                        params.append(site_id)
                    
                    query += " ORDER BY calculated_at DESC LIMIT 1"
                    
                    row = await conn.fetchrow(query, *params)
                    
                    if row:
                        data = {
                            "kpi_name": kpi_name,
                            "value": float(row['kpi_value']),
                            "unit": row['unit'],
                            "quality_score": float(row['quality_score']),
                            "timestamp": row['calculated_at'].isoformat(),
                            "metadata": row['metadata']
                        }
                        
                        yield f"data: {json.dumps(data)}\n\n"
                    
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            logger.info(f"KPI stream cancelled for {kpi_name}")
        except Exception as e:
            logger.error(f"Error in KPI stream for {kpi_name}: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

async def process_kpi_calculation(calculation_id: int, request: KPICalculationRequest):
    """Background task to process KPI calculation"""
    try:
        # This would integrate with the KPI worker
        # For now, just update the status
        async with get_connection() as conn:
            await conn.execute("""
                UPDATE kpi_calculation_requests
                SET status = 'processing', started_at = NOW()
                WHERE id = $1
            """, calculation_id)
            
            # Simulate processing time
            await asyncio.sleep(5)
            
            await conn.execute("""
                UPDATE kpi_calculation_requests
                SET status = 'completed', completed_at = NOW()
                WHERE id = $1
            """, calculation_id)
        
        logger.info(f"KPI calculation {calculation_id} completed")
        
    except Exception as e:
        logger.error(f"Error processing KPI calculation {calculation_id}: {e}")
        async with get_connection() as conn:
            await conn.execute("""
                UPDATE kpi_calculation_requests
                SET status = 'failed', error_message = $2, completed_at = NOW()
                WHERE id = $1
            """, calculation_id, str(e))

@router.get("/categories")
async def get_kpi_categories(
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get available KPI categories with counts"""
    try:
        rows = await conn.fetch("""
            SELECT category, COUNT(DISTINCT kpi_name) as kpi_count
            FROM kpi_calculations
            WHERE calculated_at >= NOW() - INTERVAL '1 day'
            GROUP BY category
            ORDER BY category
        """)
        
        categories = []
        for row in rows:
            categories.append({
                "name": row['category'],
                "kpi_count": row['kpi_count'],
                "description": get_category_description(row['category'])
            })
        
        return {"categories": categories}
        
    except Exception as e:
        logger.error(f"Error retrieving KPI categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve categories")

def get_category_description(category: str) -> str:
    """Get category description"""
    descriptions = {
        "network": "Network performance and quality metrics",
        "energy": "Energy efficiency and sustainability metrics",
        "operational": "Operational excellence and maintenance metrics",
        "financial": "Financial performance and cost metrics"
    }
    return descriptions.get(category, "Unknown category")