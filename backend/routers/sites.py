"""
Site Management API Endpoints
Site inventory, configuration and monitoring endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
import logging
import asyncpg
from datetime import datetime, timedelta

from core.database import get_database_manager, DatabaseManager
from core.auth import get_current_user
from core.models import Site, Equipment, APIResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/sites",
    tags=["Site Management"],
    dependencies=[Depends(get_current_user)]
)

@router.get("/", response_model=List[Site])
async def get_sites(
    region: Optional[str] = Query(None, description="Filter by region"),
    technology: Optional[str] = Query(None, description="Filter by technology"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get sites with filtering and pagination"""
    try:
        query = """
        SELECT site_id, site_code, site_name, latitude, longitude,
               address, region, country, technology, status, metadata
        FROM sites
        WHERE 1=1
        """
        
        params = []
        
        if region:
            query += f" AND region = ${len(params) + 1}"
            params.append(region)
            
        if technology:
            query += f" AND technology ? ${len(params) + 1}"
            params.append(technology)
            
        if status:
            query += f" AND status = ${len(params) + 1}"
            params.append(status)
        
        query += f" ORDER BY site_code LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
        params.extend([limit, offset])
        
        rows = await db.execute_query(query, *params)
        
        sites = []
        for row in rows:
            site = Site(
                site_id=row['site_id'],
                site_code=row['site_code'],
                site_name=row['site_name'],
                latitude=float(row['latitude']),
                longitude=float(row['longitude']),
                address=row['address'],
                region=row['region'],
                country=row['country'],
                technology=row['technology'] or {},
                status=row['status']
            )
            sites.append(site)
        
        logger.info(f"Retrieved {len(sites)} sites")
        return sites
        
    except Exception as e:
        logger.error(f"Error retrieving sites: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sites")

@router.get("/{site_id}", response_model=Site)
async def get_site(
    site_id: str,
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get site by ID"""
    try:
        row = await db.execute_query_one("""
            SELECT site_id, site_code, site_name, latitude, longitude,
                   address, region, country, technology, status, metadata
            FROM sites
            WHERE site_id = $1
        """, site_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Site not found")
        
        site = Site(
            site_id=row['site_id'],
            site_code=row['site_code'],
            site_name=row['site_name'],
            latitude=float(row['latitude']),
            longitude=float(row['longitude']),
            address=row['address'],
            region=row['region'],
            country=row['country'],
            technology=row['technology'] or {},
            status=row['status']
        )
        
        logger.info(f"Retrieved site {site_id}")
        return site
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving site {site_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve site")

@router.get("/{site_id}/equipment", response_model=List[Equipment])
async def get_site_equipment(
    site_id: str,
    equipment_type: Optional[str] = Query(None),
    vendor: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get equipment for a specific site"""
    try:
        # Verify site exists
        site_exists = await db.execute_query_scalar("SELECT COUNT(*) FROM sites WHERE site_id = $1", site_id)
        if not site_exists:
            raise HTTPException(status_code=404, detail="Site not found")
        
        query = """
        SELECT equipment_id, site_id, equipment_type, vendor, model,
               serial_number, software_version, installation_date, status, metadata
        FROM equipment
        WHERE site_id = $1
        """
        
        params = [site_id]
        
        if equipment_type:
            query += f" AND equipment_type = ${len(params) + 1}"
            params.append(equipment_type)
            
        if vendor:
            query += f" AND vendor = ${len(params) + 1}"
            params.append(vendor)
            
        if status:
            query += f" AND status = ${len(params) + 1}"
            params.append(status)
        
        query += " ORDER BY equipment_type, vendor, model"
        
        rows = await db.execute_query(query, *params)
        
        equipment_list = []
        for row in rows:
            equipment = Equipment(
                equipment_id=row['equipment_id'],
                site_id=row['site_id'],
                equipment_type=row['equipment_type'],
                vendor=row['vendor'],
                model=row['model'],
                serial_number=row['serial_number'],
                software_version=row['software_version'],
                installation_date=row['installation_date'],
                status=row['status']
            )
            equipment_list.append(equipment)
        
        logger.info(f"Retrieved {len(equipment_list)} equipment items for site {site_id}")
        return equipment_list
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving equipment for site {site_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve equipment")

@router.get("/{site_id}/metrics/latest")
async def get_site_latest_metrics(
    site_id: str,
    metric_names: Optional[List[str]] = Query(None),
    time_range: str = Query("1h"),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get latest metrics for a specific site"""
    try:
        # Verify site exists
        site_exists = await db.execute_query_scalar("SELECT COUNT(*) FROM sites WHERE site_id = $1", site_id)
        if not site_exists:
            raise HTTPException(status_code=404, detail="Site not found")
        
        query = """
        SELECT DISTINCT ON (metric_name)
            metric_name, metric_value, unit, technology, quality_score,
            timestamp, metadata
        FROM network_metrics
        WHERE site_id = $1
            AND timestamp >= NOW() - INTERVAL %s
        """
        
        params = [site_id, time_range.replace('m', ' minutes').replace('h', ' hours').replace('d', ' days')]
        
        if metric_names:
            placeholders = ', '.join([f"${i+3}" for i in range(len(metric_names))])
            query += f" AND metric_name IN ({placeholders})"
            params.extend(metric_names)
        
        query += " ORDER BY metric_name, timestamp DESC"
        
        # Replace %s with proper parameter number
        query = query.replace('%s', '$2')
        
        rows = await db.execute_query(query, *params)
        
        metrics = []
        for row in rows:
            metric = {
                "metric_name": row['metric_name'],
                "value": float(row['metric_value']),
                "unit": row['unit'],
                "technology": row['technology'],
                "quality_score": float(row['quality_score']),
                "timestamp": row['timestamp'].isoformat(),
                "metadata": row['metadata'] or {}
            }
            metrics.append(metric)
        
        logger.info(f"Retrieved {len(metrics)} latest metrics for site {site_id}")
        return {"site_id": site_id, "metrics": metrics}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving metrics for site {site_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@router.get("/{site_id}/kpis/latest")
async def get_site_latest_kpis(
    site_id: str,
    category: Optional[str] = Query(None, description="KPI category filter"),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get latest KPI values for a specific site"""
    try:
        # Verify site exists
        site_exists = await db.execute_query_scalar("SELECT COUNT(*) FROM sites WHERE site_id = $1", site_id)
        if not site_exists:
            raise HTTPException(status_code=404, detail="Site not found")
        
        query = """
        SELECT DISTINCT ON (kpi_name)
            kpi_name, kpi_value, unit, category, quality_score,
            target_value, calculated_at, metadata
        FROM kpi_calculations
        WHERE site_id = $1
        """
        
        params = [site_id]
        
        if category:
            query += f" AND category = ${len(params) + 1}"
            params.append(category)
        
        query += " ORDER BY kpi_name, calculated_at DESC"
        
        rows = await db.execute_query(query, *params)
        
        kpis = []
        for row in rows:
            kpi = {
                "kpi_name": row['kpi_name'],
                "value": float(row['kpi_value']),
                "unit": row['unit'],
                "category": row['category'],
                "quality_score": float(row['quality_score']),
                "target_value": float(row['target_value']) if row['target_value'] else None,
                "calculated_at": row['calculated_at'].isoformat(),
                "metadata": row['metadata'] or {}
            }
            kpis.append(kpi)
        
        logger.info(f"Retrieved {len(kpis)} latest KPIs for site {site_id}")
        return {"site_id": site_id, "kpis": kpis}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving KPIs for site {site_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve KPIs")

@router.get("/{site_id}/alerts/active")
async def get_site_active_alerts(
    site_id: str,
    severity: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get active alerts for a specific site"""
    try:
        # Verify site exists
        site_exists = await db.execute_query_scalar("SELECT COUNT(*) FROM sites WHERE site_id = $1", site_id)
        if not site_exists:
            raise HTTPException(status_code=404, detail="Site not found")
        
        query = """
        SELECT id, kpi_name, condition_type, threshold_value, current_value,
               severity, status, triggered_at, message, metadata
        FROM kpi_alerts
        WHERE site_id = $1 AND status = 'active'
        """
        
        params = [site_id]
        
        if severity:
            query += f" AND severity = ${len(params) + 1}"
            params.append(severity)
        
        query += f" ORDER BY triggered_at DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        rows = await db.execute_query(query, *params)
        
        alerts = []
        for row in rows:
            alert = {
                "id": row['id'],
                "kpi_name": row['kpi_name'],
                "condition_type": row['condition_type'],
                "threshold_value": float(row['threshold_value']),
                "current_value": float(row['current_value']),
                "severity": row['severity'],
                "status": row['status'],
                "triggered_at": row['triggered_at'].isoformat(),
                "message": row['message'],
                "metadata": row['metadata'] or {}
            }
            alerts.append(alert)
        
        logger.info(f"Retrieved {len(alerts)} active alerts for site {site_id}")
        return {"site_id": site_id, "alerts": alerts}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving alerts for site {site_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

@router.get("/{site_id}/health")
async def get_site_health(
    site_id: str,
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get overall site health status and score"""
    try:
        # Verify site exists
        site_exists = await db.execute_query_scalar("SELECT COUNT(*) FROM sites WHERE site_id = $1", site_id)
        if not site_exists:
            raise HTTPException(status_code=404, detail="Site not found")
        
        # Get site health score using the database function
        health_data = await db.execute_query_one("""
            SELECT 
                get_site_health_score($1) as health_score,
                (SELECT COUNT(*) FROM kpi_alerts WHERE site_id = $1 AND status = 'active') as active_alerts,
                (SELECT COUNT(*) FROM network_metrics 
                 WHERE site_id = $1 AND timestamp >= NOW() - INTERVAL '1 hour') as recent_metrics
        """, site_id)
        
        health_score = float(health_data['health_score']) if health_data['health_score'] else 0.0
        
        # Determine health status
        if health_score >= 0.9:
            status = "excellent"
        elif health_score >= 0.7:
            status = "good"
        elif health_score >= 0.5:
            status = "fair"
        elif health_score >= 0.3:
            status = "poor"
        else:
            status = "critical"
        
        # Get latest KPI summary
        kpi_summary = await db.execute_query("""
            SELECT category, COUNT(*) as kpi_count, AVG(quality_score) as avg_quality
            FROM kpi_calculations
            WHERE site_id = $1 AND calculated_at >= NOW() - INTERVAL '1 hour'
            GROUP BY category
        """, site_id)
        
        categories = {}
        for row in kpi_summary:
            categories[row['category']] = {
                "kpi_count": row['kpi_count'],
                "avg_quality": float(row['avg_quality'])
            }
        
        health_report = {
            "site_id": site_id,
            "health_score": health_score,
            "status": status,
            "active_alerts": health_data['active_alerts'],
            "recent_metrics": health_data['recent_metrics'],
            "categories": categories,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Retrieved health report for site {site_id}: score={health_score:.2f}")
        return health_report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving health for site {site_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve site health")

@router.get("/{site_id}/itsm/tickets")
async def get_site_itsm_tickets(
    site_id: str,
    status: Optional[str] = Query(None),
    ticket_type: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get ITSM tickets for a specific site"""
    try:
        # Verify site exists
        site_exists = await db.execute_query_scalar("SELECT COUNT(*) FROM sites WHERE site_id = $1", site_id)
        if not site_exists:
            raise HTTPException(status_code=404, detail="Site not found")
        
        query = """
        SELECT ticket_id, ticket_type, title, description, priority,
               status, assigned_to, created_at, updated_at, resolved_at, metadata
        FROM itsm_tickets
        WHERE site_id = $1
        """
        
        params = [site_id]
        
        if status:
            query += f" AND status = ${len(params) + 1}"
            params.append(status)
            
        if ticket_type:
            query += f" AND ticket_type = ${len(params) + 1}"
            params.append(ticket_type)
        
        query += f" ORDER BY created_at DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        rows = await db.execute_query(query, *params)
        
        tickets = []
        for row in rows:
            ticket = {
                "ticket_id": row['ticket_id'],
                "ticket_type": row['ticket_type'],
                "title": row['title'],
                "description": row['description'],
                "priority": row['priority'],
                "status": row['status'],
                "assigned_to": row['assigned_to'],
                "created_at": row['created_at'].isoformat(),
                "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None,
                "resolved_at": row['resolved_at'].isoformat() if row['resolved_at'] else None,
                "metadata": row['metadata'] or {}
            }
            tickets.append(ticket)
        
        logger.info(f"Retrieved {len(tickets)} ITSM tickets for site {site_id}")
        return {"site_id": site_id, "tickets": tickets}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving ITSM tickets for site {site_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tickets")

@router.get("/regions")
async def get_regions(
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get list of available regions with site counts"""
    try:
        rows = await db.execute_query("""
            SELECT region, COUNT(*) as site_count
            FROM sites
            WHERE region IS NOT NULL
            GROUP BY region
            ORDER BY region
        """)
        
        regions = []
        for row in rows:
            regions.append({
                "name": row['region'],
                "site_count": row['site_count']
            })
        
        return {"regions": regions}
        
    except Exception as e:
        logger.error(f"Error retrieving regions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve regions")

@router.get("/technologies")
async def get_technologies(
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get list of available technologies with site counts"""
    try:
        rows = await db.execute_query("""
            SELECT 
                jsonb_object_keys(technology) as tech_name,
                COUNT(*) as site_count
            FROM sites
            WHERE technology IS NOT NULL AND technology != '{}'::jsonb
            GROUP BY jsonb_object_keys(technology)
            ORDER BY tech_name
        """)
        
        technologies = []
        for row in rows:
            technologies.append({
                "name": row['tech_name'],
                "site_count": row['site_count']
            })
        
        return {"technologies": technologies}
        
    except Exception as e:
        logger.error(f"Error retrieving technologies: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve technologies")
