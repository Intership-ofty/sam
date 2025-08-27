"""
NOC (Network Operations Center) API Endpoints
Centralized operations management and incident handling
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncpg
import json
import uuid
from enum import Enum
from pydantic import BaseModel, Field

from ..core.database import get_connection
from ..core.auth import get_current_user, require_permission
from ..core.models import APIResponse, User

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/noc",
    tags=["NOC Operations"],
    dependencies=[Depends(get_current_user)]
)

class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"

class IncidentStatus(str, Enum):
    NEW = "new"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"

class IncidentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=2000)
    severity: IncidentSeverity = IncidentSeverity.MEDIUM
    category: str = Field(..., min_length=1, max_length=100)
    site_id: Optional[str] = None
    source_system: str = Field(default="manual")
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IncidentUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[IncidentSeverity] = None
    status: Optional[IncidentStatus] = None
    assigned_to: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class EscalationRuleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    time_threshold_minutes: int = Field(..., gt=0, le=1440)
    severity_levels: List[IncidentSeverity]
    priority: int = Field(default=1, ge=1, le=10)
    enabled: bool = True

@router.get("/dashboard")
async def get_noc_dashboard(
    time_range: str = Query("24h", regex="^(1h|6h|24h|7d|30d)$"),
    current_user: User = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get NOC operational dashboard data"""
    try:
        # Calculate time range
        time_delta_map = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        start_time = datetime.utcnow() - time_delta_map[time_range]

        # Get incident statistics
        incident_stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_incidents,
                COUNT(CASE WHEN status IN ('new', 'assigned', 'in_progress') THEN 1 END) as active_incidents,
                COUNT(CASE WHEN severity = 'critical' AND status NOT IN ('resolved', 'closed') THEN 1 END) as critical_incidents,
                COUNT(CASE WHEN status = 'escalated' THEN 1 END) as escalated_incidents,
                COUNT(CASE WHEN created_at >= $2 THEN 1 END) as incidents_in_range,
                AVG(CASE WHEN resolution_time IS NOT NULL THEN 
                    EXTRACT(EPOCH FROM (resolution_time - created_at))/3600 
                END) as avg_resolution_hours
            FROM noc_incidents 
            WHERE tenant_id = $1
        """, current_user.tenant_id, start_time)

        # Get site health overview
        site_health = await conn.fetch("""
            SELECT 
                s.site_id,
                s.name as site_name,
                s.location,
                COUNT(ni.id) as active_incidents,
                MAX(ni.severity) as highest_severity,
                km.availability,
                km.uptime_percentage
            FROM sites s
            LEFT JOIN noc_incidents ni ON s.site_id = ni.site_id 
                AND ni.tenant_id = $1 
                AND ni.status NOT IN ('resolved', 'closed')
            LEFT JOIN (
                SELECT DISTINCT ON (site_id) site_id, availability, uptime_percentage
                FROM kpi_metrics 
                WHERE tenant_id = $1 
                ORDER BY site_id, timestamp DESC
            ) km ON s.site_id = km.site_id
            WHERE s.tenant_id = $1
            GROUP BY s.site_id, s.name, s.location, km.availability, km.uptime_percentage
            ORDER BY active_incidents DESC, highest_severity DESC
        """, current_user.tenant_id)

        # Get recent incidents
        recent_incidents = await conn.fetch("""
            SELECT 
                ni.id, ni.title, ni.severity, ni.status, ni.category,
                ni.created_at, ni.updated_at, ni.assigned_to,
                s.name as site_name,
                u.full_name as assignee_name
            FROM noc_incidents ni
            LEFT JOIN sites s ON ni.site_id = s.site_id
            LEFT JOIN users u ON ni.assigned_to = u.id
            WHERE ni.tenant_id = $1 
            AND ni.created_at >= $2
            ORDER BY ni.created_at DESC
            LIMIT 20
        """, current_user.tenant_id, start_time)

        # Get escalation metrics
        escalation_metrics = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_escalations,
                COUNT(CASE WHEN created_at >= $2 THEN 1 END) as recent_escalations,
                AVG(EXTRACT(EPOCH FROM (updated_at - created_at))/60) as avg_escalation_time_minutes
            FROM noc_incidents 
            WHERE tenant_id = $1 
            AND status = 'escalated'
        """, current_user.tenant_id, start_time)

        # Get team performance
        team_performance = await conn.fetch("""
            SELECT 
                u.id, u.full_name,
                COUNT(ni.id) as assigned_incidents,
                COUNT(CASE WHEN ni.status = 'resolved' THEN 1 END) as resolved_incidents,
                AVG(CASE WHEN ni.resolution_time IS NOT NULL THEN 
                    EXTRACT(EPOCH FROM (ni.resolution_time - ni.created_at))/3600 
                END) as avg_resolution_hours
            FROM users u
            LEFT JOIN noc_incidents ni ON u.id = ni.assigned_to 
                AND ni.tenant_id = $1 
                AND ni.created_at >= $2
            WHERE u.tenant_id = $1 
            AND u.role IN ('engineer', 'senior_engineer', 'lead_engineer')
            AND u.is_active = true
            GROUP BY u.id, u.full_name
            ORDER BY assigned_incidents DESC
        """, current_user.tenant_id, start_time)

        # Prepare response
        dashboard_data = {
            "time_range": time_range,
            "last_updated": datetime.utcnow().isoformat(),
            "incident_statistics": {
                "total": incident_stats['total_incidents'] or 0,
                "active": incident_stats['active_incidents'] or 0,
                "critical": incident_stats['critical_incidents'] or 0,
                "escalated": incident_stats['escalated_incidents'] or 0,
                "in_time_range": incident_stats['incidents_in_range'] or 0,
                "avg_resolution_hours": round(incident_stats['avg_resolution_hours'] or 0, 2)
            },
            "site_health": [
                {
                    "site_id": row['site_id'],
                    "site_name": row['site_name'],
                    "location": row['location'],
                    "active_incidents": row['active_incidents'],
                    "highest_severity": row['highest_severity'],
                    "availability": round(row['availability'] or 0, 2),
                    "uptime_percentage": round(row['uptime_percentage'] or 0, 2),
                    "status": "critical" if row['active_incidents'] > 0 and row['highest_severity'] == 'critical' 
                             else "warning" if row['active_incidents'] > 0 
                             else "healthy"
                }
                for row in site_health
            ],
            "recent_incidents": [
                {
                    "id": row['id'],
                    "title": row['title'],
                    "severity": row['severity'],
                    "status": row['status'],
                    "category": row['category'],
                    "site_name": row['site_name'],
                    "assignee_name": row['assignee_name'],
                    "created_at": row['created_at'].isoformat(),
                    "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None
                }
                for row in recent_incidents
            ],
            "escalation_metrics": {
                "total": escalation_metrics['total_escalations'] or 0,
                "recent": escalation_metrics['recent_escalations'] or 0,
                "avg_time_minutes": round(escalation_metrics['avg_escalation_time_minutes'] or 0, 1)
            },
            "team_performance": [
                {
                    "user_id": row['id'],
                    "name": row['full_name'],
                    "assigned_incidents": row['assigned_incidents'] or 0,
                    "resolved_incidents": row['resolved_incidents'] or 0,
                    "avg_resolution_hours": round(row['avg_resolution_hours'] or 0, 2),
                    "resolution_rate": round((row['resolved_incidents'] or 0) / max(row['assigned_incidents'] or 1, 1) * 100, 1)
                }
                for row in team_performance
            ]
        }

        return dashboard_data

    except Exception as e:
        logger.error(f"Error retrieving NOC dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve NOC dashboard")

@router.post("/incidents")
async def create_incident(
    incident: IncidentCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("incidents.create")),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Create a new incident"""
    try:
        incident_id = str(uuid.uuid4())
        
        # Create incident
        await conn.execute("""
            INSERT INTO noc_incidents (
                id, tenant_id, site_id, title, description, severity, status,
                category, source_system, created_by, created_at, updated_at,
                tags, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """,
        incident_id, current_user.tenant_id, incident.site_id,
        incident.title, incident.description, incident.severity.value,
        IncidentStatus.NEW.value, incident.category, incident.source_system,
        current_user.id, datetime.utcnow(), datetime.utcnow(),
        json.dumps(incident.tags), json.dumps(incident.metadata))

        # Start background processing
        background_tasks.add_task(
            process_new_incident,
            incident_id,
            current_user.tenant_id
        )

        logger.info(f"Created incident {incident_id}")

        return {
            "incident_id": incident_id,
            "status": "created",
            "message": "Incident created successfully"
        }

    except Exception as e:
        logger.error(f"Error creating incident: {e}")
        raise HTTPException(status_code=500, detail="Failed to create incident")

@router.get("/incidents")
async def get_incidents(
    status: Optional[IncidentStatus] = Query(None),
    severity: Optional[IncidentSeverity] = Query(None),
    assigned_to: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get incidents with filtering"""
    try:
        query = """
            SELECT 
                ni.id, ni.title, ni.description, ni.severity, ni.status,
                ni.category, ni.source_system, ni.created_at, ni.updated_at,
                ni.assigned_to, ni.resolution_time, ni.tags, ni.metadata,
                s.name as site_name,
                u.full_name as assignee_name,
                creator.full_name as created_by_name,
                COUNT(*) OVER() as total_count
            FROM noc_incidents ni
            LEFT JOIN sites s ON ni.site_id = s.site_id
            LEFT JOIN users u ON ni.assigned_to = u.id
            LEFT JOIN users creator ON ni.created_by = creator.id
            WHERE ni.tenant_id = $1
        """
        
        params = [current_user.tenant_id]
        param_count = 1

        if status:
            param_count += 1
            query += f" AND ni.status = ${param_count}"
            params.append(status.value)

        if severity:
            param_count += 1
            query += f" AND ni.severity = ${param_count}"
            params.append(severity.value)

        if assigned_to:
            param_count += 1
            query += f" AND ni.assigned_to = ${param_count}"
            params.append(assigned_to)

        if site_id:
            param_count += 1
            query += f" AND ni.site_id = ${param_count}"
            params.append(site_id)

        query += f" ORDER BY ni.created_at DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
        params.extend([limit, offset])

        rows = await conn.fetch(query, *params)

        incidents = []
        total_count = 0

        for row in rows:
            if total_count == 0:
                total_count = row['total_count']

            incident_data = {
                "id": row['id'],
                "title": row['title'],
                "description": row['description'],
                "severity": row['severity'],
                "status": row['status'],
                "category": row['category'],
                "source_system": row['source_system'],
                "site_name": row['site_name'],
                "assignee_name": row['assignee_name'],
                "created_by": row['created_by_name'],
                "created_at": row['created_at'].isoformat(),
                "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None,
                "resolution_time": row['resolution_time'].isoformat() if row['resolution_time'] else None,
                "tags": json.loads(row['tags']) if row['tags'] else [],
                "metadata": json.loads(row['metadata']) if row['metadata'] else {}
            }
            incidents.append(incident_data)

        return {
            "incidents": incidents,
            "total_count": total_count,
            "page": offset // limit + 1,
            "page_size": limit,
            "has_more": (offset + limit) < total_count
        }

    except Exception as e:
        logger.error(f"Error retrieving incidents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve incidents")

@router.get("/incidents/{incident_id}")
async def get_incident(
    incident_id: str,
    current_user: User = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get specific incident details"""
    try:
        incident = await conn.fetchrow("""
            SELECT 
                ni.*, s.name as site_name,
                u.full_name as assignee_name,
                creator.full_name as created_by_name
            FROM noc_incidents ni
            LEFT JOIN sites s ON ni.site_id = s.site_id
            LEFT JOIN users u ON ni.assigned_to = u.id
            LEFT JOIN users creator ON ni.created_by = creator.id
            WHERE ni.id = $1 AND ni.tenant_id = $2
        """, incident_id, current_user.tenant_id)

        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")

        # Get incident timeline
        timeline = await conn.fetch("""
            SELECT action, details, created_at, created_by,
                   u.full_name as created_by_name
            FROM incident_timeline it
            LEFT JOIN users u ON it.created_by = u.id
            WHERE it.incident_id = $1
            ORDER BY it.created_at ASC
        """, incident_id)

        incident_data = {
            "id": incident['id'],
            "title": incident['title'],
            "description": incident['description'],
            "severity": incident['severity'],
            "status": incident['status'],
            "category": incident['category'],
            "source_system": incident['source_system'],
            "site_id": incident['site_id'],
            "site_name": incident['site_name'],
            "assigned_to": incident['assigned_to'],
            "assignee_name": incident['assignee_name'],
            "created_by": incident['created_by'],
            "created_by_name": incident['created_by_name'],
            "created_at": incident['created_at'].isoformat(),
            "updated_at": incident['updated_at'].isoformat() if incident['updated_at'] else None,
            "resolution_time": incident['resolution_time'].isoformat() if incident['resolution_time'] else None,
            "parent_incident_id": incident['parent_incident_id'],
            "tags": json.loads(incident['tags']) if incident['tags'] else [],
            "metadata": json.loads(incident['metadata']) if incident['metadata'] else {},
            "timeline": [
                {
                    "action": row['action'],
                    "details": row['details'],
                    "created_at": row['created_at'].isoformat(),
                    "created_by": row['created_by_name']
                }
                for row in timeline
            ]
        }

        return incident_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving incident {incident_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve incident")

@router.put("/incidents/{incident_id}")
async def update_incident(
    incident_id: str,
    incident_update: IncidentUpdate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("incidents.update")),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Update incident"""
    try:
        # Verify incident exists and belongs to tenant
        existing = await conn.fetchrow("""
            SELECT id, status, assigned_to FROM noc_incidents 
            WHERE id = $1 AND tenant_id = $2
        """, incident_id, current_user.tenant_id)

        if not existing:
            raise HTTPException(status_code=404, detail="Incident not found")

        # Build update query
        update_fields = []
        params = [incident_id, current_user.tenant_id]
        param_count = 2

        if incident_update.title is not None:
            param_count += 1
            update_fields.append(f"title = ${param_count}")
            params.append(incident_update.title)

        if incident_update.description is not None:
            param_count += 1
            update_fields.append(f"description = ${param_count}")
            params.append(incident_update.description)

        if incident_update.severity is not None:
            param_count += 1
            update_fields.append(f"severity = ${param_count}")
            params.append(incident_update.severity.value)

        if incident_update.status is not None:
            param_count += 1
            update_fields.append(f"status = ${param_count}")
            params.append(incident_update.status.value)
            
            # Set resolution time if resolved
            if incident_update.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                param_count += 1
                update_fields.append(f"resolution_time = ${param_count}")
                params.append(datetime.utcnow())

        if incident_update.assigned_to is not None:
            param_count += 1
            update_fields.append(f"assigned_to = ${param_count}")
            params.append(incident_update.assigned_to)

        if incident_update.tags is not None:
            param_count += 1
            update_fields.append(f"tags = ${param_count}")
            params.append(json.dumps(incident_update.tags))

        if incident_update.metadata is not None:
            param_count += 1
            update_fields.append(f"metadata = ${param_count}")
            params.append(json.dumps(incident_update.metadata))

        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields to update")

        # Add updated_at
        param_count += 1
        update_fields.append(f"updated_at = ${param_count}")
        params.append(datetime.utcnow())

        query = f"""
            UPDATE noc_incidents 
            SET {', '.join(update_fields)}
            WHERE id = $1 AND tenant_id = $2
            RETURNING id, title, status, updated_at
        """

        result = await conn.fetchrow(query, *params)

        # Log the update in timeline
        await conn.execute("""
            INSERT INTO incident_timeline (
                incident_id, action, details, created_by, created_at
            ) VALUES ($1, $2, $3, $4, $5)
        """,
        incident_id,
        "updated",
        f"Incident updated by {current_user.full_name}",
        current_user.id,
        datetime.utcnow())

        # Process notifications for significant changes
        if incident_update.status or incident_update.assigned_to or incident_update.severity:
            background_tasks.add_task(
                send_incident_update_notifications,
                incident_id,
                incident_update.dict(exclude_unset=True),
                current_user.tenant_id
            )

        logger.info(f"Updated incident {incident_id}")

        return {
            "incident_id": result['id'],
            "title": result['title'],
            "status": result['status'],
            "updated_at": result['updated_at'].isoformat(),
            "message": "Incident updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating incident {incident_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update incident")

@router.get("/escalation-rules")
async def get_escalation_rules(
    current_user: User = Depends(require_permission("escalation.view")),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get escalation rules for tenant"""
    try:
        rules = await conn.fetch("""
            SELECT er.*, u.full_name as created_by_name
            FROM escalation_rules er
            LEFT JOIN users u ON er.created_by = u.id
            WHERE er.tenant_id = $1
            ORDER BY er.priority ASC, er.created_at DESC
        """, current_user.tenant_id)

        escalation_rules = []
        for rule in rules:
            rule_data = {
                "id": rule['id'],
                "name": rule['name'],
                "description": rule['description'],
                "conditions": json.loads(rule['conditions']),
                "actions": json.loads(rule['actions']),
                "time_threshold_minutes": rule['time_threshold_minutes'],
                "severity_levels": json.loads(rule['severity_levels']),
                "priority": rule['priority'],
                "enabled": rule['enabled'],
                "created_by": rule['created_by_name'],
                "created_at": rule['created_at'].isoformat(),
                "updated_at": rule['updated_at'].isoformat() if rule['updated_at'] else None
            }
            escalation_rules.append(rule_data)

        return {"escalation_rules": escalation_rules}

    except Exception as e:
        logger.error(f"Error retrieving escalation rules: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve escalation rules")

# Background task functions
async def process_new_incident(incident_id: str, tenant_id: str):
    """Process new incident in background"""
    try:
        # This would trigger the NOC orchestrator
        logger.info(f"Processing new incident {incident_id}")
        await asyncio.sleep(1)  # Simulate processing
        
    except Exception as e:
        logger.error(f"Error processing incident {incident_id}: {e}")

async def send_incident_update_notifications(incident_id: str, updates: Dict[str, Any], tenant_id: str):
    """Send notifications for incident updates"""
    try:
        logger.info(f"Sending notifications for incident {incident_id} updates: {updates}")
        await asyncio.sleep(1)  # Simulate notification sending
        
    except Exception as e:
        logger.error(f"Error sending incident update notifications: {e}")