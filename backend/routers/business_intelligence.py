"""
Business Intelligence API Endpoints
Advanced analytics, insights, and value generation
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncpg
import json
import uuid
from enum import Enum
from pydantic import BaseModel, Field

from core.database import get_database_manager, DatabaseManager
from core.auth import get_current_user, require_permission
from core.models import APIResponse, User

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/business-intelligence",
    tags=["Business Intelligence"],
    dependencies=[Depends(get_current_user)]
)

class InsightType(str, Enum):
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    CAPACITY_PLANNING = "capacity_planning"
    ENERGY_OPTIMIZATION = "energy_optimization"
    SLA_OPTIMIZATION = "sla_optimization"
    RISK_MITIGATION = "risk_mitigation"
    AUTOMATION_OPPORTUNITY = "automation_opportunity"

class RecommendationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class OptimizationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    optimization_type: str = Field(..., pattern="^(energy|cost|performance|capacity|maintenance)$")
    affected_sites: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    objectives: List[str] = Field(default_factory=list)

@router.get("/dashboard")
async def get_business_intelligence_dashboard(
    time_range: str = Query("30d", pattern="^(7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get Business Intelligence dashboard data"""
    try:
        # Calculate time range
        time_delta_map = {
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90),
            "1y": timedelta(days=365)
        }
        start_time = datetime.utcnow() - time_delta_map[time_range]

        # Get cost analytics
        cost_analytics = await db.execute_query_one("""
            SELECT 
                SUM(energy_cost) as total_energy_cost,
                SUM(maintenance_cost) as total_maintenance_cost,
                SUM(operational_cost) as total_operational_cost,
                AVG(energy_cost) as avg_energy_cost,
                COUNT(DISTINCT site_id) as sites_with_data
            FROM kpi_metrics 
            WHERE tenant_id = $1 AND timestamp >= $2
        """, current_user.tenant_id, start_time)

        # Get performance metrics
        performance_metrics = await db.execute_query_one("""
            SELECT 
                AVG(availability) as avg_availability,
                AVG(network_performance) as avg_network_performance,
                AVG(uptime_percentage) as avg_uptime,
                AVG(energy_efficiency) as avg_energy_efficiency
            FROM kpi_metrics 
            WHERE tenant_id = $1 AND timestamp >= $2
        """, current_user.tenant_id, start_time)

        # Get incident impact
        incident_impact = await db.execute_query_one("""
            SELECT 
                COUNT(*) as total_incidents,
                AVG(EXTRACT(EPOCH FROM (COALESCE(resolution_time, NOW()) - created_at))/3600) as avg_resolution_hours,
                COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_incidents
            FROM noc_incidents 
            WHERE tenant_id = $1 AND created_at >= $2
        """, current_user.tenant_id, start_time)

        # Get active insights
        active_insights = await db.execute_query("""
            SELECT 
                insight_type, priority, COUNT(*) as count,
                SUM(potential_savings) as total_potential_savings,
                AVG(confidence_score) as avg_confidence
            FROM business_insights 
            WHERE tenant_id = $1 AND status = 'active'
            AND created_at >= $2
            GROUP BY insight_type, priority
            ORDER BY total_potential_savings DESC
        """, current_user.tenant_id, start_time)

        # Get optimization opportunities
        optimization_opportunities = await db.execute_query("""
            SELECT 
                ot.optimization_type,
                COUNT(*) as task_count,
                AVG(ot.progress_percentage) as avg_progress,
                SUM(CASE WHEN or_.estimated_savings IS NOT NULL THEN or_.estimated_savings ELSE 0 END) as total_estimated_savings
            FROM optimization_tasks ot
            LEFT JOIN optimization_results or_ ON ot.id = or_.task_id
            WHERE ot.tenant_id = $1 AND ot.created_at >= $2
            GROUP BY ot.optimization_type
        """, current_user.tenant_id, start_time)

        # Get ROI tracking
        roi_metrics = await db.execute_query_one("""
            SELECT 
                COUNT(DISTINCT bi.id) as total_insights,
                SUM(bi.potential_savings) as total_potential_savings,
                SUM(bi.implementation_cost) as total_implementation_cost,
                AVG(bi.roi_estimate) as avg_roi_estimate
            FROM business_insights bi
            WHERE bi.tenant_id = $1 AND bi.created_at >= $2
        """, current_user.tenant_id, start_time)

        # Calculate derived metrics
        total_cost = (cost_analytics['total_energy_cost'] or 0) + (cost_analytics['total_maintenance_cost'] or 0) + (cost_analytics['total_operational_cost'] or 0)
        potential_savings = roi_metrics['total_potential_savings'] or 0
        cost_reduction_percentage = (potential_savings / total_cost * 100) if total_cost > 0 else 0

        # Prepare response
        dashboard_data = {
            "time_range": time_range,
            "last_updated": datetime.utcnow().isoformat(),
            "cost_analytics": {
                "total_cost": round(total_cost, 2),
                "energy_cost": round(cost_analytics['total_energy_cost'] or 0, 2),
                "maintenance_cost": round(cost_analytics['total_maintenance_cost'] or 0, 2),
                "operational_cost": round(cost_analytics['total_operational_cost'] or 0, 2),
                "avg_cost_per_site": round(cost_analytics['avg_energy_cost'] or 0, 2),
                "potential_savings": round(potential_savings, 2),
                "cost_reduction_percentage": round(cost_reduction_percentage, 2)
            },
            "performance_overview": {
                "avg_availability": round(performance_metrics['avg_availability'] or 0, 2),
                "avg_network_performance": round(performance_metrics['avg_network_performance'] or 0, 2),
                "avg_uptime": round(performance_metrics['avg_uptime'] or 0, 2),
                "avg_energy_efficiency": round(performance_metrics['avg_energy_efficiency'] or 0, 2)
            },
            "operational_impact": {
                "total_incidents": incident_impact['total_incidents'] or 0,
                "avg_resolution_hours": round(incident_impact['avg_resolution_hours'] or 0, 2),
                "critical_incidents": incident_impact['critical_incidents'] or 0,
                "incident_cost_impact": (incident_impact['total_incidents'] or 0) * 500  # $500 per incident
            },
            "insights_summary": [
                {
                    "type": row['insight_type'],
                    "priority": row['priority'],
                    "count": row['count'],
                    "potential_savings": round(row['total_potential_savings'] or 0, 2),
                    "avg_confidence": round(row['avg_confidence'] or 0, 3)
                }
                for row in active_insights
            ],
            "optimization_status": [
                {
                    "type": row['optimization_type'],
                    "task_count": row['task_count'],
                    "avg_progress": round(row['avg_progress'] or 0, 1),
                    "estimated_savings": round(row['total_estimated_savings'] or 0, 2)
                }
                for row in optimization_opportunities
            ],
            "roi_metrics": {
                "total_insights": roi_metrics['total_insights'] or 0,
                "potential_savings": round(roi_metrics['total_potential_savings'] or 0, 2),
                "implementation_cost": round(roi_metrics['total_implementation_cost'] or 0, 2),
                "avg_roi": round(roi_metrics['avg_roi_estimate'] or 0, 2),
                "payback_period_months": round((roi_metrics['total_implementation_cost'] or 0) / max((roi_metrics['total_potential_savings'] or 0) / 12, 1), 1)
            }
        }

        return dashboard_data

    except Exception as e:
        logger.error(f"Error retrieving BI dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve business intelligence dashboard")

@router.get("/insights")
async def get_business_insights(
    insight_type: Optional[InsightType] = Query(None),
    priority: Optional[RecommendationPriority] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get business insights with filtering"""
    try:
        query = """
            SELECT 
                bi.id, bi.title, bi.description, bi.insight_type, bi.priority,
                bi.confidence_score, bi.potential_savings, bi.implementation_cost,
                bi.roi_estimate, bi.time_to_value_days, bi.affected_sites,
                bi.kpis_improved, bi.recommendations, bi.evidence_data,
                bi.created_at, bi.expires_at, bi.status,
                COUNT(*) OVER() as total_count
            FROM business_insights bi
            WHERE bi.tenant_id = $1 AND bi.status = 'active'
        """
        
        params = [current_user.tenant_id]
        param_count = 1

        if insight_type:
            param_count += 1
            query += f" AND bi.insight_type = ${param_count}"
            params.append(insight_type.value)

        if priority:
            param_count += 1
            query += f" AND bi.priority = ${param_count}"
            params.append(priority.value)

        query += f" ORDER BY bi.priority DESC, bi.potential_savings DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
        params.extend([limit, offset])

        rows = await db.execute_query(query, *params)

        insights = []
        total_count = 0

        for row in rows:
            if total_count == 0:
                total_count = row['total_count']

            insight_data = {
                "id": row['id'],
                "title": row['title'],
                "description": row['description'],
                "insight_type": row['insight_type'],
                "priority": row['priority'],
                "confidence_score": row['confidence_score'],
                "potential_savings": row['potential_savings'],
                "implementation_cost": row['implementation_cost'],
                "roi_estimate": row['roi_estimate'],
                "time_to_value_days": row['time_to_value_days'],
                "affected_sites": json.loads(row['affected_sites']) if row['affected_sites'] else [],
                "kpis_improved": json.loads(row['kpis_improved']) if row['kpis_improved'] else [],
                "recommendations": json.loads(row['recommendations']) if row['recommendations'] else [],
                "evidence_data": json.loads(row['evidence_data']) if row['evidence_data'] else {},
                "created_at": row['created_at'].isoformat(),
                "expires_at": row['expires_at'].isoformat() if row['expires_at'] else None,
                "status": row['status']
            }
            insights.append(insight_data)

        return {
            "insights": insights,
            "total_count": total_count,
            "page": offset // limit + 1,
            "page_size": limit,
            "has_more": (offset + limit) < total_count
        }

    except Exception as e:
        logger.error(f"Error retrieving business insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve business insights")

@router.get("/insights/{insight_id}")
async def get_insight_details(
    insight_id: str,
    current_user: User = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get detailed insight information"""
    try:
        insight = await db.execute_query_one("""
            SELECT bi.*, 
                   ARRAY_AGG(DISTINCT s.name) as site_names
            FROM business_insights bi
            LEFT JOIN sites s ON s.site_id = ANY(
                SELECT jsonb_array_elements_text(bi.affected_sites::jsonb)
            ) AND s.tenant_id = bi.tenant_id
            WHERE bi.id = $1 AND bi.tenant_id = $2
            GROUP BY bi.id
        """, insight_id, current_user.tenant_id)

        if not insight:
            raise HTTPException(status_code=404, detail="Insight not found")

        # Get related optimization tasks
        related_tasks = await db.execute_query("""
            SELECT id, name, optimization_type, status, progress_percentage
            FROM optimization_tasks 
            WHERE tenant_id = $1 
            AND affected_sites && $2::jsonb
            ORDER BY created_at DESC
            LIMIT 5
        """, current_user.tenant_id, insight['affected_sites'])

        insight_data = {
            "id": insight['id'],
            "title": insight['title'],
            "description": insight['description'],
            "insight_type": insight['insight_type'],
            "priority": insight['priority'],
            "confidence_score": insight['confidence_score'],
            "potential_savings": insight['potential_savings'],
            "implementation_cost": insight['implementation_cost'],
            "roi_estimate": insight['roi_estimate'],
            "time_to_value_days": insight['time_to_value_days'],
            "affected_sites": json.loads(insight['affected_sites']) if insight['affected_sites'] else [],
            "site_names": [name for name in insight['site_names'] if name],
            "kpis_improved": json.loads(insight['kpis_improved']) if insight['kpis_improved'] else [],
            "recommendations": json.loads(insight['recommendations']) if insight['recommendations'] else [],
            "evidence_data": json.loads(insight['evidence_data']) if insight['evidence_data'] else {},
            "created_at": insight['created_at'].isoformat(),
            "expires_at": insight['expires_at'].isoformat() if insight['expires_at'] else None,
            "status": insight['status'],
            "related_optimizations": [
                {
                    "id": task['id'],
                    "name": task['name'],
                    "type": task['optimization_type'],
                    "status": task['status'],
                    "progress": task['progress_percentage']
                }
                for task in related_tasks
            ]
        }

        return insight_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving insight {insight_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve insight details")

@router.post("/optimizations")
async def create_optimization(
    optimization: OptimizationCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("optimization.create")),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Create new optimization task"""
    try:
        task_id = str(uuid.uuid4())
        
        # Create optimization task
        await db.execute_command("""
            INSERT INTO optimization_tasks (
                id, tenant_id, name, description, optimization_type,
                parameters, constraints, objectives, affected_sites,
                status, progress_percentage, created_by, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """,
        task_id, current_user.tenant_id, optimization.name, optimization.description,
        optimization.optimization_type, json.dumps(optimization.parameters),
        json.dumps(optimization.constraints), json.dumps(optimization.objectives),
        json.dumps(optimization.affected_sites), 'pending', 0.0,
        current_user.id, datetime.utcnow())

        # Start optimization in background
        background_tasks.add_task(
            process_optimization_task,
            task_id,
            optimization.dict()
        )

        logger.info(f"Created optimization task {task_id}")

        return {
            "task_id": task_id,
            "status": "created",
            "message": "Optimization task created successfully"
        }

    except Exception as e:
        logger.error(f"Error creating optimization: {e}")
        raise HTTPException(status_code=500, detail="Failed to create optimization task")

@router.get("/optimizations")
async def get_optimizations(
    status: Optional[str] = Query(None),
    optimization_type: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get optimization tasks"""
    try:
        query = """
            SELECT 
                ot.id, ot.name, ot.description, ot.optimization_type, ot.status,
                ot.progress_percentage, ot.created_at, ot.started_at, ot.completed_at,
                u.full_name as created_by_name,
                or_.estimated_savings, or_.confidence_score,
                COUNT(*) OVER() as total_count
            FROM optimization_tasks ot
            LEFT JOIN users u ON ot.created_by = u.id
            LEFT JOIN optimization_results or_ ON ot.id = or_.task_id
            WHERE ot.tenant_id = $1
        """
        
        params = [current_user.tenant_id]
        param_count = 1

        if status:
            param_count += 1
            query += f" AND ot.status = ${param_count}"
            params.append(status)

        if optimization_type:
            param_count += 1
            query += f" AND ot.optimization_type = ${param_count}"
            params.append(optimization_type)

        query += f" ORDER BY ot.created_at DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
        params.extend([limit, offset])

        rows = await db.execute_query(query, *params)

        optimizations = []
        total_count = 0

        for row in rows:
            if total_count == 0:
                total_count = row['total_count']

            optimization_data = {
                "id": row['id'],
                "name": row['name'],
                "description": row['description'],
                "optimization_type": row['optimization_type'],
                "status": row['status'],
                "progress_percentage": row['progress_percentage'],
                "created_by": row['created_by_name'],
                "created_at": row['created_at'].isoformat(),
                "started_at": row['started_at'].isoformat() if row['started_at'] else None,
                "completed_at": row['completed_at'].isoformat() if row['completed_at'] else None,
                "estimated_savings": row['estimated_savings'],
                "confidence_score": row['confidence_score']
            }
            optimizations.append(optimization_data)

        return {
            "optimizations": optimizations,
            "total_count": total_count,
            "page": offset // limit + 1,
            "page_size": limit,
            "has_more": (offset + limit) < total_count
        }

    except Exception as e:
        logger.error(f"Error retrieving optimizations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve optimizations")

@router.get("/roi-analysis")
async def get_roi_analysis(
    time_range: str = Query("90d", pattern="^(30d|90d|180d|1y)$"),
    current_user: User = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get comprehensive ROI analysis"""
    try:
        # Calculate time range
        time_delta_map = {
            "30d": timedelta(days=30),
            "90d": timedelta(days=90), 
            "180d": timedelta(days=180),
            "1y": timedelta(days=365)
        }
        start_time = datetime.utcnow() - time_delta_map[time_range]

        # Get baseline metrics
        baseline_metrics = await db.execute_query_one("""
            SELECT 
                AVG(energy_cost) as baseline_energy_cost,
                AVG(maintenance_cost) as baseline_maintenance_cost,
                AVG(operational_cost) as baseline_operational_cost,
                AVG(availability) as baseline_availability,
                COUNT(DISTINCT site_id) as total_sites
            FROM kpi_metrics 
            WHERE tenant_id = $1 AND timestamp BETWEEN $2 AND $3
        """, current_user.tenant_id, start_time - timedelta(days=30), start_time)

        # Get current metrics
        current_metrics = await db.execute_query_one("""
            SELECT 
                AVG(energy_cost) as current_energy_cost,
                AVG(maintenance_cost) as current_maintenance_cost,
                AVG(operational_cost) as current_operational_cost,
                AVG(availability) as current_availability
            FROM kpi_metrics 
            WHERE tenant_id = $1 AND timestamp >= $3
        """, current_user.tenant_id, start_time, datetime.utcnow())

        # Get implemented insights value
        implemented_value = await db.execute_query_one("""
            SELECT 
                COUNT(*) as implemented_insights,
                SUM(potential_savings) as total_realized_savings,
                SUM(implementation_cost) as total_implementation_cost,
                AVG(roi_estimate) as avg_roi
            FROM business_insights 
            WHERE tenant_id = $1 AND status = 'implemented'
            AND created_at >= $2
        """, current_user.tenant_id, start_time)

        # Calculate improvements
        energy_improvement = 0
        maintenance_improvement = 0
        operational_improvement = 0
        availability_improvement = 0

        if baseline_metrics['baseline_energy_cost'] and current_metrics['current_energy_cost']:
            energy_improvement = ((baseline_metrics['baseline_energy_cost'] - current_metrics['current_energy_cost']) / baseline_metrics['baseline_energy_cost']) * 100

        if baseline_metrics['baseline_maintenance_cost'] and current_metrics['current_maintenance_cost']:
            maintenance_improvement = ((baseline_metrics['baseline_maintenance_cost'] - current_metrics['current_maintenance_cost']) / baseline_metrics['baseline_maintenance_cost']) * 100

        if baseline_metrics['baseline_operational_cost'] and current_metrics['current_operational_cost']:
            operational_improvement = ((baseline_metrics['baseline_operational_cost'] - current_metrics['current_operational_cost']) / baseline_metrics['baseline_operational_cost']) * 100

        if baseline_metrics['baseline_availability'] and current_metrics['current_availability']:
            availability_improvement = ((current_metrics['current_availability'] - baseline_metrics['baseline_availability']) / baseline_metrics['baseline_availability']) * 100

        # Calculate total value impact
        monthly_cost_baseline = (baseline_metrics['baseline_energy_cost'] or 0) + (baseline_metrics['baseline_maintenance_cost'] or 0) + (baseline_metrics['baseline_operational_cost'] or 0)
        monthly_cost_current = (current_metrics['current_energy_cost'] or 0) + (current_metrics['current_maintenance_cost'] or 0) + (current_metrics['current_operational_cost'] or 0)
        monthly_savings = monthly_cost_baseline - monthly_cost_current
        annual_savings = monthly_savings * 12

        roi_analysis = {
            "time_range": time_range,
            "analysis_date": datetime.utcnow().isoformat(),
            "baseline_period": {
                "start_date": (start_time - timedelta(days=30)).isoformat(),
                "end_date": start_time.isoformat()
            },
            "current_period": {
                "start_date": start_time.isoformat(),
                "end_date": datetime.utcnow().isoformat()
            },
            "cost_improvements": {
                "energy_cost_improvement_percentage": round(energy_improvement, 2),
                "maintenance_cost_improvement_percentage": round(maintenance_improvement, 2),
                "operational_cost_improvement_percentage": round(operational_improvement, 2),
                "total_monthly_savings": round(monthly_savings, 2),
                "annualized_savings": round(annual_savings, 2)
            },
            "performance_improvements": {
                "availability_improvement_percentage": round(availability_improvement, 2),
                "baseline_availability": round(baseline_metrics['baseline_availability'] or 0, 2),
                "current_availability": round(current_metrics['current_availability'] or 0, 2)
            },
            "investment_analysis": {
                "implemented_insights": implemented_value['implemented_insights'] or 0,
                "total_investment": round(implemented_value['total_implementation_cost'] or 0, 2),
                "realized_savings": round(implemented_value['total_realized_savings'] or 0, 2),
                "roi_percentage": round(implemented_value['avg_roi'] or 0, 2),
                "payback_period_months": round((implemented_value['total_implementation_cost'] or 0) / max(monthly_savings, 1), 1) if monthly_savings > 0 else 0
            },
            "value_metrics": {
                "cost_per_site_baseline": round(monthly_cost_baseline / max(baseline_metrics['total_sites'] or 1, 1), 2),
                "cost_per_site_current": round(monthly_cost_current / max(baseline_metrics['total_sites'] or 1, 1), 2),
                "total_sites": baseline_metrics['total_sites'] or 0,
                "value_creation_rate": round(annual_savings / max(implemented_value['total_implementation_cost'] or 1, 1), 2)
            }
        }

        return roi_analysis

    except Exception as e:
        logger.error(f"Error generating ROI analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate ROI analysis")

@router.put("/insights/{insight_id}/implement")
async def implement_insight(
    insight_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("insights.implement")),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Mark insight as implemented and track value"""
    try:
        # Update insight status
        result = await db.execute_query_one("""
            UPDATE business_insights 
            SET status = 'implemented', implemented_at = NOW(), implemented_by = $2
            WHERE id = $1 AND tenant_id = $3
            RETURNING title, potential_savings
        """, insight_id, current_user.id, current_user.tenant_id)

        if not result:
            raise HTTPException(status_code=404, detail="Insight not found")

        # Create value tracking entry
        await db.execute_command("""
            INSERT INTO value_tracking (
                insight_id, tenant_id, implemented_by, implemented_at,
                estimated_value, tracking_start_date, status
            ) VALUES ($1, $2, $3, NOW(), $4, NOW(), 'tracking')
        """, insight_id, current_user.tenant_id, current_user.id, result['potential_savings'])

        # Schedule value measurement
        background_tasks.add_task(
            schedule_value_measurement,
            insight_id,
            current_user.tenant_id
        )

        logger.info(f"Implemented insight {insight_id}")

        return {
            "insight_id": insight_id,
            "title": result['title'],
            "estimated_value": result['potential_savings'],
            "status": "implemented",
            "message": "Insight marked as implemented and value tracking started"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error implementing insight {insight_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to implement insight")

# Background task functions
async def process_optimization_task(task_id: str, optimization_data: Dict[str, Any]):
    """Process optimization task in background"""
    try:
        logger.info(f"Processing optimization task {task_id}")
        await asyncio.sleep(5)  # Simulate processing
        logger.info(f"Completed optimization task {task_id}")
        
    except Exception as e:
        logger.error(f"Error processing optimization task {task_id}: {e}")

async def schedule_value_measurement(insight_id: str, tenant_id: str):
    """Schedule value measurement for implemented insight"""
    try:
        logger.info(f"Scheduling value measurement for insight {insight_id}")
        # This would schedule periodic value measurements
        
    except Exception as e:
        logger.error(f"Error scheduling value measurement: {e}")
