"""
AIOps API Endpoints - Machine Learning Operations API
Anomaly detection, RCA, event correlation, and predictive maintenance endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncpg
import json

from core.database import get_database_manager, DatabaseManager
from core.auth import get_current_user
from core.models import APIResponse
from core.messaging import MessageProducer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/aiops",
    tags=["AIOps"],
    dependencies=[Depends(get_current_user)]
)

@router.get("/anomalies/recent")
async def get_recent_anomalies(
    site_id: Optional[str] = Query(None),
    metric_name: Optional[str] = Query(None),
    anomaly_type: Optional[str] = Query(None),
    hours: int = Query(24, le=168),
    limit: int = Query(100, le=500),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get recent anomaly detections"""
    try:
        query = """
        SELECT id, metric_name, site_id, anomaly_score, anomaly_type,
               detected_at, value, expected_range, confidence,
               contributing_factors, metadata
        FROM anomaly_detections
        WHERE detected_at >= NOW() - INTERVAL %s
        """
        
        params = [f'{hours} hours']
        
        if site_id:
            query += f" AND site_id = ${len(params) + 1}"
            params.append(site_id)
            
        if metric_name:
            query += f" AND metric_name = ${len(params) + 1}"
            params.append(metric_name)
            
        if anomaly_type:
            query += f" AND anomaly_type = ${len(params) + 1}"
            params.append(anomaly_type)
        
        query += f" ORDER BY detected_at DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        # Replace %s with proper parameter
        query = query.replace('%s', '$1')
        
        rows = await db.execute_query(query, *params)
        
        anomalies = []
        for row in rows:
            anomaly = {
                "id": row['id'],
                "metric_name": row['metric_name'],
                "site_id": row['site_id'],
                "anomaly_score": float(row['anomaly_score']),
                "anomaly_type": row['anomaly_type'],
                "detected_at": row['detected_at'].isoformat(),
                "value": float(row['value']),
                "expected_range": row['expected_range'] or {},
                "confidence": float(row['confidence']),
                "contributing_factors": row['contributing_factors'] or [],
                "metadata": row['metadata'] or {}
            }
            anomalies.append(anomaly)
        
        logger.info(f"Retrieved {len(anomalies)} anomalies")
        return {"anomalies": anomalies, "count": len(anomalies)}
        
    except Exception as e:
        logger.error(f"Error retrieving anomalies: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve anomalies")

@router.get("/rca/results")
async def get_rca_results(
    incident_id: Optional[str] = Query(None),
    site_id: Optional[str] = Query(None),
    confidence_min: Optional[float] = Query(None, ge=0.0, le=1.0),
    days: int = Query(7, le=90),
    limit: int = Query(50, le=200),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get Root Cause Analysis results"""
    try:
        query = """
        SELECT id, incident_id, primary_symptom, site_id, analysis_timestamp,
               root_causes, confidence_level, confidence_score, causal_chain,
               contributing_factors, recommendations, similar_incidents,
               analysis_duration, metadata
        FROM rca_results
        WHERE analysis_timestamp >= NOW() - INTERVAL %s
        """
        
        params = [f'{days} days']
        
        if incident_id:
            query += f" AND incident_id = ${len(params) + 1}"
            params.append(incident_id)
            
        if site_id:
            query += f" AND site_id = ${len(params) + 1}"
            params.append(site_id)
            
        if confidence_min is not None:
            query += f" AND confidence_score >= ${len(params) + 1}"
            params.append(confidence_min)
        
        query += f" ORDER BY analysis_timestamp DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        # Replace %s with proper parameter
        query = query.replace('%s', '$1')
        
        rows = await db.execute_query(query, *params)
        
        rca_results = []
        for row in rows:
            result = {
                "id": row['id'],
                "incident_id": row['incident_id'],
                "primary_symptom": row['primary_symptom'],
                "site_id": row['site_id'],
                "analysis_timestamp": row['analysis_timestamp'].isoformat(),
                "root_causes": row['root_causes'] or [],
                "confidence_level": row['confidence_level'],
                "confidence_score": float(row['confidence_score']),
                "causal_chain": row['causal_chain'] or [],
                "contributing_factors": row['contributing_factors'] or [],
                "recommendations": row['recommendations'] or [],
                "similar_incidents": row['similar_incidents'] or [],
                "analysis_duration": float(row['analysis_duration']),
                "metadata": row['metadata'] or {}
            }
            rca_results.append(result)
        
        logger.info(f"Retrieved {len(rca_results)} RCA results")
        return {"rca_results": rca_results, "count": len(rca_results)}
        
    except Exception as e:
        logger.error(f"Error retrieving RCA results: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve RCA results")

@router.post("/rca/analyze")
async def trigger_rca_analysis(
    background_tasks: BackgroundTasks,
    incident_id: str,
    primary_symptom: str,
    site_id: str,
    timestamp: Optional[str] = None,
    db: DatabaseManager = Depends(get_database_manager)
):
    """Trigger Root Cause Analysis for an incident"""
    try:
        analysis_timestamp = datetime.fromisoformat(timestamp) if timestamp else datetime.utcnow()
        
        # Store RCA request
        request_id = await db.execute_query_scalar("""
            INSERT INTO rca_requests (
                incident_id, primary_symptom, site_id, requested_at,
                status, priority
            ) VALUES ($1, $2, $3, $4, 'queued', 'normal')
            RETURNING id
        """, incident_id, primary_symptom, site_id, analysis_timestamp)
        
        # Send RCA request to AIOps workers via Kafka
        success = await MessageProducer.send_aiops_prediction_request(
            model_name="rca_analyzer",
            site_id=site_id,
            prediction_type="root_cause_analysis",
            target_time=analysis_timestamp.isoformat(),
            input_data={
                "incident_id": incident_id,
                "primary_symptom": primary_symptom,
                "request_id": request_id
            }
        )
        
        if not success:
            logger.warning(f"Failed to send RCA request to Kafka for incident {incident_id}")
        
        # Also create event for event correlator
        await MessageProducer.send_event(
            site_id=site_id,
            tenant_id="default",
            event_type="rca_requested",
            severity="high",
            source_system="aiops-api",
            title=f"RCA analysis requested for incident {incident_id}",
            description=f"Primary symptom: {primary_symptom}",
            metadata={"incident_id": incident_id, "request_id": request_id}
        )
        
        # Add background task as fallback
        background_tasks.add_task(process_rca_request, request_id, {
            'incident_id': incident_id,
            'primary_symptom': primary_symptom,
            'site_id': site_id,
            'timestamp': analysis_timestamp.isoformat()
        })
        
        logger.info(f"RCA analysis sent to workers via Kafka for incident {incident_id}")
        
        return {
            "request_id": request_id,
            "status": "queued",
            "message": "RCA analysis has been queued for processing"
        }
        
    except Exception as e:
        logger.error(f"Error triggering RCA analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger RCA analysis")

@router.get("/correlations/events")
async def get_event_correlations(
    cluster_id: Optional[str] = Query(None),
    correlation_type: Optional[str] = Query(None),
    strength_min: Optional[str] = Query(None),
    hours: int = Query(6, le=48),
    limit: int = Query(100, le=500),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get event correlation results"""
    try:
        query = """
        SELECT id, cluster_id, correlation_type, correlation_strength,
               cluster_score, events, event_count, site_count,
               time_span_seconds, pattern_signature, severity_distribution,
               created_at, metadata
        FROM event_correlations
        WHERE created_at >= NOW() - INTERVAL %s
        """
        
        params = [f'{hours} hours']
        
        if cluster_id:
            query += f" AND cluster_id = ${len(params) + 1}"
            params.append(cluster_id)
            
        if correlation_type:
            query += f" AND correlation_type = ${len(params) + 1}"
            params.append(correlation_type)
            
        if strength_min:
            query += f" AND correlation_strength >= ${len(params) + 1}::correlation_strength"
            params.append(strength_min)
        
        query += f" ORDER BY created_at DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        # Replace %s with proper parameter
        query = query.replace('%s', '$1')
        
        rows = await db.execute_query(query, *params)
        
        correlations = []
        for row in rows:
            correlation = {
                "id": row['id'],
                "cluster_id": row['cluster_id'],
                "correlation_type": row['correlation_type'],
                "correlation_strength": row['correlation_strength'],
                "cluster_score": float(row['cluster_score']),
                "events": row['events'] or [],
                "event_count": row['event_count'],
                "site_count": row['site_count'],
                "time_span_seconds": float(row['time_span_seconds']),
                "pattern_signature": row['pattern_signature'],
                "severity_distribution": row['severity_distribution'] or {},
                "created_at": row['created_at'].isoformat(),
                "metadata": row['metadata'] or {}
            }
            correlations.append(correlation)
        
        logger.info(f"Retrieved {len(correlations)} event correlations")
        return {"correlations": correlations, "count": len(correlations)}
        
    except Exception as e:
        logger.error(f"Error retrieving event correlations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve event correlations")

@router.get("/maintenance/recommendations")
async def get_maintenance_recommendations(
    site_id: Optional[str] = Query(None),
    component_id: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    maintenance_type: Optional[str] = Query(None),
    days_ahead: int = Query(90, le=365),
    limit: int = Query(100, le=500),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get predictive maintenance recommendations"""
    try:
        query = """
        SELECT id, component_id, site_id, component_type, predicted_failure_date,
               risk_level, risk_score, confidence, maintenance_type,
               recommended_actions, cost_estimate, impact_assessment,
               supporting_evidence, created_at, metadata
        FROM maintenance_recommendations
        WHERE created_at >= NOW() - INTERVAL '30 days'
            AND predicted_failure_date <= NOW() + INTERVAL %s
        """
        
        params = [f'{days_ahead} days']
        
        if site_id:
            query += f" AND site_id = ${len(params) + 1}"
            params.append(site_id)
            
        if component_id:
            query += f" AND component_id = ${len(params) + 1}"
            params.append(component_id)
            
        if risk_level:
            query += f" AND risk_level = ${len(params) + 1}"
            params.append(risk_level)
            
        if maintenance_type:
            query += f" AND maintenance_type = ${len(params) + 1}"
            params.append(maintenance_type)
        
        query += f" ORDER BY risk_score DESC, predicted_failure_date ASC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        # Replace %s with proper parameter
        query = query.replace('%s', '$1')
        
        rows = await db.execute_query(query, *params)
        
        recommendations = []
        for row in rows:
            recommendation = {
                "id": row['id'],
                "component_id": row['component_id'],
                "site_id": row['site_id'],
                "component_type": row['component_type'],
                "predicted_failure_date": row['predicted_failure_date'].isoformat() if row['predicted_failure_date'] else None,
                "risk_level": row['risk_level'],
                "risk_score": float(row['risk_score']),
                "confidence": float(row['confidence']),
                "maintenance_type": row['maintenance_type'],
                "recommended_actions": row['recommended_actions'] or [],
                "cost_estimate": float(row['cost_estimate']),
                "impact_assessment": row['impact_assessment'] or {},
                "supporting_evidence": row['supporting_evidence'] or [],
                "created_at": row['created_at'].isoformat(),
                "metadata": row['metadata'] or {}
            }
            recommendations.append(recommendation)
        
        logger.info(f"Retrieved {len(recommendations)} maintenance recommendations")
        return {"recommendations": recommendations, "count": len(recommendations)}
        
    except Exception as e:
        logger.error(f"Error retrieving maintenance recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve maintenance recommendations")

@router.get("/health/components/{site_id}")
async def get_component_health(
    site_id: str,
    component_type: Optional[str] = Query(None),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get component health status for a site"""
    try:
        query = """
        SELECT ch.component_id, ch.site_id, ch.component_type, ch.health_score,
               ch.degradation_rate, ch.remaining_useful_life_days, 
               ch.critical_parameters, ch.trends, ch.last_assessment, ch.metadata,
               e.vendor, e.model, e.installation_date
        FROM component_health ch
        LEFT JOIN equipment e ON ch.component_id = e.equipment_id AND ch.site_id = e.site_id
        WHERE ch.site_id = $1
        """
        
        params = [site_id]
        
        if component_type:
            query += f" AND ch.component_type = ${len(params) + 1}"
            params.append(component_type)
        
        query += " ORDER BY ch.health_score ASC, ch.remaining_useful_life_days ASC"
        
        rows = await db.execute_query(query, *params)
        
        components = []
        for row in rows:
            component = {
                "component_id": row['component_id'],
                "site_id": row['site_id'],
                "component_type": row['component_type'],
                "health_score": float(row['health_score']),
                "degradation_rate": float(row['degradation_rate']),
                "remaining_useful_life_days": row['remaining_useful_life_days'],
                "critical_parameters": row['critical_parameters'] or {},
                "trends": row['trends'] or {},
                "last_assessment": row['last_assessment'].isoformat(),
                "metadata": row['metadata'] or {},
                "equipment_info": {
                    "vendor": row['vendor'],
                    "model": row['model'],
                    "installation_date": row['installation_date'].isoformat() if row['installation_date'] else None
                }
            }
            components.append(component)
        
        logger.info(f"Retrieved health for {len(components)} components at site {site_id}")
        return {"components": components, "count": len(components)}
        
    except Exception as e:
        logger.error(f"Error retrieving component health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve component health")

@router.post("/health/assess/{component_id}")
async def trigger_health_assessment(
    component_id: str,
    site_id: str,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_database_manager)
):
    """Trigger health assessment for a specific component"""
    try:
        # Verify component exists
        component_exists = await db.execute_query_scalar("""
            SELECT COUNT(*) FROM equipment 
            WHERE equipment_id = $1 AND site_id = $2
        """, component_id, site_id)
        
        if not component_exists:
            raise HTTPException(status_code=404, detail="Component not found")
        
        # Queue health assessment
        assessment_id = await db.execute_query_scalar("""
            INSERT INTO health_assessment_requests (
                component_id, site_id, requested_at, status, priority
            ) VALUES ($1, $2, NOW(), 'queued', 'normal')
            RETURNING id
        """, component_id, site_id)
        
        # Add background task
        background_tasks.add_task(process_health_assessment, assessment_id, component_id, site_id)
        
        logger.info(f"Health assessment queued for component {component_id}")
        
        return {
            "assessment_id": assessment_id,
            "status": "queued",
            "message": "Health assessment has been queued for processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering health assessment: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger health assessment")

@router.get("/predictions/failures")
async def get_failure_predictions(
    site_id: Optional[str] = Query(None),
    component_type: Optional[str] = Query(None),
    risk_min: Optional[float] = Query(0.3, ge=0.0, le=1.0),
    days_ahead: int = Query(90, le=365),
    limit: int = Query(100, le=500),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get failure predictions"""
    try:
        query = """
        SELECT fp.id, fp.component_id, fp.site_id, fp.component_type,
               fp.failure_probability, fp.predicted_failure_date, fp.confidence,
               fp.contributing_factors, fp.model_version, fp.created_at,
               e.vendor, e.model
        FROM failure_predictions fp
        LEFT JOIN equipment e ON fp.component_id = e.equipment_id AND fp.site_id = e.site_id
        WHERE fp.failure_probability >= $1
            AND fp.predicted_failure_date <= NOW() + INTERVAL %s
            AND fp.created_at >= NOW() - INTERVAL '7 days'
        """
        
        params = [risk_min, f'{days_ahead} days']
        
        if site_id:
            query += f" AND fp.site_id = ${len(params) + 1}"
            params.append(site_id)
            
        if component_type:
            query += f" AND fp.component_type = ${len(params) + 1}"
            params.append(component_type)
        
        query += f" ORDER BY fp.failure_probability DESC, fp.predicted_failure_date ASC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        # Replace %s with proper parameter
        query = query.replace('%s', '$2')
        
        rows = await db.execute_query(query, *params)
        
        predictions = []
        for row in rows:
            prediction = {
                "id": row['id'],
                "component_id": row['component_id'],
                "site_id": row['site_id'],
                "component_type": row['component_type'],
                "failure_probability": float(row['failure_probability']),
                "predicted_failure_date": row['predicted_failure_date'].isoformat() if row['predicted_failure_date'] else None,
                "confidence": float(row['confidence']),
                "contributing_factors": row['contributing_factors'] or [],
                "model_version": row['model_version'],
                "created_at": row['created_at'].isoformat(),
                "equipment_info": {
                    "vendor": row['vendor'],
                    "model": row['model']
                }
            }
            predictions.append(prediction)
        
        logger.info(f"Retrieved {len(predictions)} failure predictions")
        return {"predictions": predictions, "count": len(predictions)}
        
    except Exception as e:
        logger.error(f"Error retrieving failure predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve failure predictions")

@router.get("/insights/summary")
async def get_aiops_insights_summary(
    site_id: Optional[str] = Query(None),
    hours: int = Query(24, le=168),
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get AIOps insights summary"""
    try:
        # Build time condition
        time_condition = "WHERE created_at >= NOW() - INTERVAL %s"
        site_condition = ""
        params = [f'{hours} hours']
        
        if site_id:
            site_condition = f" AND site_id = ${len(params) + 1}"
            params.append(site_id)
        
        # Get anomaly counts
        anomaly_stats = await db.execute_query_one(f"""
            SELECT 
                COUNT(*) as total_anomalies,
                COUNT(CASE WHEN anomaly_type = 'point' THEN 1 END) as point_anomalies,
                COUNT(CASE WHEN anomaly_type = 'contextual' THEN 1 END) as contextual_anomalies,
                COUNT(CASE WHEN anomaly_type = 'collective' THEN 1 END) as collective_anomalies,
                AVG(confidence) as avg_confidence
            FROM anomaly_detections
            {time_condition.replace('%s', '$1')} {site_condition}
        """, *params)
        
        # Get RCA stats
        rca_stats = await db.execute_query_one(f"""
            SELECT 
                COUNT(*) as total_rca,
                COUNT(CASE WHEN confidence_level = 'high' THEN 1 END) as high_confidence_rca,
                AVG(confidence_score) as avg_confidence_score,
                AVG(analysis_duration) as avg_analysis_time
            FROM rca_results
            {time_condition.replace('created_at', 'analysis_timestamp').replace('%s', '$1')} {site_condition}
        """, *params)
        
        # Get correlation stats
        correlation_stats = await db.execute_query_one(f"""
            SELECT 
                COUNT(*) as total_correlations,
                COUNT(CASE WHEN correlation_strength = 'strong' THEN 1 END) as strong_correlations,
                AVG(cluster_score) as avg_cluster_score,
                SUM(event_count) as total_correlated_events
            FROM event_correlations
            {time_condition.replace('%s', '$1')} {site_condition}
        """, *params)
        
        # Get maintenance stats
        maintenance_stats = await db.execute_query_one(f"""
            SELECT 
                COUNT(*) as total_recommendations,
                COUNT(CASE WHEN risk_level = 'critical' THEN 1 END) as critical_recommendations,
                COUNT(CASE WHEN risk_level = 'high' THEN 1 END) as high_risk_recommendations,
                AVG(cost_estimate) as avg_cost_estimate
            FROM maintenance_recommendations
            {time_condition.replace('%s', '$1')} {site_condition}
        """, *params)
        
        # Get top issues
        top_anomalous_metrics = await db.execute_query(f"""
            SELECT metric_name, COUNT(*) as anomaly_count
            FROM anomaly_detections
            {time_condition.replace('%s', '$1')} {site_condition}
            GROUP BY metric_name
            ORDER BY anomaly_count DESC
            LIMIT 5
        """, *params)
        
        summary = {
            "time_range_hours": hours,
            "site_id": site_id,
            "anomaly_detection": {
                "total_anomalies": anomaly_stats['total_anomalies'] or 0,
                "point_anomalies": anomaly_stats['point_anomalies'] or 0,
                "contextual_anomalies": anomaly_stats['contextual_anomalies'] or 0,
                "collective_anomalies": anomaly_stats['collective_anomalies'] or 0,
                "avg_confidence": float(anomaly_stats['avg_confidence'] or 0)
            },
            "root_cause_analysis": {
                "total_analyses": rca_stats['total_rca'] or 0,
                "high_confidence_analyses": rca_stats['high_confidence_rca'] or 0,
                "avg_confidence_score": float(rca_stats['avg_confidence_score'] or 0),
                "avg_analysis_time_seconds": float(rca_stats['avg_analysis_time'] or 0)
            },
            "event_correlation": {
                "total_correlations": correlation_stats['total_correlations'] or 0,
                "strong_correlations": correlation_stats['strong_correlations'] or 0,
                "avg_cluster_score": float(correlation_stats['avg_cluster_score'] or 0),
                "total_correlated_events": correlation_stats['total_correlated_events'] or 0
            },
            "predictive_maintenance": {
                "total_recommendations": maintenance_stats['total_recommendations'] or 0,
                "critical_recommendations": maintenance_stats['critical_recommendations'] or 0,
                "high_risk_recommendations": maintenance_stats['high_risk_recommendations'] or 0,
                "avg_cost_estimate": float(maintenance_stats['avg_cost_estimate'] or 0)
            },
            "top_anomalous_metrics": [
                {"metric_name": row['metric_name'], "anomaly_count": row['anomaly_count']}
                for row in top_anomalous_metrics
            ]
        }
        
        logger.info(f"Generated AIOps insights summary for {hours}h period")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating AIOps insights summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate insights summary")


# Background task functions
async def process_rca_request(request_id: int, incident_data: Dict[str, Any]):
    """Background task to process RCA request"""
    try:
        # This would integrate with the RCA engine
        # For now, just update the status
        logger.info(f"Processing RCA request {request_id}")
        await asyncio.sleep(5)  # Simulate processing
        logger.info(f"RCA request {request_id} completed")
        
    except Exception as e:
        logger.error(f"Error processing RCA request {request_id}: {e}")

async def process_health_assessment(assessment_id: int, component_id: str, site_id: str):
    """Background task to process health assessment"""
    try:
        # This would integrate with the predictive maintenance engine
        logger.info(f"Processing health assessment {assessment_id} for component {component_id}")
        await asyncio.sleep(3)  # Simulate processing
        logger.info(f"Health assessment {assessment_id} completed")
        
    except Exception as e:
        logger.error(f"Error processing health assessment {assessment_id}: {e}")
