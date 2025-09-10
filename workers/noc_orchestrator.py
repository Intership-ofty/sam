"""
NOC (Network Operations Center) Orchestrator
Intelligent operational center for unified monitoring and incident management
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncpg
import redis.asyncio as redis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

import signal
from aiohttp import web

# Import core modules from backend
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from core.database import init_db
from core.cache import init_redis
from core.messaging import init_kafka, start_consumer
from core.config import settings, KAFKA_TOPICS
from core.monitoring import init_monitoring

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentStatus(Enum):
    NEW = "new"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"

class WorkflowAction(Enum):
    ASSIGN = "assign"
    ESCALATE = "escalate"
    NOTIFY = "notify"
    CREATE_TICKET = "create_ticket"
    RUN_AUTOMATION = "run_automation"
    UPDATE_STATUS = "update_status"

@dataclass
class NOCIncident:
    id: str
    tenant_id: str
    site_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    category: str
    source_system: str
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str] = None
    escalated_to: Optional[str] = None
    parent_incident_id: Optional[str] = None
    child_incidents: List[str] = None
    sla_breach_time: Optional[datetime] = None
    resolution_time: Optional[datetime] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class EscalationRule:
    id: str
    name: str
    tenant_id: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    time_threshold_minutes: int
    severity_levels: List[IncidentSeverity]
    enabled: bool = True
    priority: int = 1

@dataclass
class WorkflowStep:
    id: str
    name: str
    action: WorkflowAction
    parameters: Dict[str, Any]
    conditions: Dict[str, Any]
    timeout_minutes: int
    retry_count: int = 3
    on_success: Optional[str] = None
    on_failure: Optional[str] = None

class NOCOrchestrator:
    """Intelligent NOC orchestrator for unified operations management"""
    
    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        self.ml_models = {}
        self.escalation_rules = {}
        self.workflow_definitions = {}
        
        # Initialize ML components
        self.incident_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.severity_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.resolution_time_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # NOC configuration
        self.config = {
            'max_concurrent_incidents': 1000,
            'auto_escalation_enabled': True,
            'sla_monitoring_enabled': True,
            'intelligent_routing_enabled': True,
            'automation_engine_enabled': True,
            'correlation_window_minutes': 15,
            'escalation_check_interval': 60,
        }
    
    async def initialize(self):
        """Initialize NOC orchestrator"""
        try:
            # Initialize database connection
            self.db_pool = await asyncpg.create_pool(
                "postgresql://towerco:secure_password@localhost:5432/towerco_aiops",
                min_size=5,
                max_size=20
            )
            
            # Initialize Redis connection
            self.redis_client = await redis.from_url("redis://localhost:6379", decode_responses=True)
            
            # Load escalation rules and workflows
            await self.load_escalation_rules()
            await self.load_workflow_definitions()
            
            # Train ML models with historical data
            await self.train_ml_models()
            
            logger.info("NOC Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NOC Orchestrator: {e}")
            raise
    
    async def process_incident_creation(self, incident_data: Dict[str, Any]) -> NOCIncident:
        """Process new incident creation with intelligent routing"""
        try:
            # Create incident object
            incident = NOCIncident(
                id=incident_data['id'],
                tenant_id=incident_data['tenant_id'],
                site_id=incident_data['site_id'],
                title=incident_data['title'],
                description=incident_data['description'],
                severity=IncidentSeverity(incident_data.get('severity', 'medium')),
                status=IncidentStatus.NEW,
                category=incident_data.get('category', 'general'),
                source_system=incident_data.get('source_system', 'aiops'),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                tags=incident_data.get('tags', []),
                metadata=incident_data.get('metadata', {}),
                child_incidents=[]
            )
            
            # Intelligent severity prediction
            predicted_severity = await self.predict_incident_severity(incident)
            if predicted_severity != incident.severity:
                logger.info(f"ML predicted different severity: {predicted_severity} vs {incident.severity}")
                incident.severity = predicted_severity
            
            # Check for incident correlation
            related_incidents = await self.find_related_incidents(incident)
            if related_incidents:
                incident = await self.handle_incident_correlation(incident, related_incidents)
            
            # Intelligent assignment
            assigned_user = await self.intelligent_assignment(incident)
            if assigned_user:
                incident.assigned_to = assigned_user
                incident.status = IncidentStatus.ASSIGNED
            
            # Store incident
            await self.store_incident(incident)
            
            # Start monitoring workflows
            await self.start_incident_workflows(incident)
            
            # Send notifications
            await self.send_incident_notifications(incident)
            
            logger.info(f"Processed incident creation: {incident.id}")
            return incident
            
        except Exception as e:
            logger.error(f"Error processing incident creation: {e}")
            raise
    
    async def predict_incident_severity(self, incident: NOCIncident) -> IncidentSeverity:
        """Use ML to predict incident severity"""
        try:
            # Extract features for prediction
            features = await self.extract_incident_features(incident)
            
            if features and len(features) > 0:
                features_scaled = self.scaler.transform([features])
                prediction = self.severity_predictor.predict(features_scaled)[0]
                
                severity_mapping = {0: IncidentSeverity.LOW, 1: IncidentSeverity.MEDIUM, 
                                  2: IncidentSeverity.HIGH, 3: IncidentSeverity.CRITICAL}
                
                return severity_mapping.get(prediction, incident.severity)
            
            return incident.severity
            
        except Exception as e:
            logger.error(f"Error predicting incident severity: {e}")
            return incident.severity
    
    async def extract_incident_features(self, incident: NOCIncident) -> List[float]:
        """Extract features for ML prediction"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get site metrics
                site_metrics = await conn.fetchrow("""
                    SELECT 
                        AVG(availability) as avg_availability,
                        AVG(uptime_percentage) as avg_uptime,
                        COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_alerts_24h
                    FROM kpi_metrics km
                    WHERE km.site_id = $1 
                    AND km.timestamp > NOW() - INTERVAL '24 hours'
                """, incident.site_id)
                
                # Get historical incident patterns
                incident_patterns = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_incidents_7d,
                        AVG(EXTRACT(EPOCH FROM (resolution_time - created_at))/3600) as avg_resolution_hours
                    FROM noc_incidents 
                    WHERE site_id = $1 
                    AND created_at > NOW() - INTERVAL '7 days'
                    AND status = 'resolved'
                """, incident.site_id)
                
                features = [
                    float(site_metrics['avg_availability'] or 0),
                    float(site_metrics['avg_uptime'] or 0),
                    float(site_metrics['critical_alerts_24h'] or 0),
                    float(incident_patterns['total_incidents_7d'] or 0),
                    float(incident_patterns['avg_resolution_hours'] or 0),
                    len(incident.tags) if incident.tags else 0,
                    1 if 'network' in incident.category.lower() else 0,
                    1 if 'power' in incident.category.lower() else 0,
                    1 if 'security' in incident.category.lower() else 0,
                    datetime.utcnow().hour,  # Time of day
                    datetime.utcnow().weekday(),  # Day of week
                ]
                
                return features
                
        except Exception as e:
            logger.error(f"Error extracting incident features: {e}")
            return []
    
    async def find_related_incidents(self, incident: NOCIncident) -> List[NOCIncident]:
        """Find related incidents for correlation"""
        try:
            async with self.db_pool.acquire() as conn:
                # Look for incidents in the same site within correlation window
                related_rows = await conn.fetch("""
                    SELECT * FROM noc_incidents 
                    WHERE tenant_id = $1 
                    AND site_id = $2 
                    AND status NOT IN ('resolved', 'closed')
                    AND created_at > $3
                    AND id != $4
                """, 
                incident.tenant_id,
                incident.site_id,
                datetime.utcnow() - timedelta(minutes=self.config['correlation_window_minutes']),
                incident.id)
                
                related_incidents = []
                for row in related_rows:
                    related_incident = NOCIncident(
                        id=row['id'],
                        tenant_id=row['tenant_id'],
                        site_id=row['site_id'],
                        title=row['title'],
                        description=row['description'],
                        severity=IncidentSeverity(row['severity']),
                        status=IncidentStatus(row['status']),
                        category=row['category'],
                        source_system=row['source_system'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        assigned_to=row['assigned_to'],
                        tags=json.loads(row['tags']) if row['tags'] else [],
                        metadata=json.loads(row['metadata']) if row['metadata'] else {}
                    )
                    related_incidents.append(related_incident)
                
                return related_incidents
                
        except Exception as e:
            logger.error(f"Error finding related incidents: {e}")
            return []
    
    async def handle_incident_correlation(self, incident: NOCIncident, related_incidents: List[NOCIncident]) -> NOCIncident:
        """Handle incident correlation and potential merging"""
        try:
            # Simple correlation logic - can be enhanced with NLP/ML
            correlation_score = 0
            potential_parent = None
            
            for related in related_incidents:
                score = 0
                
                # Same category
                if related.category == incident.category:
                    score += 3
                
                # Similar keywords in title/description
                incident_words = set((incident.title + " " + incident.description).lower().split())
                related_words = set((related.title + " " + related.description).lower().split())
                common_words = incident_words.intersection(related_words)
                score += len(common_words) * 2
                
                # Similar tags
                if incident.tags and related.tags:
                    common_tags = set(incident.tags).intersection(set(related.tags))
                    score += len(common_tags) * 2
                
                if score > correlation_score:
                    correlation_score = score
                    potential_parent = related
            
            # If strong correlation found, link incidents
            if correlation_score > 5 and potential_parent:
                incident.parent_incident_id = potential_parent.id
                
                # Update parent incident
                if not potential_parent.child_incidents:
                    potential_parent.child_incidents = []
                potential_parent.child_incidents.append(incident.id)
                
                await self.update_incident(potential_parent)
                
                logger.info(f"Linked incident {incident.id} to parent {potential_parent.id}")
            
            return incident
            
        except Exception as e:
            logger.error(f"Error handling incident correlation: {e}")
            return incident
    
    async def intelligent_assignment(self, incident: NOCIncident) -> Optional[str]:
        """Intelligently assign incident to best available engineer"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get available engineers for this tenant
                engineers = await conn.fetch("""
                    SELECT u.id, u.full_name, u.metadata
                    FROM users u
                    WHERE u.tenant_id = $1 
                    AND u.role IN ('engineer', 'senior_engineer', 'lead_engineer')
                    AND u.is_active = true
                """, incident.tenant_id)
                
                if not engineers:
                    return None
                
                best_engineer = None
                best_score = -1
                
                for engineer in engineers:
                    score = await self.calculate_assignment_score(engineer, incident)
                    if score > best_score:
                        best_score = score
                        best_engineer = engineer
                
                return best_engineer['id'] if best_engineer else None
                
        except Exception as e:
            logger.error(f"Error in intelligent assignment: {e}")
            return None
    
    async def calculate_assignment_score(self, engineer: Dict[str, Any], incident: NOCIncident) -> float:
        """Calculate assignment score for engineer-incident pair"""
        try:
            score = 0.0
            
            # Base score
            score += 10.0
            
            # Check current workload
            async with self.db_pool.acquire() as conn:
                current_incidents = await conn.fetchval("""
                    SELECT COUNT(*) FROM noc_incidents 
                    WHERE assigned_to = $1 
                    AND status NOT IN ('resolved', 'closed')
                """, engineer['id'])
                
                # Penalize overloaded engineers
                workload_penalty = min(current_incidents * 5, 30)
                score -= workload_penalty
                
                # Check expertise based on past incidents
                expertise_score = await conn.fetchval("""
                    SELECT COUNT(*) FROM noc_incidents 
                    WHERE assigned_to = $1 
                    AND category = $2 
                    AND status = 'resolved'
                """, engineer['id'], incident.category)
                
                score += expertise_score * 2
                
                # Check recent performance
                resolution_performance = await conn.fetchval("""
                    SELECT AVG(EXTRACT(EPOCH FROM (resolution_time - created_at))/3600)
                    FROM noc_incidents 
                    WHERE assigned_to = $1 
                    AND resolution_time IS NOT NULL
                    AND created_at > NOW() - INTERVAL '30 days'
                """, engineer['id'])
                
                if resolution_performance:
                    # Bonus for faster resolution times
                    if resolution_performance < 4:  # Less than 4 hours average
                        score += 15
                    elif resolution_performance < 8:  # Less than 8 hours average
                        score += 10
                
            return score
            
        except Exception as e:
            logger.error(f"Error calculating assignment score: {e}")
            return 0.0
    
    async def start_incident_workflows(self, incident: NOCIncident):
        """Start automated workflows for incident"""
        try:
            # Schedule escalation monitoring
            if self.config['auto_escalation_enabled']:
                await self.schedule_escalation_check(incident)
            
            # Start SLA monitoring
            if self.config['sla_monitoring_enabled']:
                await self.start_sla_monitoring(incident)
            
            # Run automation workflows
            if self.config['automation_engine_enabled']:
                await self.execute_automation_workflows(incident)
            
        except Exception as e:
            logger.error(f"Error starting incident workflows: {e}")
    
    async def schedule_escalation_check(self, incident: NOCIncident):
        """Schedule escalation check for incident"""
        try:
            # Store escalation timer in Redis
            escalation_key = f"escalation_timer:{incident.id}"
            escalation_time = datetime.utcnow() + timedelta(minutes=self.get_escalation_threshold(incident))
            
            await self.redis_client.set(
                escalation_key,
                json.dumps({
                    'incident_id': incident.id,
                    'escalation_time': escalation_time.isoformat(),
                    'current_severity': incident.severity.value
                }),
                ex=3600 * 24  # Expire after 24 hours
            )
            
        except Exception as e:
            logger.error(f"Error scheduling escalation check: {e}")
    
    def get_escalation_threshold(self, incident: NOCIncident) -> int:
        """Get escalation time threshold based on severity"""
        thresholds = {
            IncidentSeverity.CRITICAL: 15,  # 15 minutes
            IncidentSeverity.HIGH: 60,      # 1 hour
            IncidentSeverity.MEDIUM: 240,   # 4 hours
            IncidentSeverity.LOW: 480       # 8 hours
        }
        return thresholds.get(incident.severity, 240)
    
    async def monitor_escalations(self):
        """Monitor and process escalations"""
        try:
            while True:
                # Check for incidents that need escalation
                escalation_keys = await self.redis_client.keys("escalation_timer:*")
                
                for key in escalation_keys:
                    escalation_data = await self.redis_client.get(key)
                    if escalation_data:
                        data = json.loads(escalation_data)
                        escalation_time = datetime.fromisoformat(data['escalation_time'])
                        
                        if datetime.utcnow() >= escalation_time:
                            await self.process_escalation(data['incident_id'])
                            await self.redis_client.delete(key)
                
                await asyncio.sleep(self.config['escalation_check_interval'])
                
        except Exception as e:
            logger.error(f"Error in escalation monitoring: {e}")
    
    async def process_escalation(self, incident_id: str):
        """Process incident escalation"""
        try:
            incident = await self.get_incident(incident_id)
            if not incident or incident.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                return
            
            # Apply escalation rules
            for rule in self.escalation_rules.get(incident.tenant_id, []):
                if await self.evaluate_escalation_conditions(incident, rule):
                    await self.execute_escalation_actions(incident, rule)
                    break
            
        except Exception as e:
            logger.error(f"Error processing escalation for incident {incident_id}: {e}")
    
    async def store_incident(self, incident: NOCIncident):
        """Store incident in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO noc_incidents (
                        id, tenant_id, site_id, title, description, severity, status,
                        category, source_system, created_at, updated_at, assigned_to,
                        parent_incident_id, tags, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                incident.id, incident.tenant_id, incident.site_id, incident.title,
                incident.description, incident.severity.value, incident.status.value,
                incident.category, incident.source_system, incident.created_at,
                incident.updated_at, incident.assigned_to, incident.parent_incident_id,
                json.dumps(incident.tags), json.dumps(incident.metadata))
                
        except Exception as e:
            logger.error(f"Error storing incident: {e}")
            raise
    
    async def load_escalation_rules(self):
        """Load escalation rules from database"""
        try:
            async with self.db_pool.acquire() as conn:
                rules = await conn.fetch("SELECT * FROM escalation_rules WHERE enabled = true")
                
                for rule in rules:
                    tenant_id = rule['tenant_id']
                    if tenant_id not in self.escalation_rules:
                        self.escalation_rules[tenant_id] = []
                    
                    escalation_rule = EscalationRule(
                        id=rule['id'],
                        name=rule['name'],
                        tenant_id=tenant_id,
                        conditions=json.loads(rule['conditions']),
                        actions=json.loads(rule['actions']),
                        time_threshold_minutes=rule['time_threshold_minutes'],
                        severity_levels=[IncidentSeverity(s) for s in json.loads(rule['severity_levels'])],
                        enabled=rule['enabled'],
                        priority=rule['priority']
                    )
                    
                    self.escalation_rules[tenant_id].append(escalation_rule)
                
                # Sort by priority
                for tenant_rules in self.escalation_rules.values():
                    tenant_rules.sort(key=lambda x: x.priority)
                
        except Exception as e:
            logger.error(f"Error loading escalation rules: {e}")
    
    async def load_workflow_definitions(self):
        """Load workflow definitions from database"""
        try:
            async with self.db_pool.acquire() as conn:
                workflows = await conn.fetch("SELECT * FROM workflow_definitions WHERE enabled = true")
                
                for workflow in workflows:
                    workflow_id = workflow['id']
                    steps = await conn.fetch("""
                        SELECT * FROM workflow_steps 
                        WHERE workflow_id = $1 
                        ORDER BY step_order
                    """, workflow_id)
                    
                    workflow_steps = []
                    for step in steps:
                        workflow_step = WorkflowStep(
                            id=step['id'],
                            name=step['name'],
                            action=WorkflowAction(step['action']),
                            parameters=json.loads(step['parameters']),
                            conditions=json.loads(step['conditions']),
                            timeout_minutes=step['timeout_minutes'],
                            retry_count=step['retry_count'],
                            on_success=step['on_success'],
                            on_failure=step['on_failure']
                        )
                        workflow_steps.append(workflow_step)
                    
                    self.workflow_definitions[workflow_id] = {
                        'name': workflow['name'],
                        'tenant_id': workflow['tenant_id'],
                        'trigger_conditions': json.loads(workflow['trigger_conditions']),
                        'steps': workflow_steps
                    }
                
        except Exception as e:
            logger.error(f"Error loading workflow definitions: {e}")
    
    async def train_ml_models(self):
        """Train ML models with historical data"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get training data
                training_data = await conn.fetch("""
                    SELECT ni.*, 
                           EXTRACT(EPOCH FROM (COALESCE(ni.resolution_time, NOW()) - ni.created_at))/3600 as resolution_hours
                    FROM noc_incidents ni
                    WHERE ni.created_at > NOW() - INTERVAL '90 days'
                    AND ni.metadata IS NOT NULL
                """)
                
                if len(training_data) < 100:
                    logger.warning("Insufficient training data for ML models")
                    return
                
                features_list = []
                severity_labels = []
                
                for incident_row in training_data:
                    # Create incident object
                    incident = NOCIncident(
                        id=incident_row['id'],
                        tenant_id=incident_row['tenant_id'],
                        site_id=incident_row['site_id'],
                        title=incident_row['title'],
                        description=incident_row['description'],
                        severity=IncidentSeverity(incident_row['severity']),
                        status=IncidentStatus(incident_row['status']),
                        category=incident_row['category'],
                        source_system=incident_row['source_system'],
                        created_at=incident_row['created_at'],
                        updated_at=incident_row['updated_at'],
                        tags=json.loads(incident_row['tags']) if incident_row['tags'] else [],
                        metadata=json.loads(incident_row['metadata']) if incident_row['metadata'] else {}
                    )
                    
                    features = await self.extract_incident_features(incident)
                    if features:
                        features_list.append(features)
                        severity_mapping = {
                            IncidentSeverity.LOW: 0, IncidentSeverity.MEDIUM: 1,
                            IncidentSeverity.HIGH: 2, IncidentSeverity.CRITICAL: 3
                        }
                        severity_labels.append(severity_mapping[incident.severity])
                
                if len(features_list) > 50:
                    # Prepare training data
                    X = np.array(features_list)
                    y = np.array(severity_labels)
                    
                    # Fit scaler and transform features
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # Train severity predictor
                    self.severity_predictor.fit(X_scaled, y)
                    
                    logger.info(f"Trained ML models with {len(features_list)} samples")
                
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
    
    async def run(self):
        """Main run loop for NOC orchestrator"""
        try:
            await self.initialize()
            
            # Start monitoring tasks
            tasks = [
                asyncio.create_task(self.monitor_escalations()),
                asyncio.create_task(self.monitor_sla_breaches()),
                asyncio.create_task(self.process_incident_queue()),
            ]
            
            logger.info("NOC Orchestrator started successfully")
            
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            logger.info("NOC Orchestrator stopped by user")
        except Exception as e:
            logger.error(f"NOC Orchestrator error: {e}")
        finally:
            if self.db_pool:
                await self.db_pool.close()
            if self.redis_client:
                await self.redis_client.close()

# Global state
shutdown_event = asyncio.Event()
noc_orchestrator = None

async def message_handler(topic, key, value, headers, timestamp, partition, offset):
    """Handle incoming Kafka messages"""
    global noc_orchestrator
    
    logger.info(f"Received message from {topic}: {key}")
    
    try:
        if topic == KAFKA_TOPICS['alerts']:
            # Process alerts for orchestration
            alert_type = value.get('alert_type')
            site_id = value.get('site_id')
            severity = value.get('severity')
            
            if alert_type and site_id and noc_orchestrator:
                # Trigger orchestration for this alert
                asyncio.create_task(
                    noc_orchestrator.orchestrate_alert_response(value)
                )
                
        elif topic == KAFKA_TOPICS['events']:
            # Process events for orchestration
            event_type = value.get('event_type')
            
            if event_type in ['incident_created', 'optimization_requested'] and noc_orchestrator:
                # Orchestrate response to events
                asyncio.create_task(
                    noc_orchestrator.orchestrate_event_response(value)
                )
                
        elif topic == KAFKA_TOPICS['kpi_calculations']:
            # Process KPI results for operational decisions
            site_id = value.get('site_id')
            kpi_name = value.get('kpi_name')
            kpi_value = value.get('value')
            
            if site_id and kpi_name and kpi_value is not None and noc_orchestrator:
                # Check if KPI triggers operational actions
                asyncio.create_task(
                    noc_orchestrator.evaluate_kpi_for_actions(site_id, kpi_name, kpi_value)
                )
                
    except Exception as e:
        logger.error(f"Error handling message: {e}")

async def create_health_server():
    """Create health check server"""
    from aiohttp import web
    
    async def health_handler(request):
        return web.json_response({
            "status": "healthy",
            "worker": "noc_orchestrator",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    app = web.Application()
    app.router.add_get('/health', health_handler)
    
    return app

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point"""
    global noc_orchestrator, health_server
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting Towerco NOC Orchestrator...")
    
    try:
        # Initialize core services
        await init_monitoring()
        await init_db()
        await init_redis()
        await init_kafka()
        
        # Setup signal handlers
        setup_signal_handlers()
        
        # Start health server
        health_app = await create_health_server()
        health_runner = web.AppRunner(health_app)
        await health_runner.setup()
        site = web.TCPSite(health_runner, '0.0.0.0', 8007)
        await site.start()
        logger.info("Health server started on port 8007")
        
        # Initialize NOC Orchestrator
        noc_orchestrator = NOCOrchestrator()
        await noc_orchestrator.initialize()
        
        # Start Kafka consumers
        await start_consumer(
            'noc_orchestrator',
            [KAFKA_TOPICS['alerts'], KAFKA_TOPICS['events'], KAFKA_TOPICS['kpi_calculations']],
            'noc_orchestrator_group',
            message_handler
        )
        
        logger.info("NOC Orchestrator started successfully")
        
        # Main loop
        while not shutdown_event.is_set():
            await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("NOC Orchestrator stopping...")
    except Exception as e:
        logger.error(f"Fatal error in NOC Orchestrator: {e}")
        raise
    
    finally:
        logger.info("Shutting down NOC Orchestrator...")
        
        # Cleanup health server
        if health_server:
            await health_runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())