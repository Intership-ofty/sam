"""
Root Cause Analysis Engine - Automated RCA for Telecom Infrastructure
Advanced AI-powered root cause analysis with causal inference and ML
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict, deque

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb

import signal
from aiohttp import web

# Import core modules from backend
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from core.database import DatabaseManager, init_db
from core.cache import CacheManager, init_redis
from core.messaging import MessageProducer, init_kafka, start_consumer
from core.config import settings, KAFKA_TOPICS
from core.monitoring import init_monitoring

logger = logging.getLogger(__name__)

class CauseType(Enum):
    """Types of root causes"""
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_ISSUE = "software_issue"
    NETWORK_CONGESTION = "network_congestion"
    POWER_ISSUE = "power_issue"
    ENVIRONMENTAL = "environmental"
    CONFIGURATION_ERROR = "configuration_error"
    CAPACITY_LIMITATION = "capacity_limitation"
    EXTERNAL_DEPENDENCY = "external_dependency"
    HUMAN_ERROR = "human_error"
    SECURITY_INCIDENT = "security_incident"

class ConfidenceLevel(Enum):
    """Confidence levels for RCA results"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class CausalFactor:
    """Individual causal factor in RCA"""
    factor_name: str
    factor_type: str
    impact_score: float
    evidence: List[str]
    temporal_relationship: str  # precedes, concurrent, follows
    correlation_strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RCAResult:
    """Root cause analysis result"""
    incident_id: str
    primary_symptom: str
    site_id: str
    analysis_timestamp: datetime
    root_causes: List[CausalFactor]
    confidence_level: ConfidenceLevel
    confidence_score: float
    causal_chain: List[Dict[str, Any]]
    contributing_factors: List[str]
    recommendations: List[str]
    similar_incidents: List[str]
    analysis_duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class TelecomKnowledgeBase:
    """Knowledge base of telecom-specific causal relationships"""
    
    def __init__(self):
        self.causal_rules = self._build_causal_rules()
        self.symptom_patterns = self._build_symptom_patterns()
        self.dependency_graph = self._build_dependency_graph()
    
    def _build_causal_rules(self) -> Dict[str, List[Dict]]:
        """Build causal inference rules for telecom systems"""
        return {
            # Network performance issues
            "high_packet_loss": [
                {
                    "cause": "network_congestion",
                    "conditions": ["high_utilization > 85", "throughput_degraded"],
                    "confidence": 0.9
                },
                {
                    "cause": "hardware_failure", 
                    "conditions": ["interface_errors > 100", "physical_layer_issues"],
                    "confidence": 0.85
                },
                {
                    "cause": "configuration_error",
                    "conditions": ["recent_config_change", "routing_issues"],
                    "confidence": 0.75
                }
            ],
            
            # Power and energy issues
            "power_consumption_spike": [
                {
                    "cause": "cooling_failure",
                    "conditions": ["temperature_high > 45", "fan_failure"],
                    "confidence": 0.9
                },
                {
                    "cause": "hardware_malfunction",
                    "conditions": ["cpu_usage_normal", "unexpected_power_draw"],
                    "confidence": 0.8
                }
            ],
            
            # Availability issues
            "service_unavailable": [
                {
                    "cause": "hardware_failure",
                    "conditions": ["device_unreachable", "no_heartbeat"],
                    "confidence": 0.95
                },
                {
                    "cause": "software_crash",
                    "conditions": ["process_killed", "memory_exhaustion"],
                    "confidence": 0.85
                },
                {
                    "cause": "network_partition",
                    "conditions": ["partial_connectivity", "routing_loops"],
                    "confidence": 0.8
                }
            ],
            
            # Performance degradation
            "high_latency": [
                {
                    "cause": "network_congestion",
                    "conditions": ["bandwidth_saturated", "queue_depth_high"],
                    "confidence": 0.9
                },
                {
                    "cause": "routing_inefficiency", 
                    "conditions": ["suboptimal_path", "bgp_issues"],
                    "confidence": 0.8
                },
                {
                    "cause": "processing_delay",
                    "conditions": ["cpu_overload", "memory_pressure"],
                    "confidence": 0.75
                }
            ]
        }
    
    def _build_symptom_patterns(self) -> Dict[str, Dict]:
        """Build patterns of symptoms for different incident types"""
        return {
            "cascade_failure": {
                "pattern": ["initial_component_failure", "dependency_failures", "service_degradation"],
                "temporal_window": timedelta(minutes=30),
                "severity_progression": "increasing"
            },
            
            "capacity_exhaustion": {
                "pattern": ["gradual_performance_decline", "threshold_breach", "service_impact"],
                "temporal_window": timedelta(hours=2),
                "severity_progression": "gradual"
            },
            
            "configuration_propagation": {
                "pattern": ["config_deployment", "immediate_impact", "widespread_effect"],
                "temporal_window": timedelta(minutes=10),
                "severity_progression": "immediate"
            }
        }
    
    def _build_dependency_graph(self) -> nx.DiGraph:
        """Build dependency graph for telecom infrastructure"""
        G = nx.DiGraph()
        
        # Add nodes with types
        components = {
            # Physical layer
            "power_supply": {"type": "infrastructure", "criticality": "critical"},
            "cooling_system": {"type": "infrastructure", "criticality": "high"},
            "fiber_optic": {"type": "network", "criticality": "critical"},
            
            # Network equipment
            "base_station": {"type": "radio", "criticality": "critical"},
            "router": {"type": "network", "criticality": "critical"},
            "switch": {"type": "network", "criticality": "high"},
            "firewall": {"type": "security", "criticality": "high"},
            
            # Software systems
            "bss_oss": {"type": "software", "criticality": "high"},
            "monitoring_system": {"type": "software", "criticality": "medium"},
            "ems": {"type": "software", "criticality": "high"},
            
            # External dependencies
            "transmission_network": {"type": "external", "criticality": "critical"},
            "core_network": {"type": "external", "criticality": "critical"}
        }
        
        for comp_name, attrs in components.items():
            G.add_node(comp_name, **attrs)
        
        # Add dependencies (edges)
        dependencies = [
            ("power_supply", "cooling_system"),
            ("power_supply", "base_station"),
            ("power_supply", "router"),
            ("cooling_system", "base_station"),
            ("fiber_optic", "base_station"),
            ("fiber_optic", "router"),
            ("router", "base_station"),
            ("transmission_network", "router"),
            ("core_network", "router"),
            ("base_station", "bss_oss"),
            ("router", "monitoring_system"),
            ("switch", "router")
        ]
        
        G.add_edges_from(dependencies)
        return G

class RootCauseAnalysisEngine:
    """Main RCA engine with multiple analysis techniques"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.producer = MessageProducer()
        self.knowledge_base = TelecomKnowledgeBase()
        
        # ML Models for RCA
        self.pattern_classifier = None
        self.causal_model = None
        self.similarity_encoder = None
        
        # Configuration
        self.analysis_window = timedelta(hours=2)
        self.max_causal_depth = 5
        self.min_confidence_threshold = 0.6
        
        # Historical incident database
        self.incident_history = []
        
    async def initialize(self):
        """Initialize the RCA engine"""
        try:
            logger.info("Initializing Root Cause Analysis Engine...")
            
            # Load historical incident data
            await self.load_historical_incidents()
            
            # Train ML models
            await self.train_rca_models()
            
            # Setup event subscriptions
            await self.setup_event_subscriptions()
            
            logger.info("RCA Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RCA engine: {e}")
            raise
    
    async def analyze_incident(self, incident_data: Dict[str, Any]) -> RCAResult:
        """Main entry point for RCA analysis"""
        start_time = datetime.utcnow()
        
        try:
            incident_id = incident_data.get('incident_id', f"inc_{int(start_time.timestamp())}")
            primary_symptom = incident_data.get('primary_symptom', 'unknown')
            site_id = incident_data.get('site_id')
            
            logger.info(f"Starting RCA for incident {incident_id} at site {site_id}")
            
            # Phase 1: Data Collection and Context Building
            context_data = await self.collect_incident_context(incident_data)
            
            # Phase 2: Temporal Analysis
            temporal_analysis = await self.perform_temporal_analysis(incident_data, context_data)
            
            # Phase 3: Causal Inference
            causal_factors = await self.perform_causal_inference(
                incident_data, context_data, temporal_analysis
            )
            
            # Phase 4: Pattern Matching
            pattern_matches = await self.match_historical_patterns(incident_data, context_data)
            
            # Phase 5: ML-based Analysis
            ml_predictions = await self.perform_ml_analysis(incident_data, context_data)
            
            # Phase 6: Knowledge-based Analysis
            knowledge_analysis = await self.perform_knowledge_based_analysis(
                incident_data, context_data
            )
            
            # Phase 7: Dependency Analysis
            dependency_analysis = await self.analyze_dependencies(incident_data, context_data)
            
            # Phase 8: Synthesis and Ranking
            root_causes = self.synthesize_root_causes(
                causal_factors, pattern_matches, ml_predictions, 
                knowledge_analysis, dependency_analysis
            )
            
            # Phase 9: Generate Recommendations
            recommendations = await self.generate_recommendations(
                incident_data, root_causes, context_data
            )
            
            # Calculate confidence
            confidence_score, confidence_level = self.calculate_confidence(root_causes)
            
            # Build causal chain
            causal_chain = self.build_causal_chain(root_causes, context_data)
            
            # Find similar incidents
            similar_incidents = await self.find_similar_incidents(incident_data, context_data)
            
            analysis_duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Create RCA result
            rca_result = RCAResult(
                incident_id=incident_id,
                primary_symptom=primary_symptom,
                site_id=site_id,
                analysis_timestamp=datetime.utcnow(),
                root_causes=root_causes,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                causal_chain=causal_chain,
                contributing_factors=[f.factor_name for f in root_causes],
                recommendations=recommendations,
                similar_incidents=similar_incidents,
                analysis_duration=analysis_duration,
                metadata={
                    'context_data_points': len(context_data.get('metrics', [])),
                    'temporal_patterns_found': len(temporal_analysis.get('patterns', [])),
                    'ml_model_used': ml_predictions.get('model_type', 'none'),
                    'knowledge_rules_matched': len(knowledge_analysis.get('matched_rules', []))
                }
            )
            
            # Store results
            await self.store_rca_result(rca_result)
            
            # Send notifications
            await self.send_rca_notification(rca_result)
            
            logger.info(f"RCA completed for incident {incident_id} in {analysis_duration:.2f}s")
            return rca_result
            
        except Exception as e:
            logger.error(f"Error in RCA analysis: {e}")
            # Return basic result even on error
            return self.create_fallback_result(incident_data, start_time, str(e))
    
    async def collect_incident_context(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect contextual data around the incident"""
        try:
            site_id = incident_data.get('site_id')
            incident_time = datetime.fromisoformat(incident_data.get('timestamp', datetime.utcnow().isoformat()))
            
            # Define time window for context collection
            start_time = incident_time - self.analysis_window
            end_time = incident_time + timedelta(minutes=30)
            
            context = {
                'incident_time': incident_time,
                'time_window': {'start': start_time, 'end': end_time},
                'metrics': [],
                'events': [],
                'configuration_changes': [],
                'maintenance_activities': [],
                'external_events': []
            }
            
            # Collect metrics data
            metrics_query = """
            SELECT metric_name, metric_value, timestamp, unit, quality_score, metadata
            FROM network_metrics
            WHERE site_id = $1 
                AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp
            """
            
            metric_rows = await self.db.fetch_all(metrics_query, site_id, start_time, end_time)
            context['metrics'] = [dict(row) for row in metric_rows]
            
            # Collect events and alarms
            events_query = """
            SELECT event_type, severity, title, description, timestamp, 
                   source_system, acknowledged, metadata
            FROM events
            WHERE site_id = $1 
                AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp
            """
            
            event_rows = await self.db.fetch_all(events_query, site_id, start_time, end_time)
            context['events'] = [dict(row) for row in event_rows]
            
            # Collect configuration changes
            config_query = """
            SELECT config_type, configuration, timestamp, metadata
            FROM configuration_data
            WHERE site_id = $1 
                AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp
            """
            
            config_rows = await self.db.fetch_all(config_query, site_id, start_time, end_time)
            context['configuration_changes'] = [dict(row) for row in config_rows]
            
            # Collect maintenance activities
            maintenance_query = """
            SELECT ticket_id, ticket_type, title, status, created_at, resolved_at
            FROM itsm_tickets
            WHERE site_id = $1 
                AND (created_at >= $2 OR resolved_at >= $2)
                AND created_at <= $3
            ORDER BY created_at
            """
            
            maintenance_rows = await self.db.fetch_all(
                maintenance_query, site_id, start_time, end_time
            )
            context['maintenance_activities'] = [dict(row) for row in maintenance_rows]
            
            # Collect anomaly detections
            anomaly_query = """
            SELECT metric_name, anomaly_score, anomaly_type, detected_at,
                   value, confidence, contributing_factors, metadata
            FROM anomaly_detections
            WHERE site_id = $1 
                AND detected_at >= $2 AND detected_at <= $3
            ORDER BY detected_at
            """
            
            anomaly_rows = await self.db.fetch_all(anomaly_query, site_id, start_time, end_time)
            context['anomalies'] = [dict(row) for row in anomaly_rows]
            
            logger.info(f"Collected context: {len(context['metrics'])} metrics, "
                       f"{len(context['events'])} events, {len(context['anomalies'])} anomalies")
            
            return context
            
        except Exception as e:
            logger.error(f"Error collecting incident context: {e}")
            return {'error': str(e)}
    
    async def perform_temporal_analysis(self, incident_data: Dict, context_data: Dict) -> Dict[str, Any]:
        """Analyze temporal relationships between events and metrics"""
        try:
            incident_time = context_data['incident_time']
            
            # Group events by time proximity
            time_clusters = self.cluster_events_by_time(context_data['events'], incident_time)
            
            # Identify temporal patterns
            patterns = []
            
            # Look for precursor events (events before incident)
            precursors = self.find_precursor_events(context_data['events'], incident_time)
            if precursors:
                patterns.append({
                    'type': 'precursor_events',
                    'events': precursors,
                    'confidence': 0.8
                })
            
            # Look for cascading failures
            cascade_pattern = self.detect_cascade_pattern(context_data['events'], incident_time)
            if cascade_pattern:
                patterns.append({
                    'type': 'cascading_failure',
                    'pattern': cascade_pattern,
                    'confidence': 0.9
                })
            
            # Look for correlated metric changes
            metric_correlations = self.analyze_metric_correlations(
                context_data['metrics'], incident_time
            )
            if metric_correlations:
                patterns.append({
                    'type': 'metric_correlations',
                    'correlations': metric_correlations,
                    'confidence': 0.7
                })
            
            return {
                'time_clusters': time_clusters,
                'patterns': patterns,
                'precursors': precursors,
                'analysis_window': self.analysis_window.total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
            return {'error': str(e)}
    
    def cluster_events_by_time(self, events: List[Dict], incident_time: datetime, 
                              window_minutes: int = 5) -> List[List[Dict]]:
        """Cluster events by temporal proximity"""
        if not events:
            return []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x['timestamp'])
        
        clusters = []
        current_cluster = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            time_diff = (event['timestamp'] - current_cluster[-1]['timestamp']).total_seconds()
            
            if time_diff <= window_minutes * 60:
                current_cluster.append(event)
            else:
                clusters.append(current_cluster)
                current_cluster = [event]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def find_precursor_events(self, events: List[Dict], incident_time: datetime) -> List[Dict]:
        """Find events that occurred before the incident and might be precursors"""
        precursors = []
        
        for event in events:
            event_time = event['timestamp']
            time_diff = (incident_time - event_time).total_seconds()
            
            # Look for events 5 minutes to 2 hours before incident
            if 300 <= time_diff <= 7200:  # 5 minutes to 2 hours
                # Score based on severity and temporal proximity
                severity_score = {'critical': 1.0, 'major': 0.8, 'minor': 0.6, 'warning': 0.4}.get(
                    event.get('severity', '').lower(), 0.2
                )
                
                # Closer in time = higher score
                temporal_score = max(0.1, 1.0 - (time_diff / 7200))
                
                combined_score = (severity_score + temporal_score) / 2
                
                if combined_score > 0.5:
                    event['precursor_score'] = combined_score
                    precursors.append(event)
        
        return sorted(precursors, key=lambda x: x['precursor_score'], reverse=True)
    
    def detect_cascade_pattern(self, events: List[Dict], incident_time: datetime) -> Optional[Dict]:
        """Detect cascading failure patterns"""
        if len(events) < 3:
            return None
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda x: x['timestamp'])
        
        # Look for increasing severity pattern
        severity_mapping = {'warning': 1, 'minor': 2, 'major': 3, 'critical': 4}
        
        cascade_events = []
        last_severity = 0
        
        for event in sorted_events:
            event_severity = severity_mapping.get(event.get('severity', '').lower(), 0)
            
            if event_severity >= last_severity:
                cascade_events.append(event)
                last_severity = event_severity
            elif len(cascade_events) >= 3:
                # Found a cascade pattern
                return {
                    'events': cascade_events,
                    'duration_seconds': (cascade_events[-1]['timestamp'] - 
                                       cascade_events[0]['timestamp']).total_seconds(),
                    'severity_progression': [severity_mapping.get(e.get('severity', '').lower(), 0) 
                                          for e in cascade_events]
                }
        
        # Check if we found a cascade pattern
        if len(cascade_events) >= 3:
            return {
                'events': cascade_events,
                'duration_seconds': (cascade_events[-1]['timestamp'] - 
                                   cascade_events[0]['timestamp']).total_seconds(),
                'severity_progression': [severity_mapping.get(e.get('severity', '').lower(), 0) 
                                      for e in cascade_events]
            }
        
        return None
    
    def analyze_metric_correlations(self, metrics: List[Dict], 
                                  incident_time: datetime) -> List[Dict]:
        """Analyze correlations between metrics around incident time"""
        if not metrics:
            return []
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(metrics)
        if df.empty:
            return []
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Focus on metrics within 30 minutes of incident
        time_window = timedelta(minutes=30)
        mask = (df['timestamp'] >= incident_time - time_window) & \
               (df['timestamp'] <= incident_time + time_window)
        
        relevant_metrics = df[mask]
        
        if len(relevant_metrics) < 10:
            return []
        
        correlations = []
        
        # Group by metric name and calculate correlations
        metric_groups = relevant_metrics.groupby('metric_name')
        metric_series = {}
        
        for name, group in metric_groups:
            if len(group) >= 5:  # Need at least 5 data points
                group = group.sort_values('timestamp')
                metric_series[name] = group['metric_value'].values
        
        # Calculate pairwise correlations
        metric_names = list(metric_series.keys())
        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                name1, name2 = metric_names[i], metric_names[j]
                series1, series2 = metric_series[name1], metric_series[name2]
                
                # Align series length
                min_len = min(len(series1), len(series2))
                if min_len >= 5:
                    corr_coef = np.corrcoef(series1[:min_len], series2[:min_len])[0, 1]
                    
                    if not np.isnan(corr_coef) and abs(corr_coef) > 0.7:
                        correlations.append({
                            'metric1': name1,
                            'metric2': name2,
                            'correlation': corr_coef,
                            'strength': 'strong' if abs(corr_coef) > 0.8 else 'moderate'
                        })
        
        return correlations
    
    async def perform_causal_inference(self, incident_data: Dict, context_data: Dict,
                                     temporal_analysis: Dict) -> List[CausalFactor]:
        """Perform causal inference analysis"""
        try:
            causal_factors = []
            
            # Use temporal patterns to infer causality
            patterns = temporal_analysis.get('patterns', [])
            
            for pattern in patterns:
                if pattern['type'] == 'precursor_events':
                    for precursor in pattern['events']:
                        factor = CausalFactor(
                            factor_name=precursor['title'],
                            factor_type='precursor_event',
                            impact_score=precursor.get('precursor_score', 0.5),
                            evidence=[f"Event occurred {precursor['timestamp']} before incident"],
                            temporal_relationship='precedes',
                            correlation_strength=precursor.get('precursor_score', 0.5),
                            metadata=precursor
                        )
                        causal_factors.append(factor)
                
                elif pattern['type'] == 'cascading_failure':
                    cascade = pattern['pattern']
                    factor = CausalFactor(
                        factor_name='cascading_failure',
                        factor_type='system_failure',
                        impact_score=0.9,
                        evidence=[f"Cascade of {len(cascade['events'])} failures detected"],
                        temporal_relationship='precedes',
                        correlation_strength=0.9,
                        metadata=cascade
                    )
                    causal_factors.append(factor)
            
            # Analyze anomalies as causal factors
            anomalies = context_data.get('anomalies', [])
            for anomaly in anomalies:
                if anomaly['confidence'] > 0.7:
                    factor = CausalFactor(
                        factor_name=f"anomaly_{anomaly['metric_name']}",
                        factor_type='anomaly',
                        impact_score=anomaly['anomaly_score'],
                        evidence=anomaly['contributing_factors'],
                        temporal_relationship='precedes' if anomaly['detected_at'] < context_data['incident_time'] else 'concurrent',
                        correlation_strength=anomaly['confidence'],
                        metadata=anomaly
                    )
                    causal_factors.append(factor)
            
            # Analyze configuration changes
            config_changes = context_data.get('configuration_changes', [])
            for config_change in config_changes:
                # Configuration changes shortly before incident are highly suspicious
                time_diff = (context_data['incident_time'] - config_change['timestamp']).total_seconds()
                if 0 < time_diff <= 3600:  # Within 1 hour before incident
                    impact_score = max(0.3, 1.0 - (time_diff / 3600))
                    
                    factor = CausalFactor(
                        factor_name='configuration_change',
                        factor_type='configuration',
                        impact_score=impact_score,
                        evidence=[f"Configuration change at {config_change['timestamp']}"],
                        temporal_relationship='precedes',
                        correlation_strength=impact_score,
                        metadata=config_change
                    )
                    causal_factors.append(factor)
            
            return sorted(causal_factors, key=lambda x: x.impact_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error in causal inference: {e}")
            return []
    
    async def perform_ml_analysis(self, incident_data: Dict, context_data: Dict) -> Dict[str, Any]:
        """Perform ML-based root cause analysis"""
        try:
            if self.pattern_classifier is None:
                return {'error': 'ML models not available'}
            
            # Prepare features from context data
            features = self.prepare_ml_features(context_data)
            
            if features is None:
                return {'error': 'Insufficient data for ML analysis'}
            
            # Predict using trained models
            predictions = {}
            
            # Pattern classification
            if hasattr(self.pattern_classifier, 'predict_proba'):
                pattern_probs = self.pattern_classifier.predict_proba([features])[0]
                pattern_classes = self.pattern_classifier.classes_
                
                predictions['pattern_classification'] = {
                    class_name: prob for class_name, prob in zip(pattern_classes, pattern_probs)
                }
            
            # Get top predictions
            top_predictions = sorted(
                predictions.get('pattern_classification', {}).items(),
                key=lambda x: x[1], reverse=True
            )[:3]
            
            return {
                'model_type': 'ensemble',
                'predictions': predictions,
                'top_predictions': top_predictions,
                'confidence': max([prob for _, prob in top_predictions]) if top_predictions else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in ML analysis: {e}")
            return {'error': str(e)}
    
    def prepare_ml_features(self, context_data: Dict) -> Optional[List[float]]:
        """Prepare features for ML models"""
        try:
            features = []
            
            # Metrics-based features
            metrics = context_data.get('metrics', [])
            if not metrics:
                return None
            
            # Statistical features from metrics
            metric_df = pd.DataFrame(metrics)
            
            if 'metric_value' in metric_df.columns:
                features.extend([
                    metric_df['metric_value'].mean(),
                    metric_df['metric_value'].std(),
                    metric_df['metric_value'].min(),
                    metric_df['metric_value'].max(),
                    metric_df['metric_value'].median()
                ])
            else:
                features.extend([0.0] * 5)
            
            # Event-based features
            events = context_data.get('events', [])
            features.extend([
                len(events),
                len([e for e in events if e.get('severity') == 'critical']),
                len([e for e in events if e.get('severity') == 'major']),
                len([e for e in events if e.get('acknowledged', False)])
            ])
            
            # Anomaly-based features
            anomalies = context_data.get('anomalies', [])
            features.extend([
                len(anomalies),
                np.mean([a.get('confidence', 0) for a in anomalies]) if anomalies else 0,
                np.max([a.get('anomaly_score', 0) for a in anomalies]) if anomalies else 0
            ])
            
            # Configuration change features
            config_changes = context_data.get('configuration_changes', [])
            features.extend([
                len(config_changes),
                1 if config_changes else 0  # Binary: any config changes
            ])
            
            # Temporal features
            incident_time = context_data.get('incident_time', datetime.utcnow())
            features.extend([
                incident_time.hour,
                incident_time.weekday(),
                1 if incident_time.weekday() >= 5 else 0  # Weekend flag
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return None
    
    async def perform_knowledge_based_analysis(self, incident_data: Dict, 
                                             context_data: Dict) -> Dict[str, Any]:
        """Perform knowledge-based analysis using telecom domain rules"""
        try:
            primary_symptom = incident_data.get('primary_symptom', 'unknown').lower()
            matched_rules = []
            
            # Check if primary symptom matches known patterns
            if primary_symptom in self.knowledge_base.causal_rules:
                rules = self.knowledge_base.causal_rules[primary_symptom]
                
                for rule in rules:
                    conditions_met = self.evaluate_rule_conditions(rule['conditions'], context_data)
                    if conditions_met['score'] > 0.5:
                        matched_rules.append({
                            'cause': rule['cause'],
                            'base_confidence': rule['confidence'],
                            'conditions_score': conditions_met['score'],
                            'matched_conditions': conditions_met['matched'],
                            'final_confidence': rule['confidence'] * conditions_met['score']
                        })
            
            # Check for symptom patterns
            pattern_matches = []
            for pattern_name, pattern_def in self.knowledge_base.symptom_patterns.items():
                match_score = self.match_symptom_pattern(pattern_def, context_data)
                if match_score > 0.6:
                    pattern_matches.append({
                        'pattern': pattern_name,
                        'score': match_score,
                        'definition': pattern_def
                    })
            
            return {
                'matched_rules': matched_rules,
                'pattern_matches': pattern_matches,
                'knowledge_confidence': np.mean([r['final_confidence'] for r in matched_rules]) if matched_rules else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in knowledge-based analysis: {e}")
            return {'error': str(e)}
    
    def evaluate_rule_conditions(self, conditions: List[str], context_data: Dict) -> Dict[str, Any]:
        """Evaluate rule conditions against context data"""
        matched_conditions = []
        total_score = 0
        
        for condition in conditions:
            score = self.evaluate_single_condition(condition, context_data)
            if score > 0:
                matched_conditions.append(condition)
                total_score += score
        
        return {
            'score': total_score / len(conditions) if conditions else 0,
            'matched': matched_conditions,
            'total_conditions': len(conditions)
        }
    
    def evaluate_single_condition(self, condition: str, context_data: Dict) -> float:
        """Evaluate a single condition string"""
        try:
            # Parse condition (e.g., "high_utilization > 85")
            if '>' in condition:
                metric_condition, threshold = condition.split('>')
                metric_condition = metric_condition.strip()
                threshold = float(threshold.strip())
                
                # Find matching metrics
                metrics = context_data.get('metrics', [])
                matching_metrics = [m for m in metrics if metric_condition in m.get('metric_name', '')]
                
                if matching_metrics:
                    max_value = max(m['metric_value'] for m in matching_metrics)
                    return 1.0 if max_value > threshold else 0.5
            
            elif condition in ['recent_config_change']:
                config_changes = context_data.get('configuration_changes', [])
                return 1.0 if config_changes else 0.0
            
            elif condition in ['interface_errors', 'physical_layer_issues']:
                events = context_data.get('events', [])
                relevant_events = [e for e in events if condition.replace('_', ' ') in e.get('title', '').lower()]
                return 1.0 if relevant_events else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error evaluating condition '{condition}': {e}")
            return 0.0
    
    def match_symptom_pattern(self, pattern_def: Dict, context_data: Dict) -> float:
        """Match symptom patterns against context data"""
        try:
            pattern_sequence = pattern_def['pattern']
            temporal_window = pattern_def['temporal_window']
            
            events = context_data.get('events', [])
            if not events:
                return 0.0
            
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda x: x['timestamp'])
            
            # Look for pattern sequence
            pattern_matches = 0
            for i, expected_event in enumerate(pattern_sequence):
                found_match = False
                for event in sorted_events:
                    if expected_event.replace('_', ' ') in event.get('title', '').lower():
                        found_match = True
                        pattern_matches += 1
                        break
                
                if not found_match:
                    break
            
            # Calculate match score
            match_score = pattern_matches / len(pattern_sequence)
            
            # Bonus for temporal consistency
            if match_score > 0.5:
                first_event_time = sorted_events[0]['timestamp']
                last_event_time = sorted_events[-1]['timestamp']
                actual_duration = last_event_time - first_event_time
                
                if actual_duration <= temporal_window:
                    match_score += 0.2  # Temporal bonus
            
            return min(match_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error matching symptom pattern: {e}")
            return 0.0
    
    async def analyze_dependencies(self, incident_data: Dict, context_data: Dict) -> Dict[str, Any]:
        """Analyze component dependencies for impact assessment"""
        try:
            site_id = incident_data.get('site_id')
            
            # Get affected components from events
            affected_components = set()
            events = context_data.get('events', [])
            
            for event in events:
                # Extract component information from event metadata or title
                component = self.extract_component_from_event(event)
                if component:
                    affected_components.add(component)
            
            if not affected_components:
                return {'error': 'No affected components identified'}
            
            # Analyze dependency graph
            dependency_graph = self.knowledge_base.dependency_graph
            
            # Find upstream dependencies (what could have caused this)
            potential_causes = set()
            for component in affected_components:
                if component in dependency_graph:
                    predecessors = list(dependency_graph.predecessors(component))
                    potential_causes.update(predecessors)
            
            # Find downstream impacts (what this could affect)
            potential_impacts = set()
            for component in affected_components:
                if component in dependency_graph:
                    successors = list(dependency_graph.successors(component))
                    potential_impacts.update(successors)
            
            # Calculate criticality scores
            criticality_scores = {}
            for component in affected_components:
                if component in dependency_graph:
                    node_data = dependency_graph.nodes[component]
                    criticality = node_data.get('criticality', 'medium')
                    criticality_scores[component] = {
                        'critical': 1.0, 'high': 0.8, 'medium': 0.6, 'low': 0.4
                    }.get(criticality, 0.5)
            
            return {
                'affected_components': list(affected_components),
                'potential_causes': list(potential_causes),
                'potential_impacts': list(potential_impacts),
                'criticality_scores': criticality_scores,
                'dependency_analysis': True
            }
            
        except Exception as e:
            logger.error(f"Error in dependency analysis: {e}")
            return {'error': str(e)}
    
    def extract_component_from_event(self, event: Dict) -> Optional[str]:
        """Extract component name from event data"""
        title = event.get('title', '').lower()
        
        # Component keywords mapping
        component_keywords = {
            'base_station': ['base station', 'bts', 'cell', 'radio'],
            'router': ['router', 'routing', 'bgp'],
            'switch': ['switch', 'switching', 'vlan'],
            'power_supply': ['power', 'ups', 'battery'],
            'cooling_system': ['cooling', 'temperature', 'fan', 'hvac'],
            'fiber_optic': ['fiber', 'optical', 'cable'],
            'firewall': ['firewall', 'security', 'blocked']
        }
        
        for component, keywords in component_keywords.items():
            if any(keyword in title for keyword in keywords):
                return component
        
        return None
    
    def synthesize_root_causes(self, *analysis_results) -> List[CausalFactor]:
        """Synthesize results from all analysis methods"""
        all_factors = []
        
        # Collect all causal factors
        for result in analysis_results:
            if isinstance(result, list):
                all_factors.extend(result)
            elif isinstance(result, dict):
                # Convert dict results to CausalFactor objects
                if 'matched_rules' in result:
                    for rule in result['matched_rules']:
                        factor = CausalFactor(
                            factor_name=rule['cause'],
                            factor_type='knowledge_rule',
                            impact_score=rule['final_confidence'],
                            evidence=[f"Matched rule conditions with score {rule['conditions_score']:.2f}"],
                            temporal_relationship='inferred',
                            correlation_strength=rule['final_confidence'],
                            metadata=rule
                        )
                        all_factors.append(factor)
                
                if 'potential_causes' in result:
                    for cause in result['potential_causes']:
                        factor = CausalFactor(
                            factor_name=f"dependency_{cause}",
                            factor_type='dependency',
                            impact_score=result.get('criticality_scores', {}).get(cause, 0.5),
                            evidence=[f"Dependency analysis identified {cause} as potential cause"],
                            temporal_relationship='upstream',
                            correlation_strength=0.7,
                            metadata={'component': cause}
                        )
                        all_factors.append(factor)
        
        # Remove duplicates and rank by impact score
        unique_factors = {}
        for factor in all_factors:
            key = f"{factor.factor_name}_{factor.factor_type}"
            if key not in unique_factors or unique_factors[key].impact_score < factor.impact_score:
                unique_factors[key] = factor
        
        # Sort by impact score and return top factors
        ranked_factors = sorted(unique_factors.values(), 
                              key=lambda x: x.impact_score, reverse=True)
        
        return ranked_factors[:10]  # Return top 10 factors
    
    def calculate_confidence(self, root_causes: List[CausalFactor]) -> Tuple[float, ConfidenceLevel]:
        """Calculate overall confidence in RCA results"""
        if not root_causes:
            return 0.0, ConfidenceLevel.LOW
        
        # Weighted confidence based on top factors
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Decreasing weights for top 5 factors
        
        weighted_score = 0
        total_weight = 0
        
        for i, factor in enumerate(root_causes[:5]):
            weight = weights[i] if i < len(weights) else 0.1
            weighted_score += factor.impact_score * weight
            total_weight += weight
        
        confidence_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine confidence level
        if confidence_score >= 0.9:
            confidence_level = ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.75:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW
        
        return confidence_score, confidence_level
    
    def build_causal_chain(self, root_causes: List[CausalFactor], context_data: Dict) -> List[Dict[str, Any]]:
        """Build causal chain from root causes"""
        causal_chain = []
        
        # Sort causes by temporal relationship and impact
        temporal_order = {'precedes': 1, 'upstream': 2, 'concurrent': 3, 'inferred': 4}
        sorted_causes = sorted(root_causes[:5], 
                             key=lambda x: (temporal_order.get(x.temporal_relationship, 5), 
                                          -x.impact_score))
        
        for i, cause in enumerate(sorted_causes):
            chain_element = {
                'step': i + 1,
                'cause': cause.factor_name,
                'type': cause.factor_type,
                'impact_score': cause.impact_score,
                'confidence': cause.correlation_strength,
                'evidence': cause.evidence,
                'temporal_relationship': cause.temporal_relationship
            }
            causal_chain.append(chain_element)
        
        return causal_chain
    
    async def generate_recommendations(self, incident_data: Dict, root_causes: List[CausalFactor],
                                     context_data: Dict) -> List[str]:
        """Generate actionable recommendations based on root causes"""
        recommendations = []
        
        for cause in root_causes[:3]:  # Top 3 causes
            cause_type = cause.factor_type
            
            if cause_type == 'anomaly':
                recommendations.append(
                    f"Investigate {cause.factor_name} anomaly - check metric thresholds and alert rules"
                )
            
            elif cause_type == 'precursor_event':
                recommendations.append(
                    f"Review event handling procedures for {cause.factor_name}"
                )
            
            elif cause_type == 'configuration':
                recommendations.append(
                    "Review recent configuration changes and implement rollback if necessary"
                )
            
            elif cause_type == 'system_failure':
                recommendations.append(
                    "Implement circuit breakers to prevent cascading failures"
                )
            
            elif cause_type == 'dependency':
                component = cause.metadata.get('component', 'unknown')
                recommendations.append(
                    f"Check health and status of {component} component"
                )
            
            elif cause_type == 'knowledge_rule':
                recommendations.append(
                    f"Apply standard remediation procedures for {cause.factor_name}"
                )
        
        # Generic recommendations
        recommendations.extend([
            "Monitor system metrics closely for the next 24 hours",
            "Update incident documentation and lessons learned",
            "Consider implementing additional monitoring for early detection"
        ])
        
        return recommendations
    
    async def find_similar_incidents(self, incident_data: Dict, context_data: Dict) -> List[str]:
        """Find historically similar incidents"""
        try:
            # This would use ML similarity matching
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Error finding similar incidents: {e}")
            return []
    
    async def store_rca_result(self, rca_result: RCAResult):
        """Store RCA result in database"""
        try:
            query = """
            INSERT INTO rca_results (
                incident_id, primary_symptom, site_id, analysis_timestamp,
                root_causes, confidence_level, confidence_score, causal_chain,
                contributing_factors, recommendations, similar_incidents,
                analysis_duration, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """
            
            await self.db.execute(
                query,
                rca_result.incident_id,
                rca_result.primary_symptom,
                rca_result.site_id,
                rca_result.analysis_timestamp,
                json.dumps([{
                    'factor_name': f.factor_name,
                    'factor_type': f.factor_type,
                    'impact_score': f.impact_score,
                    'evidence': f.evidence,
                    'correlation_strength': f.correlation_strength
                } for f in rca_result.root_causes]),
                rca_result.confidence_level.value,
                rca_result.confidence_score,
                json.dumps(rca_result.causal_chain),
                rca_result.contributing_factors,
                rca_result.recommendations,
                rca_result.similar_incidents,
                rca_result.analysis_duration,
                json.dumps(rca_result.metadata)
            )
            
            logger.info(f"Stored RCA result for incident {rca_result.incident_id}")
            
        except Exception as e:
            logger.error(f"Error storing RCA result: {e}")
    
    async def send_rca_notification(self, rca_result: RCAResult):
        """Send RCA result notification"""
        try:
            notification = {
                'type': 'rca_completed',
                'timestamp': rca_result.analysis_timestamp.isoformat(),
                'incident_id': rca_result.incident_id,
                'site_id': rca_result.site_id,
                'confidence_level': rca_result.confidence_level.value,
                'confidence_score': rca_result.confidence_score,
                'top_causes': [f.factor_name for f in rca_result.root_causes[:3]],
                'recommendations': rca_result.recommendations[:3],
                'analysis_duration': rca_result.analysis_duration
            }
            
            await self.producer.send_message('aiops.rca_results', notification)
            logger.info(f"Sent RCA notification for incident {rca_result.incident_id}")
            
        except Exception as e:
            logger.error(f"Error sending RCA notification: {e}")
    
    def create_fallback_result(self, incident_data: Dict, start_time: datetime, error: str) -> RCAResult:
        """Create a fallback RCA result when analysis fails"""
        return RCAResult(
            incident_id=incident_data.get('incident_id', 'unknown'),
            primary_symptom=incident_data.get('primary_symptom', 'unknown'),
            site_id=incident_data.get('site_id', 'unknown'),
            analysis_timestamp=datetime.utcnow(),
            root_causes=[],
            confidence_level=ConfidenceLevel.LOW,
            confidence_score=0.0,
            causal_chain=[],
            contributing_factors=[],
            recommendations=['Manual investigation required', 'Check system logs', 'Contact support team'],
            similar_incidents=[],
            analysis_duration=(datetime.utcnow() - start_time).total_seconds(),
            metadata={'error': error, 'fallback': True}
        )
    
    async def load_historical_incidents(self):
        """Load historical incident data for training"""
        try:
            query = """
            SELECT incident_id, primary_symptom, site_id, root_causes, confidence_score
            FROM rca_results
            WHERE analysis_timestamp >= NOW() - INTERVAL '90 days'
            ORDER BY analysis_timestamp DESC
            """
            
            rows = await self.db.fetch_all(query)
            self.incident_history = [dict(row) for row in rows]
            
            logger.info(f"Loaded {len(self.incident_history)} historical incidents")
            
        except Exception as e:
            logger.error(f"Error loading historical incidents: {e}")
    
    async def train_rca_models(self):
        """Train ML models for RCA"""
        try:
            if len(self.incident_history) < 10:
                logger.warning("Insufficient historical data for ML training")
                return
            
            # This would implement actual ML model training
            # For now, create placeholder models
            self.pattern_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            
            logger.info("RCA ML models training completed")
            
        except Exception as e:
            logger.error(f"Error training RCA models: {e}")
    
    async def setup_event_subscriptions(self):
        """Setup subscriptions to relevant event streams"""
        # This would integrate with message queue to receive events
        pass


# Main worker function
# Global state
shutdown_event = asyncio.Event()
rca_engine = None

async def message_handler(topic, key, value, headers, timestamp, partition, offset):
    """Handle incoming Kafka messages"""
    global rca_engine
    
    logger.info(f"Received message from {topic}: {key}")
    
    try:
        if topic == KAFKA_TOPICS['alerts']:
            # Process alerts for root cause analysis
            alert_type = value.get('alert_type')
            site_id = value.get('site_id')
            severity = value.get('severity')
            
            if alert_type and site_id and rca_engine:
                # Trigger RCA for this alert
                asyncio.create_task(
                    rca_engine.analyze_alert_root_cause(value)
                )
                
        elif topic == KAFKA_TOPICS['events']:
            # Process events for RCA
            event_type = value.get('event_type')
            
            if event_type in ['incident_created', 'rca_requested'] and rca_engine:
                # Trigger RCA analysis
                asyncio.create_task(
                    rca_engine.analyze_event_root_cause(value)
                )
                
        elif topic == KAFKA_TOPICS['aiops_predictions']:
            # Process RCA requests from AIOps endpoints
            prediction_type = value.get('prediction_type')
            
            if prediction_type == 'root_cause_analysis' and rca_engine:
                # Handle direct RCA requests
                input_data = value.get('input_data', {})
                asyncio.create_task(
                    rca_engine.process_rca_request(input_data)
                )
                
    except Exception as e:
        logger.error(f"Error handling message: {e}")

async def create_health_server():
    """Create health check server"""
    from aiohttp import web
    
    async def health_handler(request):
        return web.json_response({
            "status": "healthy",
            "worker": "root_cause_analyzer",
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
    """Main RCA worker"""
    global rca_engine, health_server
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting Towerco Root Cause Analysis Engine...")
    
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
        site = web.TCPSite(health_runner, '0.0.0.0', 8006)
        await site.start()
        logger.info("Health server started on port 8006")
        
        # Initialize Root Cause Analysis Engine
        rca_engine = RootCauseAnalysisEngine()
        await rca_engine.initialize()
        
        # Start Kafka consumers
        await start_consumer(
            'rca_analyzer',
            [KAFKA_TOPICS['alerts'], KAFKA_TOPICS['events'], KAFKA_TOPICS['aiops_predictions']],
            'rca_analyzer_group',
            message_handler
        )
        
        logger.info("Root Cause Analysis Engine started successfully")
        
        # Main loop
        while not shutdown_event.is_set():
            await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("RCA worker stopping...")
    except Exception as e:
        logger.error(f"Fatal error in RCA worker: {e}")
        raise
    
    finally:
        logger.info("Shutting down Root Cause Analysis Engine...")
        
        # Cleanup health server
        if health_server:
            await health_runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())