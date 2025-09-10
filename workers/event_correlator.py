"""
Event Correlation Engine - Advanced event correlation and pattern detection
Multi-dimensional correlation analysis for telecom infrastructure events
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
from collections import defaultdict, deque
import networkx as nx
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import signal
from aiohttp import web

# Import core modules from backend
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from core.database import DatabaseManager, init_db
from core.cache import CacheManager, init_redis
from core.messaging import MessageProducer, MessageConsumer, init_kafka, start_consumer
from core.config import settings, KAFKA_TOPICS
from core.monitoring import init_monitoring

logger = logging.getLogger(__name__)

class CorrelationType(Enum):
    """Types of event correlations"""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CAUSAL = "causal"
    SEMANTIC = "semantic"
    STATISTICAL = "statistical"

class CorrelationStrength(Enum):
    """Correlation strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class EventCluster:
    """Cluster of correlated events"""
    cluster_id: str
    events: List[Dict[str, Any]]
    correlation_type: CorrelationType
    correlation_strength: CorrelationStrength
    cluster_score: float
    centroid: Dict[str, Any]
    time_span: timedelta
    site_count: int
    severity_distribution: Dict[str, int]
    pattern_signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrelationRule:
    """Event correlation rule"""
    rule_id: str
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    time_window: timedelta
    spatial_scope: str  # site, region, global
    confidence_threshold: float
    action: str  # alert, escalate, suppress, merge
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class EventCorrelationEngine:
    """Main event correlation engine"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.producer = MessageProducer()
        self.consumer = MessageConsumer()
        
        # Correlation parameters
        self.temporal_window = timedelta(minutes=15)
        self.spatial_radius = 50  # km
        self.min_cluster_size = 3
        self.correlation_threshold = 0.7
        
        # Event queues and processing
        self.event_buffer = deque(maxlen=10000)
        self.correlation_rules = []
        self.active_clusters = {}
        
        # ML models for correlation
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
        # Pattern templates for telecom events
        self.event_patterns = self._build_event_patterns()
        
        # Processing metrics
        self.processing_stats = {
            'events_processed': 0,
            'correlations_found': 0,
            'clusters_created': 0,
            'false_positives': 0
        }
    
    def _build_event_patterns(self) -> Dict[str, Dict]:
        """Build telecom-specific event patterns"""
        return {
            "power_cascade": {
                "events": ["power_failure", "ups_activation", "generator_start", "site_isolation"],
                "temporal_sequence": True,
                "max_time_span": timedelta(minutes=30),
                "severity_progression": "increasing"
            },
            
            "network_congestion": {
                "events": ["high_utilization", "packet_loss", "increased_latency", "connection_drops"],
                "temporal_sequence": False,
                "max_time_span": timedelta(minutes=10),
                "spatial_correlation": True
            },
            
            "equipment_failure": {
                "events": ["temperature_alarm", "fan_failure", "cpu_overload", "service_degradation"],
                "temporal_sequence": True,
                "max_time_span": timedelta(hours=2),
                "causal_chain": True
            },
            
            "configuration_error": {
                "events": ["config_change", "routing_error", "connectivity_loss", "service_impact"],
                "temporal_sequence": True,
                "max_time_span": timedelta(minutes=5),
                "immediate_impact": True
            },
            
            "security_incident": {
                "events": ["intrusion_attempt", "firewall_block", "authentication_failure", "service_disruption"],
                "temporal_sequence": False,
                "max_time_span": timedelta(hours=1),
                "geographic_spread": True
            }
        }
    
    async def initialize(self):
        """Initialize the event correlation engine"""
        try:
            logger.info("Initializing Event Correlation Engine...")
            
            # Load correlation rules
            await self.load_correlation_rules()
            
            # Initialize ML models
            await self.initialize_ml_models()
            
            # Setup event subscriptions
            await self.setup_event_subscriptions()
            
            # Start background processing
            asyncio.create_task(self.event_processing_loop())
            asyncio.create_task(self.correlation_maintenance_loop())
            
            logger.info("Event Correlation Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize event correlation engine: {e}")
            raise
    
    async def process_event(self, event_data: Dict[str, Any]) -> List[EventCluster]:
        """Process a single event and find correlations"""
        try:
            self.processing_stats['events_processed'] += 1
            
            # Add event to buffer
            self.event_buffer.append(event_data)
            
            # Find correlations
            correlations = []
            
            # Temporal correlation
            temporal_cluster = await self.find_temporal_correlations(event_data)
            if temporal_cluster:
                correlations.append(temporal_cluster)
            
            # Spatial correlation
            spatial_cluster = await self.find_spatial_correlations(event_data)
            if spatial_cluster:
                correlations.append(spatial_cluster)
            
            # Semantic correlation
            semantic_cluster = await self.find_semantic_correlations(event_data)
            if semantic_cluster:
                correlations.append(semantic_cluster)
            
            # Pattern-based correlation
            pattern_cluster = await self.find_pattern_correlations(event_data)
            if pattern_cluster:
                correlations.append(pattern_cluster)
            
            # Statistical correlation
            statistical_cluster = await self.find_statistical_correlations(event_data)
            if statistical_cluster:
                correlations.append(statistical_cluster)
            
            # Process rule-based correlations
            rule_correlations = await self.apply_correlation_rules(event_data)
            correlations.extend(rule_correlations)
            
            # Update active clusters
            await self.update_active_clusters(correlations)
            
            # Generate alerts for significant correlations
            for cluster in correlations:
                if cluster.correlation_strength in [CorrelationStrength.STRONG, CorrelationStrength.VERY_STRONG]:
                    await self.generate_correlation_alert(cluster)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error processing event correlation: {e}")
            return []
    
    async def find_temporal_correlations(self, event_data: Dict[str, Any]) -> Optional[EventCluster]:
        """Find temporally correlated events"""
        try:
            event_timestamp = datetime.fromisoformat(event_data.get('timestamp'))
            event_site = event_data.get('site_id')
            
            # Look for events in temporal window
            start_time = event_timestamp - self.temporal_window
            end_time = event_timestamp + self.temporal_window
            
            related_events = []
            for buffered_event in self.event_buffer:
                if buffered_event == event_data:
                    continue
                    
                buffered_timestamp = datetime.fromisoformat(buffered_event.get('timestamp'))
                
                if start_time <= buffered_timestamp <= end_time:
                    # Calculate temporal similarity
                    time_diff = abs((event_timestamp - buffered_timestamp).total_seconds())
                    temporal_score = max(0, 1.0 - (time_diff / self.temporal_window.total_seconds()))
                    
                    if temporal_score > 0.3:
                        buffered_event['temporal_score'] = temporal_score
                        related_events.append(buffered_event)
            
            if len(related_events) >= self.min_cluster_size - 1:  # -1 because we include current event
                # Create temporal cluster
                all_events = [event_data] + related_events
                
                cluster_score = np.mean([e.get('temporal_score', 0) for e in related_events])
                correlation_strength = self._calculate_correlation_strength(cluster_score)
                
                time_span = max(datetime.fromisoformat(e['timestamp']) for e in all_events) - \
                           min(datetime.fromisoformat(e['timestamp']) for e in all_events)
                
                site_count = len(set(e.get('site_id') for e in all_events if e.get('site_id')))
                
                severity_dist = defaultdict(int)
                for e in all_events:
                    severity_dist[e.get('severity', 'unknown')] += 1
                
                cluster = EventCluster(
                    cluster_id=f"temporal_{int(event_timestamp.timestamp())}",
                    events=all_events,
                    correlation_type=CorrelationType.TEMPORAL,
                    correlation_strength=correlation_strength,
                    cluster_score=cluster_score,
                    centroid=self._calculate_temporal_centroid(all_events),
                    time_span=time_span,
                    site_count=site_count,
                    severity_distribution=dict(severity_dist),
                    pattern_signature=self._generate_pattern_signature(all_events),
                    metadata={'temporal_window': self.temporal_window.total_seconds()}
                )
                
                self.processing_stats['correlations_found'] += 1
                return cluster
            
            return None
            
        except Exception as e:
            logger.error(f"Error in temporal correlation: {e}")
            return None
    
    async def find_spatial_correlations(self, event_data: Dict[str, Any]) -> Optional[EventCluster]:
        """Find spatially correlated events"""
        try:
            event_site = event_data.get('site_id')
            if not event_site:
                return None
            
            # Get site location
            site_location = await self.get_site_location(event_site)
            if not site_location:
                return None
            
            event_lat, event_lon = site_location['latitude'], site_location['longitude']
            
            # Look for events in spatial radius
            related_events = []
            for buffered_event in self.event_buffer:
                if buffered_event == event_data:
                    continue
                
                buffered_site = buffered_event.get('site_id')
                if not buffered_site:
                    continue
                
                buffered_location = await self.get_site_location(buffered_site)
                if not buffered_location:
                    continue
                
                # Calculate distance
                distance = self._calculate_distance(
                    event_lat, event_lon,
                    buffered_location['latitude'], buffered_location['longitude']
                )
                
                if distance <= self.spatial_radius:
                    spatial_score = max(0, 1.0 - (distance / self.spatial_radius))
                    
                    # Also consider temporal proximity for spatial correlation
                    event_time = datetime.fromisoformat(event_data['timestamp'])
                    buffered_time = datetime.fromisoformat(buffered_event['timestamp'])
                    time_diff = abs((event_time - buffered_time).total_seconds())
                    
                    if time_diff <= 3600:  # Within 1 hour
                        temporal_factor = max(0, 1.0 - (time_diff / 3600))
                        combined_score = (spatial_score + temporal_factor) / 2
                        
                        if combined_score > 0.5:
                            buffered_event['spatial_score'] = combined_score
                            buffered_event['distance'] = distance
                            related_events.append(buffered_event)
            
            if len(related_events) >= self.min_cluster_size - 1:
                all_events = [event_data] + related_events
                
                cluster_score = np.mean([e.get('spatial_score', 0) for e in related_events])
                correlation_strength = self._calculate_correlation_strength(cluster_score)
                
                time_span = max(datetime.fromisoformat(e['timestamp']) for e in all_events) - \
                           min(datetime.fromisoformat(e['timestamp']) for e in all_events)
                
                site_count = len(set(e.get('site_id') for e in all_events if e.get('site_id')))
                
                severity_dist = defaultdict(int)
                for e in all_events:
                    severity_dist[e.get('severity', 'unknown')] += 1
                
                cluster = EventCluster(
                    cluster_id=f"spatial_{event_site}_{int(datetime.fromisoformat(event_data['timestamp']).timestamp())}",
                    events=all_events,
                    correlation_type=CorrelationType.SPATIAL,
                    correlation_strength=correlation_strength,
                    cluster_score=cluster_score,
                    centroid=self._calculate_spatial_centroid(all_events),
                    time_span=time_span,
                    site_count=site_count,
                    severity_distribution=dict(severity_dist),
                    pattern_signature=self._generate_pattern_signature(all_events),
                    metadata={
                        'center_location': {'lat': event_lat, 'lon': event_lon},
                        'radius_km': self.spatial_radius,
                        'avg_distance': np.mean([e.get('distance', 0) for e in related_events])
                    }
                )
                
                self.processing_stats['correlations_found'] += 1
                return cluster
            
            return None
            
        except Exception as e:
            logger.error(f"Error in spatial correlation: {e}")
            return None
    
    async def find_semantic_correlations(self, event_data: Dict[str, Any]) -> Optional[EventCluster]:
        """Find semantically similar events using NLP"""
        try:
            event_title = event_data.get('title', '')
            event_description = event_data.get('description', '')
            event_text = f"{event_title} {event_description}".lower()
            
            if not event_text.strip():
                return None
            
            # Collect text from buffered events
            event_texts = []
            related_events = []
            
            for buffered_event in list(self.event_buffer)[-100:]:  # Last 100 events
                if buffered_event == event_data:
                    continue
                
                buffered_title = buffered_event.get('title', '')
                buffered_description = buffered_event.get('description', '')
                buffered_text = f"{buffered_title} {buffered_description}".lower()
                
                if buffered_text.strip():
                    event_texts.append(buffered_text)
                    related_events.append(buffered_event)
            
            if not event_texts:
                return None
            
            # Vectorize texts
            all_texts = [event_text] + event_texts
            
            try:
                tfidf_matrix = self.text_vectorizer.fit_transform(all_texts)
                
                # Calculate cosine similarities
                event_vector = tfidf_matrix[0:1]
                similarities = []
                
                for i in range(1, tfidf_matrix.shape[0]):
                    other_vector = tfidf_matrix[i:i+1]
                    similarity = 1 - cosine(event_vector.toarray().flatten(), 
                                          other_vector.toarray().flatten())
                    similarities.append(similarity)
                
                # Find highly similar events
                correlated_events = []
                for i, similarity in enumerate(similarities):
                    if similarity > self.correlation_threshold:
                        related_events[i]['semantic_score'] = similarity
                        correlated_events.append(related_events[i])
                
                if len(correlated_events) >= self.min_cluster_size - 1:
                    all_events = [event_data] + correlated_events
                    
                    cluster_score = np.mean([e.get('semantic_score', 0) for e in correlated_events])
                    correlation_strength = self._calculate_correlation_strength(cluster_score)
                    
                    time_span = max(datetime.fromisoformat(e['timestamp']) for e in all_events) - \
                               min(datetime.fromisoformat(e['timestamp']) for e in all_events)
                    
                    site_count = len(set(e.get('site_id') for e in all_events if e.get('site_id')))
                    
                    severity_dist = defaultdict(int)
                    for e in all_events:
                        severity_dist[e.get('severity', 'unknown')] += 1
                    
                    cluster = EventCluster(
                        cluster_id=f"semantic_{hash(event_text) % 1000000}",
                        events=all_events,
                        correlation_type=CorrelationType.SEMANTIC,
                        correlation_strength=correlation_strength,
                        cluster_score=cluster_score,
                        centroid=self._calculate_semantic_centroid(all_events),
                        time_span=time_span,
                        site_count=site_count,
                        severity_distribution=dict(severity_dist),
                        pattern_signature=self._generate_pattern_signature(all_events),
                        metadata={'similarity_threshold': self.correlation_threshold}
                    )
                    
                    self.processing_stats['correlations_found'] += 1
                    return cluster
                    
            except ValueError as e:
                logger.debug(f"TF-IDF vectorization failed: {e}")
                return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error in semantic correlation: {e}")
            return None
    
    async def find_pattern_correlations(self, event_data: Dict[str, Any]) -> Optional[EventCluster]:
        """Find pattern-based correlations using predefined templates"""
        try:
            event_type = self._classify_event_type(event_data)
            
            for pattern_name, pattern_def in self.event_patterns.items():
                if event_type in pattern_def['events']:
                    # Look for other events in this pattern
                    pattern_events = [event_data]
                    pattern_scores = []
                    
                    for buffered_event in self.event_buffer:
                        if buffered_event == event_data:
                            continue
                        
                        buffered_type = self._classify_event_type(buffered_event)
                        if buffered_type in pattern_def['events']:
                            
                            # Check temporal constraints
                            event_time = datetime.fromisoformat(event_data['timestamp'])
                            buffered_time = datetime.fromisoformat(buffered_event['timestamp'])
                            time_diff = abs((event_time - buffered_time).total_seconds())
                            
                            if time_diff <= pattern_def['max_time_span'].total_seconds():
                                # Calculate pattern score
                                pattern_score = self._calculate_pattern_score(
                                    pattern_def, event_data, buffered_event, time_diff
                                )
                                
                                if pattern_score > 0.6:
                                    buffered_event['pattern_score'] = pattern_score
                                    pattern_events.append(buffered_event)
                                    pattern_scores.append(pattern_score)
                    
                    if len(pattern_events) >= 3:  # At least 3 events for a pattern
                        cluster_score = np.mean(pattern_scores) if pattern_scores else 0.5
                        correlation_strength = self._calculate_correlation_strength(cluster_score)
                        
                        time_span = max(datetime.fromisoformat(e['timestamp']) for e in pattern_events) - \
                                   min(datetime.fromisoformat(e['timestamp']) for e in pattern_events)
                        
                        site_count = len(set(e.get('site_id') for e in pattern_events if e.get('site_id')))
                        
                        severity_dist = defaultdict(int)
                        for e in pattern_events:
                            severity_dist[e.get('severity', 'unknown')] += 1
                        
                        cluster = EventCluster(
                            cluster_id=f"pattern_{pattern_name}_{int(datetime.fromisoformat(event_data['timestamp']).timestamp())}",
                            events=pattern_events,
                            correlation_type=CorrelationType.CAUSAL,
                            correlation_strength=correlation_strength,
                            cluster_score=cluster_score,
                            centroid=self._calculate_pattern_centroid(pattern_events, pattern_name),
                            time_span=time_span,
                            site_count=site_count,
                            severity_distribution=dict(severity_dist),
                            pattern_signature=pattern_name,
                            metadata={
                                'pattern_name': pattern_name,
                                'pattern_definition': pattern_def
                            }
                        )
                        
                        self.processing_stats['correlations_found'] += 1
                        return cluster
            
            return None
            
        except Exception as e:
            logger.error(f"Error in pattern correlation: {e}")
            return None
    
    async def find_statistical_correlations(self, event_data: Dict[str, Any]) -> Optional[EventCluster]:
        """Find statistically significant correlations"""
        try:
            # Extract numerical features for statistical analysis
            event_features = self._extract_statistical_features(event_data)
            if not event_features:
                return None
            
            # Collect features from buffered events
            feature_vectors = []
            related_events = []
            
            for buffered_event in list(self.event_buffer)[-50:]:  # Last 50 events
                if buffered_event == event_data:
                    continue
                
                buffered_features = self._extract_statistical_features(buffered_event)
                if buffered_features:
                    feature_vectors.append(buffered_features)
                    related_events.append(buffered_event)
            
            if len(feature_vectors) < self.min_cluster_size - 1:
                return None
            
            # Perform clustering using DBSCAN
            all_features = [event_features] + feature_vectors
            
            try:
                scaled_features = self.scaler.fit_transform(all_features)
                
                clustering = DBSCAN(eps=0.5, min_samples=self.min_cluster_size)
                cluster_labels = clustering.fit_predict(scaled_features)
                
                # Check if current event is in a cluster (not noise)
                if cluster_labels[0] != -1:
                    cluster_id = cluster_labels[0]
                    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                    
                    if len(cluster_indices) >= self.min_cluster_size:
                        clustered_events = [event_data]
                        for idx in cluster_indices[1:]:  # Skip first (current event)
                            related_events[idx - 1]['statistical_score'] = 0.8  # DBSCAN doesn't provide distances
                            clustered_events.append(related_events[idx - 1])
                        
                        cluster_score = 0.8  # Fixed score for DBSCAN clusters
                        correlation_strength = self._calculate_correlation_strength(cluster_score)
                        
                        time_span = max(datetime.fromisoformat(e['timestamp']) for e in clustered_events) - \
                                   min(datetime.fromisoformat(e['timestamp']) for e in clustered_events)
                        
                        site_count = len(set(e.get('site_id') for e in clustered_events if e.get('site_id')))
                        
                        severity_dist = defaultdict(int)
                        for e in clustered_events:
                            severity_dist[e.get('severity', 'unknown')] += 1
                        
                        cluster = EventCluster(
                            cluster_id=f"statistical_{cluster_id}_{int(datetime.fromisoformat(event_data['timestamp']).timestamp())}",
                            events=clustered_events,
                            correlation_type=CorrelationType.STATISTICAL,
                            correlation_strength=correlation_strength,
                            cluster_score=cluster_score,
                            centroid=self._calculate_statistical_centroid(clustered_events),
                            time_span=time_span,
                            site_count=site_count,
                            severity_distribution=dict(severity_dist),
                            pattern_signature=self._generate_pattern_signature(clustered_events),
                            metadata={'cluster_method': 'dbscan', 'eps': 0.5}
                        )
                        
                        self.processing_stats['correlations_found'] += 1
                        return cluster
                        
            except ValueError as e:
                logger.debug(f"Statistical clustering failed: {e}")
                return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error in statistical correlation: {e}")
            return None
    
    async def apply_correlation_rules(self, event_data: Dict[str, Any]) -> List[EventCluster]:
        """Apply predefined correlation rules"""
        try:
            clusters = []
            
            for rule in self.correlation_rules:
                if not rule.enabled:
                    continue
                
                if self._event_matches_rule(event_data, rule):
                    # Look for related events that satisfy rule conditions
                    related_events = []
                    
                    for buffered_event in self.event_buffer:
                        if buffered_event == event_data:
                            continue
                        
                        if self._events_satisfy_rule_conditions(event_data, buffered_event, rule):
                            related_events.append(buffered_event)
                    
                    if len(related_events) >= self.min_cluster_size - 1:
                        all_events = [event_data] + related_events
                        
                        cluster = EventCluster(
                            cluster_id=f"rule_{rule.rule_id}_{int(datetime.fromisoformat(event_data['timestamp']).timestamp())}",
                            events=all_events,
                            correlation_type=CorrelationType.CAUSAL,
                            correlation_strength=CorrelationStrength.STRONG,
                            cluster_score=rule.confidence_threshold,
                            centroid=self._calculate_rule_centroid(all_events, rule),
                            time_span=max(datetime.fromisoformat(e['timestamp']) for e in all_events) - \
                                     min(datetime.fromisoformat(e['timestamp']) for e in all_events),
                            site_count=len(set(e.get('site_id') for e in all_events if e.get('site_id'))),
                            severity_distribution=defaultdict(int),
                            pattern_signature=f"rule_{rule.name}",
                            metadata={'rule': rule.__dict__}
                        )
                        
                        clusters.append(cluster)
                        self.processing_stats['correlations_found'] += 1
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error applying correlation rules: {e}")
            return []
    
    def _classify_event_type(self, event_data: Dict[str, Any]) -> str:
        """Classify event type based on title and content"""
        title = event_data.get('title', '').lower()
        description = event_data.get('description', '').lower()
        text = f"{title} {description}"
        
        # Classification rules
        if any(word in text for word in ['power', 'ups', 'battery', 'generator']):
            return 'power_failure'
        elif any(word in text for word in ['temperature', 'cooling', 'overheat', 'fan']):
            return 'temperature_alarm'
        elif any(word in text for word in ['cpu', 'processor', 'load', 'overload']):
            return 'cpu_overload'
        elif any(word in text for word in ['memory', 'ram', 'oom']):
            return 'memory_issue'
        elif any(word in text for word in ['network', 'connectivity', 'packet', 'loss']):
            return 'network_issue'
        elif any(word in text for word in ['config', 'configuration', 'change']):
            return 'config_change'
        elif any(word in text for word in ['service', 'down', 'unavailable', 'outage']):
            return 'service_unavailable'
        else:
            return 'unknown'
    
    def _calculate_pattern_score(self, pattern_def: Dict, event1: Dict, event2: Dict, 
                                time_diff: float) -> float:
        """Calculate pattern matching score"""
        score = 0.0
        
        # Temporal score
        max_time = pattern_def['max_time_span'].total_seconds()
        temporal_score = max(0, 1.0 - (time_diff / max_time))
        score += temporal_score * 0.4
        
        # Severity progression score
        if pattern_def.get('severity_progression') == 'increasing':
            severity_map = {'warning': 1, 'minor': 2, 'major': 3, 'critical': 4}
            sev1 = severity_map.get(event1.get('severity', '').lower(), 0)
            sev2 = severity_map.get(event2.get('severity', '').lower(), 0)
            
            if sev2 > sev1:
                score += 0.3
            elif sev2 == sev1:
                score += 0.1
        
        # Spatial correlation score
        if pattern_def.get('spatial_correlation'):
            if event1.get('site_id') == event2.get('site_id'):
                score += 0.3
        
        return min(score, 1.0)
    
    def _extract_statistical_features(self, event_data: Dict[str, Any]) -> Optional[List[float]]:
        """Extract numerical features from event"""
        try:
            features = []
            
            # Temporal features
            timestamp = datetime.fromisoformat(event_data['timestamp'])
            features.extend([
                timestamp.hour,
                timestamp.day,
                timestamp.weekday(),
                int(timestamp.timestamp()) % 86400  # Seconds since midnight
            ])
            
            # Severity encoding
            severity_map = {'warning': 1, 'minor': 2, 'major': 3, 'critical': 4}
            severity_score = severity_map.get(event_data.get('severity', '').lower(), 0)
            features.append(severity_score)
            
            # Event type encoding
            event_type = self._classify_event_type(event_data)
            type_hash = hash(event_type) % 1000  # Simple hash for categorical
            features.append(type_hash)
            
            # Text length features
            title_len = len(event_data.get('title', ''))
            desc_len = len(event_data.get('description', ''))
            features.extend([title_len, desc_len])
            
            # Site hash (if available)
            if event_data.get('site_id'):
                site_hash = hash(event_data['site_id']) % 1000
                features.append(site_hash)
            else:
                features.append(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting statistical features: {e}")
            return None
    
    def _calculate_correlation_strength(self, score: float) -> CorrelationStrength:
        """Calculate correlation strength from score"""
        if score >= 0.9:
            return CorrelationStrength.VERY_STRONG
        elif score >= 0.75:
            return CorrelationStrength.STRONG
        elif score >= 0.6:
            return CorrelationStrength.MODERATE
        else:
            return CorrelationStrength.WEAK
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in km"""
        from math import sin, cos, sqrt, atan2, radians
        
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def _calculate_temporal_centroid(self, events: List[Dict]) -> Dict[str, Any]:
        """Calculate centroid for temporal cluster"""
        timestamps = [datetime.fromisoformat(e['timestamp']) for e in events]
        avg_timestamp = datetime.fromtimestamp(
            np.mean([t.timestamp() for t in timestamps])
        )
        
        return {
            'centroid_time': avg_timestamp.isoformat(),
            'time_span': max(timestamps) - min(timestamps),
            'event_count': len(events)
        }
    
    def _calculate_spatial_centroid(self, events: List[Dict]) -> Dict[str, Any]:
        """Calculate centroid for spatial cluster"""
        return {
            'centroid_type': 'spatial',
            'event_count': len(events),
            'unique_sites': len(set(e.get('site_id') for e in events if e.get('site_id')))
        }
    
    def _calculate_semantic_centroid(self, events: List[Dict]) -> Dict[str, Any]:
        """Calculate centroid for semantic cluster"""
        return {
            'centroid_type': 'semantic',
            'event_count': len(events),
            'common_terms': self._extract_common_terms(events)
        }
    
    def _calculate_pattern_centroid(self, events: List[Dict], pattern_name: str) -> Dict[str, Any]:
        """Calculate centroid for pattern cluster"""
        return {
            'centroid_type': 'pattern',
            'pattern_name': pattern_name,
            'event_count': len(events)
        }
    
    def _calculate_statistical_centroid(self, events: List[Dict]) -> Dict[str, Any]:
        """Calculate centroid for statistical cluster"""
        return {
            'centroid_type': 'statistical',
            'event_count': len(events),
            'cluster_method': 'dbscan'
        }
    
    def _calculate_rule_centroid(self, events: List[Dict], rule: CorrelationRule) -> Dict[str, Any]:
        """Calculate centroid for rule-based cluster"""
        return {
            'centroid_type': 'rule',
            'rule_name': rule.name,
            'event_count': len(events)
        }
    
    def _extract_common_terms(self, events: List[Dict]) -> List[str]:
        """Extract common terms from event titles/descriptions"""
        from collections import Counter
        
        all_words = []
        for event in events:
            title = event.get('title', '')
            description = event.get('description', '')
            words = (title + ' ' + description).lower().split()
            all_words.extend(words)
        
        # Get most common words
        common_words = Counter(all_words).most_common(5)
        return [word for word, count in common_words if len(word) > 3]
    
    def _generate_pattern_signature(self, events: List[Dict]) -> str:
        """Generate a signature for the event pattern"""
        event_types = [self._classify_event_type(e) for e in events]
        type_counts = Counter(event_types)
        
        # Create signature from most common event types
        signature_parts = []
        for event_type, count in type_counts.most_common(3):
            signature_parts.append(f"{event_type}({count})")
        
        return "_".join(signature_parts)
    
    def _event_matches_rule(self, event_data: Dict[str, Any], rule: CorrelationRule) -> bool:
        """Check if event matches rule conditions"""
        # Simplified rule matching - in practice would be more complex
        for condition in rule.conditions:
            if not self._evaluate_condition(event_data, condition):
                return False
        return True
    
    def _events_satisfy_rule_conditions(self, event1: Dict, event2: Dict, rule: CorrelationRule) -> bool:
        """Check if two events satisfy rule conditions"""
        # Check temporal constraint
        time1 = datetime.fromisoformat(event1['timestamp'])
        time2 = datetime.fromisoformat(event2['timestamp'])
        time_diff = abs((time1 - time2).total_seconds())
        
        if time_diff > rule.time_window.total_seconds():
            return False
        
        # Check spatial constraint
        if rule.spatial_scope == 'site':
            return event1.get('site_id') == event2.get('site_id')
        
        return True
    
    def _evaluate_condition(self, event_data: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Evaluate a single rule condition"""
        # Simplified condition evaluation
        field = condition.get('field')
        operator = condition.get('operator')
        value = condition.get('value')
        
        if field in event_data:
            event_value = event_data[field]
            
            if operator == 'equals':
                return event_value == value
            elif operator == 'contains':
                return value in str(event_value).lower()
            elif operator == 'greater_than':
                return event_value > value
        
        return False
    
    async def get_site_location(self, site_id: str) -> Optional[Dict[str, float]]:
        """Get site location from cache or database"""
        try:
            # Try cache first
            cache_key = f"site_location:{site_id}"
            cached_location = await self.cache.get(cache_key)
            
            if cached_location:
                return cached_location
            
            # Query database
            query = "SELECT latitude, longitude FROM sites WHERE site_id = $1"
            row = await self.db.fetchrow(query, site_id)
            
            if row:
                location = {
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude'])
                }
                
                # Cache for 1 hour
                await self.cache.set(cache_key, location, expire=3600)
                return location
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting site location for {site_id}: {e}")
            return None
    
    async def update_active_clusters(self, new_clusters: List[EventCluster]):
        """Update active cluster tracking"""
        try:
            current_time = datetime.utcnow()
            
            # Add new clusters
            for cluster in new_clusters:
                self.active_clusters[cluster.cluster_id] = {
                    'cluster': cluster,
                    'created_at': current_time,
                    'last_updated': current_time
                }
            
            # Clean up old clusters (older than 1 hour)
            cutoff_time = current_time - timedelta(hours=1)
            expired_clusters = [
                cluster_id for cluster_id, data in self.active_clusters.items()
                if data['created_at'] < cutoff_time
            ]
            
            for cluster_id in expired_clusters:
                del self.active_clusters[cluster_id]
            
            logger.debug(f"Active clusters: {len(self.active_clusters)}")
            
        except Exception as e:
            logger.error(f"Error updating active clusters: {e}")
    
    async def generate_correlation_alert(self, cluster: EventCluster):
        """Generate alert for significant correlation"""
        try:
            alert = {
                'type': 'event_correlation',
                'timestamp': datetime.utcnow().isoformat(),
                'cluster_id': cluster.cluster_id,
                'correlation_type': cluster.correlation_type.value,
                'correlation_strength': cluster.correlation_strength.value,
                'cluster_score': cluster.cluster_score,
                'event_count': len(cluster.events),
                'site_count': cluster.site_count,
                'time_span_seconds': cluster.time_span.total_seconds(),
                'pattern_signature': cluster.pattern_signature,
                'severity_distribution': cluster.severity_distribution,
                'events': [
                    {
                        'event_id': e.get('event_id'),
                        'title': e.get('title'),
                        'site_id': e.get('site_id'),
                        'timestamp': e.get('timestamp'),
                        'severity': e.get('severity')
                    }
                    for e in cluster.events
                ]
            }
            
            await self.producer.send_message('aiops.correlations', alert)
            logger.info(f"Generated correlation alert: {cluster.cluster_id}")
            
        except Exception as e:
            logger.error(f"Error generating correlation alert: {e}")
    
    async def load_correlation_rules(self):
        """Load correlation rules from database"""
        try:
            query = """
            SELECT rule_id, name, description, conditions, time_window_seconds,
                   spatial_scope, confidence_threshold, action, enabled
            FROM correlation_rules
            WHERE enabled = true
            """
            
            rows = await self.db.fetch_all(query)
            
            for row in rows:
                rule = CorrelationRule(
                    rule_id=row['rule_id'],
                    name=row['name'],
                    description=row['description'],
                    conditions=json.loads(row['conditions']) if row['conditions'] else [],
                    time_window=timedelta(seconds=row['time_window_seconds']),
                    spatial_scope=row['spatial_scope'],
                    confidence_threshold=row['confidence_threshold'],
                    action=row['action'],
                    enabled=row['enabled']
                )
                self.correlation_rules.append(rule)
            
            logger.info(f"Loaded {len(self.correlation_rules)} correlation rules")
            
        except Exception as e:
            logger.error(f"Error loading correlation rules: {e}")
    
    async def initialize_ml_models(self):
        """Initialize ML models for correlation"""
        try:
            # Initialize with some sample data if available
            logger.info("ML models initialized for event correlation")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    async def setup_event_subscriptions(self):
        """Setup subscriptions to event streams"""
        try:
            # Subscribe to events topic
            await self.consumer.subscribe(['events', 'anomalies'])
            logger.info("Event subscriptions setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up event subscriptions: {e}")
    
    async def event_processing_loop(self):
        """Main event processing loop"""
        while True:
            try:
                # This would consume events from message queue
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(5)
    
    async def correlation_maintenance_loop(self):
        """Maintenance loop for correlation engine"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Log processing statistics
                logger.info(f"Correlation stats: {self.processing_stats}")
                
                # Clean up old events from buffer
                current_time = datetime.utcnow()
                cutoff_time = current_time - timedelta(hours=2)
                
                # Remove old events
                filtered_buffer = deque()
                for event in self.event_buffer:
                    event_time = datetime.fromisoformat(event['timestamp'])
                    if event_time > cutoff_time:
                        filtered_buffer.append(event)
                
                self.event_buffer = filtered_buffer
                
            except Exception as e:
                logger.error(f"Error in correlation maintenance loop: {e}")

# Main worker function
# Global state
shutdown_event = asyncio.Event()
correlator_engine = None

async def message_handler(topic, key, value, headers, timestamp, partition, offset):
    """Handle incoming Kafka messages"""
    global correlator_engine
    
    logger.info(f"Received message from {topic}: {key}")
    
    try:
        if topic == KAFKA_TOPICS['events']:
            # Process events for correlation
            event_type = value.get('event_type')
            site_id = value.get('site_id')
            severity = value.get('severity')
            
            if event_type and site_id and correlator_engine:
                # Trigger event correlation
                asyncio.create_task(
                    correlator_engine.correlate_event(value)
                )
                
        elif topic == KAFKA_TOPICS['alerts']:
            # Process alerts for correlation
            alert_type = value.get('alert_type')
            site_id = value.get('site_id')
            
            if alert_type and site_id and correlator_engine:
                # Correlate with existing events
                asyncio.create_task(
                    correlator_engine.correlate_alert(value)
                )
                
        elif topic == KAFKA_TOPICS['network_metrics']:
            # Process network events for correlation
            site_id = value.get('site_id')
            metric_name = value.get('metric_name')
            
            if site_id and metric_name and correlator_engine:
                # Check for metric-based correlations
                asyncio.create_task(
                    correlator_engine.correlate_metric_event(value)
                )
                
    except Exception as e:
        logger.error(f"Error handling message: {e}")

async def create_health_server():
    """Create health check server"""
    from aiohttp import web
    
    async def health_handler(request):
        return web.json_response({
            "status": "healthy",
            "worker": "event_correlator",
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
    """Main event correlation worker"""
    global correlator_engine, health_server
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting Towerco Event Correlation Engine...")
    
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
        site = web.TCPSite(health_runner, '0.0.0.0', 8005)
        await site.start()
        logger.info("Health server started on port 8005")
        
        # Initialize Event Correlation Engine
        correlator_engine = EventCorrelationEngine()
        await correlator_engine.initialize()
        
        # Start Kafka consumers
        await start_consumer(
            'event_correlator',
            [KAFKA_TOPICS['events'], KAFKA_TOPICS['alerts'], KAFKA_TOPICS['network_metrics']],
            'event_correlator_group',
            message_handler
        )
        
        logger.info("Event Correlation Engine started successfully")
        
        # Main loop
        while not shutdown_event.is_set():
            await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Event correlation worker stopping...")
    except Exception as e:
        logger.error(f"Fatal error in event correlation worker: {e}")
        raise
    
    finally:
        logger.info("Shutting down Event Correlation Engine...")
        
        # Cleanup health server
        if health_server:
            await health_runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())