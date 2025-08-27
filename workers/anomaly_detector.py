"""
Anomaly Detection Engine - AIOps Machine Learning
Advanced anomaly detection for telecom infrastructure monitoring
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
import joblib

from core.database import DatabaseManager
from core.cache import CacheManager
from core.messaging import MessageProducer
from core.config import settings

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies detected"""
    POINT = "point"           # Single data point anomaly
    CONTEXTUAL = "contextual" # Anomaly within context
    COLLECTIVE = "collective" # Multiple related anomalies
    SEASONAL = "seasonal"     # Seasonal pattern deviation
    TREND = "trend"          # Trend change anomaly

class AnomalySeverity(Enum):
    """Severity levels for anomalies"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AnomalyResult:
    """Anomaly detection result"""
    metric_name: str
    site_id: str
    timestamp: datetime
    value: float
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float
    anomaly_score: float
    expected_range: Dict[str, float]
    contributing_factors: List[str]
    metadata: Dict[str, Any]

class AnomalyDetectionEngine:
    """Main anomaly detection engine with multiple ML algorithms"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.producer = MessageProducer()
        
        # ML Models
        self.isolation_forest = None
        self.dbscan = None
        self.scalers = {}
        self.pca_models = {}
        
        # Configuration
        self.detection_window = timedelta(hours=24)
        self.min_data_points = 100
        self.contamination_rate = 0.1  # Expected anomaly rate
        self.sensitivity_threshold = 0.7
        
        # Metrics to monitor
        self.monitored_metrics = [
            'network_utilization', 'cpu_usage', 'memory_usage',
            'temperature', 'power_consumption', 'packet_loss',
            'latency', 'throughput', 'error_rate', 'availability'
        ]
        
    async def initialize(self):
        """Initialize the anomaly detection engine"""
        try:
            logger.info("Initializing Anomaly Detection Engine...")
            
            # Load or train initial models
            await self.load_or_train_models()
            
            # Setup metric subscriptions
            await self.setup_metric_subscriptions()
            
            logger.info("Anomaly Detection Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize anomaly detection engine: {e}")
            raise
    
    async def load_or_train_models(self):
        """Load existing models or train new ones"""
        try:
            # Try to load pre-trained models from cache
            models_loaded = await self.load_models_from_cache()
            
            if not models_loaded:
                logger.info("No cached models found, training new models...")
                await self.train_initial_models()
                await self.save_models_to_cache()
            
        except Exception as e:
            logger.error(f"Error loading/training models: {e}")
            # Fallback to basic statistical methods
            await self.setup_statistical_models()
    
    async def train_initial_models(self):
        """Train initial anomaly detection models"""
        try:
            # Get historical data for training
            training_data = await self.get_training_data()
            
            if training_data.empty:
                logger.warning("No training data available, using statistical methods")
                await self.setup_statistical_models()
                return
            
            for metric_name in self.monitored_metrics:
                metric_data = training_data[training_data['metric_name'] == metric_name]
                
                if len(metric_data) < self.min_data_points:
                    logger.warning(f"Insufficient data for {metric_name}, skipping ML training")
                    continue
                
                await self.train_metric_specific_models(metric_name, metric_data)
            
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    async def train_metric_specific_models(self, metric_name: str, data: pd.DataFrame):
        """Train models specific to a metric"""
        try:
            # Prepare features
            features = self.prepare_features(data)
            
            if features is None or len(features) < self.min_data_points:
                return
            
            # Scale features
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(features)
            self.scalers[metric_name] = scaler
            
            # Train Isolation Forest
            isolation_forest = IsolationForest(
                contamination=self.contamination_rate,
                random_state=42,
                n_jobs=-1
            )
            isolation_forest.fit(scaled_features)
            
            # Train DBSCAN for cluster-based anomaly detection
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(scaled_features)
            
            # Calculate silhouette score
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(scaled_features, cluster_labels)
                logger.info(f"DBSCAN silhouette score for {metric_name}: {silhouette_avg:.3f}")
            
            # PCA for dimensionality reduction
            pca = PCA(n_components=0.95)  # Keep 95% variance
            pca.fit(scaled_features)
            
            # Store models
            self.scalers[metric_name] = scaler
            self.isolation_forest = isolation_forest
            self.dbscan = dbscan
            self.pca_models[metric_name] = pca
            
            logger.info(f"Trained models for metric: {metric_name}")
            
        except Exception as e:
            logger.error(f"Error training models for {metric_name}: {e}")
    
    def prepare_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for ML training"""
        try:
            if data.empty:
                return None
            
            # Sort by timestamp
            data = data.sort_values('timestamp')
            
            # Create time-based features
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
            data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
            
            # Rolling statistics
            data['rolling_mean'] = data['metric_value'].rolling(window=12).mean()
            data['rolling_std'] = data['metric_value'].rolling(window=12).std()
            data['rolling_min'] = data['metric_value'].rolling(window=12).min()
            data['rolling_max'] = data['metric_value'].rolling(window=12).max()
            
            # Lag features
            for lag in [1, 6, 12, 24]:
                data[f'lag_{lag}'] = data['metric_value'].shift(lag)
            
            # Rate of change
            data['rate_of_change'] = data['metric_value'].pct_change()
            data['acceleration'] = data['rate_of_change'].diff()
            
            # Z-score
            data['z_score'] = np.abs(stats.zscore(data['metric_value'], nan_policy='omit'))
            
            # Select feature columns
            feature_columns = [
                'metric_value', 'hour', 'day_of_week', 'is_weekend',
                'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max',
                'lag_1', 'lag_6', 'lag_12', 'lag_24',
                'rate_of_change', 'acceleration', 'z_score'
            ]
            
            # Remove rows with NaN values
            features = data[feature_columns].dropna()
            
            return features.values if not features.empty else None
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    async def detect_anomalies_realtime(self, metric_data: Dict[str, Any]) -> List[AnomalyResult]:
        """Detect anomalies in real-time metric data"""
        try:
            results = []
            
            # Extract metric information
            metric_name = metric_data.get('metric_name')
            site_id = metric_data.get('site_id')
            value = float(metric_data.get('metric_value', 0))
            timestamp = datetime.fromisoformat(metric_data.get('timestamp'))
            
            if metric_name not in self.monitored_metrics:
                return results
            
            # Get historical context
            historical_data = await self.get_metric_history(metric_name, site_id)
            
            # Perform different types of anomaly detection
            anomalies = []
            
            # 1. Statistical anomaly detection
            stat_anomaly = await self.detect_statistical_anomaly(
                metric_name, site_id, value, timestamp, historical_data
            )
            if stat_anomaly:
                anomalies.append(stat_anomaly)
            
            # 2. ML-based anomaly detection
            if metric_name in self.scalers:
                ml_anomaly = await self.detect_ml_anomaly(
                    metric_name, site_id, value, timestamp, historical_data
                )
                if ml_anomaly:
                    anomalies.append(ml_anomaly)
            
            # 3. Time series anomaly detection
            ts_anomaly = await self.detect_time_series_anomaly(
                metric_name, site_id, value, timestamp, historical_data
            )
            if ts_anomaly:
                anomalies.append(ts_anomaly)
            
            # 4. Contextual anomaly detection
            ctx_anomaly = await self.detect_contextual_anomaly(
                metric_name, site_id, value, timestamp, metric_data
            )
            if ctx_anomaly:
                anomalies.append(ctx_anomaly)
            
            # Consolidate and rank anomalies
            if anomalies:
                consolidated_anomaly = self.consolidate_anomalies(anomalies)
                results.append(consolidated_anomaly)
                
                # Store anomaly result
                await self.store_anomaly(consolidated_anomaly)
                
                # Send alert if severity is high enough
                if consolidated_anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]:
                    await self.send_anomaly_alert(consolidated_anomaly)
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def detect_statistical_anomaly(
        self, metric_name: str, site_id: str, value: float, 
        timestamp: datetime, historical_data: pd.DataFrame
    ) -> Optional[AnomalyResult]:
        """Detect anomalies using statistical methods"""
        try:
            if historical_data.empty or len(historical_data) < 10:
                return None
            
            values = historical_data['metric_value'].values
            
            # Calculate statistical thresholds
            mean_val = np.mean(values)
            std_val = np.std(values)
            median_val = np.median(values)
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            
            # Z-score anomaly
            z_score = abs((value - mean_val) / std_val) if std_val > 0 else 0
            
            # IQR anomaly
            iqr_lower = q1 - 1.5 * iqr
            iqr_upper = q3 + 1.5 * iqr
            is_iqr_outlier = value < iqr_lower or value > iqr_upper
            
            # Modified Z-score (more robust)
            mad = np.median(np.abs(values - median_val))
            modified_z_score = 0.6745 * (value - median_val) / mad if mad > 0 else 0
            
            # Determine if anomalous
            is_anomalous = (
                z_score > 3 or 
                abs(modified_z_score) > 3.5 or 
                is_iqr_outlier
            )
            
            if not is_anomalous:
                return None
            
            # Calculate confidence and severity
            confidence = min(max(z_score / 5, 0.5), 1.0)
            
            if z_score > 5 or abs(modified_z_score) > 5:
                severity = AnomalySeverity.CRITICAL
            elif z_score > 4 or abs(modified_z_score) > 4:
                severity = AnomalySeverity.HIGH
            elif z_score > 3 or abs(modified_z_score) > 3:
                severity = AnomalySeverity.MEDIUM
            else:
                severity = AnomalySeverity.LOW
            
            return AnomalyResult(
                metric_name=metric_name,
                site_id=site_id,
                timestamp=timestamp,
                value=value,
                anomaly_type=AnomalyType.POINT,
                severity=severity,
                confidence=confidence,
                anomaly_score=max(z_score, abs(modified_z_score)),
                expected_range={
                    'mean': mean_val,
                    'std': std_val,
                    'lower_bound': mean_val - 2 * std_val,
                    'upper_bound': mean_val + 2 * std_val,
                    'iqr_lower': iqr_lower,
                    'iqr_upper': iqr_upper
                },
                contributing_factors=[
                    f"Z-score: {z_score:.2f}",
                    f"Modified Z-score: {modified_z_score:.2f}",
                    f"IQR outlier: {is_iqr_outlier}"
                ],
                metadata={
                    'detection_method': 'statistical',
                    'z_score': z_score,
                    'modified_z_score': modified_z_score,
                    'is_iqr_outlier': is_iqr_outlier
                }
            )
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}")
            return None
    
    async def detect_ml_anomaly(
        self, metric_name: str, site_id: str, value: float,
        timestamp: datetime, historical_data: pd.DataFrame
    ) -> Optional[AnomalyResult]:
        """Detect anomalies using ML models"""
        try:
            if (metric_name not in self.scalers or 
                self.isolation_forest is None or 
                historical_data.empty):
                return None
            
            # Prepare current data point
            current_data = historical_data.iloc[-20:].copy()  # Last 20 points for context
            current_data = current_data.append({
                'metric_name': metric_name,
                'site_id': site_id,
                'metric_value': value,
                'timestamp': timestamp
            }, ignore_index=True)
            
            # Prepare features
            features = self.prepare_features(current_data)
            if features is None or len(features) == 0:
                return None
            
            # Scale features
            scaler = self.scalers[metric_name]
            scaled_features = scaler.transform(features[-1:])  # Only current point
            
            # Isolation Forest prediction
            isolation_score = self.isolation_forest.decision_function(scaled_features)[0]
            is_anomaly = self.isolation_forest.predict(scaled_features)[0] == -1
            
            if not is_anomaly:
                return None
            
            # Calculate confidence based on isolation score
            confidence = min(abs(isolation_score) / 0.5, 1.0)
            
            # Determine severity based on isolation score
            if isolation_score < -0.4:
                severity = AnomalySeverity.CRITICAL
            elif isolation_score < -0.3:
                severity = AnomalySeverity.HIGH
            elif isolation_score < -0.2:
                severity = AnomalySeverity.MEDIUM
            else:
                severity = AnomalySeverity.LOW
            
            return AnomalyResult(
                metric_name=metric_name,
                site_id=site_id,
                timestamp=timestamp,
                value=value,
                anomaly_type=AnomalyType.POINT,
                severity=severity,
                confidence=confidence,
                anomaly_score=abs(isolation_score),
                expected_range={'isolation_score_threshold': -0.1},
                contributing_factors=[
                    f"Isolation score: {isolation_score:.4f}",
                    "Machine learning model detection"
                ],
                metadata={
                    'detection_method': 'ml_isolation_forest',
                    'isolation_score': isolation_score
                }
            )
            
        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {e}")
            return None
    
    async def detect_time_series_anomaly(
        self, metric_name: str, site_id: str, value: float,
        timestamp: datetime, historical_data: pd.DataFrame
    ) -> Optional[AnomalyResult]:
        """Detect time series anomalies (trend changes, seasonality)"""
        try:
            if historical_data.empty or len(historical_data) < 50:
                return None
            
            # Sort by timestamp
            data = historical_data.sort_values('timestamp')
            values = data['metric_value'].values
            
            # Detect trend changes using change point detection
            trend_change = self.detect_trend_change(values)
            
            # Detect seasonal anomalies
            seasonal_anomaly = self.detect_seasonal_anomaly(data, value, timestamp)
            
            if not trend_change and not seasonal_anomaly:
                return None
            
            anomaly_type = AnomalyType.TREND if trend_change else AnomalyType.SEASONAL
            confidence = 0.8 if trend_change and seasonal_anomaly else 0.6
            
            # Calculate severity
            severity = AnomalySeverity.HIGH if trend_change else AnomalySeverity.MEDIUM
            
            return AnomalyResult(
                metric_name=metric_name,
                site_id=site_id,
                timestamp=timestamp,
                value=value,
                anomaly_type=anomaly_type,
                severity=severity,
                confidence=confidence,
                anomaly_score=0.8 if trend_change else 0.6,
                expected_range={'trend_change': trend_change, 'seasonal_anomaly': seasonal_anomaly},
                contributing_factors=[
                    f"Trend change detected: {trend_change}",
                    f"Seasonal anomaly: {seasonal_anomaly}"
                ],
                metadata={
                    'detection_method': 'time_series',
                    'trend_change': trend_change,
                    'seasonal_anomaly': seasonal_anomaly
                }
            )
            
        except Exception as e:
            logger.error(f"Error in time series anomaly detection: {e}")
            return None
    
    def detect_trend_change(self, values: np.ndarray) -> bool:
        """Detect significant trend changes"""
        try:
            if len(values) < 20:
                return False
            
            # Split into two halves and compare trends
            mid = len(values) // 2
            first_half = values[:mid]
            second_half = values[mid:]
            
            # Calculate linear trends
            x1 = np.arange(len(first_half))
            x2 = np.arange(len(second_half))
            
            slope1, _, _, _, _ = stats.linregress(x1, first_half)
            slope2, _, _, _, _ = stats.linregress(x2, second_half)
            
            # Check if slopes have significantly different signs or magnitudes
            slope_diff = abs(slope1 - slope2)
            slope_threshold = np.std(values) * 0.1
            
            return slope_diff > slope_threshold
            
        except Exception as e:
            logger.error(f"Error detecting trend change: {e}")
            return False
    
    def detect_seasonal_anomaly(self, data: pd.DataFrame, value: float, timestamp: datetime) -> bool:
        """Detect seasonal anomalies"""
        try:
            if len(data) < 168:  # Need at least a week of hourly data
                return False
            
            # Extract time features
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Filter data for same hour and day of week
            similar_times = data[
                (pd.to_datetime(data['timestamp']).dt.hour == hour) &
                (pd.to_datetime(data['timestamp']).dt.dayofweek == day_of_week)
            ]
            
            if len(similar_times) < 5:
                return False
            
            # Calculate expected range for this time
            similar_values = similar_times['metric_value'].values
            expected_mean = np.mean(similar_values)
            expected_std = np.std(similar_values)
            
            # Check if current value is anomalous for this time
            if expected_std > 0:
                z_score = abs((value - expected_mean) / expected_std)
                return z_score > 2.5
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting seasonal anomaly: {e}")
            return False
    
    async def detect_contextual_anomaly(
        self, metric_name: str, site_id: str, value: float,
        timestamp: datetime, metric_data: Dict[str, Any]
    ) -> Optional[AnomalyResult]:
        """Detect contextual anomalies based on related metrics"""
        try:
            # Get related metrics at the same time
            related_metrics = await self.get_related_metrics(site_id, timestamp)
            
            if not related_metrics:
                return None
            
            # Check for metric correlations that suggest anomalies
            anomalous_correlations = self.check_metric_correlations(
                metric_name, value, related_metrics
            )
            
            if not anomalous_correlations:
                return None
            
            confidence = len(anomalous_correlations) * 0.2
            severity = AnomalySeverity.MEDIUM
            
            return AnomalyResult(
                metric_name=metric_name,
                site_id=site_id,
                timestamp=timestamp,
                value=value,
                anomaly_type=AnomalyType.CONTEXTUAL,
                severity=severity,
                confidence=min(confidence, 1.0),
                anomaly_score=confidence,
                expected_range={'correlations': anomalous_correlations},
                contributing_factors=[f"Anomalous correlation: {corr}" for corr in anomalous_correlations],
                metadata={
                    'detection_method': 'contextual',
                    'anomalous_correlations': anomalous_correlations,
                    'related_metrics': related_metrics
                }
            )
            
        except Exception as e:
            logger.error(f"Error in contextual anomaly detection: {e}")
            return None
    
    def check_metric_correlations(
        self, metric_name: str, value: float, related_metrics: List[Dict]
    ) -> List[str]:
        """Check for anomalous correlations between metrics"""
        anomalous_correlations = []
        
        # Define expected correlations for telecom metrics
        correlation_rules = {
            'cpu_usage': {
                'temperature': 'positive',
                'power_consumption': 'positive',
                'memory_usage': 'positive'
            },
            'network_utilization': {
                'throughput': 'positive',
                'latency': 'negative_when_high',
                'packet_loss': 'positive_when_high'
            },
            'temperature': {
                'power_consumption': 'positive',
                'cpu_usage': 'positive'
            },
            'power_consumption': {
                'cpu_usage': 'positive',
                'temperature': 'positive',
                'network_utilization': 'positive'
            }
        }
        
        if metric_name not in correlation_rules:
            return anomalous_correlations
        
        rules = correlation_rules[metric_name]
        
        for related_metric in related_metrics:
            related_name = related_metric['metric_name']
            related_value = related_metric['metric_value']
            
            if related_name in rules:
                expected_correlation = rules[related_name]
                
                # Check correlation violations
                if expected_correlation == 'positive':
                    if value > 80 and related_value < 20:  # Both should be high
                        anomalous_correlations.append(f"{metric_name} high but {related_name} low")
                elif expected_correlation == 'negative_when_high':
                    if value > 80 and related_value > 80:  # Should be negatively correlated
                        anomalous_correlations.append(f"{metric_name} and {related_name} both high")
                elif expected_correlation == 'positive_when_high':
                    if value > 80 and related_value < 10:
                        anomalous_correlations.append(f"{metric_name} high but {related_name} very low")
        
        return anomalous_correlations
    
    def consolidate_anomalies(self, anomalies: List[AnomalyResult]) -> AnomalyResult:
        """Consolidate multiple anomaly detections into a single result"""
        if len(anomalies) == 1:
            return anomalies[0]
        
        # Take the most severe anomaly as base
        base_anomaly = max(anomalies, key=lambda x: x.anomaly_score)
        
        # Combine confidence scores
        combined_confidence = min(sum(a.confidence for a in anomalies) / len(anomalies) * 1.2, 1.0)
        
        # Combine anomaly scores
        combined_score = max(a.anomaly_score for a in anomalies)
        
        # Combine contributing factors
        all_factors = []
        for anomaly in anomalies:
            all_factors.extend(anomaly.contributing_factors)
        
        # Determine final severity
        severities = [a.severity for a in anomalies]
        if AnomalySeverity.CRITICAL in severities:
            final_severity = AnomalySeverity.CRITICAL
        elif AnomalySeverity.HIGH in severities:
            final_severity = AnomalySeverity.HIGH
        else:
            final_severity = AnomalySeverity.MEDIUM
        
        return AnomalyResult(
            metric_name=base_anomaly.metric_name,
            site_id=base_anomaly.site_id,
            timestamp=base_anomaly.timestamp,
            value=base_anomaly.value,
            anomaly_type=AnomalyType.COLLECTIVE,
            severity=final_severity,
            confidence=combined_confidence,
            anomaly_score=combined_score,
            expected_range=base_anomaly.expected_range,
            contributing_factors=list(set(all_factors)),
            metadata={
                'detection_method': 'consolidated',
                'num_detections': len(anomalies),
                'detection_methods': [a.metadata.get('detection_method', 'unknown') for a in anomalies]
            }
        )
    
    async def get_training_data(self) -> pd.DataFrame:
        """Get historical data for model training"""
        try:
            # Get last 30 days of data for training
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)
            
            query = """
            SELECT metric_name, site_id, metric_value, timestamp, quality_score
            FROM network_metrics
            WHERE timestamp >= $1 AND timestamp <= $2
                AND quality_score >= 0.7
                AND metric_name = ANY($3)
            ORDER BY timestamp
            """
            
            rows = await self.db.fetch_all(
                query, start_time, end_time, self.monitored_metrics
            )
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = pd.DataFrame([dict(row) for row in rows])
            return data
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()
    
    async def get_metric_history(self, metric_name: str, site_id: str, hours: int = 24) -> pd.DataFrame:
        """Get historical data for a specific metric and site"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            query = """
            SELECT metric_value, timestamp, quality_score
            FROM network_metrics
            WHERE metric_name = $1 AND site_id = $2
                AND timestamp >= $3 AND timestamp <= $4
                AND quality_score >= 0.5
            ORDER BY timestamp
            """
            
            rows = await self.db.fetch_all(
                query, metric_name, site_id, start_time, end_time
            )
            
            if not rows:
                return pd.DataFrame()
            
            return pd.DataFrame([dict(row) for row in rows])
            
        except Exception as e:
            logger.error(f"Error getting metric history: {e}")
            return pd.DataFrame()
    
    async def get_related_metrics(self, site_id: str, timestamp: datetime) -> List[Dict]:
        """Get related metrics for contextual analysis"""
        try:
            # Get metrics within 5 minutes of timestamp
            start_time = timestamp - timedelta(minutes=5)
            end_time = timestamp + timedelta(minutes=5)
            
            query = """
            SELECT metric_name, metric_value, timestamp
            FROM network_metrics
            WHERE site_id = $1
                AND timestamp >= $2 AND timestamp <= $3
                AND metric_name = ANY($4)
            ORDER BY timestamp DESC
            """
            
            rows = await self.db.fetch_all(
                query, site_id, start_time, end_time, self.monitored_metrics
            )
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting related metrics: {e}")
            return []
    
    async def store_anomaly(self, anomaly: AnomalyResult):
        """Store anomaly detection result"""
        try:
            query = """
            INSERT INTO anomaly_detections (
                metric_name, site_id, anomaly_score, anomaly_type,
                detected_at, value, expected_range, confidence,
                contributing_factors, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """
            
            await self.db.execute(
                query,
                anomaly.metric_name,
                anomaly.site_id,
                anomaly.anomaly_score,
                anomaly.anomaly_type.value,
                anomaly.timestamp,
                anomaly.value,
                json.dumps(anomaly.expected_range),
                anomaly.confidence,
                anomaly.contributing_factors,
                json.dumps(anomaly.metadata)
            )
            
            logger.info(f"Stored anomaly for {anomaly.metric_name} at {anomaly.site_id}")
            
        except Exception as e:
            logger.error(f"Error storing anomaly: {e}")
    
    async def send_anomaly_alert(self, anomaly: AnomalyResult):
        """Send anomaly alert to message queue"""
        try:
            alert_message = {
                'type': 'anomaly_detected',
                'timestamp': anomaly.timestamp.isoformat(),
                'metric_name': anomaly.metric_name,
                'site_id': anomaly.site_id,
                'severity': anomaly.severity.value,
                'confidence': anomaly.confidence,
                'anomaly_score': anomaly.anomaly_score,
                'value': anomaly.value,
                'expected_range': anomaly.expected_range,
                'contributing_factors': anomaly.contributing_factors,
                'anomaly_type': anomaly.anomaly_type.value
            }
            
            await self.producer.send_message('aiops.anomalies', alert_message)
            logger.info(f"Sent anomaly alert for {anomaly.metric_name} at {anomaly.site_id}")
            
        except Exception as e:
            logger.error(f"Error sending anomaly alert: {e}")
    
    async def setup_metric_subscriptions(self):
        """Setup subscriptions to metric streams"""
        # This would integrate with the message queue to receive real-time metrics
        pass
    
    async def load_models_from_cache(self) -> bool:
        """Load trained models from cache"""
        try:
            # Try to load models from cache
            models_data = await self.cache.get("anomaly_models")
            if models_data:
                # Deserialize models
                logger.info("Loading models from cache")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading models from cache: {e}")
            return False
    
    async def save_models_to_cache(self):
        """Save trained models to cache"""
        try:
            # Serialize and save models to cache
            await self.cache.set("anomaly_models", {}, expire=86400)  # 24 hours
            logger.info("Saved models to cache")
        except Exception as e:
            logger.error(f"Error saving models to cache: {e}")
    
    async def setup_statistical_models(self):
        """Setup fallback statistical models"""
        logger.info("Setting up statistical fallback models")
        # Basic statistical thresholds as fallback


# Main worker function
async def main():
    """Main anomaly detection worker"""
    detector = AnomalyDetectionEngine()
    
    try:
        await detector.initialize()
        
        # Main processing loop
        while True:
            try:
                # This would normally process metrics from message queue
                logger.info("Anomaly detection engine running...")
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in main processing loop: {e}")
                await asyncio.sleep(30)
                
    except KeyboardInterrupt:
        logger.info("Anomaly detection worker stopping...")
    except Exception as e:
        logger.error(f"Fatal error in anomaly detection worker: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())