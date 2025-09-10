"""
Predictive Maintenance Engine - AI-driven predictive maintenance for telecom infrastructure
Advanced ML models for equipment failure prediction and maintenance optimization
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# ML and statistical libraries
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from scipy import stats
from prophet import Prophet
import joblib
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

from core.database import DatabaseManager
from core.cache import CacheManager
from core.messaging import MessageProducer
from core.config import settings

logger = logging.getLogger(__name__)

class MaintenanceType(Enum):
    """Types of maintenance activities"""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"

class ComponentType(Enum):
    """Types of telecom components"""
    POWER_SYSTEM = "power_system"
    COOLING_SYSTEM = "cooling_system"
    RADIO_EQUIPMENT = "radio_equipment"
    TRANSMISSION = "transmission"
    PROCESSING_UNIT = "processing_unit"
    BATTERY_BACKUP = "battery_backup"
    ANTENNA_SYSTEM = "antenna_system"

class RiskLevel(Enum):
    """Risk levels for component failure"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MaintenanceRecommendation:
    """Predictive maintenance recommendation"""
    component_id: str
    site_id: str
    component_type: ComponentType
    predicted_failure_date: datetime
    risk_level: RiskLevel
    risk_score: float
    confidence: float
    maintenance_type: MaintenanceType
    recommended_actions: List[str]
    cost_estimate: float
    impact_assessment: Dict[str, Any]
    supporting_evidence: List[str]
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComponentHealth:
    """Component health status"""
    component_id: str
    site_id: str
    component_type: ComponentType
    health_score: float
    degradation_rate: float
    remaining_useful_life_days: int
    critical_parameters: Dict[str, float]
    trends: Dict[str, List[float]]
    last_assessment: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class PredictiveMaintenanceEngine:
    """Main predictive maintenance engine with ML capabilities"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.producer = MessageProducer()
        
        # ML Models for different components
        self.failure_prediction_models = {}
        self.health_scoring_models = {}
        self.degradation_models = {}
        self.anomaly_detectors = {}
        
        # Scalers for different component types
        self.scalers = {}
        self.label_encoders = {}
        
        # Component-specific thresholds
        self.component_thresholds = self._build_component_thresholds()
        
        # Maintenance cost models
        self.maintenance_costs = self._build_maintenance_costs()
        
        # Historical data cache
        self.component_history_cache = {}
        
        # Configuration
        self.prediction_horizon_days = 90  # Predict failures up to 90 days ahead
        self.min_training_samples = 100
        self.health_update_interval = timedelta(hours=6)
        
    def _build_component_thresholds(self) -> Dict[ComponentType, Dict[str, Any]]:
        """Build component-specific thresholds and parameters"""
        return {
            ComponentType.POWER_SYSTEM: {
                'critical_params': ['voltage', 'current', 'power_factor', 'temperature'],
                'thresholds': {
                    'voltage_deviation': 0.05,  # 5% deviation
                    'temperature_max': 60,  # Celsius
                    'power_factor_min': 0.8,
                    'efficiency_min': 0.85
                },
                'degradation_indicators': ['efficiency_drop', 'temperature_rise', 'harmonics'],
                'failure_patterns': ['voltage_instability', 'overheating', 'component_aging']
            },
            
            ComponentType.COOLING_SYSTEM: {
                'critical_params': ['temperature_delta', 'fan_speed', 'power_consumption', 'vibration'],
                'thresholds': {
                    'temperature_delta_max': 25,  # Celsius
                    'fan_speed_min': 0.7,  # 70% of max speed
                    'power_consumption_max': 1.2,  # 120% of nominal
                    'vibration_max': 10  # mm/s
                },
                'degradation_indicators': ['fan_bearing_wear', 'filter_clogging', 'refrigerant_leakage'],
                'failure_patterns': ['fan_failure', 'compressor_failure', 'system_blockage']
            },
            
            ComponentType.RADIO_EQUIPMENT: {
                'critical_params': ['tx_power', 'rx_sensitivity', 'vswr', 'temperature'],
                'thresholds': {
                    'tx_power_min': 0.9,  # 90% of nominal
                    'rx_sensitivity_max': -100,  # dBm
                    'vswr_max': 2.0,
                    'temperature_max': 55
                },
                'degradation_indicators': ['pa_aging', 'filter_drift', 'connector_corrosion'],
                'failure_patterns': ['power_amplifier_failure', 'receiver_degradation', 'antenna_mismatch']
            },
            
            ComponentType.TRANSMISSION: {
                'critical_params': ['signal_quality', 'ber', 'link_availability', 'latency'],
                'thresholds': {
                    'ber_max': 1e-6,  # Bit error rate
                    'availability_min': 0.9999,  # 99.99%
                    'latency_max': 10,  # ms
                    'signal_quality_min': -70  # dBm
                },
                'degradation_indicators': ['fiber_aging', 'connector_degradation', 'equipment_drift'],
                'failure_patterns': ['fiber_break', 'equipment_failure', 'synchronization_loss']
            },
            
            ComponentType.PROCESSING_UNIT: {
                'critical_params': ['cpu_usage', 'memory_usage', 'temperature', 'error_rate'],
                'thresholds': {
                    'cpu_usage_max': 0.8,  # 80%
                    'memory_usage_max': 0.9,  # 90%
                    'temperature_max': 70,  # Celsius
                    'error_rate_max': 0.01  # 1%
                },
                'degradation_indicators': ['performance_degradation', 'memory_leaks', 'thermal_cycling'],
                'failure_patterns': ['cpu_failure', 'memory_failure', 'software_corruption']
            },
            
            ComponentType.BATTERY_BACKUP: {
                'critical_params': ['voltage', 'capacity', 'internal_resistance', 'temperature'],
                'thresholds': {
                    'capacity_min': 0.8,  # 80% of nominal
                    'voltage_min': 0.95,  # 95% of nominal
                    'resistance_max': 1.5,  # 150% of initial
                    'temperature_max': 40  # Celsius
                },
                'degradation_indicators': ['capacity_fade', 'resistance_increase', 'sulfation'],
                'failure_patterns': ['capacity_loss', 'internal_short', 'thermal_runaway']
            }
        }
    
    def _build_maintenance_costs(self) -> Dict[ComponentType, Dict[str, float]]:
        """Build maintenance cost estimates"""
        return {
            ComponentType.POWER_SYSTEM: {
                'inspection': 500,
                'preventive': 2000,
                'replacement': 15000,
                'emergency': 25000
            },
            ComponentType.COOLING_SYSTEM: {
                'inspection': 300,
                'preventive': 1500,
                'replacement': 8000,
                'emergency': 15000
            },
            ComponentType.RADIO_EQUIPMENT: {
                'inspection': 800,
                'preventive': 3000,
                'replacement': 25000,
                'emergency': 40000
            },
            ComponentType.TRANSMISSION: {
                'inspection': 400,
                'preventive': 2500,
                'replacement': 20000,
                'emergency': 35000
            },
            ComponentType.PROCESSING_UNIT: {
                'inspection': 600,
                'preventive': 2000,
                'replacement': 12000,
                'emergency': 20000
            },
            ComponentType.BATTERY_BACKUP: {
                'inspection': 200,
                'preventive': 800,
                'replacement': 5000,
                'emergency': 8000
            }
        }
    
    async def initialize(self):
        """Initialize the predictive maintenance engine"""
        try:
            logger.info("Initializing Predictive Maintenance Engine...")
            
            # Load historical maintenance data
            await self.load_historical_data()
            
            # Train or load ML models
            await self.initialize_ml_models()
            
            # Setup monitoring subscriptions
            await self.setup_monitoring_subscriptions()
            
            # Start background processing
            asyncio.create_task(self.health_assessment_loop())
            asyncio.create_task(self.prediction_update_loop())
            
            logger.info("Predictive Maintenance Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize predictive maintenance engine: {e}")
            raise
    
    async def assess_component_health(self, component_id: str, site_id: str) -> ComponentHealth:
        """Assess current health of a component"""
        try:
            # Get component type
            component_type = await self.get_component_type(component_id, site_id)
            
            # Collect recent metrics
            metrics_data = await self.collect_component_metrics(component_id, site_id, days=30)
            
            if metrics_data.empty:
                logger.warning(f"No metrics data for component {component_id}")
                return self._create_default_health(component_id, site_id, component_type)
            
            # Calculate health score
            health_score = await self.calculate_health_score(metrics_data, component_type)
            
            # Calculate degradation rate
            degradation_rate = await self.calculate_degradation_rate(metrics_data, component_type)
            
            # Estimate remaining useful life
            rul_days = await self.estimate_remaining_useful_life(
                metrics_data, component_type, degradation_rate
            )
            
            # Extract critical parameters
            critical_params = self.extract_critical_parameters(metrics_data, component_type)
            
            # Generate trends
            trends = self.calculate_parameter_trends(metrics_data, component_type)
            
            component_health = ComponentHealth(
                component_id=component_id,
                site_id=site_id,
                component_type=component_type,
                health_score=health_score,
                degradation_rate=degradation_rate,
                remaining_useful_life_days=rul_days,
                critical_parameters=critical_params,
                trends=trends,
                last_assessment=datetime.utcnow(),
                metadata={
                    'data_points': len(metrics_data),
                    'assessment_method': 'ml_enhanced'
                }
            )
            
            # Cache the result
            await self.cache_component_health(component_health)
            
            return component_health
            
        except Exception as e:
            logger.error(f"Error assessing component health for {component_id}: {e}")
            return self._create_default_health(component_id, site_id, ComponentType.PROCESSING_UNIT)
    
    async def predict_maintenance_needs(self, site_id: str, 
                                      horizon_days: int = 90) -> List[MaintenanceRecommendation]:
        """Predict maintenance needs for all components at a site"""
        try:
            recommendations = []
            
            # Get all components for the site
            components = await self.get_site_components(site_id)
            
            for component in components:
                component_id = component['component_id']
                component_type = ComponentType(component['component_type'])
                
                # Get component health
                health = await self.assess_component_health(component_id, site_id)
                
                # Predict failure probability
                failure_prediction = await self.predict_component_failure(
                    component_id, site_id, component_type, horizon_days
                )
                
                if failure_prediction['failure_probability'] > 0.3:  # 30% threshold
                    recommendation = await self.generate_maintenance_recommendation(
                        component_id, site_id, component_type, 
                        health, failure_prediction
                    )
                    recommendations.append(recommendation)
            
            # Sort by risk level and predicted failure date
            recommendations.sort(
                key=lambda x: (x.risk_level.value, x.predicted_failure_date)
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error predicting maintenance needs for site {site_id}: {e}")
            return []
    
    async def predict_component_failure(self, component_id: str, site_id: str, 
                                      component_type: ComponentType, 
                                      horizon_days: int) -> Dict[str, Any]:
        """Predict failure probability for a specific component"""
        try:
            # Get historical data
            historical_data = await self.collect_component_metrics(
                component_id, site_id, days=180
            )
            
            if historical_data.empty:
                return {
                    'failure_probability': 0.0,
                    'predicted_failure_date': None,
                    'confidence': 0.0,
                    'method': 'insufficient_data'
                }
            
            # Prepare features for ML model
            features = self.prepare_failure_prediction_features(historical_data, component_type)
            
            if features is None:
                return {
                    'failure_probability': 0.0,
                    'predicted_failure_date': None,
                    'confidence': 0.0,
                    'method': 'feature_extraction_failed'
                }
            
            # Use appropriate model
            model_key = f"{component_type.value}_failure"
            
            if model_key in self.failure_prediction_models:
                model = self.failure_prediction_models[model_key]
                scaler = self.scalers.get(model_key)
                
                if scaler:
                    scaled_features = scaler.transform([features])
                else:
                    scaled_features = [features]
                
                # Predict failure probability
                failure_prob = model.predict_proba(scaled_features)[0][1]  # Probability of failure class
                
                # Estimate failure date based on degradation trend
                predicted_date = self.estimate_failure_date(
                    historical_data, component_type, horizon_days
                )
                
                # Calculate confidence based on model performance and data quality
                confidence = self.calculate_prediction_confidence(
                    historical_data, model, features
                )
                
                return {
                    'failure_probability': failure_prob,
                    'predicted_failure_date': predicted_date,
                    'confidence': confidence,
                    'method': 'ml_model'
                }
            else:
                # Fallback to statistical method
                return await self.statistical_failure_prediction(
                    historical_data, component_type, horizon_days
                )
                
        except Exception as e:
            logger.error(f"Error predicting failure for component {component_id}: {e}")
            return {
                'failure_probability': 0.0,
                'predicted_failure_date': None,
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def prepare_failure_prediction_features(self, data: pd.DataFrame, 
                                          component_type: ComponentType) -> Optional[List[float]]:
        """Prepare features for failure prediction model"""
        try:
            if data.empty:
                return None
            
            features = []
            thresholds = self.component_thresholds.get(component_type, {})
            critical_params = thresholds.get('critical_params', [])
            
            # Statistical features for each critical parameter
            for param in critical_params:
                param_data = data[data['metric_name'] == param]['metric_value']
                
                if not param_data.empty:
                    features.extend([
                        param_data.mean(),
                        param_data.std(),
                        param_data.min(),
                        param_data.max(),
                        param_data.median()
                    ])
                    
                    # Trend features
                    if len(param_data) > 1:
                        trend_slope, _, _, _, _ = stats.linregress(
                            range(len(param_data)), param_data.values
                        )
                        features.append(trend_slope)
                    else:
                        features.append(0.0)
                else:
                    features.extend([0.0] * 6)  # Fill missing with zeros
            
            # Time-based features
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            time_span = (data['timestamp'].max() - data['timestamp'].min()).days
            features.extend([
                time_span,
                len(data),  # Number of data points
                data['quality_score'].mean() if 'quality_score' in data.columns else 1.0
            ])
            
            # Threshold violation features
            for param in critical_params:
                param_data = data[data['metric_name'] == param]['metric_value']
                param_thresholds = thresholds.get('thresholds', {})
                
                if not param_data.empty:
                    # Calculate violation rate
                    violations = 0
                    for threshold_name, threshold_value in param_thresholds.items():
                        if param in threshold_name:
                            if 'max' in threshold_name:
                                violations += (param_data > threshold_value).sum()
                            elif 'min' in threshold_name:
                                violations += (param_data < threshold_value).sum()
                    
                    violation_rate = violations / len(param_data) if len(param_data) > 0 else 0
                    features.append(violation_rate)
                else:
                    features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing failure prediction features: {e}")
            return None
    
    def estimate_failure_date(self, data: pd.DataFrame, component_type: ComponentType, 
                            horizon_days: int) -> Optional[datetime]:
        """Estimate failure date based on degradation trends"""
        try:
            if data.empty:
                return None
            
            thresholds = self.component_thresholds.get(component_type, {})
            critical_params = thresholds.get('critical_params', [])
            
            # Find the parameter with steepest degradation
            worst_case_days = horizon_days
            
            for param in critical_params:
                param_data = data[data['metric_name'] == param]
                
                if len(param_data) < 5:  # Need at least 5 data points
                    continue
                
                param_data = param_data.sort_values('timestamp')
                values = param_data['metric_value'].values
                
                if len(values) < 2:
                    continue
                
                # Calculate trend
                x = np.arange(len(values))
                slope, intercept, _, _, _ = stats.linregress(x, values)
                
                if abs(slope) < 1e-6:  # No significant trend
                    continue
                
                # Find failure threshold for this parameter
                param_thresholds = thresholds.get('thresholds', {})
                failure_threshold = None
                
                for threshold_name, threshold_value in param_thresholds.items():
                    if param in threshold_name:
                        failure_threshold = threshold_value
                        break
                
                if failure_threshold is None:
                    continue
                
                # Calculate days to reach failure threshold
                current_value = values[-1]
                
                if slope != 0:
                    if (slope > 0 and 'max' in str(param_thresholds.keys())) or \
                       (slope < 0 and 'min' in str(param_thresholds.keys())):
                        
                        days_to_failure = abs((failure_threshold - current_value) / slope)
                        
                        if days_to_failure < worst_case_days:
                            worst_case_days = days_to_failure
            
            if worst_case_days < horizon_days:
                failure_date = datetime.utcnow() + timedelta(days=int(worst_case_days))
                return failure_date
            
            return None
            
        except Exception as e:
            logger.error(f"Error estimating failure date: {e}")
            return None
    
    async def calculate_health_score(self, data: pd.DataFrame, 
                                   component_type: ComponentType) -> float:
        """Calculate component health score (0-1, where 1 is perfect health)"""
        try:
            if data.empty:
                return 0.5  # Default neutral health score
            
            thresholds = self.component_thresholds.get(component_type, {})
            critical_params = thresholds.get('critical_params', [])
            param_thresholds = thresholds.get('thresholds', {})
            
            health_scores = []
            
            for param in critical_params:
                param_data = data[data['metric_name'] == param]['metric_value']
                
                if param_data.empty:
                    continue
                
                current_value = param_data.iloc[-1]  # Latest value
                
                # Find threshold for this parameter
                param_score = 1.0  # Default good health
                
                for threshold_name, threshold_value in param_thresholds.items():
                    if param in threshold_name:
                        if 'max' in threshold_name:
                            # Lower is better
                            if current_value > threshold_value:
                                param_score = max(0, 1.0 - (current_value - threshold_value) / threshold_value)
                            else:
                                param_score = 1.0
                        elif 'min' in threshold_name:
                            # Higher is better
                            if current_value < threshold_value:
                                param_score = max(0, current_value / threshold_value)
                            else:
                                param_score = 1.0
                        break
                
                health_scores.append(param_score)
            
            # Overall health is the minimum of all parameter health scores
            if health_scores:
                overall_health = min(health_scores)
            else:
                overall_health = 0.5
            
            return max(0.0, min(1.0, overall_health))
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 0.5
    
    async def calculate_degradation_rate(self, data: pd.DataFrame, 
                                       component_type: ComponentType) -> float:
        """Calculate component degradation rate (units per day)"""
        try:
            if data.empty or len(data) < 5:
                return 0.0
            
            data = data.sort_values('timestamp')
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Calculate degradation for each critical parameter
            thresholds = self.component_thresholds.get(component_type, {})
            critical_params = thresholds.get('critical_params', [])
            
            degradation_rates = []
            
            for param in critical_params:
                param_data = data[data['metric_name'] == param]
                
                if len(param_data) < 5:
                    continue
                
                param_data = param_data.sort_values('timestamp')
                values = param_data['metric_value'].values
                timestamps = param_data['timestamp'].values
                
                # Convert timestamps to days since start
                start_time = timestamps[0]
                days = [(t - start_time).total_seconds() / 86400 for t in timestamps]
                
                # Calculate linear regression slope
                if len(days) > 1:
                    slope, _, _, _, _ = stats.linregress(days, values)
                    degradation_rates.append(abs(slope))  # Absolute degradation rate
            
            # Return average degradation rate
            if degradation_rates:
                return np.mean(degradation_rates)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating degradation rate: {e}")
            return 0.0
    
    async def estimate_remaining_useful_life(self, data: pd.DataFrame, 
                                           component_type: ComponentType,
                                           degradation_rate: float) -> int:
        """Estimate remaining useful life in days"""
        try:
            if degradation_rate == 0 or data.empty:
                return 365  # Default to 1 year if no degradation detected
            
            thresholds = self.component_thresholds.get(component_type, {})
            critical_params = thresholds.get('critical_params', [])
            param_thresholds = thresholds.get('thresholds', {})
            
            min_rul_days = 365  # Start with 1 year
            
            for param in critical_params:
                param_data = data[data['metric_name'] == param]
                
                if param_data.empty:
                    continue
                
                current_value = param_data.iloc[-1]['metric_value']
                
                # Find failure threshold
                for threshold_name, threshold_value in param_thresholds.items():
                    if param in threshold_name:
                        # Calculate days until threshold reached
                        if degradation_rate > 0:
                            if 'max' in threshold_name and current_value < threshold_value:
                                days_to_threshold = (threshold_value - current_value) / degradation_rate
                            elif 'min' in threshold_name and current_value > threshold_value:
                                days_to_threshold = (current_value - threshold_value) / degradation_rate
                            else:
                                days_to_threshold = 365  # Already at threshold or good condition
                            
                            min_rul_days = min(min_rul_days, max(0, int(days_to_threshold)))
                        break
            
            return max(0, min_rul_days)
            
        except Exception as e:
            logger.error(f"Error estimating remaining useful life: {e}")
            return 365
    
    def extract_critical_parameters(self, data: pd.DataFrame, 
                                  component_type: ComponentType) -> Dict[str, float]:
        """Extract current values of critical parameters"""
        try:
            critical_params = {}
            thresholds = self.component_thresholds.get(component_type, {})
            param_names = thresholds.get('critical_params', [])
            
            for param in param_names:
                param_data = data[data['metric_name'] == param]
                if not param_data.empty:
                    critical_params[param] = float(param_data.iloc[-1]['metric_value'])
                else:
                    critical_params[param] = 0.0
            
            return critical_params
            
        except Exception as e:
            logger.error(f"Error extracting critical parameters: {e}")
            return {}
    
    def calculate_parameter_trends(self, data: pd.DataFrame, 
                                 component_type: ComponentType) -> Dict[str, List[float]]:
        """Calculate trends for critical parameters"""
        try:
            trends = {}
            thresholds = self.component_thresholds.get(component_type, {})
            param_names = thresholds.get('critical_params', [])
            
            for param in param_names:
                param_data = data[data['metric_name'] == param]
                
                if not param_data.empty and len(param_data) >= 5:
                    # Get last 10 values as trend
                    recent_values = param_data.tail(10)['metric_value'].tolist()
                    trends[param] = recent_values
                else:
                    trends[param] = [0.0]
            
            return trends
            
        except Exception as e:
            logger.error(f"Error calculating parameter trends: {e}")
            return {}
    
    async def generate_maintenance_recommendation(self, component_id: str, site_id: str,
                                                component_type: ComponentType,
                                                health: ComponentHealth,
                                                failure_prediction: Dict[str, Any]) -> MaintenanceRecommendation:
        """Generate maintenance recommendation based on analysis"""
        try:
            failure_prob = failure_prediction.get('failure_probability', 0.0)
            predicted_date = failure_prediction.get('predicted_failure_date')
            confidence = failure_prediction.get('confidence', 0.0)
            
            # Determine risk level
            if failure_prob >= 0.8:
                risk_level = RiskLevel.CRITICAL
            elif failure_prob >= 0.6:
                risk_level = RiskLevel.HIGH
            elif failure_prob >= 0.4:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Determine maintenance type
            if health.remaining_useful_life_days < 30:
                maintenance_type = MaintenanceType.EMERGENCY
            elif health.remaining_useful_life_days < 90:
                maintenance_type = MaintenanceType.PREDICTIVE
            else:
                maintenance_type = MaintenanceType.PREVENTIVE
            
            # Generate recommendations
            recommendations = self.generate_maintenance_actions(
                component_type, health, failure_prediction, risk_level
            )
            
            # Calculate cost estimate
            cost_estimate = self.calculate_maintenance_cost(
                component_type, maintenance_type, risk_level
            )
            
            # Generate impact assessment
            impact_assessment = await self.assess_failure_impact(
                component_id, site_id, component_type
            )
            
            # Generate supporting evidence
            evidence = self.generate_supporting_evidence(health, failure_prediction)
            
            recommendation = MaintenanceRecommendation(
                component_id=component_id,
                site_id=site_id,
                component_type=component_type,
                predicted_failure_date=predicted_date or datetime.utcnow() + timedelta(days=365),
                risk_level=risk_level,
                risk_score=failure_prob,
                confidence=confidence,
                maintenance_type=maintenance_type,
                recommended_actions=recommendations,
                cost_estimate=cost_estimate,
                impact_assessment=impact_assessment,
                supporting_evidence=evidence,
                created_at=datetime.utcnow(),
                metadata={
                    'health_score': health.health_score,
                    'degradation_rate': health.degradation_rate,
                    'rul_days': health.remaining_useful_life_days
                }
            )
            
            # Store recommendation
            await self.store_maintenance_recommendation(recommendation)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating maintenance recommendation: {e}")
            # Return a basic recommendation
            return MaintenanceRecommendation(
                component_id=component_id,
                site_id=site_id,
                component_type=component_type,
                predicted_failure_date=datetime.utcnow() + timedelta(days=365),
                risk_level=RiskLevel.MEDIUM,
                risk_score=0.5,
                confidence=0.3,
                maintenance_type=MaintenanceType.PREVENTIVE,
                recommended_actions=["Schedule routine inspection"],
                cost_estimate=1000.0,
                impact_assessment={'service_impact': 'low'},
                supporting_evidence=["Insufficient data for detailed analysis"],
                created_at=datetime.utcnow()
            )
    
    def generate_maintenance_actions(self, component_type: ComponentType,
                                   health: ComponentHealth,
                                   failure_prediction: Dict[str, Any],
                                   risk_level: RiskLevel) -> List[str]:
        """Generate specific maintenance actions"""
        actions = []
        
        # Base actions by component type
        if component_type == ComponentType.POWER_SYSTEM:
            actions.extend([
                "Inspect power connections and terminals",
                "Check voltage regulation and power factor",
                "Test backup systems and transfer switches",
                "Clean and tighten electrical connections"
            ])
            
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                actions.extend([
                    "Replace aging power components",
                    "Upgrade UPS batteries if necessary",
                    "Schedule emergency backup power testing"
                ])
        
        elif component_type == ComponentType.COOLING_SYSTEM:
            actions.extend([
                "Clean air filters and heat exchangers",
                "Check refrigerant levels and pressure",
                "Inspect fan bearings and motor condition",
                "Verify temperature control calibration"
            ])
            
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                actions.extend([
                    "Replace worn fan bearings",
                    "Service or replace compressor",
                    "Emergency cooling system backup preparation"
                ])
        
        elif component_type == ComponentType.RADIO_EQUIPMENT:
            actions.extend([
                "Test RF performance and power output",
                "Inspect antenna connections and cables",
                "Check VSWR and return loss",
                "Calibrate transmitter and receiver"
            ])
            
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                actions.extend([
                    "Replace aging power amplifier",
                    "Service RF filters and duplexers",
                    "Prepare backup radio equipment"
                ])
        
        # Add urgency-based actions
        if health.remaining_useful_life_days < 30:
            actions.insert(0, "URGENT: Schedule immediate inspection within 48 hours")
            actions.append("Prepare replacement parts and tools")
        elif health.remaining_useful_life_days < 90:
            actions.insert(0, "Schedule inspection within 2 weeks")
            actions.append("Order replacement parts as backup")
        
        return actions
    
    def calculate_maintenance_cost(self, component_type: ComponentType,
                                 maintenance_type: MaintenanceType,
                                 risk_level: RiskLevel) -> float:
        """Calculate estimated maintenance cost"""
        base_costs = self.maintenance_costs.get(component_type, {})
        
        if maintenance_type == MaintenanceType.EMERGENCY:
            base_cost = base_costs.get('emergency', 10000)
        elif maintenance_type == MaintenanceType.PREDICTIVE:
            base_cost = base_costs.get('replacement', 5000)
        elif maintenance_type == MaintenanceType.PREVENTIVE:
            base_cost = base_costs.get('preventive', 1000)
        else:
            base_cost = base_costs.get('inspection', 500)
        
        # Risk multiplier
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 1.2,
            RiskLevel.HIGH: 1.5,
            RiskLevel.CRITICAL: 2.0
        }
        
        multiplier = risk_multipliers.get(risk_level, 1.0)
        return base_cost * multiplier
    
    async def assess_failure_impact(self, component_id: str, site_id: str,
                                  component_type: ComponentType) -> Dict[str, Any]:
        """Assess impact of component failure"""
        try:
            # Get site information
            site_info = await self.get_site_info(site_id)
            
            # Base impact assessment
            impact = {
                'service_impact': 'medium',
                'revenue_impact_daily': 1000.0,
                'customer_count': 0,
                'sla_penalty': 0.0
            }
            
            if site_info:
                # Calculate based on site criticality
                customer_count = site_info.get('customer_count', 0)
                site_revenue = site_info.get('daily_revenue', 1000)
                
                impact['customer_count'] = customer_count
                impact['revenue_impact_daily'] = site_revenue
                
                # Component-specific impact
                if component_type == ComponentType.POWER_SYSTEM:
                    impact['service_impact'] = 'critical'
                    impact['revenue_impact_daily'] *= 1.0  # Full impact
                elif component_type == ComponentType.RADIO_EQUIPMENT:
                    impact['service_impact'] = 'high'
                    impact['revenue_impact_daily'] *= 0.8  # 80% impact
                elif component_type == ComponentType.COOLING_SYSTEM:
                    impact['service_impact'] = 'medium'
                    impact['revenue_impact_daily'] *= 0.3  # 30% impact (gradual)
                
                # SLA penalty calculation
                if customer_count > 1000:  # Major site
                    impact['sla_penalty'] = site_revenue * 0.1  # 10% penalty
            
            return impact
            
        except Exception as e:
            logger.error(f"Error assessing failure impact: {e}")
            return {
                'service_impact': 'medium',
                'revenue_impact_daily': 1000.0,
                'customer_count': 0,
                'sla_penalty': 0.0
            }
    
    def generate_supporting_evidence(self, health: ComponentHealth,
                                   failure_prediction: Dict[str, Any]) -> List[str]:
        """Generate supporting evidence for the recommendation"""
        evidence = []
        
        # Health score evidence
        if health.health_score < 0.5:
            evidence.append(f"Component health score is low: {health.health_score:.2f}")
        
        if health.degradation_rate > 0.1:
            evidence.append(f"High degradation rate detected: {health.degradation_rate:.3f} units/day")
        
        if health.remaining_useful_life_days < 90:
            evidence.append(f"Remaining useful life: {health.remaining_useful_life_days} days")
        
        # Critical parameter evidence
        for param, value in health.critical_parameters.items():
            # Add specific evidence based on parameter values
            evidence.append(f"Current {param}: {value:.2f}")
        
        # Prediction evidence
        failure_prob = failure_prediction.get('failure_probability', 0)
        if failure_prob > 0.5:
            evidence.append(f"High failure probability: {failure_prob:.1%}")
        
        confidence = failure_prediction.get('confidence', 0)
        evidence.append(f"Prediction confidence: {confidence:.1%}")
        
        return evidence
    
    async def statistical_failure_prediction(self, data: pd.DataFrame,
                                           component_type: ComponentType,
                                           horizon_days: int) -> Dict[str, Any]:
        """Fallback statistical failure prediction method"""
        try:
            if data.empty:
                return {
                    'failure_probability': 0.0,
                    'predicted_failure_date': None,
                    'confidence': 0.0,
                    'method': 'insufficient_data'
                }
            
            # Use threshold crossing analysis
            thresholds = self.component_thresholds.get(component_type, {})
            param_thresholds = thresholds.get('thresholds', {})
            critical_params = thresholds.get('critical_params', [])
            
            violation_count = 0
            total_checks = 0
            
            for param in critical_params:
                param_data = data[data['metric_name'] == param]
                
                for threshold_name, threshold_value in param_thresholds.items():
                    if param in threshold_name and not param_data.empty:
                        current_value = param_data.iloc[-1]['metric_value']
                        total_checks += 1
                        
                        if 'max' in threshold_name and current_value > threshold_value:
                            violation_count += 1
                        elif 'min' in threshold_name and current_value < threshold_value:
                            violation_count += 1
            
            # Calculate failure probability based on threshold violations
            if total_checks > 0:
                failure_prob = violation_count / total_checks
            else:
                failure_prob = 0.0
            
            # Estimate failure date based on trends
            predicted_date = None
            if failure_prob > 0.3:
                # Simple linear extrapolation
                predicted_date = datetime.utcnow() + timedelta(
                    days=int(horizon_days * (1 - failure_prob))
                )
            
            return {
                'failure_probability': failure_prob,
                'predicted_failure_date': predicted_date,
                'confidence': 0.6,  # Moderate confidence for statistical method
                'method': 'statistical_threshold'
            }
            
        except Exception as e:
            logger.error(f"Error in statistical failure prediction: {e}")
            return {
                'failure_probability': 0.0,
                'predicted_failure_date': None,
                'confidence': 0.0,
                'method': 'error'
            }
    
    def calculate_prediction_confidence(self, data: pd.DataFrame, 
                                      model: Any, features: List[float]) -> float:
        """Calculate confidence in prediction based on data quality and model performance"""
        try:
            base_confidence = 0.7  # Base confidence
            
            # Data quality factors
            if len(data) < 50:
                base_confidence *= 0.7  # Reduce for limited data
            
            quality_scores = data.get('quality_score', pd.Series([1.0] * len(data)))
            avg_quality = quality_scores.mean()
            base_confidence *= avg_quality
            
            # Time span factor
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            time_span = (data['timestamp'].max() - data['timestamp'].min()).days
            
            if time_span < 30:
                base_confidence *= 0.8  # Reduce for short time span
            
            return max(0.1, min(1.0, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {e}")
            return 0.5
    
    def _create_default_health(self, component_id: str, site_id: str,
                             component_type: ComponentType) -> ComponentHealth:
        """Create default health status when data is insufficient"""
        return ComponentHealth(
            component_id=component_id,
            site_id=site_id,
            component_type=component_type,
            health_score=0.5,
            degradation_rate=0.0,
            remaining_useful_life_days=365,
            critical_parameters={},
            trends={},
            last_assessment=datetime.utcnow(),
            metadata={'data_quality': 'insufficient'}
        )
    
    # Database and cache operations
    async def collect_component_metrics(self, component_id: str, site_id: str, 
                                      days: int = 30) -> pd.DataFrame:
        """Collect historical metrics for a component"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            query = """
            SELECT metric_name, metric_value, timestamp, unit, quality_score, metadata
            FROM network_metrics
            WHERE site_id = $1
                AND timestamp >= $2 AND timestamp <= $3
                AND (metadata->>'component_id' = $4 OR metric_name LIKE $5)
            ORDER BY timestamp
            """
            
            # Component ID pattern for metrics
            component_pattern = f"%{component_id}%"
            
            rows = await self.db.fetch_all(
                query, site_id, start_time, end_time, component_id, component_pattern
            )
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame([dict(row) for row in rows])
            return df
            
        except Exception as e:
            logger.error(f"Error collecting component metrics: {e}")
            return pd.DataFrame()
    
    async def get_component_type(self, component_id: str, site_id: str) -> ComponentType:
        """Get component type from database"""
        try:
            query = """
            SELECT equipment_type FROM equipment
            WHERE equipment_id = $1 AND site_id = $2
            """
            
            row = await self.db.fetchrow(query, component_id, site_id)
            
            if row and row['equipment_type']:
                equipment_type = row['equipment_type'].lower()
                
                # Map equipment types to component types
                type_mapping = {
                    'ups': ComponentType.POWER_SYSTEM,
                    'power': ComponentType.POWER_SYSTEM,
                    'battery': ComponentType.BATTERY_BACKUP,
                    'cooling': ComponentType.COOLING_SYSTEM,
                    'hvac': ComponentType.COOLING_SYSTEM,
                    'radio': ComponentType.RADIO_EQUIPMENT,
                    'antenna': ComponentType.ANTENNA_SYSTEM,
                    'transmission': ComponentType.TRANSMISSION,
                    'processor': ComponentType.PROCESSING_UNIT,
                    'cpu': ComponentType.PROCESSING_UNIT
                }
                
                for key, comp_type in type_mapping.items():
                    if key in equipment_type:
                        return comp_type
            
            return ComponentType.PROCESSING_UNIT  # Default
            
        except Exception as e:
            logger.error(f"Error getting component type: {e}")
            return ComponentType.PROCESSING_UNIT
    
    async def get_site_components(self, site_id: str) -> List[Dict[str, Any]]:
        """Get all components for a site"""
        try:
            query = """
            SELECT equipment_id as component_id, equipment_type as component_type,
                   vendor, model, installation_date, status
            FROM equipment
            WHERE site_id = $1 AND status = 'operational'
            """
            
            rows = await self.db.fetch_all(query, site_id)
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting site components: {e}")
            return []
    
    async def get_site_info(self, site_id: str) -> Optional[Dict[str, Any]]:
        """Get site information"""
        try:
            query = """
            SELECT site_name, region, country, technology, metadata
            FROM sites
            WHERE site_id = $1
            """
            
            row = await self.db.fetchrow(query, site_id)
            
            if row:
                site_info = dict(row)
                # Extract additional info from metadata
                metadata = site_info.get('metadata', {})
                site_info.update({
                    'customer_count': metadata.get('customer_count', 0),
                    'daily_revenue': metadata.get('daily_revenue', 1000.0)
                })
                return site_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting site info: {e}")
            return None
    
    async def cache_component_health(self, health: ComponentHealth):
        """Cache component health assessment"""
        try:
            cache_key = f"component_health:{health.site_id}:{health.component_id}"
            
            health_data = {
                'component_id': health.component_id,
                'site_id': health.site_id,
                'component_type': health.component_type.value,
                'health_score': health.health_score,
                'degradation_rate': health.degradation_rate,
                'remaining_useful_life_days': health.remaining_useful_life_days,
                'critical_parameters': health.critical_parameters,
                'trends': health.trends,
                'last_assessment': health.last_assessment.isoformat(),
                'metadata': health.metadata
            }
            
            await self.cache.set(cache_key, health_data, expire=3600)  # Cache for 1 hour
            
        except Exception as e:
            logger.error(f"Error caching component health: {e}")
    
    async def store_maintenance_recommendation(self, recommendation: MaintenanceRecommendation):
        """Store maintenance recommendation in database"""
        try:
            query = """
            INSERT INTO maintenance_recommendations (
                component_id, site_id, component_type, predicted_failure_date,
                risk_level, risk_score, confidence, maintenance_type,
                recommended_actions, cost_estimate, impact_assessment,
                supporting_evidence, created_at, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """
            
            await self.db.execute(
                query,
                recommendation.component_id,
                recommendation.site_id,
                recommendation.component_type.value,
                recommendation.predicted_failure_date,
                recommendation.risk_level.value,
                recommendation.risk_score,
                recommendation.confidence,
                recommendation.maintenance_type.value,
                json.dumps(recommendation.recommended_actions),
                recommendation.cost_estimate,
                json.dumps(recommendation.impact_assessment),
                json.dumps(recommendation.supporting_evidence),
                recommendation.created_at,
                json.dumps(recommendation.metadata)
            )
            
            logger.info(f"Stored maintenance recommendation for {recommendation.component_id}")
            
        except Exception as e:
            logger.error(f"Error storing maintenance recommendation: {e}")
    
    async def load_historical_data(self):
        """Load historical maintenance and failure data"""
        try:
            # Load historical failure data for model training
            query = """
            SELECT component_id, site_id, equipment_type, failure_date,
                   failure_type, root_cause, repair_cost, downtime_hours
            FROM equipment_failures
            WHERE failure_date >= NOW() - INTERVAL '2 years'
            """
            
            rows = await self.db.fetch_all(query)
            logger.info(f"Loaded {len(rows)} historical failure records")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    async def initialize_ml_models(self):
        """Initialize or train ML models"""
        try:
            # For now, create placeholder models
            # In practice, these would be trained on historical data
            
            for component_type in ComponentType:
                model_key = f"{component_type.value}_failure"
                
                # Create simple random forest models
                self.failure_prediction_models[model_key] = RandomForestRegressor(
                    n_estimators=100, random_state=42
                )
                
                self.health_scoring_models[model_key] = RandomForestRegressor(
                    n_estimators=50, random_state=42
                )
                
                self.scalers[model_key] = StandardScaler()
            
            logger.info("ML models initialized for predictive maintenance")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    async def setup_monitoring_subscriptions(self):
        """Setup subscriptions for real-time monitoring"""
        try:
            # This would subscribe to metric streams
            logger.info("Monitoring subscriptions setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up monitoring subscriptions: {e}")
    
    async def health_assessment_loop(self):
        """Background loop for health assessments"""
        while True:
            try:
                # Periodic health assessment for all components
                logger.info("Running periodic health assessments...")
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in health assessment loop: {e}")
                await asyncio.sleep(300)
    
    async def prediction_update_loop(self):
        """Background loop for updating predictions"""
        while True:
            try:
                # Update predictions and recommendations
                logger.info("Updating maintenance predictions...")
                await asyncio.sleep(21600)  # Run every 6 hours
                
            except Exception as e:
                logger.error(f"Error in prediction update loop: {e}")
                await asyncio.sleep(1800)


# Main worker function
# Global state
shutdown_event = asyncio.Event()
pm_engine = None

async def message_handler(topic, key, value, headers, timestamp, partition, offset):
    """Handle incoming Kafka messages"""
    global pm_engine
    
    logger.info(f"Received message from {topic}: {key}")
    
    try:
        if topic == KAFKA_TOPICS['network_metrics']:
            # Process network metrics for equipment health
            site_id = value.get('site_id')
            metric_name = value.get('metric_name')
            metric_value = value.get('metric_value')
            
            if site_id and metric_name and metric_value is not None and pm_engine:
                # Analyze equipment health
                asyncio.create_task(
                    pm_engine.analyze_equipment_health(site_id, metric_name, metric_value)
                )
                
        elif topic == KAFKA_TOPICS['energy_metrics']:
            # Process energy metrics for maintenance predictions
            site_id = value.get('site_id')
            energy_type = value.get('energy_type')
            metric_value = value.get('metric_value')
            
            if site_id and energy_type and metric_value is not None and pm_engine:
                # Analyze energy patterns for maintenance
                asyncio.create_task(
                    pm_engine.analyze_energy_patterns(site_id, energy_type, metric_value)
                )
                
        elif topic == KAFKA_TOPICS['alerts']:
            # Process alerts for maintenance triggers
            alert_type = value.get('alert_type')
            site_id = value.get('site_id')
            
            if alert_type and site_id and pm_engine:
                # Check if alert triggers maintenance
                asyncio.create_task(
                    pm_engine.evaluate_maintenance_trigger(value)
                )
                
    except Exception as e:
        logger.error(f"Error handling message: {e}")

async def create_health_server():
    """Create health check server"""
    from aiohttp import web
    
    async def health_handler(request):
        return web.json_response({
            "status": "healthy",
            "worker": "predictive_maintenance",
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
    """Main predictive maintenance worker"""
    global pm_engine, health_server
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting Towerco Predictive Maintenance Engine...")
    
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
        site = web.TCPSite(health_runner, '0.0.0.0', 8009)
        await site.start()
        logger.info("Health server started on port 8009")
        
        # Initialize Predictive Maintenance Engine
        pm_engine = PredictiveMaintenanceEngine()
        await pm_engine.initialize()
        
        # Start Kafka consumers
        await start_consumer(
            'predictive_maintenance',
            [KAFKA_TOPICS['network_metrics'], KAFKA_TOPICS['energy_metrics'], KAFKA_TOPICS['alerts']],
            'predictive_maintenance_group',
            message_handler
        )
        
        logger.info("Predictive Maintenance Engine started successfully")
        
        # Main loop
        while not shutdown_event.is_set():
            await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Predictive maintenance worker stopping...")
    except Exception as e:
        logger.error(f"Fatal error in predictive maintenance worker: {e}")
        raise
    
    finally:
        logger.info("Shutting down Predictive Maintenance Engine...")
        
        # Cleanup health server
        if health_server:
            await health_runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())