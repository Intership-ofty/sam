"""
Business Intelligence Engine
Advanced analytics, insights generation, and value optimization
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
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
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

class InsightType(Enum):
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    CAPACITY_PLANNING = "capacity_planning"
    ENERGY_OPTIMIZATION = "energy_optimization"
    SLA_OPTIMIZATION = "sla_optimization"
    RISK_MITIGATION = "risk_mitigation"
    AUTOMATION_OPPORTUNITY = "automation_opportunity"

class RecommendationPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class BusinessInsight:
    id: str
    tenant_id: str
    title: str
    description: str
    insight_type: InsightType
    priority: RecommendationPriority
    confidence_score: float
    potential_savings: float
    implementation_cost: float
    roi_estimate: float
    time_to_value_days: int
    affected_sites: List[str]
    kpis_improved: List[str]
    recommendations: List[str]
    evidence_data: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    status: str = "active"

@dataclass
class ValueMetric:
    metric_name: str
    current_value: float
    target_value: float
    improvement_percentage: float
    financial_impact: float
    time_period_days: int

@dataclass
class OptimizationOpportunity:
    id: str
    name: str
    category: str
    description: str
    potential_savings: float
    implementation_effort: str  # low, medium, high
    risk_level: str  # low, medium, high
    affected_sites: List[str]
    recommended_actions: List[str]

class BusinessIntelligenceEngine:
    """Advanced Business Intelligence and Analytics Engine"""
    
    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        
        # ML Models
        self.cost_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.performance_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.energy_optimizer = RandomForestRegressor(n_estimators=100, random_state=42)
        self.capacity_planner = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Analytics components
        self.site_clusterer = KMeans(n_clusters=5, random_state=42)
        self.anomaly_detector = DBSCAN(eps=0.5, min_samples=5)
        self.pca_transformer = PCA(n_components=10)
        
        # Configuration
        self.config = {
            'analysis_lookback_days': 90,
            'prediction_horizon_days': 30,
            'min_confidence_threshold': 0.7,
            'roi_threshold': 1.5,  # Minimum ROI for recommendations
            'update_frequency_hours': 6,
            'max_insights_per_tenant': 50,
        }
        
        # Business rules and thresholds
        self.business_rules = {
            'energy_cost_threshold': 1000,  # USD per site per month
            'downtime_cost_per_hour': 500,  # USD per hour
            'maintenance_cost_threshold': 2000,  # USD per site per month
            'performance_threshold': 95,  # Minimum acceptable performance %
            'sla_breach_cost': 10000,  # USD per breach
        }
    
    async def initialize(self):
        """Initialize Business Intelligence Engine"""
        try:
            # Initialize database connection
            self.db_pool = await asyncpg.create_pool(
                "postgresql://towerco:secure_password@localhost:5432/towerco_aiops",
                min_size=5,
                max_size=20
            )
            
            # Initialize Redis connection
            self.redis_client = await redis.from_url("redis://localhost:6379", decode_responses=True)
            
            # Train ML models with historical data
            await self.train_business_models()
            
            # Initialize site clustering
            await self.perform_site_clustering()
            
            logger.info("Business Intelligence Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Business Intelligence Engine: {e}")
            raise
    
    async def generate_business_insights(self, tenant_id: str) -> List[BusinessInsight]:
        """Generate comprehensive business insights for a tenant"""
        try:
            insights = []
            
            # Cost optimization insights
            cost_insights = await self.analyze_cost_optimization(tenant_id)
            insights.extend(cost_insights)
            
            # Performance improvement insights
            performance_insights = await self.analyze_performance_optimization(tenant_id)
            insights.extend(performance_insights)
            
            # Energy optimization insights
            energy_insights = await self.analyze_energy_optimization(tenant_id)
            insights.extend(energy_insights)
            
            # Capacity planning insights
            capacity_insights = await self.analyze_capacity_planning(tenant_id)
            insights.extend(capacity_insights)
            
            # SLA optimization insights
            sla_insights = await self.analyze_sla_optimization(tenant_id)
            insights.extend(sla_insights)
            
            # Predictive maintenance insights
            maintenance_insights = await self.analyze_predictive_maintenance(tenant_id)
            insights.extend(maintenance_insights)
            
            # Risk mitigation insights
            risk_insights = await self.analyze_risk_mitigation(tenant_id)
            insights.extend(risk_insights)
            
            # Automation opportunities
            automation_insights = await self.identify_automation_opportunities(tenant_id)
            insights.extend(automation_insights)
            
            # Filter and rank insights
            filtered_insights = await self.filter_and_rank_insights(insights)
            
            # Store insights
            await self.store_business_insights(filtered_insights)
            
            logger.info(f"Generated {len(filtered_insights)} business insights for tenant {tenant_id}")
            return filtered_insights
            
        except Exception as e:
            logger.error(f"Error generating business insights: {e}")
            return []
    
    async def analyze_cost_optimization(self, tenant_id: str) -> List[BusinessInsight]:
        """Analyze cost optimization opportunities"""
        insights = []
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get cost data
                cost_data = await conn.fetch("""
                    SELECT 
                        s.site_id, s.name, s.location,
                        AVG(km.energy_cost) as avg_energy_cost,
                        AVG(km.maintenance_cost) as avg_maintenance_cost,
                        AVG(km.operational_cost) as avg_operational_cost,
                        COUNT(ni.id) as incident_count,
                        SUM(EXTRACT(EPOCH FROM (COALESCE(ni.resolution_time, NOW()) - ni.created_at))/3600) as total_downtime_hours
                    FROM sites s
                    LEFT JOIN kpi_metrics km ON s.site_id = km.site_id
                    LEFT JOIN noc_incidents ni ON s.site_id = ni.site_id 
                        AND ni.created_at > NOW() - INTERVAL '90 days'
                    WHERE s.tenant_id = $1
                    AND km.timestamp > NOW() - INTERVAL '90 days'
                    GROUP BY s.site_id, s.name, s.location
                """, tenant_id)
                
                high_cost_sites = []
                total_potential_savings = 0
                
                for site in cost_data:
                    total_cost = (site['avg_energy_cost'] or 0) + (site['avg_maintenance_cost'] or 0) + (site['avg_operational_cost'] or 0)
                    downtime_cost = (site['total_downtime_hours'] or 0) * self.business_rules['downtime_cost_per_hour']
                    
                    if total_cost > 5000 or downtime_cost > 2000:  # High cost threshold
                        high_cost_sites.append(site)
                        
                        # Calculate potential savings
                        if site['avg_energy_cost'] and site['avg_energy_cost'] > self.business_rules['energy_cost_threshold']:
                            energy_savings = site['avg_energy_cost'] * 0.15  # 15% reduction potential
                            total_potential_savings += energy_savings * 12  # Annualized
                        
                        if site['incident_count'] and site['incident_count'] > 10:
                            incident_reduction_savings = site['incident_count'] * 200 * 12  # $200 per incident avoided
                            total_potential_savings += incident_reduction_savings
                
                if high_cost_sites and total_potential_savings > 10000:
                    insight = BusinessInsight(
                        id=f"cost_opt_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                        tenant_id=tenant_id,
                        title="High-Cost Site Optimization Opportunity",
                        description=f"Identified {len(high_cost_sites)} sites with elevated operational costs that can be optimized through targeted interventions.",
                        insight_type=InsightType.COST_OPTIMIZATION,
                        priority=RecommendationPriority.HIGH if total_potential_savings > 50000 else RecommendationPriority.MEDIUM,
                        confidence_score=0.85,
                        potential_savings=total_potential_savings,
                        implementation_cost=total_potential_savings * 0.2,  # 20% of savings
                        roi_estimate=total_potential_savings / (total_potential_savings * 0.2),
                        time_to_value_days=90,
                        affected_sites=[site['site_id'] for site in high_cost_sites],
                        kpis_improved=['operational_cost', 'energy_efficiency', 'incident_rate'],
                        recommendations=[
                            "Implement energy optimization protocols",
                            "Deploy predictive maintenance schedules",
                            "Optimize operational procedures",
                            "Consolidate vendor contracts",
                            "Implement automated monitoring"
                        ],
                        evidence_data={
                            'high_cost_sites': len(high_cost_sites),
                            'average_monthly_savings': total_potential_savings / 12,
                            'cost_breakdown': [dict(site) for site in high_cost_sites[:5]]
                        },
                        created_at=datetime.utcnow()
                    )
                    insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error analyzing cost optimization: {e}")
        
        return insights
    
    async def analyze_performance_optimization(self, tenant_id: str) -> List[BusinessInsight]:
        """Analyze performance optimization opportunities"""
        insights = []
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get performance data
                performance_data = await conn.fetch("""
                    SELECT 
                        s.site_id, s.name,
                        AVG(km.availability) as avg_availability,
                        AVG(km.network_performance) as avg_network_performance,
                        AVG(km.uptime_percentage) as avg_uptime,
                        STDDEV(km.availability) as availability_variance
                    FROM sites s
                    JOIN kpi_metrics km ON s.site_id = km.site_id
                    WHERE s.tenant_id = $1
                    AND km.timestamp > NOW() - INTERVAL '90 days'
                    GROUP BY s.site_id, s.name
                    HAVING AVG(km.availability) < $2 OR STDDEV(km.availability) > 5
                """, tenant_id, self.business_rules['performance_threshold'])
                
                if performance_data:
                    underperforming_sites = len(performance_data)
                    avg_performance_gap = self.business_rules['performance_threshold'] - np.mean([row['avg_availability'] for row in performance_data])
                    
                    # Calculate value impact
                    performance_value = underperforming_sites * avg_performance_gap * 100  # Value per percentage point
                    
                    insight = BusinessInsight(
                        id=f"perf_opt_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                        tenant_id=tenant_id,
                        title="Site Performance Enhancement Opportunity",
                        description=f"Identified {underperforming_sites} sites operating below optimal performance thresholds with potential for significant improvement.",
                        insight_type=InsightType.PERFORMANCE_IMPROVEMENT,
                        priority=RecommendationPriority.HIGH if avg_performance_gap > 10 else RecommendationPriority.MEDIUM,
                        confidence_score=0.9,
                        potential_savings=performance_value * 50,  # $50 per performance point
                        implementation_cost=underperforming_sites * 2000,  # $2k per site
                        roi_estimate=(performance_value * 50) / (underperforming_sites * 2000),
                        time_to_value_days=60,
                        affected_sites=[row['site_id'] for row in performance_data],
                        kpis_improved=['availability', 'network_performance', 'uptime_percentage'],
                        recommendations=[
                            "Upgrade network infrastructure components",
                            "Implement performance monitoring and alerting",
                            "Optimize configuration parameters",
                            "Deploy redundancy mechanisms",
                            "Establish performance baselines and SLAs"
                        ],
                        evidence_data={
                            'underperforming_sites': underperforming_sites,
                            'avg_performance_gap': round(avg_performance_gap, 2),
                            'site_details': [dict(row) for row in performance_data[:10]]
                        },
                        created_at=datetime.utcnow()
                    )
                    insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error analyzing performance optimization: {e}")
        
        return insights
    
    async def analyze_energy_optimization(self, tenant_id: str) -> List[BusinessInsight]:
        """Analyze energy optimization opportunities"""
        insights = []
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get energy consumption patterns
                energy_data = await conn.fetch("""
                    SELECT 
                        s.site_id, s.name, s.location,
                        AVG(km.energy_consumption) as avg_energy_consumption,
                        AVG(km.energy_efficiency) as avg_energy_efficiency,
                        AVG(km.energy_cost) as avg_energy_cost,
                        MAX(km.energy_consumption) - MIN(km.energy_consumption) as energy_variance
                    FROM sites s
                    JOIN kpi_metrics km ON s.site_id = km.site_id
                    WHERE s.tenant_id = $1
                    AND km.timestamp > NOW() - INTERVAL '90 days'
                    AND km.energy_consumption IS NOT NULL
                    GROUP BY s.site_id, s.name, s.location
                    ORDER BY avg_energy_cost DESC
                """, tenant_id)
                
                if energy_data:
                    # Identify high energy consumption sites
                    high_energy_sites = [site for site in energy_data if site['avg_energy_cost'] > self.business_rules['energy_cost_threshold']]
                    
                    if high_energy_sites:
                        total_energy_savings = 0
                        for site in high_energy_sites:
                            # Estimate 12-20% energy savings potential
                            savings_potential = site['avg_energy_cost'] * 0.16 * 12  # Annualized
                            total_energy_savings += savings_potential
                        
                        insight = BusinessInsight(
                            id=f"energy_opt_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                            tenant_id=tenant_id,
                            title="Energy Consumption Optimization",
                            description=f"Significant energy optimization potential identified across {len(high_energy_sites)} high-consumption sites.",
                            insight_type=InsightType.ENERGY_OPTIMIZATION,
                            priority=RecommendationPriority.HIGH if total_energy_savings > 100000 else RecommendationPriority.MEDIUM,
                            confidence_score=0.8,
                            potential_savings=total_energy_savings,
                            implementation_cost=len(high_energy_sites) * 5000,  # $5k per site
                            roi_estimate=total_energy_savings / (len(high_energy_sites) * 5000),
                            time_to_value_days=120,
                            affected_sites=[site['site_id'] for site in high_energy_sites],
                            kpis_improved=['energy_efficiency', 'energy_cost', 'carbon_footprint'],
                            recommendations=[
                                "Install energy-efficient equipment",
                                "Implement smart power management systems",
                                "Deploy renewable energy solutions",
                                "Optimize cooling and HVAC systems",
                                "Establish energy consumption monitoring and alerts"
                            ],
                            evidence_data={
                                'high_energy_sites': len(high_energy_sites),
                                'average_monthly_energy_cost': np.mean([site['avg_energy_cost'] for site in high_energy_sites]),
                                'estimated_monthly_savings': total_energy_savings / 12,
                                'site_breakdown': [dict(site) for site in high_energy_sites[:5]]
                            },
                            created_at=datetime.utcnow()
                        )
                        insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error analyzing energy optimization: {e}")
        
        return insights
    
    async def analyze_capacity_planning(self, tenant_id: str) -> List[BusinessInsight]:
        """Analyze capacity planning requirements"""
        insights = []
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get capacity utilization data
                capacity_data = await conn.fetch("""
                    SELECT 
                        s.site_id, s.name,
                        AVG(km.cpu_utilization) as avg_cpu_utilization,
                        AVG(km.memory_utilization) as avg_memory_utilization,
                        AVG(km.storage_utilization) as avg_storage_utilization,
                        AVG(km.network_utilization) as avg_network_utilization,
                        MAX(km.cpu_utilization) as max_cpu_utilization,
                        COUNT(CASE WHEN km.cpu_utilization > 85 THEN 1 END) as high_cpu_incidents
                    FROM sites s
                    JOIN kpi_metrics km ON s.site_id = km.site_id
                    WHERE s.tenant_id = $1
                    AND km.timestamp > NOW() - INTERVAL '90 days'
                    GROUP BY s.site_id, s.name
                """, tenant_id)
                
                # Identify sites approaching capacity limits
                capacity_constrained_sites = []
                underutilized_sites = []
                
                for site in capacity_data:
                    avg_utilization = np.mean([
                        site['avg_cpu_utilization'] or 0,
                        site['avg_memory_utilization'] or 0,
                        site['avg_storage_utilization'] or 0,
                        site['avg_network_utilization'] or 0
                    ])
                    
                    if avg_utilization > 80 or site['high_cpu_incidents'] > 10:
                        capacity_constrained_sites.append(site)
                    elif avg_utilization < 30:
                        underutilized_sites.append(site)
                
                # Generate capacity expansion insights
                if capacity_constrained_sites:
                    expansion_cost = len(capacity_constrained_sites) * 15000  # $15k per site upgrade
                    performance_value = len(capacity_constrained_sites) * 25000  # Value of avoided performance issues
                    
                    insight = BusinessInsight(
                        id=f"capacity_exp_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                        tenant_id=tenant_id,
                        title="Capacity Expansion Planning",
                        description=f"Proactive capacity expansion recommended for {len(capacity_constrained_sites)} sites approaching resource limits.",
                        insight_type=InsightType.CAPACITY_PLANNING,
                        priority=RecommendationPriority.HIGH,
                        confidence_score=0.85,
                        potential_savings=performance_value,
                        implementation_cost=expansion_cost,
                        roi_estimate=performance_value / expansion_cost,
                        time_to_value_days=180,
                        affected_sites=[site['site_id'] for site in capacity_constrained_sites],
                        kpis_improved=['resource_utilization', 'performance', 'availability'],
                        recommendations=[
                            "Upgrade CPU and memory resources",
                            "Expand storage capacity",
                            "Implement load balancing",
                            "Deploy auto-scaling mechanisms",
                            "Optimize resource allocation"
                        ],
                        evidence_data={
                            'capacity_constrained_sites': len(capacity_constrained_sites),
                            'avg_utilization': np.mean([np.mean([
                                site['avg_cpu_utilization'] or 0,
                                site['avg_memory_utilization'] or 0,
                                site['avg_storage_utilization'] or 0
                            ]) for site in capacity_constrained_sites]),
                            'high_utilization_incidents': sum([site['high_cpu_incidents'] for site in capacity_constrained_sites])
                        },
                        created_at=datetime.utcnow()
                    )
                    insights.append(insight)
                
                # Generate consolidation insights
                if underutilized_sites and len(underutilized_sites) > 2:
                    consolidation_savings = len(underutilized_sites) * 8000  # $8k annual savings per consolidated site
                    
                    insight = BusinessInsight(
                        id=f"capacity_cons_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                        tenant_id=tenant_id,
                        title="Resource Consolidation Opportunity",
                        description=f"Resource consolidation potential identified across {len(underutilized_sites)} underutilized sites.",
                        insight_type=InsightType.COST_OPTIMIZATION,
                        priority=RecommendationPriority.MEDIUM,
                        confidence_score=0.75,
                        potential_savings=consolidation_savings,
                        implementation_cost=len(underutilized_sites) * 3000,  # $3k per site consolidation
                        roi_estimate=consolidation_savings / (len(underutilized_sites) * 3000),
                        time_to_value_days=120,
                        affected_sites=[site['site_id'] for site in underutilized_sites],
                        kpis_improved=['resource_efficiency', 'operational_cost'],
                        recommendations=[
                            "Consolidate underutilized resources",
                            "Implement virtualization",
                            "Optimize workload distribution",
                            "Consider site decommissioning",
                            "Redistribute capacity to high-demand sites"
                        ],
                        evidence_data={
                            'underutilized_sites': len(underutilized_sites),
                            'avg_utilization': np.mean([np.mean([
                                site['avg_cpu_utilization'] or 0,
                                site['avg_memory_utilization'] or 0,
                                site['avg_storage_utilization'] or 0
                            ]) for site in underutilized_sites]),
                            'potential_annual_savings': consolidation_savings
                        },
                        created_at=datetime.utcnow()
                    )
                    insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error analyzing capacity planning: {e}")
        
        return insights
    
    async def calculate_roi_metrics(self, tenant_id: str) -> Dict[str, ValueMetric]:
        """Calculate comprehensive ROI and value metrics"""
        try:
            metrics = {}
            
            async with self.db_pool.acquire() as conn:
                # Calculate cost savings metrics
                cost_data = await conn.fetchrow("""
                    SELECT 
                        SUM(energy_cost) as total_energy_cost,
                        SUM(maintenance_cost) as total_maintenance_cost,
                        SUM(operational_cost) as total_operational_cost,
                        COUNT(DISTINCT site_id) as total_sites
                    FROM kpi_metrics 
                    WHERE tenant_id = $1 
                    AND timestamp > NOW() - INTERVAL '90 days'
                """, tenant_id)
                
                if cost_data:
                    current_monthly_cost = (cost_data['total_energy_cost'] or 0) / 3  # 90 days to monthly
                    target_monthly_cost = current_monthly_cost * 0.85  # 15% reduction target
                    
                    metrics['cost_reduction'] = ValueMetric(
                        metric_name="Monthly Operational Cost Reduction",
                        current_value=current_monthly_cost,
                        target_value=target_monthly_cost,
                        improvement_percentage=15.0,
                        financial_impact=(current_monthly_cost - target_monthly_cost) * 12,
                        time_period_days=365
                    )
                
                # Calculate performance metrics
                performance_data = await conn.fetchrow("""
                    SELECT 
                        AVG(availability) as avg_availability,
                        AVG(uptime_percentage) as avg_uptime,
                        COUNT(DISTINCT site_id) as sites_count
                    FROM kpi_metrics 
                    WHERE tenant_id = $1 
                    AND timestamp > NOW() - INTERVAL '90 days'
                """, tenant_id)
                
                if performance_data:
                    current_availability = performance_data['avg_availability'] or 95
                    target_availability = 99.5
                    
                    metrics['availability_improvement'] = ValueMetric(
                        metric_name="Service Availability Enhancement",
                        current_value=current_availability,
                        target_value=target_availability,
                        improvement_percentage=((target_availability - current_availability) / current_availability) * 100,
                        financial_impact=(target_availability - current_availability) * 1000 * (performance_data['sites_count'] or 1),
                        time_period_days=365
                    )
                
                # Calculate incident reduction metrics
                incident_data = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_incidents,
                        AVG(EXTRACT(EPOCH FROM (COALESCE(resolution_time, NOW()) - created_at))/3600) as avg_resolution_hours
                    FROM noc_incidents 
                    WHERE tenant_id = $1 
                    AND created_at > NOW() - INTERVAL '90 days'
                """, tenant_id)
                
                if incident_data and incident_data['total_incidents']:
                    current_monthly_incidents = incident_data['total_incidents'] / 3  # 90 days to monthly
                    target_monthly_incidents = current_monthly_incidents * 0.6  # 40% reduction target
                    
                    metrics['incident_reduction'] = ValueMetric(
                        metric_name="Monthly Incident Count Reduction",
                        current_value=current_monthly_incidents,
                        target_value=target_monthly_incidents,
                        improvement_percentage=40.0,
                        financial_impact=(current_monthly_incidents - target_monthly_incidents) * 500 * 12,  # $500 per incident
                        time_period_days=365
                    )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating ROI metrics: {e}")
            return {}
    
    async def filter_and_rank_insights(self, insights: List[BusinessInsight]) -> List[BusinessInsight]:
        """Filter and rank insights by value and confidence"""
        try:
            # Filter by confidence and ROI thresholds
            filtered_insights = [
                insight for insight in insights
                if insight.confidence_score >= self.config['min_confidence_threshold']
                and insight.roi_estimate >= self.config['roi_threshold']
            ]
            
            # Rank by priority, ROI, and potential savings
            def ranking_score(insight):
                priority_weight = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[insight.priority.value]
                roi_weight = min(insight.roi_estimate, 10) / 10  # Cap at 10x ROI
                savings_weight = min(insight.potential_savings / 100000, 1)  # Normalize to $100k
                confidence_weight = insight.confidence_score
                
                return (priority_weight * 0.3 + roi_weight * 0.25 + 
                       savings_weight * 0.25 + confidence_weight * 0.2)
            
            filtered_insights.sort(key=ranking_score, reverse=True)
            
            # Limit to max insights per tenant
            return filtered_insights[:self.config['max_insights_per_tenant']]
            
        except Exception as e:
            logger.error(f"Error filtering and ranking insights: {e}")
            return insights[:20]  # Return top 20 as fallback
    
    async def store_business_insights(self, insights: List[BusinessInsight]):
        """Store business insights in database"""
        try:
            async with self.db_pool.acquire() as conn:
                for insight in insights:
                    await conn.execute("""
                        INSERT INTO business_insights (
                            id, tenant_id, title, description, insight_type, priority,
                            confidence_score, potential_savings, implementation_cost,
                            roi_estimate, time_to_value_days, affected_sites,
                            kpis_improved, recommendations, evidence_data,
                            created_at, expires_at, status
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                        ON CONFLICT (id) DO UPDATE SET
                            confidence_score = EXCLUDED.confidence_score,
                            potential_savings = EXCLUDED.potential_savings,
                            evidence_data = EXCLUDED.evidence_data,
                            updated_at = NOW()
                    """,
                    insight.id, insight.tenant_id, insight.title, insight.description,
                    insight.insight_type.value, insight.priority.value,
                    insight.confidence_score, insight.potential_savings,
                    insight.implementation_cost, insight.roi_estimate,
                    insight.time_to_value_days, json.dumps(insight.affected_sites),
                    json.dumps(insight.kpis_improved), json.dumps(insight.recommendations),
                    json.dumps(insight.evidence_data), insight.created_at,
                    insight.expires_at, insight.status)
                    
        except Exception as e:
            logger.error(f"Error storing business insights: {e}")
    
    async def run_continuous_analysis(self):
        """Run continuous business intelligence analysis"""
        while True:
            try:
                # Get all active tenants
                async with self.db_pool.acquire() as conn:
                    tenants = await conn.fetch("SELECT DISTINCT tenant_id FROM sites")
                
                for tenant_row in tenants:
                    tenant_id = tenant_row['tenant_id']
                    
                    try:
                        # Generate insights for each tenant
                        insights = await self.generate_business_insights(tenant_id)
                        logger.info(f"Generated {len(insights)} insights for tenant {tenant_id}")
                        
                        # Calculate and cache ROI metrics
                        roi_metrics = await self.calculate_roi_metrics(tenant_id)
                        
                        # Cache results in Redis
                        await self.redis_client.setex(
                            f"bi_insights:{tenant_id}",
                            3600 * 6,  # 6 hours cache
                            json.dumps([asdict(insight) for insight in insights], default=str)
                        )
                        
                        await self.redis_client.setex(
                            f"roi_metrics:{tenant_id}",
                            3600 * 6,  # 6 hours cache  
                            json.dumps([asdict(metric) for metric in roi_metrics.values()], default=str)
                        )
                        
                    except Exception as e:
                        logger.error(f"Error processing tenant {tenant_id}: {e}")
                        continue
                
                # Wait for next analysis cycle
                await asyncio.sleep(self.config['update_frequency_hours'] * 3600)
                
            except Exception as e:
                logger.error(f"Error in continuous analysis loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def run(self):
        """Main run loop"""
        try:
            await self.initialize()
            await self.run_continuous_analysis()
            
        except KeyboardInterrupt:
            logger.info("Business Intelligence Engine stopped by user")
        except Exception as e:
            logger.error(f"Business Intelligence Engine error: {e}")
        finally:
            if self.db_pool:
                await self.db_pool.close()
            if self.redis_client:
                await self.redis_client.close()

# Global state
shutdown_event = asyncio.Event()
bi_engine = None

async def message_handler(topic, key, value, headers, timestamp, partition, offset):
    """Handle incoming Kafka messages"""
    global bi_engine
    
    logger.info(f"Received message from {topic}: {key}")
    
    try:
        if topic == KAFKA_TOPICS['kpi_calculations']:
            # Process KPI results for business intelligence
            site_id = value.get('site_id')
            kpi_name = value.get('kpi_name')
            kpi_value = value.get('value')
            
            if site_id and kpi_name and kpi_value is not None and bi_engine:
                # Trigger business analysis for this KPI
                asyncio.create_task(
                    bi_engine.analyze_kpi_impact(site_id, kpi_name, kpi_value)
                )
                
        elif topic == KAFKA_TOPICS['events']:
            # Process events for business impact analysis
            event_type = value.get('event_type')
            site_id = value.get('site_id')
            
            if event_type in ['optimization_requested', 'insight_implemented'] and bi_engine:
                # Process business events
                asyncio.create_task(
                    bi_engine.process_business_event(value)
                )
                
        elif topic == KAFKA_TOPICS['aiops_predictions']:
            # Process AIOps predictions for BI insights
            prediction_type = value.get('prediction_type')
            
            if prediction_type == 'optimization' and bi_engine:
                # Handle optimization requests
                asyncio.create_task(
                    bi_engine.process_optimization_request(value)
                )
                
    except Exception as e:
        logger.error(f"Error handling message: {e}")

async def create_health_server():
    """Create health check server"""
    from aiohttp import web
    
    async def health_handler(request):
        return web.json_response({
            "status": "healthy",
            "worker": "business_intelligence_engine",
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
    global bi_engine, health_server
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting Towerco Business Intelligence Engine...")
    
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
        site = web.TCPSite(health_runner, '0.0.0.0', 8004)
        await site.start()
        logger.info("Health server started on port 8004")
        
        # Initialize Business Intelligence Engine
        bi_engine = BusinessIntelligenceEngine()
        await bi_engine.initialize()
        
        # Start Kafka consumers
        await start_consumer(
            'bi_triggers',
            [KAFKA_TOPICS['kpi_calculations'], KAFKA_TOPICS['events'], KAFKA_TOPICS['aiops_predictions']],
            'bi_engine_group',
            message_handler
        )
        
        logger.info("Business Intelligence Engine started successfully")
        
        # Main loop
        while not shutdown_event.is_set():
            await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Business Intelligence Engine stopping...")
    except Exception as e:
        logger.error(f"Fatal error in Business Intelligence Engine: {e}")
        raise
    
    finally:
        logger.info("Shutting down Business Intelligence Engine...")
        
        # Cleanup health server
        if health_server:
            await health_runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())