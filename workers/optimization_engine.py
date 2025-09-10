"""
Optimization Engine
Smart recommendations and automated optimization for telecom operations
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.optimization import minimize
from scipy.optimize import differential_evolution
import pulp  # Linear programming
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

class OptimizationType(Enum):
    ENERGY_OPTIMIZATION = "energy_optimization"
    COST_MINIMIZATION = "cost_minimization"
    PERFORMANCE_MAXIMIZATION = "performance_maximization"
    RESOURCE_ALLOCATION = "resource_allocation"
    MAINTENANCE_SCHEDULING = "maintenance_scheduling"
    CAPACITY_PLANNING = "capacity_planning"
    SLA_OPTIMIZATION = "sla_optimization"
    RISK_MINIMIZATION = "risk_minimization"

class OptimizationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class OptimizationTask:
    id: str
    tenant_id: str
    name: str
    description: str
    optimization_type: OptimizationType
    parameters: Dict[str, Any]
    constraints: Dict[str, Any]
    objectives: List[str]
    affected_sites: List[str]
    status: OptimizationStatus
    progress_percentage: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    recommendations: List[str] = None

@dataclass
class OptimizationResult:
    task_id: str
    optimization_type: OptimizationType
    objective_value: float
    improvement_percentage: float
    estimated_savings: float
    implementation_complexity: str
    confidence_score: float
    optimal_parameters: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]

@dataclass
class SmartRecommendation:
    id: str
    tenant_id: str
    title: str
    description: str
    category: str
    priority: str
    impact_score: float
    implementation_effort: str
    estimated_savings: float
    affected_kpis: List[str]
    actions: List[Dict[str, Any]]
    evidence: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None

class OptimizationEngine:
    """Advanced optimization engine for telecom operations"""
    
    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        
        # ML Models for optimization
        self.energy_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.performance_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Optimization solvers
        self.linear_solver = None
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
        
        # Configuration
        self.config = {
            'max_optimization_time_minutes': 60,
            'parallel_optimizations': 3,
            'convergence_tolerance': 1e-6,
            'max_iterations': 1000,
            'recommendation_validity_days': 30,
            'auto_optimization_enabled': True,
        }
        
        # Optimization templates
        self.optimization_templates = {
            'energy_efficiency': {
                'variables': ['power_settings', 'cooling_thresholds', 'equipment_schedules'],
                'constraints': ['min_performance', 'max_temperature', 'sla_requirements'],
                'objectives': ['minimize_energy_cost', 'maximize_efficiency']
            },
            'cost_optimization': {
                'variables': ['resource_allocation', 'vendor_selection', 'maintenance_schedules'],
                'constraints': ['budget_limits', 'service_levels', 'regulatory_requirements'],
                'objectives': ['minimize_total_cost', 'maximize_roi']
            },
            'performance_tuning': {
                'variables': ['bandwidth_allocation', 'queue_parameters', 'routing_weights'],
                'constraints': ['capacity_limits', 'latency_requirements', 'availability_targets'],
                'objectives': ['maximize_throughput', 'minimize_latency']
            }
        }
    
    async def initialize(self):
        """Initialize optimization engine"""
        try:
            # Initialize database connection
            self.db_pool = await asyncpg.create_pool(
                "postgresql://towerco:secure_password@localhost:5432/towerco_aiops",
                min_size=5,
                max_size=20
            )
            
            # Initialize Redis connection
            self.redis_client = await redis.from_url("redis://localhost:6379", decode_responses=True)
            
            # Train optimization models
            await self.train_optimization_models()
            
            logger.info("Optimization Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Optimization Engine: {e}")
            raise
    
    async def create_optimization_task(self, task_data: Dict[str, Any]) -> OptimizationTask:
        """Create new optimization task"""
        try:
            task = OptimizationTask(
                id=f"opt_{task_data['tenant_id']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                tenant_id=task_data['tenant_id'],
                name=task_data['name'],
                description=task_data['description'],
                optimization_type=OptimizationType(task_data['optimization_type']),
                parameters=task_data.get('parameters', {}),
                constraints=task_data.get('constraints', {}),
                objectives=task_data.get('objectives', []),
                affected_sites=task_data.get('affected_sites', []),
                status=OptimizationStatus.PENDING,
                progress_percentage=0.0,
                created_at=datetime.utcnow(),
                recommendations=[]
            )
            
            # Store task
            await self.store_optimization_task(task)
            
            # Queue for processing
            await self.queue_optimization_task(task)
            
            logger.info(f"Created optimization task {task.id}")
            return task
            
        except Exception as e:
            logger.error(f"Error creating optimization task: {e}")
            raise
    
    async def execute_energy_optimization(self, task: OptimizationTask) -> OptimizationResult:
        """Execute energy optimization"""
        try:
            await self.update_task_status(task.id, OptimizationStatus.RUNNING, 10)
            
            # Get energy consumption data
            async with self.db_pool.acquire() as conn:
                energy_data = await conn.fetch("""
                    SELECT 
                        km.site_id,
                        AVG(km.energy_consumption) as avg_consumption,
                        AVG(km.energy_efficiency) as avg_efficiency,
                        AVG(km.energy_cost) as avg_cost,
                        AVG(km.temperature) as avg_temperature,
                        AVG(km.cpu_utilization) as avg_cpu_util
                    FROM kpi_metrics km
                    JOIN sites s ON km.site_id = s.site_id
                    WHERE s.tenant_id = $1
                    AND km.timestamp > NOW() - INTERVAL '30 days'
                    AND km.site_id = ANY($2)
                    GROUP BY km.site_id
                """, task.tenant_id, task.affected_sites)
            
            if not energy_data:
                raise ValueError("No energy data available for optimization")
            
            await self.update_task_status(task.id, OptimizationStatus.RUNNING, 30)
            
            # Prepare optimization problem
            sites = [row['site_id'] for row in energy_data]
            current_consumption = [row['avg_consumption'] or 100 for row in energy_data]
            current_efficiency = [row['avg_efficiency'] or 80 for row in energy_data]
            current_costs = [row['avg_cost'] or 1000 for row in energy_data]
            
            # Create linear programming problem
            prob = pulp.LpProblem("Energy_Optimization", pulp.LpMinimize)
            
            # Decision variables - power reduction factors
            power_reduction = {}
            cooling_adjustment = {}
            schedule_optimization = {}
            
            for i, site in enumerate(sites):
                power_reduction[site] = pulp.LpVariable(f"power_reduction_{site}", 0, 0.3)  # Max 30% reduction
                cooling_adjustment[site] = pulp.LpVariable(f"cooling_adj_{site}", -5, 5)  # Temperature adjustment
                schedule_optimization[site] = pulp.LpVariable(f"schedule_opt_{site}", 0, 0.2)  # Schedule optimization
            
            await self.update_task_status(task.id, OptimizationStatus.RUNNING, 50)
            
            # Objective function - minimize total energy cost
            total_cost = 0
            for i, site in enumerate(sites):
                # Cost reduction from power optimization
                power_savings = current_costs[i] * power_reduction[site] * 0.8
                # Cost change from cooling adjustment  
                cooling_impact = current_costs[i] * 0.05 * cooling_adjustment[site]
                # Savings from schedule optimization
                schedule_savings = current_costs[i] * schedule_optimization[site] * 0.6
                
                optimized_cost = current_costs[i] - power_savings - schedule_savings + abs(cooling_impact)
                total_cost += optimized_cost
            
            prob += total_cost
            
            # Constraints
            for i, site in enumerate(sites):
                # Performance constraint - efficiency must not drop below 70%
                efficiency_impact = current_efficiency[i] * (1 - power_reduction[site] * 0.3)
                prob += efficiency_impact >= 70
                
                # Temperature constraint
                temp_impact = (energy_data[i]['avg_temperature'] or 25) + cooling_adjustment[site]
                prob += temp_impact <= 35  # Max temperature
                prob += temp_impact >= 15  # Min temperature
                
                # Total optimization constraint
                prob += power_reduction[site] + schedule_optimization[site] <= 0.4
            
            await self.update_task_status(task.id, OptimizationStatus.RUNNING, 70)
            
            # Solve optimization
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status != pulp.LpStatusOptimal:
                raise ValueError(f"Optimization failed with status: {pulp.LpStatus[prob.status]}")
            
            await self.update_task_status(task.id, OptimizationStatus.RUNNING, 90)
            
            # Calculate results
            total_current_cost = sum(current_costs)
            total_optimized_cost = pulp.value(prob.objective)
            savings = total_current_cost - total_optimized_cost
            improvement_percentage = (savings / total_current_cost) * 100
            
            # Extract optimal parameters
            optimal_params = {}
            for site in sites:
                optimal_params[site] = {
                    'power_reduction': pulp.value(power_reduction[site]),
                    'cooling_adjustment': pulp.value(cooling_adjustment[site]),
                    'schedule_optimization': pulp.value(schedule_optimization[site])
                }
            
            # Sensitivity analysis
            sensitivity = await self.perform_sensitivity_analysis(
                energy_data, optimal_params, 'energy_cost'
            )
            
            # Risk assessment
            risk_assessment = {
                'performance_risk': 'low' if max([params['power_reduction'] for params in optimal_params.values()]) < 0.2 else 'medium',
                'implementation_risk': 'low',
                'operational_risk': 'low' if improvement_percentage < 20 else 'medium'
            }
            
            result = OptimizationResult(
                task_id=task.id,
                optimization_type=OptimizationType.ENERGY_OPTIMIZATION,
                objective_value=total_optimized_cost,
                improvement_percentage=improvement_percentage,
                estimated_savings=savings * 12,  # Annualized
                implementation_complexity='medium',
                confidence_score=0.85,
                optimal_parameters=optimal_params,
                sensitivity_analysis=sensitivity,
                risk_assessment=risk_assessment
            )
            
            await self.update_task_status(task.id, OptimizationStatus.COMPLETED, 100)
            return result
            
        except Exception as e:
            await self.update_task_status(task.id, OptimizationStatus.FAILED, 0)
            logger.error(f"Error in energy optimization: {e}")
            raise
    
    async def execute_cost_optimization(self, task: OptimizationTask) -> OptimizationResult:
        """Execute comprehensive cost optimization"""
        try:
            await self.update_task_status(task.id, OptimizationStatus.RUNNING, 15)
            
            # Get cost breakdown data
            async with self.db_pool.acquire() as conn:
                cost_data = await conn.fetch("""
                    SELECT 
                        km.site_id,
                        AVG(km.energy_cost) as avg_energy_cost,
                        AVG(km.maintenance_cost) as avg_maintenance_cost,
                        AVG(km.operational_cost) as avg_operational_cost,
                        COUNT(ni.id) as incident_count,
                        AVG(km.availability) as avg_availability
                    FROM kpi_metrics km
                    JOIN sites s ON km.site_id = s.site_id
                    LEFT JOIN noc_incidents ni ON km.site_id = ni.site_id 
                        AND ni.created_at > NOW() - INTERVAL '30 days'
                    WHERE s.tenant_id = $1
                    AND km.timestamp > NOW() - INTERVAL '30 days'
                    AND km.site_id = ANY($2)
                    GROUP BY km.site_id
                """, task.tenant_id, task.affected_sites)
            
            await self.update_task_status(task.id, OptimizationStatus.RUNNING, 35)
            
            # Multi-objective optimization using evolutionary algorithm
            def objective_function(x):
                """Multi-objective cost optimization function"""
                n_sites = len(cost_data)
                
                # Extract decision variables
                energy_reduction = x[:n_sites]
                maintenance_optimization = x[n_sites:2*n_sites]
                vendor_consolidation = x[2*n_sites:3*n_sites]
                
                total_cost = 0
                total_risk = 0
                
                for i, site_data in enumerate(cost_data):
                    # Energy cost optimization
                    energy_savings = (site_data['avg_energy_cost'] or 0) * energy_reduction[i]
                    
                    # Maintenance cost optimization
                    maintenance_savings = (site_data['avg_maintenance_cost'] or 0) * maintenance_optimization[i]
                    
                    # Vendor consolidation savings
                    vendor_savings = (site_data['avg_operational_cost'] or 0) * vendor_consolidation[i] * 0.1
                    
                    # Calculate risk penalty
                    if energy_reduction[i] > 0.2:  # High energy reduction = higher risk
                        total_risk += 100
                    if maintenance_optimization[i] > 0.3:  # High maintenance optimization = higher risk
                        total_risk += 150
                    
                    site_cost = (
                        (site_data['avg_energy_cost'] or 0) - energy_savings +
                        (site_data['avg_maintenance_cost'] or 0) - maintenance_savings +
                        (site_data['avg_operational_cost'] or 0) - vendor_savings
                    )
                    total_cost += site_cost
                
                # Multi-objective: minimize cost and risk
                return total_cost + total_risk * 0.1
            
            await self.update_task_status(task.id, OptimizationStatus.RUNNING, 55)
            
            # Set bounds for optimization variables
            n_sites = len(cost_data)
            bounds = []
            
            # Energy reduction bounds (0-30%)
            bounds.extend([(0, 0.3) for _ in range(n_sites)])
            # Maintenance optimization bounds (0-40%)
            bounds.extend([(0, 0.4) for _ in range(n_sites)])
            # Vendor consolidation bounds (0-50%)
            bounds.extend([(0, 0.5) for _ in range(n_sites)])
            
            # Run optimization
            result_opt = differential_evolution(
                objective_function,
                bounds,
                maxiter=200,
                seed=42,
                atol=self.config['convergence_tolerance']
            )
            
            if not result_opt.success:
                raise ValueError("Cost optimization failed to converge")
            
            await self.update_task_status(task.id, OptimizationStatus.RUNNING, 85)
            
            # Extract optimal parameters
            x_optimal = result_opt.x
            n_sites = len(cost_data)
            
            optimal_params = {}
            total_current_cost = 0
            total_optimized_cost = 0
            
            for i, site_data in enumerate(cost_data):
                site_id = site_data['site_id']
                energy_reduction = x_optimal[i]
                maintenance_optimization = x_optimal[i + n_sites]
                vendor_consolidation = x_optimal[i + 2*n_sites]
                
                current_cost = (
                    (site_data['avg_energy_cost'] or 0) +
                    (site_data['avg_maintenance_cost'] or 0) +
                    (site_data['avg_operational_cost'] or 0)
                )
                
                energy_savings = (site_data['avg_energy_cost'] or 0) * energy_reduction
                maintenance_savings = (site_data['avg_maintenance_cost'] or 0) * maintenance_optimization
                vendor_savings = (site_data['avg_operational_cost'] or 0) * vendor_consolidation * 0.1
                
                optimized_cost = current_cost - energy_savings - maintenance_savings - vendor_savings
                
                optimal_params[site_id] = {
                    'energy_reduction_percentage': energy_reduction * 100,
                    'maintenance_optimization_percentage': maintenance_optimization * 100,
                    'vendor_consolidation_percentage': vendor_consolidation * 100,
                    'estimated_monthly_savings': current_cost - optimized_cost
                }
                
                total_current_cost += current_cost
                total_optimized_cost += optimized_cost
            
            savings = total_current_cost - total_optimized_cost
            improvement_percentage = (savings / total_current_cost) * 100 if total_current_cost > 0 else 0
            
            # Sensitivity analysis
            sensitivity = {
                'energy_price_sensitivity': 0.3,
                'maintenance_cost_sensitivity': 0.4,
                'vendor_negotiation_impact': 0.2,
                'implementation_timeline_risk': 'medium'
            }
            
            # Risk assessment
            avg_energy_reduction = np.mean([x_optimal[i] for i in range(n_sites)])
            avg_maintenance_opt = np.mean([x_optimal[i + n_sites] for i in range(n_sites)])
            
            risk_assessment = {
                'performance_risk': 'low' if avg_energy_reduction < 0.15 else 'medium',
                'operational_risk': 'low' if avg_maintenance_opt < 0.25 else 'medium',
                'implementation_risk': 'medium',
                'vendor_relationship_risk': 'low'
            }
            
            result = OptimizationResult(
                task_id=task.id,
                optimization_type=OptimizationType.COST_MINIMIZATION,
                objective_value=total_optimized_cost,
                improvement_percentage=improvement_percentage,
                estimated_savings=savings * 12,  # Annualized
                implementation_complexity='high',
                confidence_score=0.8,
                optimal_parameters=optimal_params,
                sensitivity_analysis=sensitivity,
                risk_assessment=risk_assessment
            )
            
            await self.update_task_status(task.id, OptimizationStatus.COMPLETED, 100)
            return result
            
        except Exception as e:
            await self.update_task_status(task.id, OptimizationStatus.FAILED, 0)
            logger.error(f"Error in cost optimization: {e}")
            raise
    
    async def generate_smart_recommendations(self, tenant_id: str) -> List[SmartRecommendation]:
        """Generate intelligent recommendations based on data analysis"""
        try:
            recommendations = []
            
            async with self.db_pool.acquire() as conn:
                # Get comprehensive site data
                site_data = await conn.fetch("""
                    SELECT 
                        s.site_id, s.name, s.location,
                        AVG(km.energy_consumption) as avg_energy_consumption,
                        AVG(km.energy_efficiency) as avg_energy_efficiency,
                        AVG(km.availability) as avg_availability,
                        AVG(km.cpu_utilization) as avg_cpu_utilization,
                        AVG(km.network_performance) as avg_network_performance,
                        COUNT(ni.id) as incident_count,
                        AVG(km.energy_cost) as avg_energy_cost,
                        AVG(km.maintenance_cost) as avg_maintenance_cost
                    FROM sites s
                    LEFT JOIN kpi_metrics km ON s.site_id = km.site_id
                        AND km.timestamp > NOW() - INTERVAL '30 days'
                    LEFT JOIN noc_incidents ni ON s.site_id = ni.site_id
                        AND ni.created_at > NOW() - INTERVAL '30 days'
                    WHERE s.tenant_id = $1
                    GROUP BY s.site_id, s.name, s.location
                """, tenant_id)
            
            # Analyze patterns and generate recommendations
            for site in site_data:
                site_recommendations = await self.analyze_site_for_recommendations(site)
                recommendations.extend(site_recommendations)
            
            # Generate tenant-level recommendations
            tenant_recommendations = await self.generate_tenant_level_recommendations(tenant_id, site_data)
            recommendations.extend(tenant_recommendations)
            
            # Filter and prioritize recommendations
            filtered_recommendations = await self.filter_recommendations(recommendations)
            
            return filtered_recommendations
            
        except Exception as e:
            logger.error(f"Error generating smart recommendations: {e}")
            return []
    
    async def analyze_site_for_recommendations(self, site_data: Dict) -> List[SmartRecommendation]:
        """Analyze individual site for optimization recommendations"""
        recommendations = []
        site_id = site_data['site_id']
        
        try:
            # Energy efficiency recommendation
            if site_data['avg_energy_efficiency'] and site_data['avg_energy_efficiency'] < 75:
                energy_rec = SmartRecommendation(
                    id=f"energy_rec_{site_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                    tenant_id=site_data.get('tenant_id'),
                    title=f"Energy Efficiency Improvement - {site_data['name']}",
                    description=f"Site {site_data['name']} is operating at {site_data['avg_energy_efficiency']:.1f}% energy efficiency, below the recommended 75% threshold.",
                    category="energy_optimization",
                    priority="medium" if site_data['avg_energy_efficiency'] > 60 else "high",
                    impact_score=8.5,
                    implementation_effort="medium",
                    estimated_savings=(100 - site_data['avg_energy_efficiency']) * 50,  # $50 per efficiency point
                    affected_kpis=["energy_efficiency", "energy_cost", "carbon_footprint"],
                    actions=[
                        {"action": "upgrade_hvac_system", "effort": "high", "cost": 15000},
                        {"action": "implement_smart_power_management", "effort": "medium", "cost": 5000},
                        {"action": "optimize_equipment_schedules", "effort": "low", "cost": 500}
                    ],
                    evidence={
                        "current_efficiency": site_data['avg_energy_efficiency'],
                        "benchmark_efficiency": 85,
                        "monthly_energy_cost": site_data['avg_energy_cost']
                    },
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=30)
                )
                recommendations.append(energy_rec)
            
            # Performance optimization recommendation
            if site_data['avg_availability'] and site_data['avg_availability'] < 95:
                perf_rec = SmartRecommendation(
                    id=f"perf_rec_{site_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                    tenant_id=site_data.get('tenant_id'),
                    title=f"Performance Enhancement - {site_data['name']}",
                    description=f"Site availability at {site_data['avg_availability']:.1f}% is below the 95% SLA target.",
                    category="performance_optimization",
                    priority="high" if site_data['avg_availability'] < 90 else "medium",
                    impact_score=9.2,
                    implementation_effort="medium",
                    estimated_savings=(95 - site_data['avg_availability']) * 1000,  # $1000 per availability point
                    affected_kpis=["availability", "uptime_percentage", "customer_satisfaction"],
                    actions=[
                        {"action": "implement_redundancy", "effort": "high", "cost": 20000},
                        {"action": "upgrade_monitoring", "effort": "medium", "cost": 8000},
                        {"action": "optimize_maintenance_schedule", "effort": "low", "cost": 1000}
                    ],
                    evidence={
                        "current_availability": site_data['avg_availability'],
                        "sla_target": 95,
                        "incident_count": site_data['incident_count']
                    },
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=30)
                )
                recommendations.append(perf_rec)
            
            # Maintenance optimization recommendation
            if site_data['incident_count'] and site_data['incident_count'] > 5:
                maint_rec = SmartRecommendation(
                    id=f"maint_rec_{site_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                    tenant_id=site_data.get('tenant_id'),
                    title=f"Preventive Maintenance Enhancement - {site_data['name']}",
                    description=f"Site has experienced {site_data['incident_count']} incidents this month, indicating maintenance optimization opportunities.",
                    category="maintenance_optimization",
                    priority="medium",
                    impact_score=7.8,
                    implementation_effort="low",
                    estimated_savings=site_data['incident_count'] * 500,  # $500 per incident avoided
                    affected_kpis=["incident_rate", "mtbf", "maintenance_cost"],
                    actions=[
                        {"action": "implement_predictive_maintenance", "effort": "medium", "cost": 10000},
                        {"action": "increase_inspection_frequency", "effort": "low", "cost": 2000},
                        {"action": "upgrade_aging_equipment", "effort": "high", "cost": 25000}
                    ],
                    evidence={
                        "monthly_incidents": site_data['incident_count'],
                        "average_incidents": 2,
                        "maintenance_cost": site_data['avg_maintenance_cost']
                    },
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=30)
                )
                recommendations.append(maint_rec)
        
        except Exception as e:
            logger.error(f"Error analyzing site {site_id}: {e}")
        
        return recommendations
    
    async def perform_sensitivity_analysis(self, data: List[Dict], optimal_params: Dict, objective: str) -> Dict[str, Any]:
        """Perform sensitivity analysis on optimization results"""
        try:
            sensitivity = {
                'parameter_sensitivity': {},
                'constraint_sensitivity': {},
                'robustness_score': 0.0
            }
            
            # Analyze parameter sensitivity
            for site_id, params in optimal_params.items():
                site_sensitivity = {}
                for param_name, param_value in params.items():
                    # Calculate impact of ±10% parameter change
                    if isinstance(param_value, (int, float)) and param_value != 0:
                        delta = param_value * 0.1
                        high_impact = self.calculate_objective_change(param_value + delta, objective)
                        low_impact = self.calculate_objective_change(param_value - delta, objective)
                        sensitivity_score = abs(high_impact - low_impact) / (2 * delta) if delta != 0 else 0
                        site_sensitivity[param_name] = sensitivity_score
                
                sensitivity['parameter_sensitivity'][site_id] = site_sensitivity
            
            # Calculate overall robustness
            all_sensitivities = []
            for site_sens in sensitivity['parameter_sensitivity'].values():
                all_sensitivities.extend(site_sens.values())
            
            if all_sensitivities:
                sensitivity['robustness_score'] = 1.0 / (1.0 + np.mean(all_sensitivities))
            
            return sensitivity
            
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {e}")
            return {'parameter_sensitivity': {}, 'constraint_sensitivity': {}, 'robustness_score': 0.5}
    
    def calculate_objective_change(self, param_value: float, objective: str) -> float:
        """Calculate objective function change for sensitivity analysis"""
        # Simplified calculation - in practice this would re-evaluate the full objective
        if objective == 'energy_cost':
            return param_value * 100  # Energy cost impact
        elif objective == 'performance':
            return param_value * 50   # Performance impact
        else:
            return param_value * 75   # General impact
    
    async def store_optimization_task(self, task: OptimizationTask):
        """Store optimization task in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO optimization_tasks (
                        id, tenant_id, name, description, optimization_type,
                        parameters, constraints, objectives, affected_sites,
                        status, progress_percentage, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                task.id, task.tenant_id, task.name, task.description,
                task.optimization_type.value, json.dumps(task.parameters),
                json.dumps(task.constraints), json.dumps(task.objectives),
                json.dumps(task.affected_sites), task.status.value,
                task.progress_percentage, task.created_at)
                
        except Exception as e:
            logger.error(f"Error storing optimization task: {e}")
    
    async def update_task_status(self, task_id: str, status: OptimizationStatus, progress: float):
        """Update optimization task status"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE optimization_tasks 
                    SET status = $2, progress_percentage = $3, updated_at = NOW()
                    WHERE id = $1
                """, task_id, status.value, progress)
                
                if status == OptimizationStatus.RUNNING and progress == 0:
                    await conn.execute("""
                        UPDATE optimization_tasks SET started_at = NOW() WHERE id = $1
                    """, task_id)
                elif status == OptimizationStatus.COMPLETED:
                    await conn.execute("""
                        UPDATE optimization_tasks SET completed_at = NOW() WHERE id = $1
                    """, task_id)
                    
        except Exception as e:
            logger.error(f"Error updating task status: {e}")

# Global state
shutdown_event = asyncio.Event()
optimization_engine = None

async def message_handler(topic, key, value, headers, timestamp, partition, offset):
    """Handle incoming Kafka messages"""
    global optimization_engine
    
    logger.info(f"Received message from {topic}: {key}")
    
    try:
        if topic == KAFKA_TOPICS['aiops_predictions']:
            # Process optimization requests
            prediction_type = value.get('prediction_type')
            
            if prediction_type == 'optimization' and optimization_engine:
                # Handle optimization requests from BI
                input_data = value.get('input_data', {})
                asyncio.create_task(
                    optimization_engine.process_optimization_request(input_data)
                )
                
        elif topic == KAFKA_TOPICS['kpi_calculations']:
            # Process KPI results for optimization opportunities
            site_id = value.get('site_id')
            kpi_name = value.get('kpi_name')
            kpi_value = value.get('value')
            
            if site_id and kpi_name and kpi_value is not None and optimization_engine:
                # Check for optimization opportunities
                asyncio.create_task(
                    optimization_engine.evaluate_optimization_opportunity(site_id, kpi_name, kpi_value)
                )
                
    except Exception as e:
        logger.error(f"Error handling message: {e}")

async def create_health_server():
    """Create health check server"""
    from aiohttp import web
    
    async def health_handler(request):
        return web.json_response({
            "status": "healthy",
            "worker": "optimization_engine",
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
    global optimization_engine, health_server
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting Towerco Optimization Engine...")
    
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
        site = web.TCPSite(health_runner, '0.0.0.0', 8008)
        await site.start()
        logger.info("Health server started on port 8008")
        
        # Initialize Optimization Engine
        optimization_engine = OptimizationEngine()
        await optimization_engine.initialize()
        
        # Start Kafka consumers
        await start_consumer(
            'optimization_engine',
            [KAFKA_TOPICS['aiops_predictions'], KAFKA_TOPICS['kpi_calculations']],
            'optimization_engine_group',
            message_handler
        )
        
        logger.info("Optimization Engine started successfully")
        
        # Main loop
        while not shutdown_event.is_set():
            await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Optimization Engine stopping...")
    except Exception as e:
        logger.error(f"Fatal error in Optimization Engine: {e}")
        raise
    
    finally:
        logger.info("Shutting down Optimization Engine...")
        
        # Cleanup health server
        if health_server:
            await health_runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())