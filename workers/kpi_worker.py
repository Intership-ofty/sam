#!/usr/bin/env python3
"""
Towerco AIOps - KPI Calculation Engine Worker
Real-time KPI calculation with 50+ pre-configured telecom formulas
"""

import asyncio
import logging
import signal
import time
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import traceback

import aiohttp
from aiohttp import web
import psutil
import numpy as np
from prophet import Prophet
import pandas as pd

# Import core modules from backend
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from core.config import settings, KAFKA_TOPICS
from core.database import init_db, DatabaseManager
from core.cache import init_redis, CacheManager, CacheKeys
from core.messaging import init_kafka, MessageProducer, MessageConsumer, start_consumer
from core.monitoring import init_monitoring, MetricsCollector

logger = logging.getLogger(__name__)

# Global state
shutdown_event = asyncio.Event()
kpi_engine = None
health_server = None


class KPICategory(Enum):
    """KPI Categories"""
    NETWORK = "NETWORK"
    ENERGY = "ENERGY" 
    OPERATIONAL = "OPERATIONAL"
    FINANCIAL = "FINANCIAL"


class AggregationType(Enum):
    """KPI Aggregation types"""
    SUM = "SUM"
    AVG = "AVG"
    MAX = "MAX"
    MIN = "MIN"
    COUNT = "COUNT"
    RATIO = "RATIO"
    PERCENTAGE = "PERCENTAGE"
    WEIGHTED_AVG = "WEIGHTED_AVG"


@dataclass
class KPIFormula:
    """KPI Formula definition"""
    kpi_id: str
    kpi_name: str
    category: KPICategory
    description: str
    formula: str
    unit: str
    target_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    calculation_interval: int = 300  # seconds
    aggregation_type: AggregationType = AggregationType.AVG
    input_metrics: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    tenant_specific: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class KPICalculationEngine:
    """Real-time KPI Calculation Engine with 50+ telecom formulas"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.producer = MessageProducer()
        self.metrics = MetricsCollector()
        
        # KPI formulas registry
        self.kpi_formulas: Dict[str, KPIFormula] = {}
        self.calculation_tasks: Dict[str, asyncio.Task] = {}
        self.last_calculation_times: Dict[str, datetime] = {}
        
        # Prediction models
        self.prediction_models: Dict[str, Prophet] = {}
        
        # Initialize 50+ pre-configured KPI formulas
        self.initialize_telecom_kpis()
    
    def initialize_telecom_kpis(self):
        """Initialize 50+ pre-configured telecom KPI formulas"""
        
        # =============================================================================
        # NETWORK PERFORMANCE KPIs (20 KPIs)
        # =============================================================================
        
        # 1. Network Unavailability Rate (NUR) - Critical telecom KPI
        self.kpi_formulas["nur"] = KPIFormula(
            kpi_id="nur",
            kpi_name="Network Unavailability Rate",
            category=KPICategory.NETWORK,
            description="Percentage of time network is unavailable",
            formula="(total_downtime_minutes / total_time_minutes) * 100",
            unit="%",
            target_value=0.1,  # 99.9% availability
            warning_threshold=0.5,
            critical_threshold=1.0,
            calculation_interval=300,
            input_metrics=["availability_pct", "service_uptime"],
            aggregation_type=AggregationType.WEIGHTED_AVG
        )
        
        # 2. Call Success Rate (CSR)
        self.kpi_formulas["csr"] = KPIFormula(
            kpi_id="csr",
            kpi_name="Call Success Rate",
            category=KPICategory.NETWORK,
            description="Percentage of successful call attempts",
            formula="(successful_calls / total_call_attempts) * 100",
            unit="%",
            target_value=98.5,
            warning_threshold=95.0,
            critical_threshold=90.0,
            input_metrics=["call_attempts", "call_successes", "call_failures"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 3. Call Drop Rate (CDR)
        self.kpi_formulas["cdr"] = KPIFormula(
            kpi_id="cdr",
            kpi_name="Call Drop Rate",
            category=KPICategory.NETWORK,
            description="Percentage of calls that are dropped",
            formula="(dropped_calls / established_calls) * 100",
            unit="%",
            target_value=1.0,
            warning_threshold=2.0,
            critical_threshold=3.0,
            input_metrics=["dropped_calls", "established_calls"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 4. Data Throughput
        self.kpi_formulas["data_throughput"] = KPIFormula(
            kpi_id="data_throughput",
            kpi_name="Average Data Throughput",
            category=KPICategory.NETWORK,
            description="Average data throughput per site",
            formula="AVG(throughput_mbps)",
            unit="Mbps",
            target_value=100.0,
            warning_threshold=50.0,
            critical_threshold=20.0,
            input_metrics=["throughput_mbps"],
            aggregation_type=AggregationType.AVG
        )
        
        # 5. Latency
        self.kpi_formulas["network_latency"] = KPIFormula(
            kpi_id="network_latency",
            kpi_name="Network Latency",
            category=KPICategory.NETWORK,
            description="Average network latency",
            formula="AVG(latency_ms)",
            unit="ms",
            target_value=20.0,
            warning_threshold=50.0,
            critical_threshold=100.0,
            input_metrics=["latency_ms"],
            aggregation_type=AggregationType.AVG
        )
        
        # 6. Packet Loss Rate
        self.kpi_formulas["packet_loss"] = KPIFormula(
            kpi_id="packet_loss",
            kpi_name="Packet Loss Rate",
            category=KPICategory.NETWORK,
            description="Percentage of packets lost",
            formula="(lost_packets / total_packets) * 100",
            unit="%",
            target_value=0.1,
            warning_threshold=1.0,
            critical_threshold=5.0,
            input_metrics=["lost_packets", "total_packets"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 7. Handover Success Rate
        self.kpi_formulas["handover_success"] = KPIFormula(
            kpi_id="handover_success",
            kpi_name="Handover Success Rate",
            category=KPICategory.NETWORK,
            description="Percentage of successful handovers",
            formula="(successful_handovers / total_handover_attempts) * 100",
            unit="%",
            target_value=98.0,
            warning_threshold=95.0,
            critical_threshold=90.0,
            input_metrics=["handover_attempts", "handover_successes"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 8. VSWR (Voltage Standing Wave Ratio)
        self.kpi_formulas["vswr"] = KPIFormula(
            kpi_id="vswr",
            kpi_name="VSWR",
            category=KPICategory.NETWORK,
            description="Voltage Standing Wave Ratio",
            formula="AVG(vswr_ratio)",
            unit="ratio",
            target_value=1.2,
            warning_threshold=1.5,
            critical_threshold=2.0,
            input_metrics=["vswr_ratio"],
            aggregation_type=AggregationType.AVG
        )
        
        # 9. Signal Quality (RSRP for 4G/5G)
        self.kpi_formulas["signal_quality"] = KPIFormula(
            kpi_id="signal_quality",
            kpi_name="Signal Quality (RSRP)",
            category=KPICategory.NETWORK,
            description="Reference Signal Received Power",
            formula="AVG(rsrp_dbm)",
            unit="dBm",
            target_value=-85.0,
            warning_threshold=-100.0,
            critical_threshold=-110.0,
            input_metrics=["rsrp_dbm"],
            aggregation_type=AggregationType.AVG
        )
        
        # 10. Spectral Efficiency
        self.kpi_formulas["spectral_efficiency"] = KPIFormula(
            kpi_id="spectral_efficiency",
            kpi_name="Spectral Efficiency",
            category=KPICategory.NETWORK,
            description="Bits per second per Hertz",
            formula="total_throughput_bps / total_bandwidth_hz",
            unit="bps/Hz",
            target_value=5.0,
            warning_threshold=3.0,
            critical_threshold=2.0,
            input_metrics=["throughput_bps", "bandwidth_hz"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 11-20: Additional network KPIs
        additional_network_kpis = [
            ("rf_quality", "RF Quality Index", "Composite RF quality score", "ratio", 0.9, 0.7, 0.5),
            ("interference_level", "Interference Level", "Average interference", "dBm", -100.0, -90.0, -80.0),
            ("cell_availability", "Cell Availability", "Percentage of time cell is available", "%", 99.5, 99.0, 98.0),
            ("traffic_volume", "Traffic Volume", "Total traffic volume", "GB", 1000.0, 500.0, 200.0),
            ("user_throughput", "User Throughput", "Average throughput per user", "Mbps", 10.0, 5.0, 2.0),
            ("rrc_setup_success", "RRC Setup Success Rate", "Percentage of successful RRC setups", "%", 99.0, 97.0, 95.0),
            ("bearer_setup_success", "Bearer Setup Success Rate", "Percentage of successful bearer setups", "%", 98.0, 95.0, 90.0),
            ("retainability", "Session Retainability", "Percentage of sessions retained", "%", 99.5, 99.0, 98.0),
            ("mobility_success", "Mobility Success Rate", "Overall mobility success rate", "%", 97.0, 95.0, 90.0),
            ("voip_quality", "VoIP Quality (MOS)", "Mean Opinion Score for VoIP", "score", 4.0, 3.5, 3.0)
        ]
        
        for i, (kpi_id, name, desc, unit, target, warn, crit) in enumerate(additional_network_kpis, 11):
            self.kpi_formulas[kpi_id] = KPIFormula(
                kpi_id=kpi_id,
                kpi_name=name,
                category=KPICategory.NETWORK,
                description=desc,
                formula=f"AVG({kpi_id})",
                unit=unit,
                target_value=target,
                warning_threshold=warn,
                critical_threshold=crit,
                input_metrics=[kpi_id],
                aggregation_type=AggregationType.AVG
            )
        
        # =============================================================================
        # ENERGY & SUSTAINABILITY KPIs (15 KPIs)
        # =============================================================================
        
        # 21. Power Usage Effectiveness (PUE)
        self.kpi_formulas["pue"] = KPIFormula(
            kpi_id="pue",
            kpi_name="Power Usage Effectiveness",
            category=KPICategory.ENERGY,
            description="Total facility energy / IT equipment energy",
            formula="total_power_consumption / it_power_consumption",
            unit="ratio",
            target_value=1.5,
            warning_threshold=2.0,
            critical_threshold=2.5,
            input_metrics=["total_power", "it_power"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 22. Energy Efficiency
        self.kpi_formulas["energy_efficiency"] = KPIFormula(
            kpi_id="energy_efficiency",
            kpi_name="Energy Efficiency",
            category=KPICategory.ENERGY,
            description="Data processed per unit of energy consumed",
            formula="(total_data_gb / total_energy_kwh) * 100",
            unit="GB/kWh",
            target_value=100.0,
            warning_threshold=50.0,
            critical_threshold=25.0,
            input_metrics=["data_volume_gb", "energy_consumption_kwh"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 23. Battery Health Score
        self.kpi_formulas["battery_health"] = KPIFormula(
            kpi_id="battery_health",
            kpi_name="Battery Health Score",
            category=KPICategory.ENERGY,
            description="Composite battery health indicator",
            formula="(battery_capacity_pct * 0.4 + battery_efficiency_pct * 0.3 + (100 - battery_age_months) * 0.3)",
            unit="%",
            target_value=90.0,
            warning_threshold=70.0,
            critical_threshold=50.0,
            input_metrics=["battery_capacity", "battery_efficiency", "battery_age"],
            aggregation_type=AggregationType.WEIGHTED_AVG
        )
        
        # 24. Fuel Consumption Rate
        self.kpi_formulas["fuel_consumption"] = KPIFormula(
            kpi_id="fuel_consumption",
            kpi_name="Fuel Consumption Rate",
            category=KPICategory.ENERGY,
            description="Daily fuel consumption rate",
            formula="SUM(fuel_consumption_liters_per_day)",
            unit="L/day",
            target_value=50.0,
            warning_threshold=75.0,
            critical_threshold=100.0,
            input_metrics=["fuel_consumption"],
            aggregation_type=AggregationType.SUM
        )
        
        # 25. Solar Efficiency
        self.kpi_formulas["solar_efficiency"] = KPIFormula(
            kpi_id="solar_efficiency",
            kpi_name="Solar Panel Efficiency",
            category=KPICategory.ENERGY,
            description="Solar energy conversion efficiency",
            formula="(actual_solar_output_kwh / theoretical_max_kwh) * 100",
            unit="%",
            target_value=20.0,
            warning_threshold=15.0,
            critical_threshold=10.0,
            input_metrics=["solar_output", "solar_irradiance"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 26-35: Additional energy KPIs
        additional_energy_kpis = [
            ("carbon_footprint", "Carbon Footprint", "CO2 emissions", "kg CO2/day", 100.0, 150.0, 200.0),
            ("renewable_ratio", "Renewable Energy Ratio", "Percentage of renewable energy", "%", 30.0, 20.0, 10.0),
            ("cooling_efficiency", "Cooling Efficiency", "Cooling system efficiency", "ratio", 3.0, 2.5, 2.0),
            ("ups_efficiency", "UPS Efficiency", "UPS system efficiency", "%", 95.0, 90.0, 85.0),
            ("generator_efficiency", "Generator Efficiency", "Generator fuel efficiency", "%", 35.0, 30.0, 25.0),
            ("power_factor", "Power Factor", "AC power efficiency", "ratio", 0.95, 0.90, 0.85),
            ("energy_cost", "Energy Cost per GB", "Cost per gigabyte processed", "$/GB", 0.10, 0.15, 0.20),
            ("peak_demand", "Peak Power Demand", "Maximum power demand", "kW", 100.0, 150.0, 200.0),
            ("load_factor", "Load Factor", "Average vs peak load ratio", "ratio", 0.8, 0.6, 0.4),
            ("temperature_efficiency", "Temperature Efficiency", "Equipment temperature vs optimal", "score", 1.0, 0.8, 0.6)
        ]
        
        for i, (kpi_id, name, desc, unit, target, warn, crit) in enumerate(additional_energy_kpis, 26):
            self.kpi_formulas[kpi_id] = KPIFormula(
                kpi_id=kpi_id,
                kpi_name=name,
                category=KPICategory.ENERGY,
                description=desc,
                formula=f"AVG({kpi_id})",
                unit=unit,
                target_value=target,
                warning_threshold=warn,
                critical_threshold=crit,
                input_metrics=[kpi_id],
                aggregation_type=AggregationType.AVG
            )
        
        # =============================================================================
        # OPERATIONAL EXCELLENCE KPIs (10 KPIs)
        # =============================================================================
        
        # 36. Mean Time To Repair (MTTR)
        self.kpi_formulas["mttr"] = KPIFormula(
            kpi_id="mttr",
            kpi_name="Mean Time To Repair",
            category=KPICategory.OPERATIONAL,
            description="Average time to repair incidents",
            formula="AVG(resolution_time_minutes)",
            unit="minutes",
            target_value=120.0,  # 2 hours
            warning_threshold=240.0,  # 4 hours
            critical_threshold=480.0,  # 8 hours
            input_metrics=["incident_resolution_time"],
            aggregation_type=AggregationType.AVG
        )
        
        # 37. Mean Time Between Failures (MTBF)
        self.kpi_formulas["mtbf"] = KPIFormula(
            kpi_id="mtbf",
            kpi_name="Mean Time Between Failures",
            category=KPICategory.OPERATIONAL,
            description="Average time between equipment failures",
            formula="total_operational_time / number_of_failures",
            unit="hours",
            target_value=8760.0,  # 1 year
            warning_threshold=4380.0,  # 6 months
            critical_threshold=2190.0,  # 3 months
            input_metrics=["operational_time", "failure_count"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 38. First Call Resolution Rate
        self.kpi_formulas["fcr"] = KPIFormula(
            kpi_id="fcr",
            kpi_name="First Call Resolution Rate",
            category=KPICategory.OPERATIONAL,
            description="Percentage of issues resolved on first contact",
            formula="(first_call_resolutions / total_calls) * 100",
            unit="%",
            target_value=80.0,
            warning_threshold=60.0,
            critical_threshold=40.0,
            input_metrics=["first_call_resolutions", "total_support_calls"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 39. Preventive Maintenance Compliance
        self.kpi_formulas["pm_compliance"] = KPIFormula(
            kpi_id="pm_compliance",
            kpi_name="Preventive Maintenance Compliance",
            category=KPICategory.OPERATIONAL,
            description="Percentage of scheduled maintenance completed on time",
            formula="(completed_pm_tasks / scheduled_pm_tasks) * 100",
            unit="%",
            target_value=95.0,
            warning_threshold=85.0,
            critical_threshold=75.0,
            input_metrics=["pm_completed", "pm_scheduled"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 40. Incident Recurrence Rate
        self.kpi_formulas["incident_recurrence"] = KPIFormula(
            kpi_id="incident_recurrence",
            kpi_name="Incident Recurrence Rate",
            category=KPICategory.OPERATIONAL,
            description="Percentage of incidents that recur within 30 days",
            formula="(recurring_incidents / total_incidents) * 100",
            unit="%",
            target_value=5.0,
            warning_threshold=10.0,
            critical_threshold=15.0,
            input_metrics=["recurring_incidents", "total_incidents"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 41-45: Additional operational KPIs
        additional_operational_kpis = [
            ("change_success_rate", "Change Success Rate", "Successful changes percentage", "%", 95.0, 90.0, 85.0),
            ("sla_compliance", "SLA Compliance", "Overall SLA compliance", "%", 99.0, 95.0, 90.0),
            ("resource_utilization", "Resource Utilization", "Technical resource utilization", "%", 80.0, 60.0, 40.0),
            ("knowledge_base_usage", "Knowledge Base Usage", "KB article usage rate", "%", 70.0, 50.0, 30.0),
            ("staff_productivity", "Staff Productivity", "Tasks completed per technician", "tasks/day", 10.0, 8.0, 5.0)
        ]
        
        for i, (kpi_id, name, desc, unit, target, warn, crit) in enumerate(additional_operational_kpis, 41):
            self.kpi_formulas[kpi_id] = KPIFormula(
                kpi_id=kpi_id,
                kpi_name=name,
                category=KPICategory.OPERATIONAL,
                description=desc,
                formula=f"AVG({kpi_id})",
                unit=unit,
                target_value=target,
                warning_threshold=warn,
                critical_threshold=crit,
                input_metrics=[kpi_id],
                aggregation_type=AggregationType.AVG
            )
        
        # =============================================================================
        # FINANCIAL KPIs (10 KPIs)
        # =============================================================================
        
        # 46. Revenue per Site
        self.kpi_formulas["revenue_per_site"] = KPIFormula(
            kpi_id="revenue_per_site",
            kpi_name="Revenue per Site",
            category=KPICategory.FINANCIAL,
            description="Monthly revenue generated per site",
            formula="total_revenue / number_of_sites",
            unit="$/month",
            target_value=10000.0,
            warning_threshold=7500.0,
            critical_threshold=5000.0,
            input_metrics=["monthly_revenue", "active_sites"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 47. OPEX per Site
        self.kpi_formulas["opex_per_site"] = KPIFormula(
            kpi_id="opex_per_site",
            kpi_name="OPEX per Site",
            category=KPICategory.FINANCIAL,
            description="Monthly operational expenditure per site",
            formula="total_opex / number_of_sites",
            unit="$/month",
            target_value=3000.0,
            warning_threshold=4000.0,
            critical_threshold=5000.0,
            input_metrics=["monthly_opex", "active_sites"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 48. SLA Penalty Cost
        self.kpi_formulas["sla_penalty"] = KPIFormula(
            kpi_id="sla_penalty",
            kpi_name="SLA Penalty Cost",
            category=KPICategory.FINANCIAL,
            description="Monthly SLA penalty costs",
            formula="SUM(sla_penalty_amount)",
            unit="$/month",
            target_value=0.0,
            warning_threshold=5000.0,
            critical_threshold=10000.0,
            input_metrics=["sla_penalties"],
            aggregation_type=AggregationType.SUM
        )
        
        # 49. Cost per Incident
        self.kpi_formulas["cost_per_incident"] = KPIFormula(
            kpi_id="cost_per_incident",
            kpi_name="Cost per Incident",
            category=KPICategory.FINANCIAL,
            description="Average cost to resolve an incident",
            formula="(labor_cost + parts_cost + penalty_cost) / number_of_incidents",
            unit="$/incident",
            target_value=500.0,
            warning_threshold=750.0,
            critical_threshold=1000.0,
            input_metrics=["incident_costs", "incident_count"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 50. ROI on Energy Efficiency
        self.kpi_formulas["energy_roi"] = KPIFormula(
            kpi_id="energy_roi",
            kpi_name="ROI on Energy Efficiency",
            category=KPICategory.FINANCIAL,
            description="Return on investment for energy efficiency initiatives",
            formula="((energy_savings - investment_cost) / investment_cost) * 100",
            unit="%",
            target_value=20.0,
            warning_threshold=10.0,
            critical_threshold=5.0,
            input_metrics=["energy_savings", "efficiency_investment"],
            aggregation_type=AggregationType.RATIO
        )
        
        # 51-55: Additional financial KPIs
        additional_financial_kpis = [
            ("profit_margin", "Profit Margin", "Net profit margin per site", "%", 25.0, 15.0, 10.0),
            ("capex_efficiency", "CAPEX Efficiency", "Revenue per CAPEX invested", "ratio", 3.0, 2.0, 1.5),
            ("customer_acquisition_cost", "Customer Acquisition Cost", "Cost to acquire new customer", "$", 1000.0, 1500.0, 2000.0),
            ("churn_cost", "Customer Churn Cost", "Monthly cost due to customer churn", "$", 5000.0, 10000.0, 15000.0),
            ("maintenance_cost_ratio", "Maintenance Cost Ratio", "Maintenance cost vs revenue", "%", 15.0, 20.0, 25.0)
        ]
        
        for i, (kpi_id, name, desc, unit, target, warn, crit) in enumerate(additional_financial_kpis, 51):
            self.kpi_formulas[kpi_id] = KPIFormula(
                kpi_id=kpi_id,
                kpi_name=name,
                category=KPICategory.FINANCIAL,
                description=desc,
                formula=f"AVG({kpi_id})",
                unit=unit,
                target_value=target,
                warning_threshold=warn,
                critical_threshold=crit,
                input_metrics=[kpi_id],
                aggregation_type=AggregationType.AVG
            )
        
        logger.info(f"Initialized {len(self.kpi_formulas)} KPI formulas across 4 categories")
    
    async def initialize(self):
        """Initialize KPI calculation engine"""
        logger.info("Initializing KPI Calculation Engine...")
        
        # Load additional KPIs from database
        await self.load_custom_kpis()
        
        # Start KPI calculation loops for each formula
        for kpi_id, formula in self.kpi_formulas.items():
            if formula.enabled:
                task = asyncio.create_task(self.kpi_calculation_loop(formula))
                self.calculation_tasks[kpi_id] = task
        
        # Start prediction model training
        asyncio.create_task(self.model_training_loop())
        
        # Start health monitoring
        asyncio.create_task(self.health_monitoring_loop())
        
        logger.info(f"KPI Calculation Engine initialized with {len(self.calculation_tasks)} active calculations")
    
    async def load_custom_kpis(self):
        """Load custom KPI definitions from database"""
        try:
            query = """
            SELECT kpi_id, kpi_name, kpi_category, calculation_formula, 
                   unit, target_value, warning_threshold, critical_threshold,
                   calculation_interval, enabled, tenant_specific, metadata
            FROM kpi_definitions
            WHERE enabled = true
            """
            
            custom_kpis = await self.db.execute_query(query)
            
            for kpi_data in custom_kpis:
                if kpi_data['kpi_id'] not in self.kpi_formulas:
                    # Convert database record to KPIFormula
                    formula = KPIFormula(
                        kpi_id=kpi_data['kpi_id'],
                        kpi_name=kpi_data['kpi_name'],
                        category=KPICategory(kpi_data['kpi_category']),
                        description=kpi_data.get('description', ''),
                        formula=kpi_data['calculation_formula'],
                        unit=kpi_data['unit'] or '',
                        target_value=kpi_data.get('target_value'),
                        warning_threshold=kpi_data.get('warning_threshold'),
                        critical_threshold=kpi_data.get('critical_threshold'),
                        calculation_interval=kpi_data.get('calculation_interval', 300),
                        enabled=kpi_data.get('enabled', True),
                        tenant_specific=kpi_data.get('tenant_specific', False),
                        metadata=kpi_data.get('metadata', {})
                    )
                    
                    self.kpi_formulas[kpi_data['kpi_id']] = formula
                    logger.info(f"Loaded custom KPI: {kpi_data['kpi_name']}")
        
        except Exception as e:
            logger.error(f"Error loading custom KPIs: {e}")
    
    async def kpi_calculation_loop(self, formula: KPIFormula):
        """Continuous KPI calculation loop for a specific formula"""
        logger.info(f"Starting KPI calculation loop for: {formula.kpi_name}")
        
        while not shutdown_event.is_set():
            try:
                # Check if it's time to calculate this KPI
                if self.should_calculate_kpi(formula):
                    await self.calculate_kpi(formula)
                
                # Wait for next calculation cycle
                await asyncio.sleep(min(formula.calculation_interval, 60))
                
            except Exception as e:
                logger.error(f"Error in KPI calculation loop for {formula.kpi_id}: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def should_calculate_kpi(self, formula: KPIFormula) -> bool:
        """Check if KPI should be calculated now"""
        if not formula.enabled:
            return False
        
        last_calc = self.last_calculation_times.get(formula.kpi_id)
        if not last_calc:
            return True
        
        time_since_calc = (datetime.utcnow() - last_calc).total_seconds()
        return time_since_calc >= formula.calculation_interval
    
    async def calculate_kpi(self, formula: KPIFormula):
        """Calculate KPI value for all applicable sites"""
        start_time = time.time()
        
        try:
            logger.debug(f"Calculating KPI: {formula.kpi_name}")
            
            # Get sites to calculate KPI for
            sites = await self.get_applicable_sites(formula)
            
            calculated_count = 0
            
            for site in sites:
                try:
                    # Calculate KPI for this site
                    kpi_result = await self.calculate_site_kpi(formula, site)
                    
                    if kpi_result:
                        # Store KPI value
                        await self.store_kpi_value(formula, site, kpi_result)
                        
                        # Send to Kafka for real-time processing
                        await self.publish_kpi_value(formula, site, kpi_result)
                        
                        calculated_count += 1
                
                except Exception as e:
                    logger.error(f"Error calculating KPI {formula.kpi_id} for site {site.get('site_id')}: {e}")
            
            # Update calculation timestamp
            self.last_calculation_times[formula.kpi_id] = datetime.utcnow()
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_kpi_calculation(formula.kpi_id, duration)
            
            logger.debug(f"KPI {formula.kpi_name} calculated for {calculated_count} sites in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error calculating KPI {formula.kpi_id}: {e}")
    
    async def get_applicable_sites(self, formula: KPIFormula) -> List[Dict[str, Any]]:
        """Get sites applicable for KPI calculation"""
        try:
            query = """
            SELECT site_id, site_code, site_name, tenant_id, technology, status
            FROM sites
            WHERE status = 'active'
            """
            
            params = []
            
            # Add tenant filter for tenant-specific KPIs
            if formula.tenant_specific and 'tenant_id' in formula.filters:
                query += " AND tenant_id = $1"
                params.append(formula.filters['tenant_id'])
            
            # Add technology filter if specified
            if 'technology' in formula.filters:
                tech_filter = formula.filters['technology']
                if isinstance(tech_filter, list):
                    placeholders = ','.join([f'${i+len(params)+1}' for i in range(len(tech_filter))])
                    query += f" AND technology ?| array[{placeholders}]"
                    params.extend(tech_filter)
                else:
                    query += f" AND technology ? ${len(params)+1}"
                    params.append(tech_filter)
            
            sites = await self.db.execute_query(query, *params)
            return [dict(site) for site in sites]
            
        except Exception as e:
            logger.error(f"Error getting applicable sites for KPI {formula.kpi_id}: {e}")
            return []
    
    async def calculate_site_kpi(self, formula: KPIFormula, site: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate KPI value for a specific site"""
        try:
            # Get input data for KPI calculation
            input_data = await self.get_input_data(formula, site)
            
            if not input_data:
                logger.debug(f"No input data for KPI {formula.kpi_id} at site {site['site_code']}")
                return None
            
            # Calculate KPI value based on formula
            kpi_value = await self.execute_kpi_formula(formula, input_data)
            
            if kpi_value is None:
                return None
            
            # Calculate deviation from target
            deviation_pct = None
            if formula.target_value:
                deviation_pct = ((kpi_value - formula.target_value) / formula.target_value) * 100
            
            # Determine quality score
            quality_score = self.calculate_quality_score(kpi_value, formula)
            
            # Determine trend
            trend = await self.calculate_trend(formula, site, kpi_value)
            
            return {
                'kpi_value': kpi_value,
                'target_value': formula.target_value,
                'deviation_pct': deviation_pct,
                'quality_score': quality_score,
                'trend': trend,
                'calculation_time': datetime.utcnow(),
                'input_data_points': len(input_data)
            }
            
        except Exception as e:
            logger.error(f"Error calculating site KPI {formula.kpi_id} for {site['site_code']}: {e}")
            return None
    
    async def get_input_data(self, formula: KPIFormula, site: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get input data for KPI calculation"""
        try:
            # Time window for data collection
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(seconds=formula.calculation_interval * 2)  # 2x window for reliability
            
            input_data = []
            
            # Collect network metrics
            if any(metric in ['availability_pct', 'throughput_mbps', 'latency_ms'] for metric in formula.input_metrics):
                network_query = """
                SELECT time, metric_name, metric_value, quality_score
                FROM network_metrics
                WHERE site_id = $1 AND time >= $2 AND time <= $3
                AND metric_name = ANY($4)
                ORDER BY time DESC
                """
                
                network_metrics = formula.input_metrics
                network_data = await self.db.execute_query(
                    network_query, 
                    site['site_id'], start_time, end_time, network_metrics
                )
                input_data.extend([dict(row) for row in network_data])
            
            # Collect energy metrics
            if any(metric in ['power_consumption', 'battery_soc', 'fuel_level'] for metric in formula.input_metrics):
                energy_query = """
                SELECT time, metric_name, metric_value, efficiency_score
                FROM energy_metrics
                WHERE site_id = $1 AND time >= $2 AND time <= $3
                AND metric_name = ANY($4)
                ORDER BY time DESC
                """
                
                energy_metrics = [m for m in formula.input_metrics if 'energy' in m or 'power' in m or 'battery' in m]
                if energy_metrics:
                    energy_data = await self.db.execute_query(
                        energy_query,
                        site['site_id'], start_time, end_time, energy_metrics
                    )
                    input_data.extend([dict(row) for row in energy_data])
            
            # Collect event/incident data
            if any(metric in ['incidents', 'alarms', 'outages'] for metric in formula.input_metrics):
                events_query = """
                SELECT time, event_type, severity, resolved, 
                       EXTRACT(EPOCH FROM (COALESCE(resolved_at, NOW()) - time))/60 as duration_minutes
                FROM events
                WHERE site_id = $1 AND time >= $2 AND time <= $3
                ORDER BY time DESC
                """
                
                events_data = await self.db.execute_query(
                    events_query,
                    site['site_id'], start_time, end_time
                )
                input_data.extend([dict(row) for row in events_data])
            
            return input_data
            
        except Exception as e:
            logger.error(f"Error getting input data for KPI {formula.kpi_id}: {e}")
            return []
    
    async def execute_kpi_formula(self, formula: KPIFormula, input_data: List[Dict[str, Any]]) -> Optional[float]:
        """Execute KPI calculation formula"""
        try:
            if not input_data:
                return None
            
            # Convert to pandas DataFrame for easier manipulation
            df = pd.DataFrame(input_data)
            
            # Execute based on aggregation type
            if formula.aggregation_type == AggregationType.AVG:
                if 'metric_value' in df.columns:
                    return float(df['metric_value'].mean())
            
            elif formula.aggregation_type == AggregationType.SUM:
                if 'metric_value' in df.columns:
                    return float(df['metric_value'].sum())
            
            elif formula.aggregation_type == AggregationType.MAX:
                if 'metric_value' in df.columns:
                    return float(df['metric_value'].max())
            
            elif formula.aggregation_type == AggregationType.MIN:
                if 'metric_value' in df.columns:
                    return float(df['metric_value'].min())
            
            elif formula.aggregation_type == AggregationType.COUNT:
                return float(len(df))
            
            elif formula.aggregation_type == AggregationType.RATIO:
                # Execute specific ratio calculations
                return await self.calculate_ratio_kpi(formula, df)
            
            elif formula.aggregation_type == AggregationType.WEIGHTED_AVG:
                return await self.calculate_weighted_avg_kpi(formula, df)
            
            # Custom formula execution for complex calculations
            return await self.execute_custom_formula(formula, df)
            
        except Exception as e:
            logger.error(f"Error executing KPI formula {formula.kpi_id}: {e}")
            return None
    
    async def calculate_ratio_kpi(self, formula: KPIFormula, df: pd.DataFrame) -> Optional[float]:
        """Calculate ratio-based KPIs"""
        try:
            if formula.kpi_id == "csr":  # Call Success Rate
                successful = df[df['event_type'] == 'call_success']['metric_value'].sum() or 0
                total = df['metric_value'].sum() or 0
                return (successful / total * 100) if total > 0 else None
            
            elif formula.kpi_id == "cdr":  # Call Drop Rate
                dropped = df[df['event_type'] == 'call_dropped']['metric_value'].sum() or 0
                established = df[df['event_type'] == 'call_established']['metric_value'].sum() or 0
                return (dropped / established * 100) if established > 0 else None
            
            elif formula.kpi_id == "packet_loss":
                lost = df[df['metric_name'] == 'lost_packets']['metric_value'].sum() or 0
                total = df[df['metric_name'] == 'total_packets']['metric_value'].sum() or 0
                return (lost / total * 100) if total > 0 else None
            
            # Generic ratio calculation
            numerator_metrics = [m for m in formula.input_metrics if 'success' in m or 'good' in m]
            denominator_metrics = [m for m in formula.input_metrics if 'total' in m or 'attempts' in m]
            
            if numerator_metrics and denominator_metrics:
                numerator = df[df['metric_name'].isin(numerator_metrics)]['metric_value'].sum()
                denominator = df[df['metric_name'].isin(denominator_metrics)]['metric_value'].sum()
                return (numerator / denominator * 100) if denominator > 0 else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating ratio KPI {formula.kpi_id}: {e}")
            return None
    
    async def calculate_weighted_avg_kpi(self, formula: KPIFormula, df: pd.DataFrame) -> Optional[float]:
        """Calculate weighted average KPIs"""
        try:
            if formula.kpi_id == "nur":  # Network Unavailability Rate
                # Weight by traffic volume or connection count
                if 'quality_score' in df.columns and 'metric_value' in df.columns:
                    weights = df['quality_score'].fillna(1.0)
                    values = df['metric_value']
                    return float(np.average(values, weights=weights))
            
            elif formula.kpi_id == "battery_health":
                # Weighted battery health calculation
                capacity_weight = 0.4
                efficiency_weight = 0.3
                age_weight = 0.3
                
                capacity_data = df[df['metric_name'] == 'battery_capacity']['metric_value']
                efficiency_data = df[df['metric_name'] == 'battery_efficiency']['metric_value']
                age_data = df[df['metric_name'] == 'battery_age']['metric_value']
                
                if not capacity_data.empty and not efficiency_data.empty:
                    capacity_avg = capacity_data.mean()
                    efficiency_avg = efficiency_data.mean()
                    age_avg = age_data.mean() if not age_data.empty else 0
                    
                    health_score = (
                        capacity_avg * capacity_weight +
                        efficiency_avg * efficiency_weight +
                        (100 - age_avg) * age_weight
                    )
                    return float(health_score)
            
            # Generic weighted average
            if 'metric_value' in df.columns and 'quality_score' in df.columns:
                weights = df['quality_score'].fillna(1.0)
                values = df['metric_value']
                return float(np.average(values, weights=weights))
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating weighted avg KPI {formula.kpi_id}: {e}")
            return None
    
    async def execute_custom_formula(self, formula: KPIFormula, df: pd.DataFrame) -> Optional[float]:
        """Execute custom formula calculations"""
        try:
            # For complex formulas, implement specific logic
            if formula.kpi_id == "mttr":
                # Calculate mean time to repair from incident data
                if 'duration_minutes' in df.columns:
                    resolved_incidents = df[df['resolved'] == True]
                    if not resolved_incidents.empty:
                        return float(resolved_incidents['duration_minutes'].mean())
            
            elif formula.kpi_id == "mtbf":
                # Calculate mean time between failures
                failure_events = df[df['severity'].isin(['CRITICAL', 'MAJOR'])]
                if len(failure_events) > 1:
                    time_diffs = failure_events['time'].diff().dt.total_seconds() / 3600  # hours
                    return float(time_diffs.mean())
            
            # Default to average if no specific logic
            if 'metric_value' in df.columns:
                return float(df['metric_value'].mean())
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing custom formula for KPI {formula.kpi_id}: {e}")
            return None
    
    def calculate_quality_score(self, kpi_value: float, formula: KPIFormula) -> float:
        """Calculate quality score based on KPI value and thresholds"""
        try:
            if not formula.target_value:
                return 1.0
            
            # Determine if higher or lower values are better
            higher_is_better = formula.kpi_id in ['csr', 'throughput', 'efficiency', 'availability']
            
            if higher_is_better:
                if formula.critical_threshold and kpi_value <= formula.critical_threshold:
                    return 0.0
                elif formula.warning_threshold and kpi_value <= formula.warning_threshold:
                    return 0.5
                elif kpi_value >= formula.target_value:
                    return 1.0
                else:
                    # Linear interpolation between warning and target
                    if formula.warning_threshold:
                        ratio = (kpi_value - formula.warning_threshold) / (formula.target_value - formula.warning_threshold)
                        return max(0.5, min(1.0, 0.5 + ratio * 0.5))
                    return 0.8
            else:  # Lower is better
                if formula.critical_threshold and kpi_value >= formula.critical_threshold:
                    return 0.0
                elif formula.warning_threshold and kpi_value >= formula.warning_threshold:
                    return 0.5
                elif kpi_value <= formula.target_value:
                    return 1.0
                else:
                    # Linear interpolation between target and warning
                    if formula.warning_threshold:
                        ratio = (formula.warning_threshold - kpi_value) / (formula.warning_threshold - formula.target_value)
                        return max(0.5, min(1.0, 0.5 + ratio * 0.5))
                    return 0.8
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5
    
    async def calculate_trend(self, formula: KPIFormula, site: Dict[str, Any], current_value: float) -> str:
        """Calculate KPI trend (IMPROVING, STABLE, DEGRADING)"""
        try:
            # Get historical values
            query = """
            SELECT kpi_value, time
            FROM kpi_values kv
            JOIN kpi_definitions kd ON kv.kpi_id = kd.kpi_id
            WHERE kd.kpi_name = $1 AND kv.site_id = $2
            AND kv.time >= $3
            ORDER BY kv.time DESC
            LIMIT 10
            """
            
            historical_data = await self.db.execute_query(
                query,
                formula.kpi_name,
                site['site_id'],
                datetime.utcnow() - timedelta(hours=24)
            )
            
            if len(historical_data) < 3:
                return "STABLE"
            
            values = [float(row['kpi_value']) for row in historical_data]
            values.append(current_value)
            
            # Calculate trend using linear regression
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            
            # Determine trend direction
            threshold = abs(current_value * 0.05)  # 5% threshold
            
            if abs(slope) < threshold:
                return "STABLE"
            elif slope > 0:
                # Higher is better for most KPIs
                higher_is_better = formula.kpi_id in ['csr', 'throughput', 'efficiency', 'availability']
                return "IMPROVING" if higher_is_better else "DEGRADING"
            else:
                higher_is_better = formula.kpi_id in ['csr', 'throughput', 'efficiency', 'availability']
                return "DEGRADING" if higher_is_better else "IMPROVING"
                
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return "STABLE"
    
    async def store_kpi_value(self, formula: KPIFormula, site: Dict[str, Any], kpi_result: Dict[str, Any]):
        """Store calculated KPI value in database"""
        try:
            # Get KPI definition ID
            kpi_def_query = "SELECT kpi_id FROM kpi_definitions WHERE kpi_name = $1"
            kpi_def = await self.db.execute_query_one(kpi_def_query, formula.kpi_name)
            
            if not kpi_def:
                # Create KPI definition if it doesn't exist
                kpi_def_id = await self.create_kpi_definition(formula)
            else:
                kpi_def_id = kpi_def['kpi_id']
            
            # Insert KPI value
            insert_query = """
            INSERT INTO kpi_values (
                time, kpi_id, site_id, tenant_id, kpi_value, target_value,
                deviation_pct, quality_score, trend, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """
            
            await self.db.execute_command(
                insert_query,
                kpi_result['calculation_time'],
                kpi_def_id,
                site['site_id'],
                site['tenant_id'],
                kpi_result['kpi_value'],
                kpi_result.get('target_value'),
                kpi_result.get('deviation_pct'),
                kpi_result['quality_score'],
                kpi_result['trend'],
                {
                    'input_data_points': kpi_result.get('input_data_points', 0),
                    'calculation_method': formula.aggregation_type.value
                }
            )
            
        except Exception as e:
            logger.error(f"Error storing KPI value: {e}")
    
    async def create_kpi_definition(self, formula: KPIFormula) -> str:
        """Create KPI definition in database"""
        try:
            insert_query = """
            INSERT INTO kpi_definitions (
                kpi_name, kpi_category, calculation_formula, unit,
                target_value, warning_threshold, critical_threshold,
                calculation_interval, enabled, tenant_specific, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING kpi_id
            """
            
            kpi_id = await self.db.execute_query_scalar(
                insert_query,
                formula.kpi_name,
                formula.category.value,
                formula.formula,
                formula.unit,
                formula.target_value,
                formula.warning_threshold,
                formula.critical_threshold,
                formula.calculation_interval,
                formula.enabled,
                formula.tenant_specific,
                formula.metadata
            )
            
            return kpi_id
            
        except Exception as e:
            logger.error(f"Error creating KPI definition: {e}")
            return None
    
    async def publish_kpi_value(self, formula: KPIFormula, site: Dict[str, Any], kpi_result: Dict[str, Any]):
        """Publish KPI value to Kafka for real-time processing"""
        try:
            message = {
                'kpi_id': formula.kpi_id,
                'kpi_name': formula.kpi_name,
                'kpi_category': formula.category.value,
                'site_id': site['site_id'],
                'site_code': site['site_code'],
                'tenant_id': site['tenant_id'],
                'kpi_value': kpi_result['kpi_value'],
                'target_value': kpi_result.get('target_value'),
                'quality_score': kpi_result['quality_score'],
                'trend': kpi_result['trend'],
                'unit': formula.unit,
                'calculation_time': kpi_result['calculation_time'].isoformat(),
                'deviation_pct': kpi_result.get('deviation_pct')
            }
            
            await self.producer.send_message(
                KAFKA_TOPICS['kpi_calculations'],
                message,
                key=f"{formula.kpi_id}:{site['site_id']}"
            )
            
            # Cache for immediate access
            cache_key = CacheKeys.kpi_values(site['site_id'])
            await self.cache.hset(cache_key, formula.kpi_id, message)
            await self.cache.expire(cache_key, 3600)  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"Error publishing KPI value: {e}")
    
    async def model_training_loop(self):
        """Train prediction models for KPIs"""
        while not shutdown_event.is_set():
            try:
                logger.info("Starting KPI prediction model training...")
                
                for kpi_id, formula in self.kpi_formulas.items():
                    if formula.enabled and formula.category in [KPICategory.NETWORK, KPICategory.ENERGY]:
                        await self.train_prediction_model(formula)
                
                # Wait 24 hours before next training cycle
                await asyncio.sleep(24 * 3600)
                
            except Exception as e:
                logger.error(f"Error in model training loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def train_prediction_model(self, formula: KPIFormula):
        """Train Prophet prediction model for a KPI"""
        try:
            # Get historical KPI data (last 30 days)
            query = """
            SELECT time as ds, AVG(kpi_value) as y
            FROM kpi_values kv
            JOIN kpi_definitions kd ON kv.kpi_id = kd.kpi_id
            WHERE kd.kpi_name = $1
            AND kv.time >= $2
            GROUP BY time
            ORDER BY time
            """
            
            historical_data = await self.db.execute_query(
                query,
                formula.kpi_name,
                datetime.utcnow() - timedelta(days=30)
            )
            
            if len(historical_data) < 100:  # Need sufficient data
                logger.debug(f"Insufficient data for KPI {formula.kpi_id} prediction model")
                return
            
            # Prepare data for Prophet
            df = pd.DataFrame([{
                'ds': pd.to_datetime(row['ds']),
                'y': float(row['y'])
            } for row in historical_data])
            
            # Train Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            model.fit(df)
            
            # Store model for predictions
            self.prediction_models[formula.kpi_id] = model
            
            logger.info(f"Trained prediction model for KPI: {formula.kpi_name}")
            
        except Exception as e:
            logger.error(f"Error training prediction model for KPI {formula.kpi_id}: {e}")
    
    async def predict_kpi_values(self, kpi_id: str, site_id: str = None, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Predict future KPI values"""
        try:
            if kpi_id not in self.prediction_models:
                logger.warning(f"No prediction model available for KPI {kpi_id}")
                return []
            
            model = self.prediction_models[kpi_id]
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=hours_ahead, freq='H')
            
            # Make predictions
            forecast = model.predict(future)
            
            # Extract predictions for future periods only
            predictions = []
            for i in range(len(forecast) - hours_ahead, len(forecast)):
                prediction = {
                    'kpi_id': kpi_id,
                    'site_id': site_id,
                    'prediction_time': forecast.iloc[i]['ds'].isoformat(),
                    'predicted_value': forecast.iloc[i]['yhat'],
                    'lower_bound': forecast.iloc[i]['yhat_lower'],
                    'upper_bound': forecast.iloc[i]['yhat_upper'],
                    'confidence': 0.8  # Prophet default confidence interval
                }
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting KPI values for {kpi_id}: {e}")
            return []
    
    async def health_monitoring_loop(self):
        """Monitor KPI engine health"""
        while not shutdown_event.is_set():
            try:
                # Count active calculations
                active_calculations = len([t for t in self.calculation_tasks.values() if not t.done()])
                
                # Count recent calculations
                recent_calculations = len([
                    t for t in self.last_calculation_times.values()
                    if (datetime.utcnow() - t).total_seconds() < 3600
                ])
                
                logger.info(f"KPI Engine Health: {active_calculations} active calculations, {recent_calculations} recent calculations")
                
                # Restart failed tasks
                for kpi_id, task in list(self.calculation_tasks.items()):
                    if task.done() and not shutdown_event.is_set():
                        logger.warning(f"Restarting failed KPI calculation for {kpi_id}")
                        formula = self.kpi_formulas[kpi_id]
                        new_task = asyncio.create_task(self.kpi_calculation_loop(formula))
                        self.calculation_tasks[kpi_id] = new_task
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(300)


# Health check web server
async def create_health_server():
    """Create health check web server"""
    app = web.Application()
    
    async def health_handler(request):
        """Health check endpoint"""
        global kpi_engine
        
        health_info = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'active_kpis': len(kpi_engine.calculation_tasks) if kpi_engine else 0,
            'prediction_models': len(kpi_engine.prediction_models) if kpi_engine else 0,
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
        }
        
        return web.json_response(health_info)
    
    app.router.add_get('/health', health_handler)
    
    return app


async def message_handler(topic, key, value, headers, timestamp, partition, offset):
    """Handle incoming Kafka messages"""
    global kpi_engine
    
    logger.info(f"Received message from {topic}: {key}")
    
    try:
        if topic == KAFKA_TOPICS['network_metrics'] or topic == KAFKA_TOPICS['energy_metrics']:
            # Trigger KPI calculations for affected site
            site_id = value.get('site_id')
            if site_id and kpi_engine:
                # Trigger immediate KPI calculation for this site
                for formula in kpi_engine.kpi_formulas.values():
                    if formula.enabled:
                        asyncio.create_task(
                            kpi_engine.calculate_site_kpi(formula, {'site_id': site_id})
                        )
    
    except Exception as e:
        logger.error(f"Error handling message: {e}")


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main application entry point"""
    global kpi_engine, health_server
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting Towerco KPI Calculation Worker...")
    
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
        site = web.TCPSite(health_runner, '0.0.0.0', 8002)
        await site.start()
        logger.info("Health server started on port 8002")
        
        # Initialize KPI Calculation Engine
        kpi_engine = KPICalculationEngine()
        await kpi_engine.initialize()
        
        # Start Kafka consumers
        await start_consumer(
            'kpi_triggers',
            [KAFKA_TOPICS['network_metrics'], KAFKA_TOPICS['energy_metrics']],
            'kpi_worker_group',
            message_handler
        )
        
        logger.info("KPI Calculation Worker started successfully")
        
        # Main loop
        while not shutdown_event.is_set():
            await asyncio.sleep(1)
    
    except Exception as e:
        logger.error(f"Failed to start KPI Calculation Worker: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()
    
    finally:
        logger.info("Shutting down KPI Calculation Worker...")
        
        # Stop calculation tasks
        if kpi_engine:
            for task in kpi_engine.calculation_tasks.values():
                task.cancel()
        
        # Cleanup health server
        if health_server:
            await health_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())