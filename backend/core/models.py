"""
Pydantic Models for Towerco AIOps Platform
Core data models and validation schemas
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum

# Enums
class SeverityLevel(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    MAJOR = "major" 
    MINOR = "minor"
    WARNING = "warning"

class EventType(str, Enum):
    """Event types"""
    ALARM = "ALARM"
    ALERT = "ALERT"
    NOTIFICATION = "NOTIFICATION"
    MAINTENANCE = "MAINTENANCE"

class DataType(str, Enum):
    """Data types for ingestion"""
    METRIC = "network_metric"
    EVENT = "event"
    CONFIGURATION = "configuration"

# Base Models
class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    created_at: datetime
    updated_at: Optional[datetime] = None

class MetadataMixin(BaseModel):
    """Mixin for metadata fields"""
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Site Management Models
class Site(BaseModel):
    """Site information model"""
    site_id: str = Field(..., description="Unique site identifier")
    site_code: str = Field(..., description="Site code/name")
    site_name: str = Field(..., description="Human readable site name")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    address: Optional[str] = None
    region: Optional[str] = None
    country: str
    technology: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="active")

class Equipment(BaseModel):
    """Equipment model"""
    equipment_id: str
    site_id: str
    equipment_type: str
    vendor: str
    model: str
    serial_number: Optional[str] = None
    software_version: Optional[str] = None
    installation_date: Optional[datetime] = None
    status: str = Field(default="operational")

# Data Ingestion Models
class NetworkMetric(BaseModel):
    """Network metric data model"""
    timestamp: datetime
    site_id: str
    technology: str
    metric_name: str
    metric_value: float
    unit: str
    quality_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Event(BaseModel):
    """Event/alarm data model"""
    timestamp: datetime
    event_type: EventType
    severity: SeverityLevel
    title: str
    description: Optional[str] = None
    site_id: Optional[str] = None
    source_system: str
    event_id: Optional[str] = None
    acknowledged: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Configuration(BaseModel):
    """Configuration data model"""
    timestamp: datetime
    config_type: str
    site_id: Optional[str] = None
    configuration: Dict[str, Any]
    version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# KPI Models
class KPICalculationRequest(BaseModel):
    """KPI calculation request model"""
    site_ids: Optional[List[str]] = None
    kpi_names: Optional[List[str]] = None
    time_range: str = Field(default="1h", description="Time range for calculation")
    priority: str = Field(default="normal", regex="^(low|normal|high|urgent)$")
    requested_by: str
    recalculate: bool = Field(default=False, description="Force recalculation of existing KPIs")

class KPIDefinition(BaseModel):
    """KPI definition model"""
    name: str = Field(..., description="KPI name/identifier")
    display_name: str
    description: str
    category: str = Field(..., regex="^(network|energy|operational|financial)$")
    unit: str
    formula: str = Field(..., description="Calculation formula")
    target_value: Optional[float] = None
    thresholds: Dict[str, float] = Field(default_factory=dict)
    calculation_interval: str = Field(default="5m")
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

class KPIAlert(BaseModel):
    """KPI alert model"""
    id: int
    kpi_name: str
    site_id: Optional[str] = None
    condition_type: str
    threshold_value: float
    current_value: float
    severity: SeverityLevel
    status: str = Field(regex="^(active|acknowledged|resolved)$")
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class KPIDashboard(BaseModel):
    """KPI dashboard model"""
    id: int
    name: str
    description: Optional[str] = None
    kpi_list: List[str]
    layout_config: Dict[str, Any] = Field(default_factory=dict)
    filters: Dict[str, Any] = Field(default_factory=dict)
    refresh_interval: int = Field(default=30, ge=5)
    is_public: bool = False
    created_at: datetime
    updated_at: datetime

# Data Source Models
class DataSourceConfig(BaseModel):
    """Data source configuration"""
    name: str = Field(..., description="Unique data source name")
    source_type: str = Field(..., description="Type: oss_network, snmp, rest_api, etc.")
    enabled: bool = True
    connection_config: Dict[str, Any]
    sync_interval: str = Field(default="5m")
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    retry_config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('sync_interval')
    def validate_interval(cls, v):
        """Validate sync interval format"""
        if not any(v.endswith(suffix) for suffix in ['s', 'm', 'h', 'd']):
            raise ValueError('Interval must end with s, m, h, or d')
        return v

class DataQualityReport(BaseModel):
    """Data quality assessment report"""
    source_name: str
    timestamp: datetime
    total_records: int
    valid_records: int
    quality_score: float = Field(ge=0.0, le=1.0)
    quality_issues: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    @validator('quality_score', always=True)
    def calculate_quality_score(cls, v, values):
        """Calculate quality score if not provided"""
        if v is None and 'total_records' in values and values['total_records'] > 0:
            return values['valid_records'] / values['total_records']
        return v

# Authentication Models
class User(BaseModel):
    """User model"""
    id: int
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    last_login: Optional[datetime] = None
    created_at: datetime
    
class UserCreate(BaseModel):
    """User creation model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    roles: List[str] = Field(default_factory=list)

class Token(BaseModel):
    """JWT token model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None

# Alert and Notification Models
class AlertRule(BaseModel):
    """Alert rule definition"""
    id: int
    name: str
    description: Optional[str] = None
    condition: str = Field(..., description="Alert condition (SQL-like)")
    severity: SeverityLevel
    enabled: bool = True
    notification_channels: List[str] = Field(default_factory=list)
    cooldown_period: str = Field(default="5m")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

class Notification(BaseModel):
    """Notification model"""
    id: int
    title: str
    message: str
    severity: SeverityLevel
    channels: List[str]
    recipients: List[str]
    status: str = Field(default="pending")
    sent_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ITSM Integration Models
class ITSMTicket(BaseModel):
    """ITSM ticket model"""
    ticket_id: str
    ticket_type: str = Field(regex="^(incident|request|change|problem)$")
    title: str
    description: str
    priority: str = Field(regex="^(low|medium|high|critical)$")
    status: str
    assigned_to: Optional[str] = None
    site_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ITSMIntegration(BaseModel):
    """ITSM integration configuration"""
    name: str
    system_type: str = Field(default="servicenow")
    endpoint: str
    credentials: Dict[str, str]
    sync_enabled: bool = True
    sync_interval: str = Field(default="5m")
    field_mappings: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Analytics and Reporting Models
class Report(BaseModel):
    """Report model"""
    id: int
    name: str
    description: Optional[str] = None
    report_type: str = Field(regex="^(kpi|sla|availability|performance|financial)$")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    schedule: Optional[str] = None
    recipients: List[str] = Field(default_factory=list)
    format: str = Field(default="pdf", regex="^(pdf|xlsx|csv|json)$")
    enabled: bool = True
    last_generated: Optional[datetime] = None
    created_at: datetime

class SLADefinition(BaseModel):
    """SLA definition model"""
    id: int
    name: str
    description: str
    site_ids: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(..., description="List of KPI names")
    targets: Dict[str, float] = Field(..., description="Target values for metrics")
    measurement_period: str = Field(default="monthly")
    penalty_rules: Dict[str, Any] = Field(default_factory=dict)
    active: bool = True
    start_date: datetime
    end_date: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Root Cause Analysis Models
class RCAResult(BaseModel):
    """Root Cause Analysis result"""
    id: int
    incident_id: str
    analysis_type: str = Field(default="automated")
    root_causes: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    contributing_factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    analysis_duration: float = Field(..., description="Analysis time in seconds")
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AnomalyDetection(BaseModel):
    """Anomaly detection result"""
    id: int
    metric_name: str
    site_id: str
    anomaly_score: float = Field(ge=0.0, le=1.0)
    anomaly_type: str = Field(regex="^(point|contextual|collective)$")
    detected_at: datetime
    value: float
    expected_range: Dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# System Health Models
class SystemHealth(BaseModel):
    """System health status"""
    component: str
    status: str = Field(regex="^(healthy|degraded|unhealthy|unknown)$")
    last_check: datetime
    response_time: Optional[float] = None
    error_rate: Optional[float] = None
    message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PerformanceMetrics(BaseModel):
    """System performance metrics"""
    timestamp: datetime
    cpu_usage: float = Field(ge=0.0, le=100.0)
    memory_usage: float = Field(ge=0.0, le=100.0)
    disk_usage: float = Field(ge=0.0, le=100.0)
    network_io: Dict[str, float] = Field(default_factory=dict)
    response_times: Dict[str, float] = Field(default_factory=dict)
    error_counts: Dict[str, int] = Field(default_factory=dict)

# Validation Helpers
def validate_time_range(time_range: str) -> bool:
    """Validate time range format"""
    valid_units = ['s', 'm', 'h', 'd', 'w']
    if not any(time_range.endswith(unit) for unit in valid_units):
        return False
    
    try:
        int(time_range[:-1])
        return True
    except ValueError:
        return False

def validate_cron_expression(cron_expr: str) -> bool:
    """Basic cron expression validation"""
    parts = cron_expr.strip().split()
    return len(parts) == 5

# Response Models
class APIResponse(BaseModel):
    """Standard API response"""
    success: bool = True
    message: str = "Request processed successfully"
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PaginatedResponse(BaseModel):
    """Paginated API response"""
    items: List[Any]
    total_count: int
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=1000)
    total_pages: int
    has_next: bool
    has_prev: bool
    
    @validator('total_pages', always=True)
    def calculate_total_pages(cls, v, values):
        """Calculate total pages"""
        if 'total_count' in values and 'page_size' in values:
            import math
            return math.ceil(values['total_count'] / values['page_size'])
        return v
        
    @validator('has_next', always=True)
    def calculate_has_next(cls, v, values):
        """Calculate has_next"""
        if 'page' in values and 'total_pages' in values:
            return values['page'] < values['total_pages']
        return False
        
    @validator('has_prev', always=True)
    def calculate_has_prev(cls, v, values):
        """Calculate has_prev"""
        if 'page' in values:
            return values['page'] > 1
        return False