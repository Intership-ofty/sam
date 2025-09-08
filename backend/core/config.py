"""
Configuration management for Towerco AIOps Platform
"""

from typing import List, Optional, Any, Dict
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Towerco AIOps Platform"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Security
    SECRET_KEY: str = Field(env="SECRET_KEY", default="your-super-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    AUTH_MODE: str = Field(default="keycloak", env="AUTH_MODE")
    
    # CORS
    ALLOWED_ORIGINS: str = Field(
        default="*",
        env="ALLOWED_ORIGINS",
        description="Comma-separated list of allowed origins for CORS"
    )
    
    # Database
    DATABASE_URL: str = Field(env="DATABASE_URL")
    DB_POOL_SIZE: int = Field(default=10, env="DB_POOL_SIZE")
    DB_POOL_MAX_OVERFLOW: int = Field(default=20, env="DB_POOL_MAX_OVERFLOW")
    DB_POOL_TIMEOUT: int = Field(default=30, env="DB_POOL_TIMEOUT")
    
    # Redis
    REDIS_URL: str = Field(env="REDIS_URL")
    REDIS_MAX_CONNECTIONS: int = Field(default=100, env="REDIS_MAX_CONNECTIONS")
    
    # Kafka/Redpanda
    KAFKA_BROKERS: str = Field(env="KAFKA_BROKERS")
    KAFKA_TOPIC_PREFIX: str = Field(default="towerco", env="KAFKA_TOPIC_PREFIX")
    
    # MinIO/S3
    MINIO_ENDPOINT: str = Field(env="MINIO_ENDPOINT")
    MINIO_ACCESS_KEY: str = Field(env="MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = Field(env="MINIO_SECRET_KEY")
    MINIO_BUCKET_PREFIX: str = Field(default="towerco", env="MINIO_BUCKET_PREFIX")
    MINIO_SECURE: bool = Field(default=False, env="MINIO_SECURE")
    
    # Keycloak
    KEYCLOAK_SERVER_URL: str = Field(env="KEYCLOAK_SERVER_URL")
    KEYCLOAK_REALM: str = Field(default="towerco", env="KEYCLOAK_REALM")
    KEYCLOAK_CLIENT_ID: str = Field(default="towerco-aiops", env="KEYCLOAK_CLIENT_ID")
    KEYCLOAK_CLIENT_SECRET: Optional[str] = Field(default=None, env="KEYCLOAK_CLIENT_SECRET")
    
    # ServiceNow
    SERVICENOW_INSTANCE: Optional[str] = Field(default=None, env="SERVICENOW_INSTANCE")
    SERVICENOW_USERNAME: Optional[str] = Field(default=None, env="SERVICENOW_USERNAME")
    SERVICENOW_PASSWORD: Optional[str] = Field(default=None, env="SERVICENOW_PASSWORD")
    
    # Email
    SMTP_HOST: Optional[str] = Field(default=None, env="SMTP_HOST")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USERNAME: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    SMTP_FROM_EMAIL: str = Field(default="noreply@towerco.com", env="SMTP_FROM_EMAIL")
    
    # OpenTelemetry
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = Field(default=None, env="OTEL_EXPORTER_OTLP_ENDPOINT")
    OTEL_SERVICE_NAME: str = Field(default="towerco-backend-api", env="OTEL_SERVICE_NAME")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=1000, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    
    # KPI Calculation
    KPI_CALCULATION_INTERVAL_SECONDS: int = Field(default=60, env="KPI_CALCULATION_INTERVAL_SECONDS")
    KPI_RETENTION_DAYS: int = Field(default=90, env="KPI_RETENTION_DAYS")
    
    # AIOps
    AIOPS_MODEL_RETRAIN_HOURS: int = Field(default=24, env="AIOPS_MODEL_RETRAIN_HOURS")
    AIOPS_PREDICTION_HORIZON_HOURS: int = Field(default=48, env="AIOPS_PREDICTION_HORIZON_HOURS")
    AIOPS_CORRELATION_WINDOW_MINUTES: int = Field(default=30, env="AIOPS_CORRELATION_WINDOW_MINUTES")
    
    # Data Retention
    METRICS_RETENTION_DAYS: int = Field(default=30, env="METRICS_RETENTION_DAYS")
    EVENTS_RETENTION_DAYS: int = Field(default=90, env="EVENTS_RETENTION_DAYS")
    LOGS_RETENTION_DAYS: int = Field(default=7, env="LOGS_RETENTION_DAYS")
    
    @validator("KAFKA_BROKERS")
    def validate_kafka_brokers(cls, v):
        if not v or not v.strip():
            raise ValueError("KAFKA_BROKERS cannot be empty")
        return v.strip()
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if not v or not v.strip():
            raise ValueError("DATABASE_URL cannot be empty")
        if not v.startswith(("postgresql://", "postgresql+asyncpg://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL")
        return v.strip()
    
    @validator("REDIS_URL")
    def validate_redis_url(cls, v):
        if not v or not v.strip():
            raise ValueError("REDIS_URL cannot be empty")
        return v.strip()
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_allowed_origins(cls, v):
        if v is None:
            return "*"
        if isinstance(v, str):
            return v.strip() if v.strip() else "*"
        if isinstance(v, list):
            return ",".join(v)
        return "*"
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Get ALLOWED_ORIGINS as a list"""
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]
    
    @property
    def kafka_brokers_list(self) -> List[str]:
        """Get Kafka brokers as a list"""
        return [broker.strip() for broker in self.KAFKA_BROKERS.split(",")]
    
    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL for migrations"""
        return self.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT.lower() == "development"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Kafka Topics Configuration
KAFKA_TOPICS = {
    "network_metrics": f"{os.getenv('KAFKA_TOPIC_PREFIX', 'towerco')}.network.metrics",
    "energy_metrics": f"{os.getenv('KAFKA_TOPIC_PREFIX', 'towerco')}.energy.metrics",
    "events": f"{os.getenv('KAFKA_TOPIC_PREFIX', 'towerco')}.events",
    "kpi_calculations": f"{os.getenv('KAFKA_TOPIC_PREFIX', 'towerco')}.kpi.calculations",
    "aiops_predictions": f"{os.getenv('KAFKA_TOPIC_PREFIX', 'towerco')}.aiops.predictions",
    "alerts": f"{os.getenv('KAFKA_TOPIC_PREFIX', 'towerco')}.alerts",
    "notifications": f"{os.getenv('KAFKA_TOPIC_PREFIX', 'towerco')}.notifications",
}

# MinIO Buckets Configuration
MINIO_BUCKETS = {
    "models": f"{os.getenv('MINIO_BUCKET_PREFIX', 'towerco')}-ml-models",
    "reports": f"{os.getenv('MINIO_BUCKET_PREFIX', 'towerco')}-reports",
    "backups": f"{os.getenv('MINIO_BUCKET_PREFIX', 'towerco')}-backups",
    "uploads": f"{os.getenv('MINIO_BUCKET_PREFIX', 'towerco')}-uploads",
}

# KPI Definitions
KPI_CATEGORIES = {
    "NETWORK": "Network Performance KPIs",
    "ENERGY": "Energy Efficiency KPIs", 
    "OPERATIONAL": "Operational Excellence KPIs",
    "FINANCIAL": "Financial Performance KPIs",
}

# Alert Severities
ALERT_SEVERITIES = ["CRITICAL", "MAJOR", "MINOR", "WARNING", "INFO"]

# Event Types
EVENT_TYPES = [
    "ALARM", "FAULT", "MAINTENANCE", "CONFIGURATION", 
    "PERFORMANCE", "SECURITY", "CAPACITY", "AVAILABILITY"
]

# Technology Types
TECHNOLOGY_TYPES = ["2G", "3G", "4G", "5G", "NB-IOT", "WIFI"]

# Site Types
SITE_TYPES = [
    "BTS", "NODEBS", "ENODEB", "GNODEB", "MACRO", "MICRO", 
    "PICO", "FEMTO", "SMALL_CELL", "DAS", "WIFI_AP"
]

# Energy Types
ENERGY_TYPES = ["GRID", "BATTERY", "GENERATOR", "SOLAR", "WIND", "HYBRID"]


# Create global settings instance
settings = Settings()