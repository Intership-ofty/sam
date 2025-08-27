"""
Monitoring and observability configuration
"""

import logging
import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from opentelemetry import trace, metrics as otel_metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from .config import settings

logger = logging.getLogger(__name__)

# Prometheus metrics registry
registry = CollectorRegistry()

# Prometheus metrics
class Metrics:
    """Prometheus metrics collection"""
    
    def __init__(self):
        # HTTP metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=registry
        )
        
        # Database metrics
        self.db_connections_active = Gauge(
            'db_connections_active',
            'Number of active database connections',
            registry=registry
        )
        
        self.db_connections_idle = Gauge(
            'db_connections_idle',
            'Number of idle database connections',
            registry=registry
        )
        
        self.db_query_duration_seconds = Histogram(
            'db_query_duration_seconds',
            'Database query duration in seconds',
            ['query_type'],
            registry=registry
        )
        
        # Redis metrics
        self.redis_operations_total = Counter(
            'redis_operations_total',
            'Total number of Redis operations',
            ['operation', 'status'],
            registry=registry
        )
        
        self.redis_memory_used_bytes = Gauge(
            'redis_memory_used_bytes',
            'Redis memory usage in bytes',
            registry=registry
        )
        
        # Kafka metrics
        self.kafka_messages_sent_total = Counter(
            'kafka_messages_sent_total',
            'Total number of Kafka messages sent',
            ['topic'],
            registry=registry
        )
        
        self.kafka_messages_received_total = Counter(
            'kafka_messages_received_total',
            'Total number of Kafka messages received',
            ['topic'],
            registry=registry
        )
        
        self.kafka_message_processing_duration_seconds = Histogram(
            'kafka_message_processing_duration_seconds',
            'Kafka message processing duration in seconds',
            ['topic'],
            registry=registry
        )
        
        # Business metrics
        self.sites_total = Gauge(
            'sites_total',
            'Total number of sites',
            ['tenant_id'],
            registry=registry
        )
        
        self.active_events_total = Gauge(
            'active_events_total',
            'Total number of active events',
            ['severity'],
            registry=registry
        )
        
        self.kpi_calculations_total = Counter(
            'kpi_calculations_total',
            'Total number of KPI calculations',
            ['kpi_name'],
            registry=registry
        )
        
        self.aiops_predictions_total = Counter(
            'aiops_predictions_total',
            'Total number of AIOps predictions',
            ['model_name', 'prediction_type'],
            registry=registry
        )
        
        # System metrics
        self.system_cpu_usage_percent = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=registry
        )
        
        self.system_memory_usage_percent = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=registry
        )
        
        self.background_tasks_active = Gauge(
            'background_tasks_active',
            'Number of active background tasks',
            ['task_type'],
            registry=registry
        )


# Global metrics instance
metrics = Metrics()

# OpenTelemetry setup
tracer_provider = None
meter_provider = None
tracer = None
meter = None


async def init_monitoring():
    """Initialize monitoring and observability"""
    global tracer_provider, meter_provider, tracer, meter
    
    try:
        # Initialize OpenTelemetry if endpoint is configured
        if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
            # Tracer setup
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)
            
            # OTLP span exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
                insecure=True
            )
            
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
            
            # Meter setup
            otlp_metric_exporter = OTLPMetricExporter(
                endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
                insecure=True
            )
            
            metric_reader = PeriodicExportingMetricReader(
                exporter=otlp_metric_exporter,
                export_interval_millis=30000
            )
            
            meter_provider = MeterProvider(metric_readers=[metric_reader])
            otel_metrics.set_meter_provider(meter_provider)
            
            # Get tracer and meter instances
            tracer = trace.get_tracer(__name__)
            meter = otel_metrics.get_meter(__name__)
            
            # Enable auto-instrumentation
            enable_auto_instrumentation()
            
            logger.info("OpenTelemetry monitoring initialized")
        else:
            logger.info("OpenTelemetry endpoint not configured, using Prometheus only")
        
        # Initialize custom metrics
        await init_custom_metrics()
        
        logger.info("Monitoring initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize monitoring: {e}")
        raise


def enable_auto_instrumentation():
    """Enable automatic instrumentation for popular libraries"""
    try:
        # Database instrumentation
        AsyncPGInstrumentor().instrument()
        SQLAlchemyInstrumentor().instrument()
        
        # Redis instrumentation
        RedisInstrumentor().instrument()
        
        # HTTP instrumentation
        RequestsInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        
        logger.info("Auto-instrumentation enabled")
        
    except Exception as e:
        logger.warning(f"Failed to enable some auto-instrumentation: {e}")


async def init_custom_metrics():
    """Initialize custom business metrics"""
    try:
        # Initialize system metrics monitoring
        import psutil
        
        def update_system_metrics():
            metrics.system_cpu_usage_percent.set(psutil.cpu_percent())
            metrics.system_memory_usage_percent.set(psutil.virtual_memory().percent)
        
        # Update system metrics initially
        update_system_metrics()
        
        logger.info("Custom metrics initialized")
        
    except ImportError:
        logger.warning("psutil not available, system metrics disabled")
    except Exception as e:
        logger.error(f"Failed to initialize custom metrics: {e}")


def instrument_fastapi_app(app):
    """Instrument FastAPI application"""
    try:
        if tracer_provider:
            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=tracer_provider,
                excluded_urls="/health,/metrics"
            )
            logger.info("FastAPI application instrumented")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI app: {e}")


class MonitoringMiddleware:
    """Custom monitoring middleware for additional metrics"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            # Process request
            await self.app(scope, receive, send)
            
            # Record metrics
            duration = time.time() - start_time
            path = scope.get("path", "")
            method = scope.get("method", "")
            
            # Update custom metrics
            if path and method:
                metrics.http_request_duration_seconds.labels(
                    method=method,
                    endpoint=path
                ).observe(duration)
        else:
            await self.app(scope, receive, send)


class MetricsCollector:
    """Custom metrics collection utilities"""
    
    @staticmethod
    async def update_business_metrics():
        """Update business-specific metrics"""
        try:
            from .database import DatabaseManager
            db = DatabaseManager()
            
            # Count active events by severity
            events = await db.get_active_events()
            severity_counts = {}
            
            for event in events:
                severity = event.get('severity', 'UNKNOWN')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Update metrics
            for severity in ['CRITICAL', 'MAJOR', 'MINOR', 'WARNING', 'INFO']:
                count = severity_counts.get(severity, 0)
                metrics.active_events_total.labels(severity=severity).set(count)
            
        except Exception as e:
            logger.error(f"Failed to update business metrics: {e}")
    
    @staticmethod
    def record_kpi_calculation(kpi_name: str, duration: float):
        """Record KPI calculation metrics"""
        metrics.kpi_calculations_total.labels(kpi_name=kpi_name).inc()
        
        # Record duration if available
        if hasattr(metrics, 'kpi_calculation_duration_seconds'):
            metrics.kpi_calculation_duration_seconds.labels(
                kpi_name=kpi_name
            ).observe(duration)
    
    @staticmethod
    def record_aiops_prediction(model_name: str, prediction_type: str):
        """Record AIOps prediction metrics"""
        metrics.aiops_predictions_total.labels(
            model_name=model_name,
            prediction_type=prediction_type
        ).inc()
    
    @staticmethod
    def record_kafka_message_sent(topic: str):
        """Record Kafka message sent"""
        metrics.kafka_messages_sent_total.labels(topic=topic).inc()
    
    @staticmethod
    def record_kafka_message_received(topic: str):
        """Record Kafka message received"""
        metrics.kafka_messages_received_total.labels(topic=topic).inc()
    
    @staticmethod
    def record_kafka_processing_time(topic: str, duration: float):
        """Record Kafka message processing time"""
        metrics.kafka_message_processing_duration_seconds.labels(
            topic=topic
        ).observe(duration)
    
    @staticmethod
    def record_db_query_time(query_type: str, duration: float):
        """Record database query time"""
        metrics.db_query_duration_seconds.labels(
            query_type=query_type
        ).observe(duration)
    
    @staticmethod
    def record_redis_operation(operation: str, success: bool):
        """Record Redis operation"""
        status = "success" if success else "error"
        metrics.redis_operations_total.labels(
            operation=operation,
            status=status
        ).inc()


def get_prometheus_metrics() -> str:
    """Get Prometheus metrics in text format"""
    return generate_latest(registry).decode('utf-8')


def get_health_status() -> Dict[str, Any]:
    """Get health status information"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "monitoring": {
            "prometheus": True,
            "opentelemetry": settings.OTEL_EXPORTER_OTLP_ENDPOINT is not None,
        }
    }


class TracingUtils:
    """OpenTelemetry tracing utilities"""
    
    @staticmethod
    def start_span(name: str, attributes: Dict[str, Any] = None):
        """Start a new span"""
        if tracer:
            span = tracer.start_span(name)
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            return span
        return None
    
    @staticmethod
    def add_event(span, name: str, attributes: Dict[str, Any] = None):
        """Add event to span"""
        if span:
            span.add_event(name, attributes or {})
    
    @staticmethod
    def set_status(span, status: str, description: str = None):
        """Set span status"""
        if span:
            from opentelemetry.trace import Status, StatusCode
            
            status_code = StatusCode.ERROR if status == "error" else StatusCode.OK
            span.set_status(Status(status_code, description))
    
    @staticmethod
    def end_span(span):
        """End span"""
        if span:
            span.end()


# Dependency for FastAPI
def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance"""
    return MetricsCollector()