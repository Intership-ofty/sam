#!/usr/bin/env python3
"""
Towerco AIOps - Universal Data Ingestor Worker
Intelligent data ingestion with auto-discovery and universal connectors
"""

import asyncio
import logging
import signal
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import traceback

import aiohttp
from aiohttp import web
import psutil

# Import core modules from backend
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from core.config import settings, KAFKA_TOPICS
from core.database import init_db, DatabaseManager
from core.cache import init_redis, CacheManager
from core.messaging import init_kafka, MessageProducer, MessageConsumer, start_consumer
from core.monitoring import init_monitoring, MetricsCollector

# Import connectors
from connectors import (
    OSS_NetworkConnector, 
    ITSM_ServiceNowConnector,
    Energy_IOTConnector,
    Site_ManagementConnector,
    GenericRestConnector,
    SNMPConnector
)

logger = logging.getLogger(__name__)

# Global state
shutdown_event = asyncio.Event()
active_connectors: Dict[str, 'DataConnector'] = {}
health_server = None


@dataclass
class DataSource:
    """Data source configuration"""
    source_id: str
    source_name: str
    source_type: str
    connection_config: Dict[str, Any]
    mapping_rules: Dict[str, Any]
    enabled: bool
    polling_interval: int = 300  # 5 minutes default
    last_sync: Optional[datetime] = None
    error_count: int = 0
    max_errors: int = 5


class UniversalDataFabric:
    """Universal Data Fabric - Core ingestion engine"""
    
    def __init__(self):
        self.data_sources: Dict[str, DataSource] = {}
        self.connectors_registry = {
            'oss_network': OSS_NetworkConnector,
            'itsm_servicenow': ITSM_ServiceNowConnector, 
            'energy_iot': Energy_IOTConnector,
            'site_management': Site_ManagementConnector,
            'rest_api': GenericRestConnector,
            'snmp': SNMPConnector
        }
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.producer = MessageProducer()
        self.metrics = MetricsCollector()
        self.quality_validator = DataQualityValidator()
    
    async def initialize(self):
        """Initialize the data fabric"""
        logger.info("Initializing Universal Data Fabric...")
        
        # Load data sources from database
        await self.load_data_sources()
        
        # Initialize connectors
        await self.initialize_connectors()
        
        # Start background tasks
        asyncio.create_task(self.data_ingestion_loop())
        asyncio.create_task(self.health_monitoring_loop())
        asyncio.create_task(self.cleanup_loop())
        
        logger.info("Universal Data Fabric initialized successfully")
    
    async def load_data_sources(self):
        """Load data source configurations from database"""
        try:
            sources_query = """
            SELECT source_id, source_name, source_type, connection_config, 
                   mapping_rules, status, last_sync
            FROM data_sources 
            WHERE status = 'active'
            ORDER BY source_name
            """
            
            sources_data = await self.db.execute_query(sources_query)
            
            for source_row in sources_data:
                source = DataSource(
                    source_id=source_row['source_id'],
                    source_name=source_row['source_name'],
                    source_type=source_row['source_type'],
                    connection_config=source_row['connection_config'],
                    mapping_rules=source_row['mapping_rules'] or {},
                    enabled=True,
                    last_sync=source_row['last_sync']
                )
                
                self.data_sources[source.source_id] = source
                logger.info(f"Loaded data source: {source.source_name} ({source.source_type})")
        
        except Exception as e:
            logger.error(f"Failed to load data sources: {e}")
    
    async def initialize_connectors(self):
        """Initialize data connectors"""
        global active_connectors
        
        for source_id, source in self.data_sources.items():
            try:
                connector_class = self.connectors_registry.get(source.source_type)
                if not connector_class:
                    logger.error(f"Unknown connector type: {source.source_type}")
                    continue
                
                connector = connector_class(source, self.producer, self.quality_validator)
                await connector.initialize()
                
                active_connectors[source_id] = connector
                logger.info(f"Initialized connector: {source.source_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize connector {source.source_name}: {e}")
    
    async def data_ingestion_loop(self):
        """Main data ingestion loop"""
        logger.info("Starting data ingestion loop...")
        
        while not shutdown_event.is_set():
            try:
                # Process each data source
                ingestion_tasks = []
                
                for source_id, source in self.data_sources.items():
                    if not source.enabled:
                        continue
                    
                    # Check if it's time to sync
                    if self.should_sync_source(source):
                        connector = active_connectors.get(source_id)
                        if connector:
                            task = asyncio.create_task(
                                self.ingest_from_source(connector, source)
                            )
                            ingestion_tasks.append(task)
                
                # Execute ingestion tasks concurrently
                if ingestion_tasks:
                    results = await asyncio.gather(*ingestion_tasks, return_exceptions=True)
                    
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"Ingestion task failed: {result}")
                
                # Wait before next cycle
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in data ingestion loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def should_sync_source(self, source: DataSource) -> bool:
        """Check if source should be synchronized"""
        if not source.last_sync:
            return True
        
        time_since_sync = datetime.utcnow() - source.last_sync
        return time_since_sync.total_seconds() >= source.polling_interval
    
    async def ingest_from_source(self, connector: 'DataConnector', source: DataSource):
        """Ingest data from a specific source"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting ingestion from: {source.source_name}")
            
            # Fetch data from connector
            data_batch = await connector.fetch_data()
            
            if not data_batch:
                logger.warning(f"No data received from {source.source_name}")
                return
            
            # Process each data item
            processed_count = 0
            error_count = 0
            
            for data_item in data_batch:
                try:
                    # Validate data quality
                    validation_result = await self.quality_validator.validate(data_item, source)
                    
                    if not validation_result.is_valid:
                        logger.warning(f"Data validation failed for {source.source_name}: {validation_result.errors}")
                        error_count += 1
                        continue
                    
                    # Apply mapping rules
                    mapped_data = await self.apply_mapping_rules(data_item, source)
                    
                    # Route to appropriate Kafka topic
                    await self.route_data(mapped_data, source)
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing data item from {source.source_name}: {e}")
                    error_count += 1
            
            # Update sync status
            await self.update_sync_status(source, processed_count, error_count)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_kafka_processing_time(f"ingestion_{source.source_type}", duration)
            
            logger.info(
                f"Ingestion completed for {source.source_name}: "
                f"{processed_count} processed, {error_count} errors in {duration:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Ingestion failed for {source.source_name}: {e}")
            await self.handle_ingestion_error(source, str(e))
    
    async def apply_mapping_rules(self, data: Dict[str, Any], source: DataSource) -> Dict[str, Any]:
        """Apply mapping rules to transform data"""
        if not source.mapping_rules:
            return data
        
        try:
            mapped_data = {}
            
            # Apply field mappings
            field_mappings = source.mapping_rules.get('field_mappings', {})
            for target_field, source_field in field_mappings.items():
                if isinstance(source_field, str):
                    # Simple field mapping
                    mapped_data[target_field] = data.get(source_field)
                elif isinstance(source_field, dict):
                    # Complex mapping with transformations
                    value = data.get(source_field.get('source'))
                    
                    # Apply transformations
                    transforms = source_field.get('transforms', [])
                    for transform in transforms:
                        value = await self.apply_transform(value, transform)
                    
                    mapped_data[target_field] = value
            
            # Add unmapped fields if configured
            if source.mapping_rules.get('include_unmapped', False):
                for key, value in data.items():
                    if key not in [v if isinstance(v, str) else v.get('source') 
                                   for v in field_mappings.values()]:
                        mapped_data[key] = value
            
            return mapped_data
            
        except Exception as e:
            logger.error(f"Error applying mapping rules: {e}")
            return data
    
    async def apply_transform(self, value: Any, transform: Dict[str, Any]) -> Any:
        """Apply data transformation"""
        transform_type = transform.get('type')
        
        if transform_type == 'unit_conversion':
            factor = transform.get('factor', 1)
            return float(value) * factor if value is not None else None
        
        elif transform_type == 'string_format':
            format_str = transform.get('format')
            return format_str.format(value) if value is not None else None
        
        elif transform_type == 'mapping':
            mapping = transform.get('mapping', {})
            return mapping.get(str(value), value)
        
        elif transform_type == 'calculation':
            formula = transform.get('formula')
            # Simple formula evaluation (could be extended)
            if formula and isinstance(value, (int, float)):
                return eval(formula.replace('x', str(value)))
        
        return value
    
    async def route_data(self, data: Dict[str, Any], source: DataSource):
        """Route data to appropriate Kafka topic"""
        data_type = data.get('data_type') or self.infer_data_type(data, source)
        
        if data_type == 'network_metric':
            await self.producer.send_network_metric(**data)
        
        elif data_type == 'energy_metric':
            await self.producer.send_energy_metric(**data)
        
        elif data_type == 'event':
            await self.producer.send_event(**data)
        
        else:
            # Send to generic events topic
            await self.producer.send_message(
                KAFKA_TOPICS['events'],
                {
                    **data,
                    'source_type': source.source_type,
                    'source_name': source.source_name
                }
            )
    
    def infer_data_type(self, data: Dict[str, Any], source: DataSource) -> str:
        """Infer data type from content and source type"""
        if source.source_type in ['oss_network', 'snmp']:
            if 'metric_name' in data and 'metric_value' in data:
                return 'network_metric'
        
        elif source.source_type == 'energy_iot':
            if 'energy_type' in data and 'metric_value' in data:
                return 'energy_metric'
        
        elif source.source_type == 'itsm_servicenow':
            if 'severity' in data or 'event_type' in data:
                return 'event'
        
        # Check for common metric patterns
        if all(key in data for key in ['metric_name', 'metric_value', 'site_id']):
            return 'network_metric'
        
        if 'severity' in data or 'alarm' in str(data).lower():
            return 'event'
        
        return 'unknown'
    
    async def update_sync_status(self, source: DataSource, processed: int, errors: int):
        """Update synchronization status"""
        try:
            source.last_sync = datetime.utcnow()
            source.error_count = errors
            
            # Update database
            await self.db.execute_command(
                """
                UPDATE data_sources 
                SET last_sync = $1, metadata = jsonb_set(
                    COALESCE(metadata, '{}'),
                    '{last_ingestion}',
                    $2
                )
                WHERE source_id = $3
                """,
                source.last_sync,
                json.dumps({
                    'processed_count': processed,
                    'error_count': errors,
                    'duration': time.time()
                }),
                source.source_id
            )
            
        except Exception as e:
            logger.error(f"Failed to update sync status: {e}")
    
    async def handle_ingestion_error(self, source: DataSource, error_msg: str):
        """Handle ingestion errors"""
        source.error_count += 1
        
        # Disable source if too many errors
        if source.error_count >= source.max_errors:
            source.enabled = False
            logger.error(f"Disabling source {source.source_name} due to excessive errors")
            
            # Update database
            await self.db.execute_command(
                "UPDATE data_sources SET status = 'error' WHERE source_id = $1",
                source.source_id
            )
        
        # Store error for monitoring
        await self.producer.send_alert(
            alert_type="data_ingestion_error",
            severity="MAJOR" if source.error_count >= 3 else "MINOR",
            title=f"Data ingestion error: {source.source_name}",
            description=error_msg,
            metadata={
                'source_id': source.source_id,
                'source_type': source.source_type,
                'error_count': source.error_count
            }
        )
    
    async def health_monitoring_loop(self):
        """Monitor health of data sources and connectors"""
        while not shutdown_event.is_set():
            try:
                for source_id, connector in active_connectors.items():
                    try:
                        health = await connector.health_check()
                        
                        # Update health metrics
                        self.metrics.record_kafka_message_sent(f"health_{source_id}")
                        
                        if not health['healthy']:
                            logger.warning(f"Connector {source_id} health check failed: {health}")
                    
                    except Exception as e:
                        logger.error(f"Health check failed for {source_id}: {e}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(120)
    
    async def cleanup_loop(self):
        """Periodic cleanup tasks"""
        while not shutdown_event.is_set():
            try:
                # Clean up old cache entries
                await self.cache.delete("ingestion:*:old")
                
                # Log statistics
                logger.info(f"Active connectors: {len(active_connectors)}")
                logger.info(f"Active data sources: {len([s for s in self.data_sources.values() if s.enabled])}")
                
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)


class DataQualityValidator:
    """Data quality validation engine"""
    
    def __init__(self):
        self.validation_rules = {}
    
    async def validate(self, data: Dict[str, Any], source: DataSource) -> 'ValidationResult':
        """Validate data quality"""
        errors = []
        warnings = []
        
        try:
            # Basic structure validation
            if not isinstance(data, dict):
                errors.append("Data must be a dictionary")
                return ValidationResult(False, errors, warnings)
            
            # Required fields validation
            required_fields = source.mapping_rules.get('required_fields', [])
            for field in required_fields:
                if field not in data or data[field] is None:
                    errors.append(f"Required field missing: {field}")
            
            # Data type validation
            field_types = source.mapping_rules.get('field_types', {})
            for field, expected_type in field_types.items():
                if field in data and data[field] is not None:
                    if not self._validate_type(data[field], expected_type):
                        errors.append(f"Invalid type for field {field}: expected {expected_type}")
            
            # Range validation for numeric fields
            field_ranges = source.mapping_rules.get('field_ranges', {})
            for field, range_config in field_ranges.items():
                if field in data and isinstance(data[field], (int, float)):
                    min_val = range_config.get('min')
                    max_val = range_config.get('max')
                    
                    if min_val is not None and data[field] < min_val:
                        warnings.append(f"Value below minimum for {field}: {data[field]} < {min_val}")
                    
                    if max_val is not None and data[field] > max_val:
                        warnings.append(f"Value above maximum for {field}: {data[field]} > {max_val}")
            
            # Custom validation rules
            custom_rules = source.mapping_rules.get('validation_rules', [])
            for rule in custom_rules:
                result = await self._apply_custom_rule(data, rule)
                if result:
                    if rule.get('severity') == 'error':
                        errors.append(result)
                    else:
                        warnings.append(result)
            
            is_valid = len(errors) == 0
            return ValidationResult(is_valid, errors, warnings)
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return ValidationResult(False, [f"Validation error: {str(e)}"], warnings)
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate data type"""
        type_mapping = {
            'string': str,
            'integer': int,
            'float': (int, float),
            'boolean': bool,
            'list': list,
            'dict': dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True
    
    async def _apply_custom_rule(self, data: Dict[str, Any], rule: Dict[str, Any]) -> Optional[str]:
        """Apply custom validation rule"""
        rule_type = rule.get('type')
        
        if rule_type == 'regex':
            field = rule.get('field')
            pattern = rule.get('pattern')
            if field in data and pattern:
                import re
                if not re.match(pattern, str(data[field])):
                    return f"Field {field} does not match pattern {pattern}"
        
        elif rule_type == 'range':
            field = rule.get('field')
            min_val = rule.get('min')
            max_val = rule.get('max')
            
            if field in data and isinstance(data[field], (int, float)):
                if min_val is not None and data[field] < min_val:
                    return f"Field {field} below minimum: {data[field]} < {min_val}"
                if max_val is not None and data[field] > max_val:
                    return f"Field {field} above maximum: {data[field]} > {max_val}"
        
        return None


@dataclass
class ValidationResult:
    """Data validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


# Health check web server
async def create_health_server():
    """Create health check web server"""
    app = web.Application()
    
    async def health_handler(request):
        """Health check endpoint"""
        system_info = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'active_connectors': len(active_connectors),
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
        }
        return web.json_response(system_info)
    
    app.router.add_get('/health', health_handler)
    
    return app


async def message_handler(topic, key, value, headers, timestamp, partition, offset):
    """Handle incoming Kafka messages"""
    logger.info(f"Received message from {topic}: {key}")
    
    try:
        # Process configuration updates
        if topic == 'towerco.config.updates':
            if value.get('type') == 'data_source_update':
                # Reload data sources
                fabric = active_connectors.get('fabric')
                if fabric:
                    await fabric.load_data_sources()
        
        # Process data source commands
        elif topic == 'towerco.data.commands':
            command = value.get('command')
            source_id = value.get('source_id')
            
            if command == 'sync_now' and source_id in active_connectors:
                connector = active_connectors[source_id]
                # Trigger immediate sync
                asyncio.create_task(connector.sync_now())
    
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
    global health_server
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting Towerco Data Ingestor Worker...")
    
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
        site = web.TCPSite(health_runner, '0.0.0.0', 8001)
        await site.start()
        logger.info("Health server started on port 8001")
        
        # Initialize Universal Data Fabric
        fabric = UniversalDataFabric()
        await fabric.initialize()
        
        # Start Kafka consumers
        await start_consumer(
            'config_updates',
            ['towerco.config.updates', 'towerco.data.commands'],
            'data_ingestor_group',
            message_handler
        )
        
        logger.info("Data Ingestor Worker started successfully")
        
        # Main loop
        while not shutdown_event.is_set():
            await asyncio.sleep(1)
    
    except Exception as e:
        logger.error(f"Failed to start Data Ingestor Worker: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()
    
    finally:
        logger.info("Shutting down Data Ingestor Worker...")
        
        # Cleanup
        if health_server:
            await health_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())