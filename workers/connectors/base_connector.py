"""
Base Data Connector - Abstract base class for all data connectors
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import traceback

logger = logging.getLogger(__name__)


class DataConnector(ABC):
    """Abstract base class for data connectors"""
    
    def __init__(self, data_source, message_producer, quality_validator):
        self.data_source = data_source
        self.producer = message_producer
        self.validator = quality_validator
        self.connection = None
        self.last_sync = None
        self.error_count = 0
        self.max_retries = 3
        self.retry_delay = 30  # seconds
        self.connection_timeout = 30
        self.read_timeout = 60
        self._initialized = False
    
    async def initialize(self):
        """Initialize the connector"""
        try:
            logger.info(f"Initializing connector for {self.data_source.source_name}")
            
            # Validate configuration
            await self.validate_config()
            
            # Establish connection
            await self.connect()
            
            # Test connection
            await self.test_connection()
            
            self._initialized = True
            logger.info(f"Connector {self.data_source.source_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connector {self.data_source.source_name}: {e}")
            raise
    
    @abstractmethod
    async def validate_config(self):
        """Validate connector configuration"""
        pass
    
    @abstractmethod
    async def connect(self):
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to data source"""
        pass
    
    @abstractmethod
    async def test_connection(self):
        """Test connection to data source"""
        pass
    
    @abstractmethod
    async def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch data from source"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check connector health"""
        try:
            if not self._initialized:
                return {
                    'healthy': False,
                    'status': 'not_initialized',
                    'last_check': datetime.utcnow().isoformat()
                }
            
            # Test connection
            await self.test_connection()
            
            return {
                'healthy': True,
                'status': 'connected',
                'last_sync': self.last_sync.isoformat() if self.last_sync else None,
                'error_count': self.error_count,
                'last_check': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'status': 'connection_failed',
                'error': str(e),
                'error_count': self.error_count,
                'last_check': datetime.utcnow().isoformat()
            }
    
    async def sync_now(self):
        """Trigger immediate synchronization"""
        try:
            logger.info(f"Manual sync triggered for {self.data_source.source_name}")
            data = await self.fetch_data()
            return len(data) if data else 0
        except Exception as e:
            logger.error(f"Manual sync failed for {self.data_source.source_name}: {e}")
            raise
    
    async def retry_operation(self, operation, *args, **kwargs):
        """Retry operation with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{self.max_retries + 1}) "
                        f"for {self.data_source.source_name}: {e}. "
                        f"Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Operation failed after {self.max_retries + 1} attempts "
                        f"for {self.data_source.source_name}: {e}"
                    )
        
        # If we get here, all retries failed
        self.error_count += 1
        raise last_exception
    
    def get_connection_config(self, key: str, default=None):
        """Get connection configuration value"""
        return self.data_source.connection_config.get(key, default)
    
    def get_mapping_rule(self, key: str, default=None):
        """Get mapping rule value"""
        return self.data_source.mapping_rules.get(key, default)
    
    async def transform_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """Transform raw data into standardized format"""
        try:
            # Default transformation - override in subclasses
            if isinstance(raw_data, list):
                return raw_data
            elif isinstance(raw_data, dict):
                return [raw_data]
            else:
                return [{'raw_data': raw_data}]
                
        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            return []
    
    async def enrich_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich data with additional context"""
        try:
            # Add source information
            data['_source'] = {
                'source_id': self.data_source.source_id,
                'source_name': self.data_source.source_name,
                'source_type': self.data_source.source_type,
                'ingestion_time': datetime.utcnow().isoformat()
            }
            
            # Add tenant information if available
            tenant_mapping = self.get_mapping_rule('tenant_mapping')
            if tenant_mapping:
                if isinstance(tenant_mapping, str):
                    # Static tenant ID
                    data['tenant_id'] = tenant_mapping
                elif isinstance(tenant_mapping, dict):
                    # Dynamic tenant mapping based on data
                    field = tenant_mapping.get('field')
                    mapping = tenant_mapping.get('mapping', {})
                    if field in data:
                        data['tenant_id'] = mapping.get(str(data[field]))
            
            # Add site information if available
            site_mapping = self.get_mapping_rule('site_mapping')
            if site_mapping and isinstance(site_mapping, dict):
                site_field = site_mapping.get('field')
                if site_field in data:
                    # Map site identifier to site_id
                    site_identifier = data[site_field]
                    # This would typically involve a database lookup
                    # For now, we'll just pass through the identifier
                    data['site_code'] = site_identifier
            
            return data
            
        except Exception as e:
            logger.error(f"Data enrichment failed: {e}")
            return data
    
    async def handle_connection_error(self, error: Exception):
        """Handle connection errors"""
        logger.error(f"Connection error for {self.data_source.source_name}: {error}")
        
        # Increment error count
        self.error_count += 1
        
        # Try to reconnect if not too many errors
        if self.error_count < 5:
            try:
                logger.info(f"Attempting to reconnect to {self.data_source.source_name}")
                await self.disconnect()
                await asyncio.sleep(5)  # Wait before reconnecting
                await self.connect()
                await self.test_connection()
                
                logger.info(f"Reconnected to {self.data_source.source_name}")
                self.error_count = 0  # Reset error count on successful reconnection
                
            except Exception as reconnect_error:
                logger.error(f"Reconnection failed: {reconnect_error}")
        else:
            logger.error(f"Too many connection errors for {self.data_source.source_name}, marking as unhealthy")
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.data_source.source_name})"
    
    def __repr__(self):
        return self.__str__()


class HttpConnectorMixin:
    """Mixin for HTTP-based connectors"""
    
    async def create_http_session(self):
        """Create HTTP session with proper configuration"""
        import aiohttp
        
        timeout = aiohttp.ClientTimeout(
            total=self.connection_timeout + self.read_timeout,
            connect=self.connection_timeout,
            sock_read=self.read_timeout
        )
        
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        headers = {
            'User-Agent': 'Towerco-AIOps-DataConnector/1.0'
        }
        
        # Add authentication headers if configured
        auth_config = self.get_connection_config('authentication', {})
        auth_type = auth_config.get('type')
        
        if auth_type == 'basic':
            import base64
            username = auth_config.get('username')
            password = auth_config.get('password')
            if username and password:
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers['Authorization'] = f'Basic {credentials}'
        
        elif auth_type == 'bearer':
            token = auth_config.get('token')
            if token:
                headers['Authorization'] = f'Bearer {token}'
        
        elif auth_type == 'api_key':
            api_key = auth_config.get('api_key')
            key_header = auth_config.get('key_header', 'X-API-Key')
            if api_key:
                headers[key_header] = api_key
        
        return aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers
        )


class DatabaseConnectorMixin:
    """Mixin for database-based connectors"""
    
    async def create_db_connection(self):
        """Create database connection"""
        db_config = self.get_connection_config('database', {})
        db_type = db_config.get('type', 'postgresql')
        
        if db_type == 'postgresql':
            import asyncpg
            
            connection = await asyncpg.connect(
                host=db_config.get('host'),
                port=db_config.get('port', 5432),
                user=db_config.get('username'),
                password=db_config.get('password'),
                database=db_config.get('database'),
                command_timeout=self.read_timeout
            )
            
            return connection
            
        elif db_type == 'mysql':
            import aiomysql
            
            connection = await aiomysql.connect(
                host=db_config.get('host'),
                port=db_config.get('port', 3306),
                user=db_config.get('username'),
                password=db_config.get('password'),
                db=db_config.get('database'),
                connect_timeout=self.connection_timeout
            )
            
            return connection
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")


class FileConnectorMixin:
    """Mixin for file-based connectors"""
    
    async def read_file(self, file_path: str, encoding: str = 'utf-8'):
        """Read file contents"""
        import aiofiles
        
        try:
            async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                content = await f.read()
            return content
            
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise
    
    async def parse_csv(self, content: str, delimiter: str = ','):
        """Parse CSV content"""
        import csv
        import io
        
        reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
        return list(reader)
    
    async def parse_json(self, content: str):
        """Parse JSON content"""
        import json
        return json.loads(content)
    
    async def parse_xml(self, content: str):
        """Parse XML content"""
        import xmltodict
        return xmltodict.parse(content)