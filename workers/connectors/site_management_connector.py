"""
Site Management Connector - For site inventory and configuration systems
Integrates with various site management platforms and databases
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from .base_connector import DataConnector, HttpConnectorMixin, DatabaseConnectorMixin

logger = logging.getLogger(__name__)


class Site_ManagementConnector(DataConnector, HttpConnectorMixin, DatabaseConnectorMixin):
    """Site Management Connector for inventory and configuration data"""
    
    def __init__(self, data_source, message_producer, quality_validator):
        super().__init__(data_source, message_producer, quality_validator)
        
        # Site management specific configuration
        self.platform_type = self.get_connection_config('platform_type', 'api')  # api, database, csv
        self.sync_types = self.get_connection_config('sync_types', ['sites', 'equipment', 'inventory'])
        self.site_fields = self.get_connection_config('site_fields', [
            'site_id', 'site_code', 'site_name', 'latitude', 'longitude',
            'address', 'region', 'technology', 'equipment'
        ])
        
        # Connection objects
        self.session = None
        self.db_connection = None
        
    async def validate_config(self):
        """Validate Site Management connector configuration"""
        if self.platform_type == 'api':
            if not self.get_connection_config('base_url'):
                raise ValueError("API platform requires base_url")
        
        elif self.platform_type == 'database':
            db_config = self.get_connection_config('database', {})
            required_fields = ['host', 'database', 'username', 'password']
            for field in required_fields:
                if not db_config.get(field):
                    raise ValueError(f"Database platform requires {field}")
        
        elif self.platform_type == 'csv':
            if not self.get_connection_config('csv_files'):
                raise ValueError("CSV platform requires csv_files configuration")
        
        else:
            raise ValueError(f"Unsupported platform type: {self.platform_type}")
        
        logger.info(f"Site Management connector configuration validated for platform: {self.platform_type}")
    
    async def connect(self):
        """Establish connection based on platform type"""
        if self.platform_type == 'api':
            await self.connect_api()
        elif self.platform_type == 'database':
            await self.connect_database()
        elif self.platform_type == 'csv':
            await self.connect_csv()
    
    async def connect_api(self):
        """Connect to Site Management API"""
        self.session = await self.create_http_session()
        logger.info(f"Connected to Site Management API: {self.get_connection_config('base_url')}")
    
    async def connect_database(self):
        """Connect to Site Management Database"""
        self.db_connection = await self.create_db_connection()
        logger.info("Connected to Site Management Database")
    
    async def connect_csv(self):
        """Initialize CSV file processing"""
        csv_files = self.get_connection_config('csv_files', {})
        logger.info(f"Initialized CSV processing for {len(csv_files)} file types")
    
    async def disconnect(self):
        """Close connections"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            if self.db_connection:
                await self.db_connection.close()
                self.db_connection = None
            
            logger.info("Disconnected from Site Management system")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Site Management system: {e}")
    
    async def test_connection(self):
        """Test connection based on platform type"""
        if self.platform_type == 'api':
            return await self.test_api_connection()
        elif self.platform_type == 'database':
            return await self.test_database_connection()
        elif self.platform_type == 'csv':
            return await self.test_csv_files()
    
    async def test_api_connection(self):
        """Test API connection"""
        if not self.session:
            raise ConnectionError("No active API session")
        
        base_url = self.get_connection_config('base_url')
        test_endpoint = self.get_connection_config('test_endpoint', '/api/sites')
        
        async with self.session.get(f"{base_url}{test_endpoint}", params={'limit': 1}) as response:
            if response.status < 400:
                logger.info("Site Management API connection test successful")
                return True
            else:
                raise ConnectionError(f"API test failed: HTTP {response.status}")
    
    async def test_database_connection(self):
        """Test database connection"""
        if not self.db_connection:
            raise ConnectionError("No database connection")
        
        # Test with a simple query
        result = await self.db_connection.fetchval("SELECT 1")
        if result == 1:
            logger.info("Site Management database connection test successful")
            return True
        else:
            raise ConnectionError("Database test query failed")
    
    async def test_csv_files(self):
        """Test CSV file accessibility"""
        csv_files = self.get_connection_config('csv_files', {})
        
        for file_type, file_path in csv_files.items():
            try:
                import os
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"CSV file not found: {file_path}")
                
                # Test file readability
                with open(file_path, 'r') as f:
                    f.readline()  # Read first line
                    
            except Exception as e:
                raise ConnectionError(f"CSV file test failed for {file_type}: {e}")
        
        logger.info("CSV files connection test successful")
        return True
    
    async def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch site management data"""
        try:
            all_data = []
            
            for sync_type in self.sync_types:
                try:
                    logger.info(f"Fetching {sync_type} data from Site Management system")
                    
                    if self.platform_type == 'api':
                        data = await self.fetch_api_data(sync_type)
                    elif self.platform_type == 'database':
                        data = await self.fetch_database_data(sync_type)
                    elif self.platform_type == 'csv':
                        data = await self.fetch_csv_data(sync_type)
                    else:
                        data = []
                    
                    # Transform and enrich data
                    for item in data:
                        transformed_item = await self.transform_site_data(item, sync_type)
                        enriched_item = await self.enrich_data(transformed_item)
                        all_data.append(enriched_item)
                    
                    logger.info(f"Fetched {len(data)} {sync_type} records")
                    
                except Exception as e:
                    logger.error(f"Error fetching {sync_type} data: {e}")
            
            return all_data
            
        except Exception as e:
            logger.error(f"Error fetching Site Management data: {e}")
            await self.handle_connection_error(e)
            raise
    
    async def fetch_api_data(self, sync_type: str) -> List[Dict[str, Any]]:
        """Fetch data via API"""
        base_url = self.get_connection_config('base_url')
        endpoints = self.get_connection_config('endpoints', {})
        
        endpoint = endpoints.get(sync_type, f'/api/{sync_type}')
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Build query parameters
        params = {}
        
        # Add pagination
        limit = self.get_connection_config('page_size', 1000)
        params['limit'] = limit
        
        # Add time-based filtering if supported
        if self.get_connection_config('incremental_sync', False) and self.last_sync:
            params['updated_since'] = self.last_sync.isoformat()
        
        all_data = []
        offset = 0
        
        while True:
            params['offset'] = offset
            
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract records based on response structure
                        records = self.extract_records(data, sync_type)
                        
                        if not records:
                            break  # No more records
                        
                        all_data.extend(records)
                        
                        # Check if we got fewer records than requested (last page)
                        if len(records) < limit:
                            break
                        
                        offset += len(records)
                        
                    else:
                        logger.error(f"API request failed: HTTP {response.status}")
                        break
                        
            except Exception as e:
                logger.error(f"API request error: {e}")
                break
        
        return all_data
    
    async def fetch_database_data(self, sync_type: str) -> List[Dict[str, Any]]:
        """Fetch data from database"""
        tables = self.get_connection_config('tables', {})
        table = tables.get(sync_type, sync_type)
        
        # Build query
        fields = self.get_connection_config('fields', {}).get(sync_type, ['*'])
        field_list = ', '.join(fields) if isinstance(fields, list) else '*'
        
        query = f"SELECT {field_list} FROM {table}"
        
        # Add WHERE conditions
        where_conditions = []
        params = []
        
        # Add incremental sync condition
        if self.get_connection_config('incremental_sync', False) and self.last_sync:
            updated_field = self.get_connection_config('updated_field', 'updated_at')
            where_conditions.append(f"{updated_field} > $1")
            params.append(self.last_sync)
        
        # Add custom conditions
        custom_conditions = self.get_connection_config('conditions', {}).get(sync_type)
        if custom_conditions:
            where_conditions.append(custom_conditions)
        
        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)
        
        # Add ordering
        order_field = self.get_connection_config('order_field', 'id')
        query += f" ORDER BY {order_field}"
        
        # Add limit
        limit = self.get_connection_config('limit', 10000)
        query += f" LIMIT {limit}"
        
        try:
            if params:
                rows = await self.db_connection.fetch(query, *params)
            else:
                rows = await self.db_connection.fetch(query)
            
            # Convert rows to dictionaries
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Database query error for {sync_type}: {e}")
            return []
    
    async def fetch_csv_data(self, sync_type: str) -> List[Dict[str, Any]]:
        """Fetch data from CSV files"""
        csv_files = self.get_connection_config('csv_files', {})
        file_path = csv_files.get(sync_type)
        
        if not file_path:
            logger.warning(f"No CSV file configured for {sync_type}")
            return []
        
        try:
            # Read and parse CSV
            content = await self.read_file(file_path)
            data = await self.parse_csv(content)
            
            # Apply filters if configured
            filters = self.get_connection_config('csv_filters', {}).get(sync_type, {})
            if filters:
                data = self.apply_csv_filters(data, filters)
            
            return data
            
        except Exception as e:
            logger.error(f"CSV processing error for {sync_type}: {e}")
            return []
    
    def extract_records(self, response_data: Dict[str, Any], sync_type: str) -> List[Dict[str, Any]]:
        """Extract records from API response"""
        # Common response structures
        if 'data' in response_data:
            return response_data['data']
        elif 'results' in response_data:
            return response_data['results']
        elif sync_type in response_data:
            return response_data[sync_type]
        elif isinstance(response_data, list):
            return response_data
        else:
            return [response_data] if response_data else []
    
    def apply_csv_filters(self, data: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to CSV data"""
        filtered_data = []
        
        for record in data:
            include_record = True
            
            for field, expected_value in filters.items():
                if field not in record:
                    include_record = False
                    break
                
                if isinstance(expected_value, list):
                    if record[field] not in expected_value:
                        include_record = False
                        break
                else:
                    if record[field] != expected_value:
                        include_record = False
                        break
            
            if include_record:
                filtered_data.append(record)
        
        return filtered_data
    
    async def transform_site_data(self, item: Dict[str, Any], sync_type: str) -> Dict[str, Any]:
        """Transform site management data to standard format"""
        try:
            if sync_type == 'sites':
                return await self.transform_site_record(item)
            elif sync_type == 'equipment':
                return await self.transform_equipment_record(item)
            elif sync_type == 'inventory':
                return await self.transform_inventory_record(item)
            else:
                return await self.transform_generic_record(item, sync_type)
                
        except Exception as e:
            logger.error(f"Error transforming {sync_type} data: {e}")
            return {
                'data_type': 'configuration',
                'config_type': sync_type,
                'timestamp': datetime.utcnow().isoformat(),
                'raw_data': item,
                'transform_error': str(e)
            }
    
    async def transform_site_record(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Transform site record to standard format"""
        # Field mappings based on configuration
        field_mappings = self.get_connection_config('field_mappings', {}).get('sites', {})
        
        transformed = {
            'data_type': 'configuration',
            'config_type': 'site',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Map standard site fields
        site_fields = {
            'site_id': item.get(field_mappings.get('site_id', 'site_id')),
            'site_code': item.get(field_mappings.get('site_code', 'site_code')),
            'site_name': item.get(field_mappings.get('site_name', 'site_name')),
            'site_type': item.get(field_mappings.get('site_type', 'site_type')),
            'latitude': self.safe_float(item.get(field_mappings.get('latitude', 'latitude'))),
            'longitude': self.safe_float(item.get(field_mappings.get('longitude', 'longitude'))),
            'address': item.get(field_mappings.get('address', 'address')),
            'region': item.get(field_mappings.get('region', 'region')),
            'country': item.get(field_mappings.get('country', 'country')),
            'technology': item.get(field_mappings.get('technology', 'technology'))
        }
        
        transformed.update(site_fields)
        
        # Add metadata
        transformed['metadata'] = {
            'source_platform': self.platform_type,
            'sync_type': 'sites',
            'last_updated': item.get('updated_at', item.get('last_modified')),
            'status': item.get('status', 'active'),
            'owner': item.get('owner'),
            'maintenance_contract': item.get('maintenance_contract')
        }
        
        # Add technology-specific information
        if isinstance(transformed.get('technology'), str):
            # Convert string to structured format
            tech_list = [t.strip() for t in transformed['technology'].split(',')]
            transformed['technology'] = {tech: {'enabled': True} for tech in tech_list}
        
        return transformed
    
    async def transform_equipment_record(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Transform equipment record to standard format"""
        field_mappings = self.get_connection_config('field_mappings', {}).get('equipment', {})
        
        transformed = {
            'data_type': 'configuration',
            'config_type': 'equipment',
            'timestamp': datetime.utcnow().isoformat(),
            'site_id': item.get(field_mappings.get('site_id', 'site_id')),
            'equipment_id': item.get(field_mappings.get('equipment_id', 'equipment_id')),
            'equipment_type': item.get(field_mappings.get('equipment_type', 'equipment_type')),
            'vendor': item.get(field_mappings.get('vendor', 'vendor')),
            'model': item.get(field_mappings.get('model', 'model')),
            'serial_number': item.get(field_mappings.get('serial_number', 'serial_number')),
            'software_version': item.get(field_mappings.get('software_version', 'software_version')),
            'hardware_version': item.get(field_mappings.get('hardware_version', 'hardware_version')),
            'installation_date': item.get(field_mappings.get('installation_date', 'installation_date')),
            'warranty_expiry': item.get(field_mappings.get('warranty_expiry', 'warranty_expiry')),
            'metadata': {
                'source_platform': self.platform_type,
                'sync_type': 'equipment',
                'operational_status': item.get('operational_status', 'unknown'),
                'administrative_status': item.get('administrative_status', 'unknown'),
                'last_maintenance': item.get('last_maintenance'),
                'next_maintenance': item.get('next_maintenance')
            }
        }
        
        return transformed
    
    async def transform_inventory_record(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Transform inventory record to standard format"""
        field_mappings = self.get_connection_config('field_mappings', {}).get('inventory', {})
        
        transformed = {
            'data_type': 'configuration',
            'config_type': 'inventory',
            'timestamp': datetime.utcnow().isoformat(),
            'site_id': item.get(field_mappings.get('site_id', 'site_id')),
            'item_id': item.get(field_mappings.get('item_id', 'item_id')),
            'item_type': item.get(field_mappings.get('item_type', 'item_type')),
            'description': item.get(field_mappings.get('description', 'description')),
            'quantity': self.safe_int(item.get(field_mappings.get('quantity', 'quantity'))),
            'unit': item.get(field_mappings.get('unit', 'unit')),
            'location': item.get(field_mappings.get('location', 'location')),
            'condition': item.get(field_mappings.get('condition', 'condition')),
            'metadata': {
                'source_platform': self.platform_type,
                'sync_type': 'inventory',
                'supplier': item.get('supplier'),
                'purchase_date': item.get('purchase_date'),
                'cost': item.get('cost'),
                'depreciation': item.get('depreciation')
            }
        }
        
        return transformed
    
    async def transform_generic_record(self, item: Dict[str, Any], sync_type: str) -> Dict[str, Any]:
        """Generic transformation for unknown sync types"""
        transformed = {
            'data_type': 'configuration',
            'config_type': sync_type,
            'timestamp': datetime.utcnow().isoformat(),
            'raw_data': item,
            'metadata': {
                'source_platform': self.platform_type,
                'sync_type': sync_type,
                'transformation': 'generic'
            }
        }
        
        # Extract common fields if available
        common_fields = ['id', 'site_id', 'name', 'type', 'status', 'updated_at']
        for field in common_fields:
            if field in item:
                transformed[field] = item[field]
        
        return transformed
    
    def safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None or value == '':
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def safe_int(self, value) -> Optional[int]:
        """Safely convert value to int"""
        if value is None or value == '':
            return None
        
        try:
            return int(float(value))  # Handle string numbers like "5.0"
        except (ValueError, TypeError):
            return None