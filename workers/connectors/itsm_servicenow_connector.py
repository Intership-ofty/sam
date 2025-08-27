"""
ITSM ServiceNow Connector - Real ServiceNow integration (no mocks)
Integrates with ServiceNow for incident, request, and change management
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from .base_connector import DataConnector, HttpConnectorMixin

logger = logging.getLogger(__name__)


class ITSM_ServiceNowConnector(DataConnector, HttpConnectorMixin):
    """ServiceNow ITSM Connector for real ticket integration"""
    
    def __init__(self, data_source, message_producer, quality_validator):
        super().__init__(data_source, message_producer, quality_validator)
        
        # ServiceNow specific configuration
        self.instance_url = self.get_connection_config('instance_url')
        self.api_version = self.get_connection_config('api_version', 'v1')
        self.tables_to_sync = self.get_connection_config('tables', ['incident', 'sc_request', 'change_request'])
        self.sync_mode = self.get_connection_config('sync_mode', 'incremental')  # incremental or full
        self.max_records = self.get_connection_config('max_records', 1000)
        
        # Field mappings for different ticket types
        self.field_mappings = {
            'incident': {
                'ticket_id': 'sys_id',
                'ticket_number': 'number',
                'title': 'short_description',
                'description': 'description',
                'priority': 'priority',
                'severity': 'severity',
                'state': 'state',
                'assigned_to': 'assigned_to.name',
                'opened_by': 'opened_by.name',
                'opened_at': 'opened_at',
                'closed_at': 'closed_at',
                'category': 'category',
                'subcategory': 'subcategory',
                'ci_name': 'cmdb_ci.name',
                'location': 'location.name'
            },
            'sc_request': {
                'ticket_id': 'sys_id',
                'ticket_number': 'number',
                'title': 'short_description',
                'description': 'description',
                'priority': 'priority',
                'state': 'state',
                'requested_by': 'requested_by.name',
                'opened_at': 'opened_at',
                'closed_at': 'closed_at',
                'category': 'category'
            },
            'change_request': {
                'ticket_id': 'sys_id',
                'ticket_number': 'number',
                'title': 'short_description',
                'description': 'description',
                'priority': 'priority',
                'state': 'state',
                'risk': 'risk',
                'type': 'type',
                'requested_by': 'requested_by.name',
                'planned_start': 'start_date',
                'planned_end': 'end_date',
                'category': 'category'
            }
        }
        
        # State mappings
        self.state_mappings = {
            'incident': {
                '1': 'New',
                '2': 'In Progress', 
                '3': 'On Hold',
                '6': 'Resolved',
                '7': 'Closed',
                '8': 'Canceled'
            },
            'sc_request': {
                '1': 'Open',
                '2': 'Work in Progress',
                '3': 'Closed Complete',
                '4': 'Closed Incomplete',
                '5': 'Closed Skipped'
            },
            'change_request': {
                '-5': 'New',
                '-4': 'Assess',
                '-3': 'Authorize',
                '-2': 'Scheduled', 
                '-1': 'Implement',
                '0': 'Review',
                '3': 'Closed'
            }
        }
        
        self.session = None
        self.last_sync_times = {}  # Track last sync time per table
    
    async def validate_config(self):
        """Validate ServiceNow connector configuration"""
        required_fields = ['instance_url', 'authentication']
        
        for field in required_fields:
            if not self.get_connection_config(field):
                raise ValueError(f"Missing required configuration: {field}")
        
        # Validate instance URL format
        if not self.instance_url.startswith('https://'):
            raise ValueError("ServiceNow instance URL must use HTTPS")
        
        if not self.instance_url.endswith('.service-now.com'):
            logger.warning("Instance URL doesn't follow standard ServiceNow format")
        
        # Validate authentication
        auth_config = self.get_connection_config('authentication', {})
        auth_type = auth_config.get('type', 'basic')
        
        if auth_type == 'basic':
            if not auth_config.get('username') or not auth_config.get('password'):
                raise ValueError("Basic authentication requires username and password")
        elif auth_type == 'oauth2':
            required_oauth_fields = ['client_id', 'client_secret', 'refresh_token']
            for field in required_oauth_fields:
                if not auth_config.get(field):
                    raise ValueError(f"OAuth2 authentication requires {field}")
        else:
            raise ValueError(f"Unsupported authentication type: {auth_type}")
        
        logger.info("ServiceNow connector configuration validated")
    
    async def connect(self):
        """Establish connection to ServiceNow"""
        try:
            self.session = await self.create_http_session()
            
            # Set ServiceNow specific headers
            self.session.headers.update({
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            })
            
            # Handle OAuth2 if configured
            auth_config = self.get_connection_config('authentication', {})
            if auth_config.get('type') == 'oauth2':
                await self.authenticate_oauth2()
            
            logger.info(f"Connected to ServiceNow instance: {self.instance_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ServiceNow: {e}")
            raise
    
    async def disconnect(self):
        """Close connection to ServiceNow"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            logger.info("Disconnected from ServiceNow")
            
        except Exception as e:
            logger.error(f"Error disconnecting from ServiceNow: {e}")
    
    async def test_connection(self):
        """Test connection to ServiceNow"""
        if not self.session:
            raise ConnectionError("No active session")
        
        try:
            # Test with a simple query
            test_url = f"{self.instance_url}/api/now/table/sys_user"
            params = {'sysparm_limit': 1}
            
            async with self.session.get(test_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"ServiceNow connection test successful: {response.status}")
                    return True
                else:
                    error_text = await response.text()
                    raise ConnectionError(f"ServiceNow test failed: HTTP {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"ServiceNow connection test failed: {e}")
            raise
    
    async def authenticate_oauth2(self):
        """OAuth2 authentication for ServiceNow"""
        auth_config = self.get_connection_config('authentication', {})
        
        token_url = f"{self.instance_url}/oauth_token.do"
        
        data = {
            'grant_type': 'refresh_token',
            'client_id': auth_config['client_id'],
            'client_secret': auth_config['client_secret'],
            'refresh_token': auth_config['refresh_token']
        }
        
        async with self.session.post(token_url, data=data) as response:
            if response.status == 200:
                token_data = await response.json()
                access_token = token_data['access_token']
                
                # Update session headers
                self.session.headers.update({
                    'Authorization': f'Bearer {access_token}'
                })
                
                logger.info("OAuth2 authentication successful")
            else:
                error_text = await response.text()
                raise ConnectionError(f"OAuth2 authentication failed: {error_text}")
    
    async def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch data from ServiceNow tables"""
        try:
            all_data = []
            
            for table in self.tables_to_sync:
                try:
                    logger.info(f"Fetching data from ServiceNow table: {table}")
                    
                    # Fetch records from table
                    records = await self.fetch_table_records(table)
                    
                    # Transform records to standard format
                    for record in records:
                        transformed_record = await self.transform_ticket(record, table)
                        enriched_record = await self.enrich_data(transformed_record)
                        all_data.append(enriched_record)
                    
                    logger.info(f"Fetched {len(records)} records from {table}")
                    
                except Exception as e:
                    logger.error(f"Error fetching from table {table}: {e}")
            
            return all_data
            
        except Exception as e:
            logger.error(f"Error fetching data from ServiceNow: {e}")
            await self.handle_connection_error(e)
            raise
    
    async def fetch_table_records(self, table: str) -> List[Dict[str, Any]]:
        """Fetch records from a specific ServiceNow table"""
        url = f"{self.instance_url}/api/now/table/{table}"
        
        # Build query parameters
        params = {
            'sysparm_limit': self.max_records,
            'sysparm_display_value': 'all',  # Get display values for reference fields
            'sysparm_exclude_reference_link': 'true'
        }
        
        # Add incremental sync filter if configured
        if self.sync_mode == 'incremental':
            last_sync = self.last_sync_times.get(table)
            if last_sync:
                # Query for records updated since last sync
                params['sysparm_query'] = f"sys_updated_on>{last_sync.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Add table-specific filters
        table_filters = self.get_connection_config('table_filters', {})
        if table in table_filters:
            additional_filter = table_filters[table]
            if 'sysparm_query' in params:
                params['sysparm_query'] += f"^{additional_filter}"
            else:
                params['sysparm_query'] = additional_filter
        
        # Add field selection if configured
        field_mappings = self.field_mappings.get(table, {})
        if field_mappings:
            # Request only the fields we need
            fields = list(field_mappings.values())
            params['sysparm_fields'] = ','.join(fields)
        
        records = []
        offset = 0
        
        while True:
            params['sysparm_offset'] = offset
            
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        batch_records = data.get('result', [])
                        
                        if not batch_records:
                            break  # No more records
                        
                        records.extend(batch_records)
                        
                        # Check if we got fewer records than requested (last page)
                        if len(batch_records) < self.max_records:
                            break
                        
                        offset += len(batch_records)
                        
                        # Avoid fetching too many records in one sync
                        if offset >= 10000:
                            logger.warning(f"Reached maximum fetch limit for table {table}")
                            break
                            
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited, waiting {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to fetch from {table}: HTTP {response.status} - {error_text}")
                        break
                        
            except Exception as e:
                logger.error(f"Error in batch fetch from {table}: {e}")
                break
        
        # Update last sync time
        self.last_sync_times[table] = datetime.utcnow()
        
        return records
    
    async def transform_ticket(self, record: Dict[str, Any], table: str) -> Dict[str, Any]:
        """Transform ServiceNow ticket to standard format"""
        try:
            field_mapping = self.field_mappings.get(table, {})
            state_mapping = self.state_mappings.get(table, {})
            
            # Base ticket structure
            ticket = {
                'data_type': 'event',
                'event_type': self.map_table_to_event_type(table),
                'source_system': 'ServiceNow',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Map fields based on configuration
            for standard_field, snow_field in field_mapping.items():
                value = self.get_nested_field(record, snow_field)
                
                if standard_field == 'severity':
                    ticket['severity'] = self.map_priority_to_severity(value, table)
                elif standard_field == 'state':
                    ticket['status'] = state_mapping.get(str(value), value)
                elif standard_field in ['opened_at', 'closed_at', 'planned_start', 'planned_end']:
                    ticket[standard_field] = self.parse_servicenow_datetime(value)
                else:
                    ticket[standard_field] = value
            
            # Set title and description
            ticket['title'] = ticket.get('title') or f"{table.title()}: {ticket.get('ticket_number', 'Unknown')}"
            
            # Map to site if possible
            site_mapping = self.get_mapping_rule('site_field_mapping')
            if site_mapping:
                site_identifier = self.get_nested_field(record, site_mapping)
                if site_identifier:
                    ticket['site_code'] = site_identifier
            
            # Add ServiceNow specific metadata
            ticket['metadata'] = {
                'servicenow_table': table,
                'servicenow_sys_id': record.get('sys_id'),
                'servicenow_url': f"{self.instance_url}/nav_to.do?uri={table}.do?sys_id={record.get('sys_id')}",
                'category': record.get('category', ''),
                'subcategory': record.get('subcategory', ''),
                'business_service': record.get('business_service', ''),
                'assignment_group': record.get('assignment_group', {}).get('display_value', '')
            }
            
            return ticket
            
        except Exception as e:
            logger.error(f"Error transforming ticket from {table}: {e}")
            return {
                'data_type': 'event',
                'event_type': 'ITSM_ERROR',
                'title': f"Error transforming {table} record",
                'description': str(e),
                'severity': 'WARNING',
                'source_system': 'ServiceNow',
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': {'raw_record': record}
            }
    
    def map_table_to_event_type(self, table: str) -> str:
        """Map ServiceNow table to event type"""
        table_mappings = {
            'incident': 'INCIDENT',
            'sc_request': 'REQUEST',
            'change_request': 'CHANGE',
            'problem': 'PROBLEM',
            'kb_knowledge': 'KNOWLEDGE'
        }
        
        return table_mappings.get(table, 'ITSM')
    
    def map_priority_to_severity(self, priority: str, table: str) -> str:
        """Map ServiceNow priority to standard severity"""
        # ServiceNow priority: 1=Critical, 2=High, 3=Moderate, 4=Low, 5=Planning
        priority_mapping = {
            '1': 'CRITICAL',
            '2': 'MAJOR',
            '3': 'MINOR', 
            '4': 'WARNING',
            '5': 'INFO'
        }
        
        # For incidents, also consider severity field if available
        if table == 'incident' and priority:
            return priority_mapping.get(str(priority), 'WARNING')
        
        return priority_mapping.get(str(priority), 'WARNING')
    
    def get_nested_field(self, record: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value using dot notation"""
        try:
            value = record
            for part in field_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return value
        except:
            return None
    
    def parse_servicenow_datetime(self, dt_string: str) -> Optional[str]:
        """Parse ServiceNow datetime format"""
        if not dt_string:
            return None
        
        try:
            # ServiceNow datetime format: YYYY-MM-DD HH:MM:SS
            dt = datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S')
            return dt.isoformat()
        except:
            # Return as-is if parsing fails
            return dt_string
    
    async def create_ticket(self, ticket_data: Dict[str, Any]) -> Optional[str]:
        """Create a new ticket in ServiceNow"""
        try:
            table = ticket_data.get('table', 'incident')
            url = f"{self.instance_url}/api/now/table/{table}"
            
            # Transform data to ServiceNow format
            snow_data = await self.transform_to_servicenow_format(ticket_data, table)
            
            async with self.session.post(url, json=snow_data) as response:
                if response.status == 201:
                    result = await response.json()
                    ticket_id = result['result']['sys_id']
                    ticket_number = result['result']['number']
                    
                    logger.info(f"Created ServiceNow ticket: {ticket_number} ({ticket_id})")
                    return ticket_id
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to create ServiceNow ticket: HTTP {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error creating ServiceNow ticket: {e}")
            return None
    
    async def update_ticket(self, ticket_id: str, update_data: Dict[str, Any]) -> bool:
        """Update existing ticket in ServiceNow"""
        try:
            table = update_data.get('table', 'incident')
            url = f"{self.instance_url}/api/now/table/{table}/{ticket_id}"
            
            # Transform data to ServiceNow format
            snow_data = await self.transform_to_servicenow_format(update_data, table)
            
            async with self.session.patch(url, json=snow_data) as response:
                if response.status == 200:
                    logger.info(f"Updated ServiceNow ticket: {ticket_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to update ServiceNow ticket: HTTP {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating ServiceNow ticket {ticket_id}: {e}")
            return False
    
    async def transform_to_servicenow_format(self, data: Dict[str, Any], table: str) -> Dict[str, Any]:
        """Transform standard ticket format to ServiceNow format"""
        field_mapping = self.field_mappings.get(table, {})
        snow_data = {}
        
        # Reverse the field mapping
        for standard_field, snow_field in field_mapping.items():
            if standard_field in data:
                # Handle nested fields (only first level for now)
                if '.' in snow_field:
                    parent_field = snow_field.split('.')[0]
                    snow_data[parent_field] = data[standard_field]
                else:
                    snow_data[snow_field] = data[standard_field]
        
        return snow_data
    
    async def get_ticket_updates(self, ticket_id: str, table: str = 'incident') -> List[Dict[str, Any]]:
        """Get updates/comments for a specific ticket"""
        try:
            # Get journal entries (updates/comments)
            url = f"{self.instance_url}/api/now/table/sys_journal_field"
            params = {
                'sysparm_query': f'element_id={ticket_id}^element!=comments^ORDERBYsys_created_on',
                'sysparm_limit': 100
            }
            
            updates = []
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for entry in data.get('result', []):
                        update = {
                            'timestamp': self.parse_servicenow_datetime(entry.get('sys_created_on')),
                            'user': entry.get('sys_created_by'),
                            'field': entry.get('element'),
                            'value': entry.get('value'),
                            'new_value': entry.get('new_value')
                        }
                        updates.append(update)
            
            return updates
            
        except Exception as e:
            logger.error(f"Error fetching ticket updates for {ticket_id}: {e}")
            return []