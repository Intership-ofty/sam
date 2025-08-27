"""
OSS Network Connector - Universal connector for Network OSS systems
Supports: Ericsson OSS-RC, Huawei U2000, Nokia NetAct, ZTE ZXONE, etc.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import json

from .base_connector import DataConnector, HttpConnectorMixin

logger = logging.getLogger(__name__)


class OSS_NetworkConnector(DataConnector, HttpConnectorMixin):
    """OSS Network Connector for telecom network management systems"""
    
    def __init__(self, data_source, message_producer, quality_validator):
        super().__init__(data_source, message_producer, quality_validator)
        self.oss_type = self.get_connection_config('oss_type', 'generic')
        self.api_version = self.get_connection_config('api_version', 'v1')
        self.polling_endpoints = self.get_connection_config('polling_endpoints', [])
        self.supported_technologies = self.get_connection_config('technologies', ['2G', '3G', '4G', '5G'])
        
        # OSS-specific configurations
        self.oss_configs = {
            'ericsson': {
                'auth_method': 'oauth2',
                'performance_endpoints': ['/pm/measurements', '/fm/alarms'],
                'config_endpoints': ['/cm/configurations'],
                'default_namespace': 'urn:3gpp:sa5:nrm'
            },
            'huawei': {
                'auth_method': 'session',
                'performance_endpoints': ['/restconf/data/performance', '/restconf/data/alarms'],
                'config_endpoints': ['/restconf/data/config'],
                'default_namespace': 'urn:huawei:yang'
            },
            'nokia': {
                'auth_method': 'basic',
                'performance_endpoints': ['/rest/api/v1/measurements', '/rest/api/v1/alarms'],
                'config_endpoints': ['/rest/api/v1/config'],
                'default_namespace': 'urn:nokia:nsp'
            }
        }
        
        self.session = None
        self.auth_token = None
        self.last_measurement_time = None
    
    async def validate_config(self):
        """Validate OSS connector configuration"""
        required_fields = ['base_url', 'oss_type']
        
        for field in required_fields:
            if not self.get_connection_config(field):
                raise ValueError(f"Missing required configuration: {field}")
        
        # Validate OSS type
        if self.oss_type not in self.oss_configs and self.oss_type != 'generic':
            raise ValueError(f"Unsupported OSS type: {self.oss_type}")
        
        logger.info(f"OSS connector configuration validated for {self.oss_type}")
    
    async def connect(self):
        """Establish connection to OSS system"""
        try:
            self.session = await self.create_http_session()
            
            # Perform OSS-specific authentication
            if self.oss_type in self.oss_configs:
                await self.authenticate_oss()
            
            logger.info(f"Connected to OSS system: {self.data_source.source_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to OSS system: {e}")
            raise
    
    async def disconnect(self):
        """Close connection to OSS system"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
                self.auth_token = None
            
            logger.info(f"Disconnected from OSS system: {self.data_source.source_name}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from OSS system: {e}")
    
    async def test_connection(self):
        """Test connection to OSS system"""
        if not self.session:
            raise ConnectionError("No active session")
        
        try:
            # Test with a simple status endpoint
            base_url = self.get_connection_config('base_url')
            test_endpoint = self.get_connection_config('test_endpoint', '/status')
            
            async with self.session.get(f"{base_url}{test_endpoint}") as response:
                if response.status == 200:
                    logger.info(f"OSS connection test successful: {response.status}")
                    return True
                else:
                    raise ConnectionError(f"OSS test failed with status: {response.status}")
                    
        except Exception as e:
            logger.error(f"OSS connection test failed: {e}")
            raise
    
    async def authenticate_oss(self):
        """Perform OSS-specific authentication"""
        oss_config = self.oss_configs[self.oss_type]
        auth_method = oss_config['auth_method']
        
        if auth_method == 'oauth2':
            await self.authenticate_oauth2()
        elif auth_method == 'session':
            await self.authenticate_session()
        elif auth_method == 'basic':
            # Basic auth is handled in HTTP session creation
            pass
        
        logger.info(f"Authentication completed for {self.oss_type}")
    
    async def authenticate_oauth2(self):
        """OAuth2 authentication (Ericsson OSS-RC)"""
        base_url = self.get_connection_config('base_url')
        auth_config = self.get_connection_config('authentication', {})
        
        token_endpoint = auth_config.get('token_endpoint', '/auth/oauth2/token')
        client_id = auth_config.get('client_id')
        client_secret = auth_config.get('client_secret')
        
        if not client_id or not client_secret:
            raise ValueError("OAuth2 requires client_id and client_secret")
        
        data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret,
            'scope': 'read write'
        }
        
        async with self.session.post(f"{base_url}{token_endpoint}", data=data) as response:
            if response.status == 200:
                token_data = await response.json()
                self.auth_token = token_data['access_token']
                
                # Update session headers
                self.session.headers.update({
                    'Authorization': f'Bearer {self.auth_token}'
                })
            else:
                error_text = await response.text()
                raise ConnectionError(f"OAuth2 authentication failed: {error_text}")
    
    async def authenticate_session(self):
        """Session-based authentication (Huawei U2000)"""
        base_url = self.get_connection_config('base_url')
        auth_config = self.get_connection_config('authentication', {})
        
        login_endpoint = auth_config.get('login_endpoint', '/rest/login')
        username = auth_config.get('username')
        password = auth_config.get('password')
        
        if not username or not password:
            raise ValueError("Session authentication requires username and password")
        
        login_data = {
            'username': username,
            'password': password
        }
        
        async with self.session.post(f"{base_url}{login_endpoint}", json=login_data) as response:
            if response.status == 200:
                session_data = await response.json()
                session_id = session_data.get('sessionId')
                
                if session_id:
                    self.session.headers.update({
                        'X-Session-ID': session_id
                    })
                else:
                    raise ConnectionError("Session ID not received")
            else:
                error_text = await response.text()
                raise ConnectionError(f"Session authentication failed: {error_text}")
    
    async def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch data from OSS system"""
        try:
            all_data = []
            
            # Fetch performance measurements
            pm_data = await self.fetch_performance_measurements()
            all_data.extend(pm_data)
            
            # Fetch fault management data
            fm_data = await self.fetch_fault_management()
            all_data.extend(fm_data)
            
            # Fetch configuration data if requested
            if self.get_connection_config('include_configuration', False):
                cm_data = await self.fetch_configuration_management()
                all_data.extend(cm_data)
            
            # Transform and enrich data
            transformed_data = []
            for item in all_data:
                enriched_item = await self.enrich_data(item)
                transformed_data.append(enriched_item)
            
            logger.info(f"Fetched {len(transformed_data)} items from OSS system")
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error fetching data from OSS system: {e}")
            await self.handle_connection_error(e)
            raise
    
    async def fetch_performance_measurements(self) -> List[Dict[str, Any]]:
        """Fetch performance measurement data"""
        data = []
        base_url = self.get_connection_config('base_url')
        
        # Determine endpoints based on OSS type
        if self.oss_type in self.oss_configs:
            endpoints = self.oss_configs[self.oss_type]['performance_endpoints']
        else:
            endpoints = self.polling_endpoints or ['/api/measurements']
        
        for endpoint in endpoints:
            try:
                # Build query parameters
                params = await self.build_measurement_params()
                
                async with self.session.get(f"{base_url}{endpoint}", params=params) as response:
                    if response.status == 200:
                        content_type = response.headers.get('Content-Type', '')
                        
                        if 'application/json' in content_type:
                            raw_data = await response.json()
                        elif 'application/xml' in content_type or 'text/xml' in content_type:
                            xml_text = await response.text()
                            raw_data = await self.parse_xml_measurements(xml_text)
                        else:
                            # Try JSON first, then XML
                            text_data = await response.text()
                            try:
                                raw_data = json.loads(text_data)
                            except:
                                raw_data = await self.parse_xml_measurements(text_data)
                        
                        # Transform to standard format
                        measurements = await self.transform_measurements(raw_data, endpoint)
                        data.extend(measurements)
                        
                    else:
                        logger.warning(f"Failed to fetch from {endpoint}: HTTP {response.status}")
                        
            except Exception as e:
                logger.error(f"Error fetching from endpoint {endpoint}: {e}")
        
        return data
    
    async def fetch_fault_management(self) -> List[Dict[str, Any]]:
        """Fetch fault management data (alarms/events)"""
        data = []
        base_url = self.get_connection_config('base_url')
        
        # Get alarm endpoints
        endpoints = ['/api/alarms', '/api/events']
        if self.oss_type in self.oss_configs:
            # Use OSS-specific endpoints if available
            pass
        
        for endpoint in endpoints:
            try:
                params = {
                    'state': 'active',
                    'limit': self.get_connection_config('max_alarms', 1000)
                }
                
                # Add time filter for incremental sync
                if self.last_sync:
                    params['since'] = self.last_sync.isoformat()
                
                async with self.session.get(f"{base_url}{endpoint}", params=params) as response:
                    if response.status == 200:
                        raw_data = await response.json()
                        
                        # Transform to standard event format
                        events = await self.transform_alarms(raw_data, endpoint)
                        data.extend(events)
                        
                    else:
                        logger.warning(f"Failed to fetch alarms from {endpoint}: HTTP {response.status}")
                        
            except Exception as e:
                logger.error(f"Error fetching alarms from {endpoint}: {e}")
        
        return data
    
    async def fetch_configuration_management(self) -> List[Dict[str, Any]]:
        """Fetch configuration management data"""
        data = []
        base_url = self.get_connection_config('base_url')
        
        # Configuration endpoints
        endpoints = ['/api/config/nodes', '/api/config/sites']
        
        for endpoint in endpoints:
            try:
                async with self.session.get(f"{base_url}{endpoint}") as response:
                    if response.status == 200:
                        raw_data = await response.json()
                        
                        # Transform configuration data
                        config_items = await self.transform_configuration(raw_data, endpoint)
                        data.extend(config_items)
                        
            except Exception as e:
                logger.error(f"Error fetching configuration from {endpoint}: {e}")
        
        return data
    
    async def build_measurement_params(self) -> Dict[str, Any]:
        """Build parameters for measurement queries"""
        params = {}
        
        # Time range
        end_time = datetime.utcnow()
        if self.last_measurement_time:
            start_time = self.last_measurement_time
        else:
            start_time = end_time - timedelta(minutes=15)  # Default 15 minutes
        
        params['startTime'] = start_time.isoformat()
        params['endTime'] = end_time.isoformat()
        
        # Update last measurement time
        self.last_measurement_time = end_time
        
        # Technologies filter
        if self.supported_technologies:
            params['technologies'] = ','.join(self.supported_technologies)
        
        # Measurement types
        measurement_types = self.get_connection_config('measurement_types', [])
        if measurement_types:
            params['measurementTypes'] = ','.join(measurement_types)
        
        # Site filter
        site_filter = self.get_connection_config('site_filter')
        if site_filter:
            params['sites'] = site_filter
        
        return params
    
    async def parse_xml_measurements(self, xml_text: str) -> Dict[str, Any]:
        """Parse XML measurement data"""
        try:
            root = ET.fromstring(xml_text)
            
            # Generic XML to dict conversion
            def xml_to_dict(element):
                result = {}
                
                # Add attributes
                if element.attrib:
                    result.update(element.attrib)
                
                # Add text content
                if element.text and element.text.strip():
                    if len(element) == 0:  # Leaf node
                        return element.text.strip()
                    else:
                        result['text'] = element.text.strip()
                
                # Add child elements
                for child in element:
                    child_data = xml_to_dict(child)
                    
                    if child.tag in result:
                        # Multiple elements with same tag
                        if not isinstance(result[child.tag], list):
                            result[child.tag] = [result[child.tag]]
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = child_data
                
                return result
            
            return xml_to_dict(root)
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return {}
    
    async def transform_measurements(self, raw_data: Any, endpoint: str) -> List[Dict[str, Any]]:
        """Transform raw measurement data to standard format"""
        measurements = []
        
        try:
            # Handle different data structures based on OSS type
            if isinstance(raw_data, dict):
                if 'measurements' in raw_data:
                    data_items = raw_data['measurements']
                elif 'data' in raw_data:
                    data_items = raw_data['data']
                elif 'results' in raw_data:
                    data_items = raw_data['results']
                else:
                    data_items = [raw_data]
            elif isinstance(raw_data, list):
                data_items = raw_data
            else:
                data_items = [{'raw_data': raw_data}]
            
            for item in data_items:
                # Extract measurement data
                measurement = {
                    'data_type': 'network_metric',
                    'timestamp': datetime.utcnow().isoformat(),
                    'technology': item.get('technology', 'unknown'),
                    'metric_name': item.get('measurementType', item.get('counter', 'unknown')),
                    'metric_value': self.safe_float(item.get('value', item.get('measurement'))),
                    'unit': item.get('unit', ''),
                    'site_id': item.get('siteId', item.get('nodeId')),
                    'quality_score': self.calculate_quality_score(item),
                    'metadata': {
                        'endpoint': endpoint,
                        'oss_type': self.oss_type,
                        'raw_timestamp': item.get('timestamp'),
                        'node_name': item.get('nodeName'),
                        'cell_id': item.get('cellId')
                    }
                }
                
                # Add technology-specific fields
                if item.get('technology') == '5G':
                    measurement['metadata']['slice_id'] = item.get('sliceId')
                    measurement['metadata']['beam_id'] = item.get('beamId')
                elif item.get('technology') == '4G':
                    measurement['metadata']['pci'] = item.get('pci')
                    measurement['metadata']['earfcn'] = item.get('earfcn')
                
                measurements.append(measurement)
            
            return measurements
            
        except Exception as e:
            logger.error(f"Error transforming measurement data: {e}")
            return []
    
    async def transform_alarms(self, raw_data: Any, endpoint: str) -> List[Dict[str, Any]]:
        """Transform alarm data to standard event format"""
        events = []
        
        try:
            if isinstance(raw_data, dict):
                if 'alarms' in raw_data:
                    data_items = raw_data['alarms']
                elif 'events' in raw_data:
                    data_items = raw_data['events']
                else:
                    data_items = [raw_data]
            elif isinstance(raw_data, list):
                data_items = raw_data
            else:
                data_items = [raw_data]
            
            for item in data_items:
                event = {
                    'data_type': 'event',
                    'timestamp': item.get('timestamp', datetime.utcnow().isoformat()),
                    'event_type': item.get('eventType', 'ALARM'),
                    'severity': self.map_severity(item.get('severity', 'UNKNOWN')),
                    'title': item.get('alarmText', item.get('description', 'Unknown alarm')),
                    'description': item.get('additionalText'),
                    'site_id': item.get('siteId', item.get('nodeId')),
                    'source_system': f"OSS-{self.oss_type}",
                    'acknowledged': item.get('acknowledged', False),
                    'metadata': {
                        'alarm_id': item.get('alarmId'),
                        'probable_cause': item.get('probableCause'),
                        'specific_problem': item.get('specificProblem'),
                        'managed_object': item.get('managedObject'),
                        'endpoint': endpoint
                    }
                }
                
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error transforming alarm data: {e}")
            return []
    
    async def transform_configuration(self, raw_data: Any, endpoint: str) -> List[Dict[str, Any]]:
        """Transform configuration data"""
        config_items = []
        
        try:
            # Configuration data is usually for inventory/topology
            # Transform to site/equipment information
            if isinstance(raw_data, list):
                data_items = raw_data
            elif isinstance(raw_data, dict):
                data_items = raw_data.get('nodes', raw_data.get('sites', [raw_data]))
            else:
                data_items = [raw_data]
            
            for item in data_items:
                config_item = {
                    'data_type': 'configuration',
                    'timestamp': datetime.utcnow().isoformat(),
                    'node_name': item.get('nodeName', item.get('name')),
                    'site_id': item.get('siteId'),
                    'node_type': item.get('nodeType'),
                    'technology': item.get('technology'),
                    'location': {
                        'latitude': item.get('latitude'),
                        'longitude': item.get('longitude')
                    },
                    'metadata': {
                        'vendor': item.get('vendor'),
                        'software_version': item.get('swVersion'),
                        'hardware_version': item.get('hwVersion'),
                        'operational_state': item.get('operationalState'),
                        'administrative_state': item.get('administrativeState')
                    }
                }
                
                config_items.append(config_item)
            
            return config_items
            
        except Exception as e:
            logger.error(f"Error transforming configuration data: {e}")
            return []
    
    def safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def calculate_quality_score(self, item: Dict[str, Any]) -> float:
        """Calculate data quality score"""
        score = 1.0
        
        # Reduce score for missing required fields
        required_fields = ['value', 'timestamp', 'measurementType']
        for field in required_fields:
            if not item.get(field):
                score -= 0.2
        
        # Check for suspicious values
        value = item.get('value')
        if value is not None:
            try:
                num_value = float(value)
                if num_value < 0 and item.get('measurementType') not in ['temperature', 'power']:
                    score -= 0.1  # Negative values might be suspicious for some metrics
            except:
                score -= 0.3  # Non-numeric values reduce quality
        
        return max(0.0, min(1.0, score))
    
    def map_severity(self, oss_severity: str) -> str:
        """Map OSS-specific severity to standard severity"""
        severity_mapping = {
            'critical': 'CRITICAL',
            'major': 'MAJOR', 
            'minor': 'MINOR',
            'warning': 'WARNING',
            'cleared': 'INFO',
            'indeterminate': 'WARNING',
            '1': 'CRITICAL',  # Numeric severity levels
            '2': 'MAJOR',
            '3': 'MINOR',
            '4': 'WARNING',
            '5': 'INFO'
        }
        
        return severity_mapping.get(str(oss_severity).lower(), 'WARNING')