"""
Generic REST API Connector - Universal connector for REST APIs
Supports various authentication methods and data formats
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from .base_connector import DataConnector, HttpConnectorMixin

logger = logging.getLogger(__name__)


class GenericRestConnector(DataConnector, HttpConnectorMixin):
    """Generic REST API connector for various systems"""
    
    def __init__(self, data_source, message_producer, quality_validator):
        super().__init__(data_source, message_producer, quality_validator)
        
        # REST connector configuration
        self.base_url = self.get_connection_config('base_url')
        self.endpoints = self.get_connection_config('endpoints', [])
        self.data_format = self.get_connection_config('data_format', 'json')  # json, xml, csv
        self.pagination_type = self.get_connection_config('pagination', 'none')  # none, offset, cursor, page
        self.rate_limit = self.get_connection_config('rate_limit', {})
        
        self.session = None
        self.request_count = 0
        self.last_request_time = None
    
    async def validate_config(self):
        """Validate REST connector configuration"""
        required_fields = ['base_url', 'endpoints']
        
        for field in required_fields:
            if not self.get_connection_config(field):
                raise ValueError(f"Missing required configuration: {field}")
        
        # Validate base URL
        if not self.base_url.startswith(('http://', 'https://')):
            raise ValueError("base_url must be a valid HTTP/HTTPS URL")
        
        # Validate endpoints
        if not isinstance(self.endpoints, list) or len(self.endpoints) == 0:
            raise ValueError("endpoints must be a non-empty list")
        
        logger.info("Generic REST connector configuration validated")
    
    async def connect(self):
        """Establish HTTP connection"""
        try:
            self.session = await self.create_http_session()
            logger.info(f"Connected to REST API: {self.base_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to REST API: {e}")
            raise
    
    async def disconnect(self):
        """Close HTTP connection"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            logger.info("Disconnected from REST API")
            
        except Exception as e:
            logger.error(f"Error disconnecting from REST API: {e}")
    
    async def test_connection(self):
        """Test REST API connection"""
        if not self.session:
            raise ConnectionError("No active session")
        
        try:
            # Use first endpoint or health check endpoint for testing
            test_endpoint = self.get_connection_config('health_endpoint')
            if not test_endpoint and self.endpoints:
                test_endpoint = self.endpoints[0] if isinstance(self.endpoints[0], str) else self.endpoints[0].get('path')
            
            if not test_endpoint:
                raise ValueError("No endpoint available for connection test")
            
            url = f"{self.base_url.rstrip('/')}/{test_endpoint.lstrip('/')}"
            
            async with self.session.get(url) as response:
                if response.status < 400:
                    logger.info(f"REST API connection test successful: {response.status}")
                    return True
                else:
                    raise ConnectionError(f"REST API test failed: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"REST API connection test failed: {e}")
            raise
    
    async def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch data from REST API endpoints"""
        try:
            all_data = []
            
            for endpoint_config in self.endpoints:
                try:
                    # Handle rate limiting
                    await self.handle_rate_limit()
                    
                    # Fetch data from endpoint
                    endpoint_data = await self.fetch_endpoint_data(endpoint_config)
                    all_data.extend(endpoint_data)
                    
                except Exception as e:
                    logger.error(f"Error fetching from endpoint {endpoint_config}: {e}")
            
            # Transform and enrich data
            transformed_data = []
            for item in all_data:
                enriched_item = await self.enrich_data(item)
                transformed_data.append(enriched_item)
            
            logger.info(f"Fetched {len(transformed_data)} items from REST API")
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error fetching data from REST API: {e}")
            await self.handle_connection_error(e)
            raise
    
    async def fetch_endpoint_data(self, endpoint_config: Any) -> List[Dict[str, Any]]:
        """Fetch data from a single endpoint"""
        # Handle both string and dict endpoint configurations
        if isinstance(endpoint_config, str):
            endpoint_config = {'path': endpoint_config}
        
        endpoint_path = endpoint_config['path']
        method = endpoint_config.get('method', 'GET').upper()
        params = endpoint_config.get('params', {})
        headers = endpoint_config.get('headers', {})
        data_path = endpoint_config.get('data_path', 'data')  # Path to data in response
        
        url = f"{self.base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"
        
        # Add time-based parameters if configured
        if endpoint_config.get('time_based', False):
            time_params = await self.build_time_params(endpoint_config)
            params.update(time_params)
        
        all_data = []
        
        # Handle pagination
        if self.pagination_type == 'none':
            data = await self.make_request(method, url, params, headers)
            items = self.extract_data_from_response(data, data_path)
            all_data.extend(items)
        
        else:
            # Paginated requests
            page_data = await self.fetch_paginated_data(method, url, params, headers, data_path, endpoint_config)
            all_data.extend(page_data)
        
        # Transform data based on endpoint configuration
        transformed_data = []
        for item in all_data:
            transformed_item = await self.transform_rest_data(item, endpoint_config)
            if transformed_item:
                transformed_data.append(transformed_item)
        
        return transformed_data
    
    async def make_request(self, method: str, url: str, params: Dict = None, headers: Dict = None, data: Any = None) -> Any:
        """Make HTTP request with error handling"""
        try:
            request_kwargs = {}
            if params:
                request_kwargs['params'] = params
            if headers:
                self.session.headers.update(headers)
            if data:
                request_kwargs['json'] = data
            
            async with self.session.request(method, url, **request_kwargs) as response:
                self.request_count += 1
                self.last_request_time = datetime.utcnow()
                
                if response.status < 400:
                    # Parse response based on content type
                    content_type = response.headers.get('Content-Type', '').lower()
                    
                    if 'application/json' in content_type:
                        return await response.json()
                    elif 'application/xml' in content_type or 'text/xml' in content_type:
                        text = await response.text()
                        return await self.parse_xml_response(text)
                    elif 'text/csv' in content_type:
                        text = await response.text()
                        return await self.parse_csv_response(text)
                    else:
                        # Try JSON first, then text
                        try:
                            return await response.json()
                        except:
                            return await response.text()
                else:
                    error_text = await response.text()
                    logger.error(f"HTTP request failed: {response.status} - {error_text}")
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Request error for {method} {url}: {e}")
            raise
    
    async def fetch_paginated_data(self, method: str, url: str, params: Dict, headers: Dict, 
                                  data_path: str, endpoint_config: Dict) -> List[Dict[str, Any]]:
        """Fetch data with pagination support"""
        all_data = []
        page = 1
        offset = 0
        cursor = None
        max_pages = endpoint_config.get('max_pages', 100)
        
        while page <= max_pages:
            # Update pagination parameters
            paginated_params = params.copy()
            
            if self.pagination_type == 'offset':
                page_size = endpoint_config.get('page_size', 100)
                paginated_params['limit'] = page_size
                paginated_params['offset'] = offset
                
            elif self.pagination_type == 'page':
                page_size = endpoint_config.get('page_size', 100)
                paginated_params['page'] = page
                paginated_params['page_size'] = page_size
                
            elif self.pagination_type == 'cursor':
                if cursor:
                    paginated_params['cursor'] = cursor
                page_size = endpoint_config.get('page_size', 100)
                paginated_params['limit'] = page_size
            
            try:
                response_data = await self.make_request(method, url, paginated_params, headers)
                items = self.extract_data_from_response(response_data, data_path)
                
                if not items:
                    break  # No more data
                
                all_data.extend(items)
                
                # Update pagination variables
                if self.pagination_type == 'offset':
                    offset += len(items)
                    if len(items) < page_size:
                        break  # Last page
                        
                elif self.pagination_type == 'page':
                    page += 1
                    if len(items) < page_size:
                        break  # Last page
                        
                elif self.pagination_type == 'cursor':
                    # Extract cursor from response
                    cursor = response_data.get('next_cursor') or response_data.get('pagination', {}).get('next_cursor')
                    if not cursor:
                        break  # No more pages
                
                # Respect rate limits
                await self.handle_rate_limit()
                
            except Exception as e:
                logger.error(f"Error in paginated request (page {page}): {e}")
                break
        
        return all_data
    
    def extract_data_from_response(self, response: Any, data_path: str) -> List[Dict[str, Any]]:
        """Extract data from response using data path"""
        try:
            if not response:
                return []
            
            # Navigate to data using dot notation
            current = response
            for part in data_path.split('.'):
                if part == '':
                    continue
                    
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, list) and part.isdigit():
                    idx = int(part)
                    current = current[idx] if idx < len(current) else None
                else:
                    current = None
                    
                if current is None:
                    break
            
            # Ensure we return a list
            if current is None:
                return []
            elif isinstance(current, list):
                return current
            elif isinstance(current, dict):
                return [current]
            else:
                return [{'value': current}]
                
        except Exception as e:
            logger.error(f"Error extracting data from response: {e}")
            return []
    
    async def transform_rest_data(self, item: Dict[str, Any], endpoint_config: Dict) -> Optional[Dict[str, Any]]:
        """Transform REST API data to standard format"""
        try:
            # Apply endpoint-specific transformations
            transform_config = endpoint_config.get('transform', {})
            data_type = transform_config.get('data_type', 'unknown')
            
            if data_type == 'metric':
                return await self.transform_to_metric(item, transform_config)
            elif data_type == 'event':
                return await self.transform_to_event(item, transform_config)
            elif data_type == 'configuration':
                return await self.transform_to_configuration(item, transform_config)
            else:
                # Generic transformation
                return await self.transform_generic(item, transform_config)
                
        except Exception as e:
            logger.error(f"Error transforming REST data: {e}")
            return None
    
    async def transform_to_metric(self, item: Dict[str, Any], config: Dict) -> Dict[str, Any]:
        """Transform to metric format"""
        return {
            'data_type': 'network_metric',
            'timestamp': item.get(config.get('timestamp_field', 'timestamp'), datetime.utcnow().isoformat()),
            'site_id': item.get(config.get('site_field', 'site_id')),
            'technology': item.get(config.get('technology_field', 'technology'), 'unknown'),
            'metric_name': item.get(config.get('metric_name_field', 'metric_name')),
            'metric_value': self.safe_float(item.get(config.get('value_field', 'value'))),
            'unit': item.get(config.get('unit_field', 'unit'), ''),
            'quality_score': 1.0,
            'metadata': {
                'source': 'rest_api',
                'endpoint': config.get('endpoint_name', ''),
                'raw_data': item
            }
        }
    
    async def transform_to_event(self, item: Dict[str, Any], config: Dict) -> Dict[str, Any]:
        """Transform to event format"""
        return {
            'data_type': 'event',
            'timestamp': item.get(config.get('timestamp_field', 'timestamp'), datetime.utcnow().isoformat()),
            'event_type': item.get(config.get('event_type_field', 'event_type'), 'UNKNOWN'),
            'severity': item.get(config.get('severity_field', 'severity'), 'WARNING'),
            'title': item.get(config.get('title_field', 'title'), 'Unknown event'),
            'description': item.get(config.get('description_field', 'description')),
            'site_id': item.get(config.get('site_field', 'site_id')),
            'source_system': 'REST-API',
            'metadata': {
                'source': 'rest_api',
                'endpoint': config.get('endpoint_name', ''),
                'raw_data': item
            }
        }
    
    async def transform_to_configuration(self, item: Dict[str, Any], config: Dict) -> Dict[str, Any]:
        """Transform to configuration format"""
        return {
            'data_type': 'configuration',
            'timestamp': datetime.utcnow().isoformat(),
            'config_type': config.get('config_type', 'unknown'),
            'site_id': item.get(config.get('site_field', 'site_id')),
            'configuration': item,
            'metadata': {
                'source': 'rest_api',
                'endpoint': config.get('endpoint_name', ''),
                'config_version': item.get('version')
            }
        }
    
    async def transform_generic(self, item: Dict[str, Any], config: Dict) -> Dict[str, Any]:
        """Generic transformation"""
        return {
            'data_type': config.get('default_type', 'unknown'),
            'timestamp': datetime.utcnow().isoformat(),
            'raw_data': item,
            'metadata': {
                'source': 'rest_api',
                'endpoint': config.get('endpoint_name', ''),
                'transform_type': 'generic'
            }
        }
    
    async def build_time_params(self, endpoint_config: Dict) -> Dict[str, str]:
        """Build time-based query parameters"""
        params = {}
        
        time_config = endpoint_config.get('time_config', {})
        
        # Calculate time range
        end_time = datetime.utcnow()
        time_window = time_config.get('window', '15m')
        
        # Parse time window (e.g., '15m', '1h', '1d')
        if time_window.endswith('m'):
            minutes = int(time_window[:-1])
            start_time = end_time - timedelta(minutes=minutes)
        elif time_window.endswith('h'):
            hours = int(time_window[:-1])
            start_time = end_time - timedelta(hours=hours)
        elif time_window.endswith('d'):
            days = int(time_window[:-1])
            start_time = end_time - timedelta(days=days)
        else:
            start_time = end_time - timedelta(minutes=15)  # Default
        
        # Format times according to API requirements
        time_format = time_config.get('format', 'iso')
        
        if time_format == 'iso':
            start_str = start_time.isoformat()
            end_str = end_time.isoformat()
        elif time_format == 'epoch':
            start_str = str(int(start_time.timestamp()))
            end_str = str(int(end_time.timestamp()))
        elif time_format == 'custom':
            format_str = time_config.get('custom_format', '%Y-%m-%d %H:%M:%S')
            start_str = start_time.strftime(format_str)
            end_str = end_time.strftime(format_str)
        else:
            start_str = start_time.isoformat()
            end_str = end_time.isoformat()
        
        # Add to parameters using configured field names
        start_param = time_config.get('start_param', 'start_time')
        end_param = time_config.get('end_param', 'end_time')
        
        params[start_param] = start_str
        params[end_param] = end_str
        
        return params
    
    async def handle_rate_limit(self):
        """Handle API rate limiting"""
        if not self.rate_limit:
            return
        
        requests_per_minute = self.rate_limit.get('requests_per_minute')
        if not requests_per_minute:
            return
        
        if self.last_request_time:
            time_since_last = (datetime.utcnow() - self.last_request_time).total_seconds()
            min_interval = 60.0 / requests_per_minute
            
            if time_since_last < min_interval:
                delay = min_interval - time_since_last
                logger.info(f"Rate limiting: waiting {delay:.2f} seconds")
                await asyncio.sleep(delay)
    
    async def parse_xml_response(self, xml_text: str) -> Dict[str, Any]:
        """Parse XML response"""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
            
            def xml_to_dict(element):
                result = {}
                
                if element.attrib:
                    result.update(element.attrib)
                
                if element.text and element.text.strip():
                    if len(element) == 0:
                        return element.text.strip()
                    else:
                        result['text'] = element.text.strip()
                
                for child in element:
                    child_data = xml_to_dict(child)
                    
                    if child.tag in result:
                        if not isinstance(result[child.tag], list):
                            result[child.tag] = [result[child.tag]]
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = child_data
                
                return result
            
            return xml_to_dict(root)
            
        except Exception as e:
            logger.error(f"XML parsing error: {e}")
            return {'raw_xml': xml_text}
    
    async def parse_csv_response(self, csv_text: str) -> List[Dict[str, Any]]:
        """Parse CSV response"""
        try:
            import csv
            import io
            
            reader = csv.DictReader(io.StringIO(csv_text))
            return list(reader)
            
        except Exception as e:
            logger.error(f"CSV parsing error: {e}")
            return [{'raw_csv': csv_text}]
    
    def safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return None