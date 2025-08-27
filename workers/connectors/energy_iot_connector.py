"""
Energy IoT Connector - For energy management systems and IoT sensors
Supports MQTT, InfluxDB, and REST APIs for energy data collection
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from .base_connector import DataConnector, HttpConnectorMixin

logger = logging.getLogger(__name__)


class Energy_IOTConnector(DataConnector, HttpConnectorMixin):
    """Energy IoT Connector for power, battery, and generator data"""
    
    def __init__(self, data_source, message_producer, quality_validator):
        super().__init__(data_source, message_producer, quality_validator)
        
        # Energy IoT specific configuration
        self.protocol = self.get_connection_config('protocol', 'http')  # http, mqtt, influxdb
        self.energy_types = self.get_connection_config('energy_types', ['GRID', 'BATTERY', 'GENERATOR', 'SOLAR'])
        self.metric_types = self.get_connection_config('metric_types', [
            'voltage', 'current', 'power', 'energy', 'frequency', 
            'temperature', 'fuel_level', 'battery_soc', 'efficiency'
        ])
        
        # Protocol-specific clients
        self.session = None
        self.mqtt_client = None
        self.influx_client = None
        
    async def validate_config(self):
        """Validate Energy IoT connector configuration"""
        if self.protocol == 'http':
            if not self.get_connection_config('base_url'):
                raise ValueError("HTTP protocol requires base_url")
        
        elif self.protocol == 'mqtt':
            if not self.get_connection_config('broker_url'):
                raise ValueError("MQTT protocol requires broker_url")
            if not self.get_connection_config('topics'):
                raise ValueError("MQTT protocol requires topics configuration")
        
        elif self.protocol == 'influxdb':
            required_fields = ['url', 'token', 'org', 'bucket']
            for field in required_fields:
                if not self.get_connection_config(field):
                    raise ValueError(f"InfluxDB protocol requires {field}")
        
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol}")
        
        logger.info(f"Energy IoT connector configuration validated for protocol: {self.protocol}")
    
    async def connect(self):
        """Establish connection based on protocol"""
        if self.protocol == 'http':
            await self.connect_http()
        elif self.protocol == 'mqtt':
            await self.connect_mqtt()
        elif self.protocol == 'influxdb':
            await self.connect_influxdb()
    
    async def connect_http(self):
        """Connect via HTTP/REST API"""
        self.session = await self.create_http_session()
        logger.info(f"Connected to Energy HTTP API: {self.get_connection_config('base_url')}")
    
    async def connect_mqtt(self):
        """Connect via MQTT broker"""
        import paho.mqtt.client as mqtt
        
        broker_config = self.get_connection_config('mqtt_config', {})
        broker_url = self.get_connection_config('broker_url')
        broker_port = self.get_connection_config('broker_port', 1883)
        
        self.mqtt_client = mqtt.Client()
        
        # Set credentials if provided
        username = broker_config.get('username')
        password = broker_config.get('password')
        if username and password:
            self.mqtt_client.username_pw_set(username, password)
        
        # Set callbacks
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
        
        # Connect to broker
        self.mqtt_client.connect(broker_url, broker_port, 60)
        self.mqtt_client.loop_start()
        
        logger.info(f"Connected to MQTT broker: {broker_url}:{broker_port}")
    
    async def connect_influxdb(self):
        """Connect to InfluxDB"""
        from influxdb_client import InfluxDBClient
        
        url = self.get_connection_config('url')
        token = self.get_connection_config('token')
        org = self.get_connection_config('org')
        
        self.influx_client = InfluxDBClient(url=url, token=token, org=org)
        
        # Test connection
        try:
            health = self.influx_client.health()
            if health.status == "pass":
                logger.info(f"Connected to InfluxDB: {url}")
            else:
                raise ConnectionError(f"InfluxDB health check failed: {health.status}")
        except Exception as e:
            await self.influx_client.close()
            raise ConnectionError(f"Failed to connect to InfluxDB: {e}")
    
    async def disconnect(self):
        """Close connections"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                self.mqtt_client = None
            
            if self.influx_client:
                await self.influx_client.close()
                self.influx_client = None
            
            logger.info("Disconnected from Energy IoT system")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Energy IoT system: {e}")
    
    async def test_connection(self):
        """Test connection based on protocol"""
        if self.protocol == 'http':
            return await self.test_http_connection()
        elif self.protocol == 'mqtt':
            return await self.test_mqtt_connection()
        elif self.protocol == 'influxdb':
            return await self.test_influxdb_connection()
    
    async def test_http_connection(self):
        """Test HTTP connection"""
        if not self.session:
            raise ConnectionError("No active HTTP session")
        
        base_url = self.get_connection_config('base_url')
        test_endpoint = self.get_connection_config('test_endpoint', '/status')
        
        async with self.session.get(f"{base_url}{test_endpoint}") as response:
            if response.status == 200:
                logger.info("Energy HTTP connection test successful")
                return True
            else:
                raise ConnectionError(f"HTTP test failed: {response.status}")
    
    async def test_mqtt_connection(self):
        """Test MQTT connection"""
        if not self.mqtt_client or not self.mqtt_client.is_connected():
            raise ConnectionError("MQTT client not connected")
        
        logger.info("Energy MQTT connection test successful")
        return True
    
    async def test_influxdb_connection(self):
        """Test InfluxDB connection"""
        if not self.influx_client:
            raise ConnectionError("No InfluxDB client")
        
        health = self.influx_client.health()
        if health.status == "pass":
            logger.info("Energy InfluxDB connection test successful")
            return True
        else:
            raise ConnectionError(f"InfluxDB health check failed: {health.status}")
    
    async def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch energy data based on protocol"""
        if self.protocol == 'http':
            return await self.fetch_http_data()
        elif self.protocol == 'mqtt':
            return await self.fetch_mqtt_data()
        elif self.protocol == 'influxdb':
            return await self.fetch_influxdb_data()
        else:
            return []
    
    async def fetch_http_data(self) -> List[Dict[str, Any]]:
        """Fetch data via HTTP API"""
        all_data = []
        base_url = self.get_connection_config('base_url')
        
        # Get endpoints for different energy types
        endpoints = self.get_connection_config('endpoints', {})
        
        for energy_type in self.energy_types:
            endpoint = endpoints.get(energy_type.lower(), f'/api/{energy_type.lower()}')
            
            try:
                params = await self.build_query_params()
                
                async with self.session.get(f"{base_url}{endpoint}", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Transform data
                        transformed_data = await self.transform_energy_data(data, energy_type, 'http')
                        all_data.extend(transformed_data)
                        
                    else:
                        logger.warning(f"Failed to fetch {energy_type} data: HTTP {response.status}")
                        
            except Exception as e:
                logger.error(f"Error fetching {energy_type} data: {e}")
        
        return all_data
    
    async def fetch_mqtt_data(self) -> List[Dict[str, Any]]:
        """Fetch data from MQTT messages (return cached messages)"""
        # MQTT data is received asynchronously via callbacks
        # Return any cached messages from recent subscriptions
        return getattr(self, '_mqtt_messages', [])
    
    async def fetch_influxdb_data(self) -> List[Dict[str, Any]]:
        """Fetch data from InfluxDB"""
        all_data = []
        
        try:
            query_api = self.influx_client.query_api()
            bucket = self.get_connection_config('bucket')
            
            # Build Flux query
            time_range = self.get_connection_config('time_range', '1h')
            
            for energy_type in self.energy_types:
                measurement = self.get_connection_config('measurements', {}).get(
                    energy_type.lower(), energy_type.lower()
                )
                
                query = f'''
                from(bucket: "{bucket}")
                |> range(start: -{time_range})
                |> filter(fn: (r) => r["_measurement"] == "{measurement}")
                |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
                |> yield(name: "mean")
                '''
                
                result = query_api.query(org=self.get_connection_config('org'), query=query)
                
                for table in result:
                    for record in table.records:
                        data_point = {
                            'timestamp': record.get_time().isoformat(),
                            'measurement': record.get_measurement(),
                            'field': record.get_field(),
                            'value': record.get_value(),
                            'tags': record.values
                        }
                        
                        # Transform to standard format
                        transformed_data = await self.transform_energy_data([data_point], energy_type, 'influxdb')
                        all_data.extend(transformed_data)
            
            return all_data
            
        except Exception as e:
            logger.error(f"Error fetching data from InfluxDB: {e}")
            return []
    
    async def build_query_params(self) -> Dict[str, Any]:
        """Build query parameters for HTTP requests"""
        params = {}
        
        # Time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=15)  # Last 15 minutes
        
        params['start_time'] = start_time.isoformat()
        params['end_time'] = end_time.isoformat()
        
        # Site filter
        site_filter = self.get_connection_config('site_filter')
        if site_filter:
            params['sites'] = site_filter
        
        # Metric filter
        if self.metric_types:
            params['metrics'] = ','.join(self.metric_types)
        
        return params
    
    async def transform_energy_data(self, raw_data: Any, energy_type: str, source_protocol: str) -> List[Dict[str, Any]]:
        """Transform raw energy data to standard format"""
        energy_metrics = []
        
        try:
            # Handle different data structures
            if isinstance(raw_data, dict):
                if 'data' in raw_data:
                    data_items = raw_data['data']
                elif 'measurements' in raw_data:
                    data_items = raw_data['measurements']
                elif 'results' in raw_data:
                    data_items = raw_data['results']
                else:
                    data_items = [raw_data]
            elif isinstance(raw_data, list):
                data_items = raw_data
            else:
                data_items = [{'raw_data': raw_data}]
            
            for item in data_items:
                # Extract common energy metric fields
                metric = {
                    'data_type': 'energy_metric',
                    'timestamp': item.get('timestamp', datetime.utcnow().isoformat()),
                    'energy_type': energy_type,
                    'metric_name': item.get('metric_name', item.get('field', 'unknown')),
                    'metric_value': self.safe_float(item.get('value', item.get('metric_value'))),
                    'unit': item.get('unit', self.infer_unit(item.get('metric_name', ''))),
                    'site_id': item.get('site_id', item.get('location_id')),
                    'efficiency_score': self.calculate_efficiency_score(item),
                    'metadata': {
                        'source_protocol': source_protocol,
                        'energy_type': energy_type,
                        'device_id': item.get('device_id'),
                        'sensor_type': item.get('sensor_type'),
                        'raw_timestamp': item.get('timestamp')
                    }
                }
                
                # Add energy-type specific fields
                if energy_type == 'BATTERY':
                    metric['metadata'].update({
                        'state_of_charge': item.get('soc'),
                        'charge_cycles': item.get('cycles'),
                        'health_percentage': item.get('health')
                    })
                
                elif energy_type == 'GENERATOR':
                    metric['metadata'].update({
                        'fuel_level': item.get('fuel_level'),
                        'runtime_hours': item.get('runtime'),
                        'maintenance_due': item.get('maintenance_due')
                    })
                
                elif energy_type == 'SOLAR':
                    metric['metadata'].update({
                        'irradiance': item.get('irradiance'),
                        'panel_temperature': item.get('panel_temp'),
                        'weather_conditions': item.get('weather')
                    })
                
                elif energy_type == 'GRID':
                    metric['metadata'].update({
                        'grid_frequency': item.get('frequency'),
                        'power_factor': item.get('power_factor'),
                        'thd': item.get('thd')  # Total Harmonic Distortion
                    })
                
                energy_metrics.append(metric)
            
            return energy_metrics
            
        except Exception as e:
            logger.error(f"Error transforming energy data: {e}")
            return []
    
    def safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def infer_unit(self, metric_name: str) -> str:
        """Infer unit from metric name"""
        metric_name = metric_name.lower()
        
        unit_mappings = {
            'voltage': 'V',
            'current': 'A', 
            'power': 'W',
            'energy': 'kWh',
            'frequency': 'Hz',
            'temperature': '°C',
            'fuel_level': '%',
            'battery_soc': '%',
            'efficiency': '%'
        }
        
        for key, unit in unit_mappings.items():
            if key in metric_name:
                return unit
        
        return ''
    
    def calculate_efficiency_score(self, item: Dict[str, Any]) -> float:
        """Calculate energy efficiency score"""
        score = 1.0
        
        # Reduce score based on various factors
        metric_name = item.get('metric_name', '').lower()
        value = self.safe_float(item.get('value'))
        
        if value is None:
            return 0.5
        
        # Efficiency scoring based on metric type
        if 'efficiency' in metric_name:
            # Direct efficiency measurement
            score = min(1.0, value / 100.0) if value <= 100 else 0.5
        
        elif 'temperature' in metric_name:
            # Temperature efficiency (optimal around 25°C for most equipment)
            optimal_temp = 25
            if value is not None:
                temp_deviation = abs(value - optimal_temp)
                if temp_deviation > 20:
                    score = 0.5
                elif temp_deviation > 10:
                    score = 0.7
                else:
                    score = 1.0
        
        elif 'fuel_level' in metric_name:
            # Fuel level efficiency
            if value is not None:
                if value < 10:  # Low fuel
                    score = 0.3
                elif value < 25:
                    score = 0.7
                else:
                    score = 1.0
        
        elif 'battery_soc' in metric_name or 'soc' in metric_name:
            # Battery state of charge efficiency
            if value is not None:
                if value < 20:  # Low battery
                    score = 0.3
                elif value < 50:
                    score = 0.7
                else:
                    score = 1.0
        
        return max(0.0, min(1.0, score))
    
    # MQTT callback methods
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connect callback"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            
            # Subscribe to configured topics
            topics = self.get_connection_config('topics', [])
            for topic in topics:
                client.subscribe(topic)
                logger.info(f"Subscribed to MQTT topic: {topic}")
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            # Decode message
            payload = json.loads(msg.payload.decode('utf-8'))
            
            # Add topic information
            payload['_mqtt_topic'] = msg.topic
            payload['_mqtt_timestamp'] = datetime.utcnow().isoformat()
            
            # Cache message for batch processing
            if not hasattr(self, '_mqtt_messages'):
                self._mqtt_messages = []
            
            self._mqtt_messages.append(payload)
            
            # Keep only recent messages (last 100)
            if len(self._mqtt_messages) > 100:
                self._mqtt_messages = self._mqtt_messages[-100:]
            
            logger.debug(f"Received MQTT message from {msg.topic}")
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback"""
        logger.warning(f"Disconnected from MQTT broker: {rc}")
        
        # Try to reconnect
        if rc != 0:
            logger.info("Attempting to reconnect to MQTT broker...")
            try:
                client.reconnect()
            except Exception as e:
                logger.error(f"Failed to reconnect to MQTT broker: {e}")