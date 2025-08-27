"""
SNMP Connector - For SNMP-based network monitoring
Supports SNMPv1, v2c, and v3 protocols
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import socket

from .base_connector import DataConnector

logger = logging.getLogger(__name__)


class SNMPConnector(DataConnector):
    """SNMP Connector for network device monitoring"""
    
    def __init__(self, data_source, message_producer, quality_validator):
        super().__init__(data_source, message_producer, quality_validator)
        
        # SNMP configuration
        self.version = self.get_connection_config('version', 'v2c')  # v1, v2c, v3
        self.targets = self.get_connection_config('targets', [])  # List of devices to monitor
        self.community = self.get_connection_config('community', 'public')
        self.port = self.get_connection_config('port', 161)
        self.timeout = self.get_connection_config('timeout', 5)
        self.retries = self.get_connection_config('retries', 3)
        
        # SNMPv3 specific
        self.v3_user = self.get_connection_config('v3_user')
        self.v3_auth_key = self.get_connection_config('v3_auth_key')
        self.v3_priv_key = self.get_connection_config('v3_priv_key')
        self.v3_auth_protocol = self.get_connection_config('v3_auth_protocol', 'SHA')
        self.v3_priv_protocol = self.get_connection_config('v3_priv_protocol', 'AES')
        
        # OID configurations for different metrics
        self.oids = self.get_connection_config('oids', {
            'system': {
                'sysDescr': '1.3.6.1.2.1.1.1.0',
                'sysUpTime': '1.3.6.1.2.1.1.3.0',
                'sysName': '1.3.6.1.2.1.1.5.0'
            },
            'interfaces': {
                'ifNumber': '1.3.6.1.2.1.2.1.0',
                'ifDescr': '1.3.6.1.2.1.2.2.1.2',
                'ifSpeed': '1.3.6.1.2.1.2.2.1.5',
                'ifOperStatus': '1.3.6.1.2.1.2.2.1.8',
                'ifInOctets': '1.3.6.1.2.1.2.2.1.10',
                'ifOutOctets': '1.3.6.1.2.1.2.2.1.16',
                'ifInErrors': '1.3.6.1.2.1.2.2.1.14',
                'ifOutErrors': '1.3.6.1.2.1.2.2.1.20'
            },
            'cpu': {
                'cpuUsage': '1.3.6.1.4.1.9.2.1.58.0'  # Cisco specific
            },
            'memory': {
                'memoryUsed': '1.3.6.1.4.1.9.2.1.8.0',  # Cisco specific
                'memoryFree': '1.3.6.1.4.1.9.2.1.9.0'
            }
        })
        
        self.snmp_engines = {}  # SNMP engines per target
    
    async def validate_config(self):
        """Validate SNMP connector configuration"""
        if not self.targets:
            raise ValueError("SNMP connector requires at least one target")
        
        if self.version not in ['v1', 'v2c', 'v3']:
            raise ValueError(f"Unsupported SNMP version: {self.version}")
        
        if self.version == 'v3':
            if not self.v3_user:
                raise ValueError("SNMPv3 requires username")
        
        # Validate target format
        for target in self.targets:
            if isinstance(target, str):
                # Simple IP address
                try:
                    socket.inet_aton(target)
                except socket.error:
                    raise ValueError(f"Invalid IP address: {target}")
            elif isinstance(target, dict):
                # Target with additional config
                if 'host' not in target:
                    raise ValueError("Target dict must contain 'host' field")
            else:
                raise ValueError("Target must be IP string or dict with host")
        
        logger.info(f"SNMP connector configuration validated for {len(self.targets)} targets")
    
    async def connect(self):
        """Initialize SNMP engines for each target"""
        try:
            from pysnmp.hlapi.asyncio import *
            
            for target in self.targets:
                target_info = self.parse_target(target)
                host = target_info['host']
                port = target_info.get('port', self.port)
                
                # Create SNMP engine
                if self.version in ['v1', 'v2c']:
                    auth_data = CommunityData(target_info.get('community', self.community))
                else:  # v3
                    auth_data = UsmUserData(
                        self.v3_user,
                        authKey=self.v3_auth_key,
                        privKey=self.v3_priv_key,
                        authProtocol=getattr(usmHMACSHAAuthProtocol if self.v3_auth_protocol == 'SHA' else usmHMACMD5AuthProtocol, 'serviceID'),
                        privProtocol=getattr(usmAesCfb128Protocol if self.v3_priv_protocol == 'AES' else usmDESPrivProtocol, 'serviceID')
                    )
                
                transport = UdpTransportTarget((host, port), timeout=self.timeout, retries=self.retries)
                
                self.snmp_engines[host] = {
                    'auth': auth_data,
                    'transport': transport,
                    'target_info': target_info
                }
            
            logger.info(f"SNMP engines initialized for {len(self.snmp_engines)} targets")
            
        except ImportError:
            raise ImportError("pysnmp library is required for SNMP connector")
        except Exception as e:
            logger.error(f"Failed to initialize SNMP engines: {e}")
            raise
    
    async def disconnect(self):
        """Clean up SNMP engines"""
        self.snmp_engines.clear()
        logger.info("SNMP engines disconnected")
    
    async def test_connection(self):
        """Test SNMP connectivity to all targets"""
        from pysnmp.hlapi.asyncio import *
        
        if not self.snmp_engines:
            raise ConnectionError("No SNMP engines initialized")
        
        failed_targets = []
        
        for host, engine_info in self.snmp_engines.items():
            try:
                # Try to get system description
                oid = self.oids['system']['sysDescr']
                
                iterator = getCmd(
                    SnmpEngine(),
                    engine_info['auth'],
                    engine_info['transport'],
                    ContextData(),
                    ObjectType(ObjectIdentity(oid))
                )
                
                errorIndication, errorStatus, errorIndex, varBinds = await iterator
                
                if errorIndication:
                    failed_targets.append(f"{host}: {errorIndication}")
                elif errorStatus:
                    failed_targets.append(f"{host}: {errorStatus}")
                else:
                    logger.info(f"SNMP connection test successful for {host}")
                    
            except Exception as e:
                failed_targets.append(f"{host}: {str(e)}")
        
        if failed_targets:
            raise ConnectionError(f"SNMP test failed for targets: {', '.join(failed_targets)}")
        
        return True
    
    async def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch SNMP data from all targets"""
        try:
            all_data = []
            
            # Fetch data from each target concurrently
            tasks = []
            for host in self.snmp_engines.keys():
                task = asyncio.create_task(self.fetch_target_data(host))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"SNMP fetch error: {result}")
                else:
                    all_data.extend(result)
            
            # Transform and enrich data
            transformed_data = []
            for item in all_data:
                enriched_item = await self.enrich_data(item)
                transformed_data.append(enriched_item)
            
            logger.info(f"Fetched {len(transformed_data)} SNMP metrics")
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error fetching SNMP data: {e}")
            await self.handle_connection_error(e)
            raise
    
    async def fetch_target_data(self, host: str) -> List[Dict[str, Any]]:
        """Fetch data from a single SNMP target"""
        from pysnmp.hlapi.asyncio import *
        
        engine_info = self.snmp_engines[host]
        target_data = []
        
        try:
            # Fetch system information
            system_data = await self.fetch_system_info(host, engine_info)
            target_data.extend(system_data)
            
            # Fetch interface statistics
            interface_data = await self.fetch_interface_stats(host, engine_info)
            target_data.extend(interface_data)
            
            # Fetch performance metrics (CPU, Memory) if supported
            perf_data = await self.fetch_performance_metrics(host, engine_info)
            target_data.extend(perf_data)
            
        except Exception as e:
            logger.error(f"Error fetching data from SNMP target {host}: {e}")
        
        return target_data
    
    async def fetch_system_info(self, host: str, engine_info: Dict) -> List[Dict[str, Any]]:
        """Fetch system information via SNMP"""
        from pysnmp.hlapi.asyncio import *
        
        data = []
        system_oids = self.oids['system']
        
        try:
            # Get system information
            oids_to_fetch = [
                ObjectType(ObjectIdentity(system_oids['sysDescr'])),
                ObjectType(ObjectIdentity(system_oids['sysUpTime'])),
                ObjectType(ObjectIdentity(system_oids['sysName']))
            ]
            
            iterator = getCmd(
                SnmpEngine(),
                engine_info['auth'],
                engine_info['transport'],
                ContextData(),
                *oids_to_fetch
            )
            
            errorIndication, errorStatus, errorIndex, varBinds = await iterator
            
            if not errorIndication and not errorStatus:
                system_info = {
                    'data_type': 'configuration',
                    'timestamp': datetime.utcnow().isoformat(),
                    'host': host,
                    'device_type': 'network_device',
                    'protocol': 'snmp',
                    'metadata': {
                        'snmp_version': self.version,
                        'system_description': str(varBinds[0][1]),
                        'system_uptime': int(varBinds[1][1]),
                        'system_name': str(varBinds[2][1])
                    }
                }
                
                data.append(system_info)
            
        except Exception as e:
            logger.error(f"Error fetching system info from {host}: {e}")
        
        return data
    
    async def fetch_interface_stats(self, host: str, engine_info: Dict) -> List[Dict[str, Any]]:
        """Fetch interface statistics via SNMP"""
        from pysnmp.hlapi.asyncio import *
        
        data = []
        interface_oids = self.oids['interfaces']
        
        try:
            # First, get number of interfaces
            iterator = getCmd(
                SnmpEngine(),
                engine_info['auth'],
                engine_info['transport'],
                ContextData(),
                ObjectType(ObjectIdentity(interface_oids['ifNumber']))
            )
            
            errorIndication, errorStatus, errorIndex, varBinds = await iterator
            
            if errorIndication or errorStatus:
                logger.warning(f"Could not get interface count for {host}")
                return data
            
            num_interfaces = int(varBinds[0][1])
            
            # Fetch data for each interface
            for if_index in range(1, min(num_interfaces + 1, 50)):  # Limit to 50 interfaces
                try:
                    interface_data = await self.fetch_single_interface(host, engine_info, if_index)
                    if interface_data:
                        data.append(interface_data)
                except Exception as e:
                    logger.debug(f"Error fetching interface {if_index} from {host}: {e}")
            
        except Exception as e:
            logger.error(f"Error fetching interface stats from {host}: {e}")
        
        return data
    
    async def fetch_single_interface(self, host: str, engine_info: Dict, if_index: int) -> Optional[Dict[str, Any]]:
        """Fetch statistics for a single interface"""
        from pysnmp.hlapi.asyncio import *
        
        interface_oids = self.oids['interfaces']
        
        try:
            # Build OIDs for this interface
            oids_to_fetch = [
                ObjectType(ObjectIdentity(f"{interface_oids['ifDescr']}.{if_index}")),
                ObjectType(ObjectIdentity(f"{interface_oids['ifSpeed']}.{if_index}")),
                ObjectType(ObjectIdentity(f"{interface_oids['ifOperStatus']}.{if_index}")),
                ObjectType(ObjectIdentity(f"{interface_oids['ifInOctets']}.{if_index}")),
                ObjectType(ObjectIdentity(f"{interface_oids['ifOutOctets']}.{if_index}")),
                ObjectType(ObjectIdentity(f"{interface_oids['ifInErrors']}.{if_index}")),
                ObjectType(ObjectIdentity(f"{interface_oids['ifOutErrors']}.{if_index}"))
            ]
            
            iterator = getCmd(
                SnmpEngine(),
                engine_info['auth'],
                engine_info['transport'],
                ContextData(),
                *oids_to_fetch
            )
            
            errorIndication, errorStatus, errorIndex, varBinds = await iterator
            
            if errorIndication or errorStatus:
                return None
            
            # Parse interface data
            interface_desc = str(varBinds[0][1])
            interface_speed = int(varBinds[1][1]) if varBinds[1][1] != 0 else None
            oper_status = int(varBinds[2][1])
            in_octets = int(varBinds[3][1])
            out_octets = int(varBinds[4][1])
            in_errors = int(varBinds[5][1])
            out_errors = int(varBinds[6][1])
            
            # Create multiple metrics for this interface
            metrics = []
            
            # Interface utilization (if speed is known)
            if interface_speed and interface_speed > 0:
                # Note: This is instantaneous, would need delta calculation for real utilization
                in_utilization = (in_octets * 8 / interface_speed) * 100
                out_utilization = (out_octets * 8 / interface_speed) * 100
                
                metrics.extend([
                    self.create_interface_metric(host, if_index, interface_desc, 'in_utilization', in_utilization, '%'),
                    self.create_interface_metric(host, if_index, interface_desc, 'out_utilization', out_utilization, '%')
                ])
            
            # Traffic metrics
            metrics.extend([
                self.create_interface_metric(host, if_index, interface_desc, 'in_octets', in_octets, 'bytes'),
                self.create_interface_metric(host, if_index, interface_desc, 'out_octets', out_octets, 'bytes'),
                self.create_interface_metric(host, if_index, interface_desc, 'in_errors', in_errors, 'count'),
                self.create_interface_metric(host, if_index, interface_desc, 'out_errors', out_errors, 'count'),
                self.create_interface_metric(host, if_index, interface_desc, 'operational_status', oper_status, 'status')
            ])
            
            return {
                'data_type': 'interface_metrics',
                'metrics': metrics,
                'interface': {
                    'index': if_index,
                    'description': interface_desc,
                    'speed': interface_speed,
                    'status': 'up' if oper_status == 1 else 'down'
                }
            }
            
        except Exception as e:
            logger.debug(f"Error fetching interface {if_index} from {host}: {e}")
            return None
    
    def create_interface_metric(self, host: str, if_index: int, if_desc: str, metric_name: str, value: float, unit: str) -> Dict[str, Any]:
        """Create a standardized interface metric"""
        return {
            'data_type': 'network_metric',
            'timestamp': datetime.utcnow().isoformat(),
            'host': host,
            'technology': 'SNMP',
            'metric_name': f"interface_{metric_name}",
            'metric_value': value,
            'unit': unit,
            'quality_score': 1.0,
            'metadata': {
                'interface_index': if_index,
                'interface_description': if_desc,
                'snmp_version': self.version,
                'protocol': 'snmp'
            }
        }
    
    async def fetch_performance_metrics(self, host: str, engine_info: Dict) -> List[Dict[str, Any]]:
        """Fetch performance metrics (CPU, Memory) if supported"""
        from pysnmp.hlapi.asyncio import *
        
        data = []
        
        # Try to fetch CPU usage (vendor-specific OIDs)
        cpu_oids = self.oids.get('cpu', {})
        memory_oids = self.oids.get('memory', {})
        
        try:
            # Attempt to get CPU usage
            if 'cpuUsage' in cpu_oids:
                iterator = getCmd(
                    SnmpEngine(),
                    engine_info['auth'],
                    engine_info['transport'],
                    ContextData(),
                    ObjectType(ObjectIdentity(cpu_oids['cpuUsage']))
                )
                
                errorIndication, errorStatus, errorIndex, varBinds = await iterator
                
                if not errorIndication and not errorStatus:
                    cpu_usage = float(varBinds[0][1])
                    
                    cpu_metric = {
                        'data_type': 'network_metric',
                        'timestamp': datetime.utcnow().isoformat(),
                        'host': host,
                        'technology': 'SNMP',
                        'metric_name': 'cpu_usage',
                        'metric_value': cpu_usage,
                        'unit': '%',
                        'quality_score': 1.0,
                        'metadata': {
                            'metric_type': 'performance',
                            'snmp_version': self.version
                        }
                    }
                    
                    data.append(cpu_metric)
            
            # Attempt to get memory usage
            if 'memoryUsed' in memory_oids and 'memoryFree' in memory_oids:
                iterator = getCmd(
                    SnmpEngine(),
                    engine_info['auth'],
                    engine_info['transport'],
                    ContextData(),
                    ObjectType(ObjectIdentity(memory_oids['memoryUsed'])),
                    ObjectType(ObjectIdentity(memory_oids['memoryFree']))
                )
                
                errorIndication, errorStatus, errorIndex, varBinds = await iterator
                
                if not errorIndication and not errorStatus:
                    memory_used = float(varBinds[0][1])
                    memory_free = float(varBinds[1][1])
                    total_memory = memory_used + memory_free
                    
                    if total_memory > 0:
                        memory_usage_pct = (memory_used / total_memory) * 100
                        
                        memory_metric = {
                            'data_type': 'network_metric',
                            'timestamp': datetime.utcnow().isoformat(),
                            'host': host,
                            'technology': 'SNMP',
                            'metric_name': 'memory_usage',
                            'metric_value': memory_usage_pct,
                            'unit': '%',
                            'quality_score': 1.0,
                            'metadata': {
                                'metric_type': 'performance',
                                'memory_used_bytes': memory_used,
                                'memory_free_bytes': memory_free,
                                'snmp_version': self.version
                            }
                        }
                        
                        data.append(memory_metric)
            
        except Exception as e:
            logger.debug(f"Could not fetch performance metrics from {host}: {e}")
        
        return data
    
    def parse_target(self, target: Any) -> Dict[str, Any]:
        """Parse target configuration"""
        if isinstance(target, str):
            return {'host': target}
        elif isinstance(target, dict):
            return target
        else:
            raise ValueError(f"Invalid target format: {target}")
    
    async def walk_oid(self, host: str, base_oid: str) -> List[Tuple[str, Any]]:
        """Perform SNMP walk on an OID"""
        from pysnmp.hlapi.asyncio import *
        
        engine_info = self.snmp_engines.get(host)
        if not engine_info:
            raise ValueError(f"No SNMP engine for host {host}")
        
        results = []
        
        try:
            for (errorIndication, errorStatus, errorIndex, varBinds) in nextCmd(
                SnmpEngine(),
                engine_info['auth'],
                engine_info['transport'],
                ContextData(),
                ObjectType(ObjectIdentity(base_oid)),
                lexicographicMode=False
            ):
                if errorIndication:
                    logger.error(f"SNMP walk error: {errorIndication}")
                    break
                elif errorStatus:
                    logger.error(f"SNMP walk error: {errorStatus.prettyPrint()} at {errorIndex}")
                    break
                else:
                    for varBind in varBinds:
                        oid, value = varBind
                        results.append((str(oid), value))
            
        except Exception as e:
            logger.error(f"Error in SNMP walk for {host}, OID {base_oid}: {e}")
        
        return results