# ADR-010: WebSocket Real-time Notifications

## Status
Accepted

## Context
The Towerco AIOps platform requires real-time communication capabilities for several critical use cases:

- **Incident Alerts**: Immediate notification of critical incidents requiring operator attention
- **KPI Updates**: Real-time dashboard updates with live metrics and performance data
- **System Status Changes**: Instant notification of site status changes, maintenance events
- **Collaborative Features**: Multi-user incident management with live updates
- **Alarm Escalation**: Time-sensitive escalation notifications with audio/visual alerts
- **Operational Updates**: Live NOC dashboard updates and team coordination

Traditional polling-based approaches would create excessive server load and introduce unacceptable latency for critical telecom operations where seconds matter.

## Decision
We will implement **WebSocket-based real-time notifications** using a hybrid approach that combines WebSocket connections for real-time updates with HTTP APIs for reliable delivery.

### Architecture Components:
1. **WebSocket Server** - Real-time bidirectional communication
2. **Connection Manager** - Handle connections, authentication, and tenant isolation
3. **Message Broker Integration** - Connect to event streams (Kafka/Redis)
4. **Fallback Mechanisms** - HTTP polling and push notifications for reliability
5. **Client-side WebSocket Management** - Auto-reconnection and state synchronization

## Alternatives Considered

### 1. HTTP Long Polling
- **Pros**: Simple implementation, works through firewalls, no special infrastructure
- **Cons**: Higher server resource usage, connection timeouts, not true bidirectional

### 2. Server-Sent Events (SSE)
- **Pros**: Simple, built-in browser support, automatic reconnection
- **Cons**: Unidirectional only, limited browser connection pool, no binary support

### 3. Push Notifications Only (FCM/APNS)
- **Pros**: Works when app is closed, OS-level integration, reliable delivery
- **Cons**: Requires mobile app, external service dependency, not suitable for real-time dashboards

### 4. gRPC Streaming
- **Pros**: Efficient binary protocol, built-in flow control, strong typing
- **Cons**: Complex browser support, requires HTTP/2, steeper learning curve

## Consequences

### Positive
- **True Real-time**: Sub-second latency for critical alerts
- **Bidirectional**: Both server-to-client and client-to-server communication
- **Efficient**: Lower overhead than polling for high-frequency updates
- **Scalable**: Can handle thousands of concurrent connections
- **Interactive**: Enable collaborative features and live updates
- **Battery Friendly**: More efficient than constant polling on mobile devices

### Negative
- **Connection Management**: Complex handling of connection drops and reconnections
- **Firewall Issues**: Some corporate firewalls may block WebSocket connections
- **State Synchronization**: Managing state consistency across connection drops
- **Resource Usage**: Persistent connections consume server resources
- **Debugging Complexity**: More difficult to debug than simple HTTP requests

## Implementation

### WebSocket Server Architecture

#### FastAPI WebSocket Integration
```python
# backend/routers/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Set
import json
import asyncio
import logging

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        # Store active connections by tenant:user
        self.active_connections: Dict[str, WebSocket] = {}
        # Store connections by tenant for broadcasting
        self.tenant_connections: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, connection_key: str, tenant_id: str):
        await websocket.accept()
        self.active_connections[connection_key] = websocket
        
        if tenant_id not in self.tenant_connections:
            self.tenant_connections[tenant_id] = set()
        self.tenant_connections[tenant_id].add(connection_key)
        
        logger.info(f"WebSocket connected: {connection_key}")
        
        # Send welcome message with connection info
        await self.send_personal_message(
            connection_key,
            {
                "type": "connection_established",
                "timestamp": datetime.utcnow().isoformat(),
                "connection_key": connection_key
            }
        )
    
    def disconnect(self, connection_key: str, tenant_id: str):
        if connection_key in self.active_connections:
            del self.active_connections[connection_key]
            
        if tenant_id in self.tenant_connections:
            self.tenant_connections[tenant_id].discard(connection_key)
            
        logger.info(f"WebSocket disconnected: {connection_key}")
    
    async def send_personal_message(self, connection_key: str, message: dict):
        if connection_key in self.active_connections:
            try:
                websocket = self.active_connections[connection_key]
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {connection_key}: {e}")
                # Remove broken connection
                tenant_id = connection_key.split(':')[0]
                self.disconnect(connection_key, tenant_id)
    
    async def broadcast_to_tenant(self, tenant_id: str, message: dict):
        if tenant_id in self.tenant_connections:
            connections = self.tenant_connections[tenant_id].copy()
            for connection_key in connections:
                await self.send_personal_message(connection_key, message)

manager = ConnectionManager()

@router.websocket("/ws/{tenant_id}/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    tenant_id: str, 
    user_id: str,
    token: str = Query(...)
):
    # Authenticate user
    try:
        user = await authenticate_websocket_user(token, tenant_id)
        if user.id != user_id:
            raise HTTPException(403, "Invalid user")
    except Exception as e:
        await websocket.close(code=1008, reason="Authentication failed")
        return
    
    connection_key = f"{tenant_id}:{user_id}"
    await manager.connect(websocket, connection_key, tenant_id)
    
    try:
        while True:
            # Listen for client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            await handle_client_message(connection_key, message, user)
            
    except WebSocketDisconnect:
        manager.disconnect(connection_key, tenant_id)
    except Exception as e:
        logger.error(f"WebSocket error for {connection_key}: {e}")
        manager.disconnect(connection_key, tenant_id)
```

#### Message Types and Routing
```python
# Message type definitions
class MessageType(str, Enum):
    INCIDENT_ALERT = "incident_alert"
    KPI_UPDATE = "kpi_update"
    SITE_STATUS = "site_status"
    SYSTEM_NOTIFICATION = "system_notification"
    USER_ACTION = "user_action"
    HEARTBEAT = "heartbeat"

async def handle_client_message(connection_key: str, message: dict, user: User):
    message_type = message.get("type")
    
    if message_type == MessageType.HEARTBEAT:
        # Respond to heartbeat
        await manager.send_personal_message(connection_key, {
            "type": "heartbeat_ack",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    elif message_type == "subscribe_kpis":
        # Subscribe to specific KPI updates
        site_ids = message.get("site_ids", [])
        await subscribe_to_kpi_updates(connection_key, site_ids)
    
    elif message_type == "incident_action":
        # Handle incident management actions
        await handle_incident_action(message, user)
    
    else:
        logger.warning(f"Unknown message type: {message_type}")

async def broadcast_incident_alert(tenant_id: str, incident_data: dict):
    """Broadcast critical incident to all connected users in tenant"""
    message = {
        "type": MessageType.INCIDENT_ALERT,
        "priority": "critical",
        "incident": incident_data,
        "timestamp": datetime.utcnow().isoformat(),
        "requires_acknowledgment": True
    }
    
    await manager.broadcast_to_tenant(tenant_id, message)

async def send_kpi_update(tenant_id: str, site_id: str, kpi_data: dict):
    """Send real-time KPI updates to subscribed users"""
    message = {
        "type": MessageType.KPI_UPDATE,
        "site_id": site_id,
        "kpi_data": kpi_data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Send to users subscribed to this site's KPIs
    connections = manager.tenant_connections.get(tenant_id, set())
    for connection_key in connections:
        if await is_subscribed_to_site(connection_key, site_id):
            await manager.send_personal_message(connection_key, message)
```

### Client-side WebSocket Management

#### React WebSocket Hook
```typescript
// hooks/useWebSocket.ts
import { useEffect, useRef, useState, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useTenant } from '../contexts/TenantContext';

interface WebSocketMessage {
  type: string;
  timestamp: string;
  [key: string]: any;
}

interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
}

export const useWebSocket = (options: UseWebSocketOptions = {}) => {
  const { user, isAuthenticated } = useAuth();
  const { tenant } = useTenant();
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttempts = useRef(0);
  
  const {
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    autoReconnect = true,
    maxReconnectAttempts = 5,
    reconnectInterval = 3000
  } = options;

  const connect = useCallback(() => {
    if (!user || !tenant || !isAuthenticated) {
      return;
    }

    const token = localStorage.getItem('accessToken');
    if (!token) {
      console.error('No access token available for WebSocket');
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/notifications/ws/${tenant.id}/${user.id}?token=${token}`;
    
    try {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        setIsConnected(true);
        reconnectAttempts.current = 0;
        console.log('WebSocket connected');
        onConnect?.();
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);
          onMessage?.(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      wsRef.current.onclose = (event) => {
        setIsConnected(false);
        console.log('WebSocket disconnected:', event.code, event.reason);
        onDisconnect?.();

        // Auto-reconnect if enabled and not manually closed
        if (autoReconnect && event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          console.log(`Attempting to reconnect (${reconnectAttempts.current}/${maxReconnectAttempts})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval * reconnectAttempts.current); // Exponential backoff
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.(error);
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
    }
  }, [user, tenant, isAuthenticated, onMessage, onConnect, onDisconnect, onError, autoReconnect, maxReconnectAttempts, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    } else {
      console.warn('WebSocket not connected, cannot send message');
      return false;
    }
  }, []);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Heartbeat to keep connection alive
  useEffect(() => {
    if (!isConnected) return;

    const heartbeatInterval = setInterval(() => {
      sendMessage({ type: 'heartbeat' });
    }, 30000); // Send heartbeat every 30 seconds

    return () => clearInterval(heartbeatInterval);
  }, [isConnected, sendMessage]);

  return {
    isConnected,
    lastMessage,
    sendMessage,
    connect,
    disconnect
  };
};
```

#### Real-time Notification Component
```typescript
// components/Notifications/RealTimeNotifications.tsx
import React, { useEffect, useState } from 'react';
import { toast } from 'react-hot-toast';
import { useWebSocket } from '../../hooks/useWebSocket';
import { BellIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';

interface RealTimeNotificationsProps {
  onIncidentAlert?: (incident: any) => void;
  onKPIUpdate?: (kpiData: any) => void;
}

export const RealTimeNotifications: React.FC<RealTimeNotificationsProps> = ({
  onIncidentAlert,
  onKPIUpdate
}) => {
  const [notifications, setNotifications] = useState<any[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);

  const { isConnected, sendMessage } = useWebSocket({
    onMessage: (message) => {
      switch (message.type) {
        case 'incident_alert':
          handleIncidentAlert(message);
          break;
        case 'kpi_update':
          handleKPIUpdate(message);
          break;
        case 'system_notification':
          handleSystemNotification(message);
          break;
        default:
          console.log('Received message:', message);
      }
    },
    
    onConnect: () => {
      console.log('Real-time notifications connected');
      toast.success('Connected to real-time updates');
      
      // Subscribe to relevant updates
      sendMessage({
        type: 'subscribe',
        topics: ['incidents', 'kpis', 'system_notifications']
      });
    },
    
    onDisconnect: () => {
      console.log('Real-time notifications disconnected');
      toast.error('Lost connection to real-time updates');
    }
  });

  const handleIncidentAlert = (message: any) => {
    const { incident, priority } = message;
    
    // Add to notifications list
    const notification = {
      id: Date.now(),
      type: 'incident',
      title: `Critical Incident: ${incident.title}`,
      message: incident.description,
      priority,
      timestamp: new Date().toISOString(),
      read: false
    };
    
    setNotifications(prev => [notification, ...prev]);
    setUnreadCount(prev => prev + 1);
    
    // Show toast notification
    toast.error(notification.title, {
      duration: 8000,
      icon: <ExclamationTriangleIcon className="w-6 h-6" />
    });
    
    // Play alert sound for critical incidents
    if (priority === 'critical') {
      playAlertSound();
    }
    
    // Call callback
    onIncidentAlert?.(incident);
  };

  const handleKPIUpdate = (message: any) => {
    const { site_id, kpi_data } = message;
    
    // Call callback for dashboard updates
    onKPIUpdate?.(kpi_data);
    
    // Show notification for threshold breaches
    if (kpi_data.threshold_breach) {
      toast.warning(`KPI threshold breach at site ${site_id}`, {
        duration: 5000
      });
    }
  };

  const handleSystemNotification = (message: any) => {
    const notification = {
      id: Date.now(),
      type: 'system',
      title: message.title,
      message: message.message,
      priority: message.priority || 'info',
      timestamp: new Date().toISOString(),
      read: false
    };
    
    setNotifications(prev => [notification, ...prev]);
    setUnreadCount(prev => prev + 1);
    
    // Show appropriate toast based on priority
    if (message.priority === 'error') {
      toast.error(message.title);
    } else if (message.priority === 'warning') {
      toast.warning(message.title);
    } else {
      toast(message.title, {
        icon: <BellIcon className="w-5 h-5" />
      });
    }
  };

  const playAlertSound = () => {
    // Create and play alert sound
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800; // High frequency for alert
    gainNode.gain.value = 0.3;
    
    oscillator.start();
    oscillator.stop(audioContext.currentTime + 0.3);
  };

  const markAllAsRead = () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
    setUnreadCount(0);
  };

  return (
    <>
      {/* Connection Status Indicator */}
      <div className={`fixed top-4 right-4 z-50 px-3 py-2 rounded-full text-sm font-medium ${
        isConnected 
          ? 'bg-green-100 text-green-800' 
          : 'bg-red-100 text-red-800'
      }`}>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            isConnected ? 'bg-green-500' : 'bg-red-500'
          } ${isConnected ? 'animate-pulse' : ''}`} />
          <span>
            {isConnected ? 'Live Updates' : 'Reconnecting...'}
          </span>
        </div>
      </div>

      {/* Notification Badge */}
      {unreadCount > 0 && (
        <div className="fixed top-16 right-4 z-50">
          <button
            onClick={markAllAsRead}
            className="bg-blue-600 text-white px-3 py-2 rounded-full shadow-lg hover:bg-blue-700 transition-colors"
          >
            <div className="flex items-center space-x-2">
              <BellIcon className="w-4 h-4" />
              <span className="text-sm font-medium">{unreadCount}</span>
            </div>
          </button>
        </div>
      )}
    </>
  );
};
```

### Message Broker Integration

#### Event Stream to WebSocket Bridge
```python
# workers/websocket_bridge.py
import asyncio
import json
import logging
from typing import Dict, Any
import redis.asyncio as redis
from kafka import KafkaConsumer
from .websocket_manager import manager

logger = logging.getLogger(__name__)

class WebSocketBridge:
    def __init__(self):
        self.redis_client = None
        self.kafka_consumer = None
        self.running = False
    
    async def initialize(self):
        # Initialize Redis connection
        self.redis_client = await redis.from_url("redis://localhost:6379")
        
        # Initialize Kafka consumer
        self.kafka_consumer = KafkaConsumer(
            'towerco.incidents.alerts',
            'towerco.kpis.updates', 
            'towerco.sites.status',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
    
    async def start_bridge(self):
        self.running = True
        
        # Start Kafka message processing
        kafka_task = asyncio.create_task(self.process_kafka_messages())
        
        # Start Redis pub/sub processing
        redis_task = asyncio.create_task(self.process_redis_messages())
        
        await asyncio.gather(kafka_task, redis_task)
    
    async def process_kafka_messages(self):
        while self.running:
            try:
                # Poll for messages
                message_pack = self.kafka_consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        await self.handle_kafka_message(
                            topic_partition.topic, 
                            message.value
                        )
                        
            except Exception as e:
                logger.error(f"Error processing Kafka messages: {e}")
                await asyncio.sleep(5)
    
    async def handle_kafka_message(self, topic: str, message: Dict[str, Any]):
        tenant_id = message.get('tenant_id')
        if not tenant_id:
            return
        
        if topic == 'towerco.incidents.alerts':
            await manager.broadcast_to_tenant(tenant_id, {
                'type': 'incident_alert',
                'incident': message['payload'],
                'priority': message['payload'].get('severity', 'medium'),
                'timestamp': message['timestamp']
            })
            
        elif topic == 'towerco.kpis.updates':
            await manager.broadcast_to_tenant(tenant_id, {
                'type': 'kpi_update',
                'site_id': message['site_id'],
                'kpi_data': message['payload'],
                'timestamp': message['timestamp']
            })
            
        elif topic == 'towerco.sites.status':
            await manager.broadcast_to_tenant(tenant_id, {
                'type': 'site_status',
                'site_id': message['site_id'],
                'status': message['payload']['status'],
                'timestamp': message['timestamp']
            })
    
    async def process_redis_messages(self):
        # Subscribe to Redis pub/sub for system notifications
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe('system_notifications')
        
        try:
            while self.running:
                message = await pubsub.get_message()
                if message and message['type'] == 'message':
                    data = json.loads(message['data'])
                    tenant_id = data.get('tenant_id')
                    
                    if tenant_id:
                        await manager.broadcast_to_tenant(tenant_id, {
                            'type': 'system_notification',
                            'title': data['title'],
                            'message': data['message'],
                            'priority': data.get('priority', 'info'),
                            'timestamp': data['timestamp']
                        })
                        
        except Exception as e:
            logger.error(f"Error processing Redis messages: {e}")
        finally:
            await pubsub.unsubscribe('system_notifications')

# Start the bridge
bridge = WebSocketBridge()

async def start_websocket_bridge():
    await bridge.initialize()
    await bridge.start_bridge()
```

### Reliability and Fallback

#### HTTP Polling Fallback
```typescript
// hooks/useNotificationFallback.ts
import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

export const useNotificationFallback = (isWebSocketConnected: boolean) => {
  const [shouldPoll, setShouldPoll] = useState(false);

  // Enable polling fallback when WebSocket is disconnected
  useEffect(() => {
    if (!isWebSocketConnected) {
      // Wait 10 seconds before falling back to polling
      const fallbackTimeout = setTimeout(() => {
        setShouldPoll(true);
      }, 10000);
      
      return () => clearTimeout(fallbackTimeout);
    } else {
      setShouldPoll(false);
    }
  }, [isWebSocketConnected]);

  // HTTP polling as fallback
  const { data: polledNotifications } = useQuery({
    queryKey: ['notifications', 'polling'],
    queryFn: () => fetch('/api/v1/notifications?limit=10').then(r => r.json()),
    enabled: shouldPoll,
    refetchInterval: 5000, // Poll every 5 seconds
  });

  return {
    shouldPoll,
    polledNotifications: shouldPoll ? polledNotifications : null
  };
};
```

## Security Considerations

### Authentication and Authorization
- WebSocket connections must include valid JWT tokens
- Tenant isolation enforced at connection level  
- Message filtering based on user permissions
- Rate limiting to prevent abuse

### Data Validation
- All incoming messages validated against schemas
- Sanitize data before broadcasting
- Prevent cross-tenant message leakage
- Audit logging for sensitive operations

## Monitoring and Operations

### Connection Monitoring
- Track active connection counts per tenant
- Monitor connection establishment/drop rates
- Alert on unusual connection patterns
- Measure message latency and throughput

### Performance Metrics
- WebSocket connection success rates
- Message delivery confirmation rates
- Fallback polling activation frequency
- Resource usage per connection

## Review Date
January 2025

## References
- [WebSocket API Specification](https://websockets.spec.whatwg.org/)
- [FastAPI WebSocket Documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [React WebSocket Best Practices](https://blog.logrocket.com/websocket-tutorial-real-time-node-react/)