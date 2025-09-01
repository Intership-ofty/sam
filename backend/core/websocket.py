"""
WebSocket Connection Manager
Handles real-time connections for notifications and live updates
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Optional
import logging
import json
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time communication"""
    
    def __init__(self):
        # Store active connections by user/tenant key
        self.active_connections: Dict[str, WebSocket] = {}
        # Store connections by tenant for broadcasting
        self.tenant_connections: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, connection_key: str):
        """Accept a WebSocket connection and store it"""
        await websocket.accept()
        self.active_connections[connection_key] = websocket
        
        # Extract tenant_id from connection_key (format: "tenant_id:user_id")
        tenant_id = connection_key.split(':')[0]
        if tenant_id not in self.tenant_connections:
            self.tenant_connections[tenant_id] = []
        self.tenant_connections[tenant_id].append(connection_key)
        
        logger.info(f"WebSocket connected: {connection_key}")
        
        # Send welcome message
        await self.send_personal_message(
            json.dumps({
                "type": "connection",
                "status": "connected",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Real-time notifications enabled"
            }),
            connection_key
        )
    
    def disconnect(self, connection_key: str):
        """Remove a WebSocket connection"""
        if connection_key in self.active_connections:
            del self.active_connections[connection_key]
            
            # Remove from tenant connections
            tenant_id = connection_key.split(':')[0]
            if tenant_id in self.tenant_connections:
                if connection_key in self.tenant_connections[tenant_id]:
                    self.tenant_connections[tenant_id].remove(connection_key)
                    
                # Clean up empty tenant list
                if not self.tenant_connections[tenant_id]:
                    del self.tenant_connections[tenant_id]
            
            logger.info(f"WebSocket disconnected: {connection_key}")
    
    async def send_personal_message(self, message: str, connection_key: str):
        """Send a message to a specific connection"""
        if connection_key in self.active_connections:
            try:
                websocket = self.active_connections[connection_key]
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {connection_key}: {e}")
                # Remove broken connection
                self.disconnect(connection_key)
    
    async def broadcast_to_tenant(self, message: str, tenant_id: str):
        """Broadcast a message to all connections in a tenant"""
        if tenant_id in self.tenant_connections:
            # Create a copy of the list to avoid modification during iteration
            connection_keys = self.tenant_connections[tenant_id].copy()
            
            for connection_key in connection_keys:
                await self.send_personal_message(message, connection_key)
    
    async def broadcast_to_all(self, message: str):
        """Broadcast a message to all active connections"""
        # Create a copy to avoid modification during iteration
        connection_keys = list(self.active_connections.keys())
        
        for connection_key in connection_keys:
            await self.send_personal_message(message, connection_key)
    
    async def send_notification(self, notification_data: dict, target_users: List[str], tenant_id: str):
        """Send a notification to specific users in a tenant"""
        message = json.dumps({
            "type": "notification",
            **notification_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        for user_id in target_users:
            connection_key = f"{tenant_id}:{user_id}"
            await self.send_personal_message(message, connection_key)
    
    async def send_alert(self, alert_data: dict, tenant_id: str):
        """Send an alert to all users in a tenant"""
        message = json.dumps({
            "type": "alert",
            **alert_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        await self.broadcast_to_tenant(message, tenant_id)
    
    async def send_system_update(self, update_data: dict, tenant_id: Optional[str] = None):
        """Send a system update (maintenance, outage, etc.)"""
        message = json.dumps({
            "type": "system_update",
            **update_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        if tenant_id:
            await self.broadcast_to_tenant(message, tenant_id)
        else:
            await self.broadcast_to_all(message)
    
    async def send_kpi_update(self, kpi_data: dict, tenant_id: str):
        """Send real-time KPI updates"""
        message = json.dumps({
            "type": "kpi_update",
            **kpi_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        await self.broadcast_to_tenant(message, tenant_id)
    
    def get_connection_stats(self) -> dict:
        """Get statistics about active connections"""
        tenant_stats = {}
        for tenant_id, connections in self.tenant_connections.items():
            tenant_stats[tenant_id] = len(connections)
        
        return {
            "total_connections": len(self.active_connections),
            "tenant_connections": tenant_stats,
            "active_tenants": len(self.tenant_connections)
        }
    
    async def ping_all_connections(self):
        """Send ping to all connections to keep them alive"""
        ping_message = json.dumps({
            "type": "ping",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        connection_keys = list(self.active_connections.keys())
        for connection_key in connection_keys:
            await self.send_personal_message(ping_message, connection_key)

# Global connection manager instance
connection_manager = ConnectionManager()

# Background task to ping connections periodically
async def websocket_keepalive():
    """Background task to keep WebSocket connections alive"""
    while True:
        try:
            await connection_manager.ping_all_connections()
            await asyncio.sleep(30)  # Ping every 30 seconds
        except Exception as e:
            logger.error(f"Error in WebSocket keepalive: {e}")
            await asyncio.sleep(60)  # Wait longer on error