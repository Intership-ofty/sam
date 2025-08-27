"""
Client Notification System API Endpoints
Real-time notifications, alerts, and communication management
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncpg
import json
import asyncio
from enum import Enum
from pydantic import BaseModel, Field

from ..core.database import get_connection
from ..core.auth import get_current_user, require_permission
from ..core.models import APIResponse, User
from ..core.websocket import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/notifications",
    tags=["Client Notifications"],
    dependencies=[Depends(get_current_user)]
)

# WebSocket connection manager
manager = ConnectionManager()

class NotificationType(str, Enum):
    ALERT = "alert"
    SLA_BREACH = "sla_breach"
    MAINTENANCE = "maintenance"
    SYSTEM = "system"
    ANNOUNCEMENT = "announcement"
    INCIDENT = "incident"
    RECOVERY = "recovery"

class NotificationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationChannel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    SLACK = "slack"
    TEAMS = "teams"

class NotificationCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=2000)
    type: NotificationType
    priority: NotificationPriority = NotificationPriority.MEDIUM
    channels: List[NotificationChannel]
    target_users: Optional[List[str]] = None
    target_roles: Optional[List[str]] = None
    site_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    action_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NotificationPreferences(BaseModel):
    email_enabled: bool = True
    sms_enabled: bool = False
    webhook_enabled: bool = False
    in_app_enabled: bool = True
    notification_types: Dict[NotificationType, Dict[NotificationChannel, bool]] = Field(default_factory=dict)
    quiet_hours_start: Optional[str] = None  # HH:MM format
    quiet_hours_end: Optional[str] = None    # HH:MM format
    timezone: str = Field(default="UTC")

class NotificationRule(BaseModel):
    name: str
    description: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    enabled: bool = True
    priority_threshold: NotificationPriority = NotificationPriority.MEDIUM

@router.websocket("/ws/{tenant_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, tenant_id: str, user_id: str):
    """WebSocket endpoint for real-time notifications"""
    await manager.connect(websocket, f"{tenant_id}:{user_id}")
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Echo back for heartbeat
            await websocket.send_text(f"heartbeat:{datetime.utcnow().isoformat()}")
    except WebSocketDisconnect:
        manager.disconnect(f"{tenant_id}:{user_id}")
        logger.info(f"WebSocket disconnected for user {user_id} in tenant {tenant_id}")

@router.post("/send")
async def send_notification(
    notification: NotificationCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("notifications.send")),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Send a new notification"""
    try:
        # Create notification record
        notification_id = await conn.fetchval("""
            INSERT INTO notifications (
                tenant_id, title, message, type, priority, channels,
                created_by, site_id, expires_at, action_url, metadata, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
            RETURNING id
        """, 
        current_user.tenant_id, notification.title, notification.message,
        notification.type.value, notification.priority.value, 
        json.dumps([ch.value for ch in notification.channels]),
        current_user.id, notification.site_id, notification.expires_at,
        notification.action_url, json.dumps(notification.metadata))

        # Determine target recipients
        target_user_ids = []
        
        if notification.target_users:
            target_user_ids.extend(notification.target_users)
        
        if notification.target_roles:
            role_users = await conn.fetch("""
                SELECT id FROM users 
                WHERE tenant_id = $1 AND role = ANY($2) AND is_active = true
            """, current_user.tenant_id, notification.target_roles)
            target_user_ids.extend([row['id'] for row in role_users])
        
        if not target_user_ids:
            # Send to all active users in tenant if no specific targets
            all_users = await conn.fetch("""
                SELECT id FROM users 
                WHERE tenant_id = $1 AND is_active = true
            """, current_user.tenant_id)
            target_user_ids = [row['id'] for row in all_users]

        # Create notification recipients
        for user_id in set(target_user_ids):  # Remove duplicates
            await conn.execute("""
                INSERT INTO notification_recipients (
                    notification_id, user_id, status, created_at
                ) VALUES ($1, $2, 'pending', NOW())
            """, notification_id, user_id)

        # Send notifications via different channels
        background_tasks.add_task(
            process_notification,
            notification_id,
            notification,
            target_user_ids,
            current_user.tenant_id
        )

        logger.info(f"Created notification {notification_id} for {len(target_user_ids)} recipients")

        return {
            "notification_id": notification_id,
            "recipients_count": len(target_user_ids),
            "message": "Notification sent successfully"
        }

    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail="Failed to send notification")

@router.get("/")
async def get_notifications(
    limit: int = 50,
    offset: int = 0,
    type_filter: Optional[NotificationType] = None,
    priority_filter: Optional[NotificationPriority] = None,
    unread_only: bool = False,
    current_user: User = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get notifications for current user"""
    try:
        query = """
            SELECT n.id, n.title, n.message, n.type, n.priority, n.channels,
                   n.site_id, n.expires_at, n.action_url, n.metadata, n.created_at,
                   nr.status, nr.read_at, nr.delivered_at,
                   s.name as site_name,
                   u.full_name as created_by_name
            FROM notifications n
            JOIN notification_recipients nr ON n.id = nr.notification_id
            LEFT JOIN sites s ON n.site_id = s.site_id
            LEFT JOIN users u ON n.created_by = u.id
            WHERE nr.user_id = $1 AND n.tenant_id = $2
        """
        
        params = [current_user.id, current_user.tenant_id]
        param_count = 2
        
        if type_filter:
            param_count += 1
            query += f" AND n.type = ${param_count}"
            params.append(type_filter.value)
        
        if priority_filter:
            param_count += 1
            query += f" AND n.priority = ${param_count}"
            params.append(priority_filter.value)
        
        if unread_only:
            query += " AND nr.read_at IS NULL"
        
        query += " AND (n.expires_at IS NULL OR n.expires_at > NOW())"
        query += f" ORDER BY n.created_at DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
        params.extend([limit, offset])
        
        rows = await conn.fetch(query, *params)
        
        notifications = []
        for row in rows:
            notification_data = {
                "id": row['id'],
                "title": row['title'],
                "message": row['message'],
                "type": row['type'],
                "priority": row['priority'],
                "channels": json.loads(row['channels']) if row['channels'] else [],
                "site_id": row['site_id'],
                "site_name": row['site_name'],
                "expires_at": row['expires_at'].isoformat() if row['expires_at'] else None,
                "action_url": row['action_url'],
                "metadata": json.loads(row['metadata']) if row['metadata'] else {},
                "created_at": row['created_at'].isoformat(),
                "created_by": row['created_by_name'],
                "status": row['status'],
                "is_read": row['read_at'] is not None,
                "read_at": row['read_at'].isoformat() if row['read_at'] else None,
                "delivered_at": row['delivered_at'].isoformat() if row['delivered_at'] else None
            }
            notifications.append(notification_data)
        
        # Get total count for pagination
        count_query = """
            SELECT COUNT(*)
            FROM notifications n
            JOIN notification_recipients nr ON n.id = nr.notification_id
            WHERE nr.user_id = $1 AND n.tenant_id = $2
        """
        
        count_params = [current_user.id, current_user.tenant_id]
        if type_filter:
            count_query += " AND n.type = $3"
            count_params.append(type_filter.value)
        if unread_only:
            count_query += " AND nr.read_at IS NULL"
        
        total_count = await conn.fetchval(count_query, *count_params)
        
        return {
            "notifications": notifications,
            "total_count": total_count,
            "unread_count": await get_unread_count(current_user.id, current_user.tenant_id, conn),
            "page": offset // limit + 1,
            "page_size": limit,
            "has_more": (offset + limit) < total_count
        }

    except Exception as e:
        logger.error(f"Error retrieving notifications: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve notifications")

@router.put("/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Mark a notification as read"""
    try:
        result = await conn.execute("""
            UPDATE notification_recipients 
            SET status = 'read', read_at = NOW()
            WHERE notification_id = $1 AND user_id = $2 AND read_at IS NULL
        """, notification_id, current_user.id)
        
        if result == "UPDATE 0":
            raise HTTPException(status_code=404, detail="Notification not found or already read")
        
        return {"message": "Notification marked as read"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark notification as read")

@router.put("/read-all")
async def mark_all_notifications_read(
    current_user: User = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Mark all notifications as read for current user"""
    try:
        result = await conn.execute("""
            UPDATE notification_recipients nr
            SET status = 'read', read_at = NOW()
            FROM notifications n
            WHERE nr.notification_id = n.id 
            AND nr.user_id = $1 
            AND n.tenant_id = $2
            AND nr.read_at IS NULL
        """, current_user.id, current_user.tenant_id)
        
        count = int(result.split()[-1])  # Extract count from "UPDATE X"
        
        return {
            "message": f"Marked {count} notifications as read",
            "count": count
        }

    except Exception as e:
        logger.error(f"Error marking all notifications as read: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark notifications as read")

@router.get("/preferences")
async def get_notification_preferences(
    current_user: User = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get user notification preferences"""
    try:
        prefs = await conn.fetchrow("""
            SELECT preferences FROM user_notification_preferences
            WHERE user_id = $1 AND tenant_id = $2
        """, current_user.id, current_user.tenant_id)
        
        if prefs and prefs['preferences']:
            return json.loads(prefs['preferences'])
        else:
            # Return default preferences
            return {
                "email_enabled": True,
                "sms_enabled": False,
                "webhook_enabled": False,
                "in_app_enabled": True,
                "notification_types": {
                    "alert": {"email": True, "in_app": True},
                    "sla_breach": {"email": True, "in_app": True, "sms": False},
                    "maintenance": {"email": True, "in_app": True},
                    "system": {"email": False, "in_app": True},
                    "announcement": {"email": True, "in_app": True}
                },
                "quiet_hours_start": None,
                "quiet_hours_end": None,
                "timezone": "UTC"
            }

    except Exception as e:
        logger.error(f"Error retrieving notification preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve preferences")

@router.put("/preferences")
async def update_notification_preferences(
    preferences: NotificationPreferences,
    current_user: User = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Update user notification preferences"""
    try:
        await conn.execute("""
            INSERT INTO user_notification_preferences (
                user_id, tenant_id, preferences, updated_at
            ) VALUES ($1, $2, $3, NOW())
            ON CONFLICT (user_id, tenant_id) 
            DO UPDATE SET preferences = $3, updated_at = NOW()
        """, current_user.id, current_user.tenant_id, json.dumps(preferences.dict()))
        
        return {"message": "Notification preferences updated successfully"}

    except Exception as e:
        logger.error(f"Error updating notification preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")

@router.get("/unread-count")
async def get_unread_notifications_count(
    current_user: User = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get count of unread notifications"""
    try:
        count = await get_unread_count(current_user.id, current_user.tenant_id, conn)
        return {"unread_count": count}

    except Exception as e:
        logger.error(f"Error getting unread count: {e}")
        raise HTTPException(status_code=500, detail="Failed to get unread count")

@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: int,
    current_user: User = Depends(require_permission("notifications.manage")),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Delete a notification (admin only)"""
    try:
        # Verify notification belongs to user's tenant
        notification = await conn.fetchrow("""
            SELECT id FROM notifications 
            WHERE id = $1 AND tenant_id = $2
        """, notification_id, current_user.tenant_id)
        
        if not notification:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        # Delete notification and recipients
        await conn.execute("DELETE FROM notification_recipients WHERE notification_id = $1", notification_id)
        await conn.execute("DELETE FROM notifications WHERE id = $1", notification_id)
        
        return {"message": "Notification deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting notification: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete notification")

# Background task functions
async def process_notification(
    notification_id: int,
    notification: NotificationCreate,
    target_user_ids: List[str],
    tenant_id: str
):
    """Process notification delivery across different channels"""
    try:
        conn = await asyncpg.connect("postgresql://towerco:secure_password@localhost:5432/towerco_aiops")
        
        # Get user preferences for each recipient
        user_prefs = await conn.fetch("""
            SELECT user_id, preferences FROM user_notification_preferences
            WHERE user_id = ANY($1) AND tenant_id = $2
        """, target_user_ids, tenant_id)
        
        prefs_map = {row['user_id']: json.loads(row['preferences']) if row['preferences'] else {} for row in user_prefs}
        
        for user_id in target_user_ids:
            user_preferences = prefs_map.get(user_id, {})
            
            # Send real-time WebSocket notification
            if NotificationChannel.IN_APP in notification.channels:
                await send_websocket_notification(tenant_id, user_id, notification_id, notification)
            
            # Send email notification
            if NotificationChannel.EMAIL in notification.channels and user_preferences.get('email_enabled', True):
                await send_email_notification(user_id, notification)
            
            # Send SMS notification
            if NotificationChannel.SMS in notification.channels and user_preferences.get('sms_enabled', False):
                await send_sms_notification(user_id, notification)
            
            # Send webhook notification
            if NotificationChannel.WEBHOOK in notification.channels and user_preferences.get('webhook_enabled', False):
                await send_webhook_notification(user_id, notification)
        
        # Update notification status
        await conn.execute("""
            UPDATE notification_recipients 
            SET status = 'delivered', delivered_at = NOW()
            WHERE notification_id = $1
        """, notification_id)
        
        await conn.close()
        
    except Exception as e:
        logger.error(f"Error processing notification {notification_id}: {e}")

async def send_websocket_notification(tenant_id: str, user_id: str, notification_id: int, notification: NotificationCreate):
    """Send real-time WebSocket notification"""
    try:
        message = {
            "type": "notification",
            "id": notification_id,
            "title": notification.title,
            "message": notification.message,
            "priority": notification.priority.value,
            "notification_type": notification.type.value,
            "action_url": notification.action_url,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await manager.send_personal_message(json.dumps(message), f"{tenant_id}:{user_id}")
        
    except Exception as e:
        logger.error(f"Error sending WebSocket notification: {e}")

async def send_email_notification(user_id: str, notification: NotificationCreate):
    """Send email notification"""
    try:
        # This would integrate with email service (SendGrid, SES, etc.)
        logger.info(f"Sending email notification to user {user_id}: {notification.title}")
        await asyncio.sleep(1)  # Simulate email sending
        
    except Exception as e:
        logger.error(f"Error sending email notification: {e}")

async def send_sms_notification(user_id: str, notification: NotificationCreate):
    """Send SMS notification"""
    try:
        # This would integrate with SMS service (Twilio, SNS, etc.)
        logger.info(f"Sending SMS notification to user {user_id}: {notification.title}")
        await asyncio.sleep(1)  # Simulate SMS sending
        
    except Exception as e:
        logger.error(f"Error sending SMS notification: {e}")

async def send_webhook_notification(user_id: str, notification: NotificationCreate):
    """Send webhook notification"""
    try:
        # This would send HTTP POST to configured webhook URL
        logger.info(f"Sending webhook notification to user {user_id}: {notification.title}")
        await asyncio.sleep(1)  # Simulate webhook sending
        
    except Exception as e:
        logger.error(f"Error sending webhook notification: {e}")

async def get_unread_count(user_id: str, tenant_id: str, conn: asyncpg.Connection) -> int:
    """Get unread notification count for user"""
    return await conn.fetchval("""
        SELECT COUNT(*)
        FROM notifications n
        JOIN notification_recipients nr ON n.id = nr.notification_id
        WHERE nr.user_id = $1 AND n.tenant_id = $2 
        AND nr.read_at IS NULL
        AND (n.expires_at IS NULL OR n.expires_at > NOW())
    """, user_id, tenant_id)