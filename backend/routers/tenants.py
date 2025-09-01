"""
Multi-tenant Management API Endpoints
Tenant configuration, user management, and subscription handling
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import asyncpg
import json
from pydantic import BaseModel, Field

from ..core.database import get_connection
from ..core.auth import get_current_user, require_permission
from ..core.models import APIResponse, User

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/tenants",
    tags=["Multi-tenant Management"],
    dependencies=[Depends(get_current_user)]
)

class TenantCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    display_name: str = Field(..., min_length=2, max_length=150)
    domain: str = Field(..., min_length=3, max_length=100)
    contact_email: str
    subscription_plan: str = Field(default="basic")
    settings: Dict[str, Any] = Field(default_factory=dict)
    branding: Dict[str, Any] = Field(default_factory=dict)

class TenantUpdate(BaseModel):
    display_name: Optional[str] = None
    contact_email: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    branding: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class TenantUserInvite(BaseModel):
    email: str
    role: str = Field(..., pattern="^(admin|manager|operator|viewer)$")
    permissions: List[str] = Field(default_factory=list)
    expires_in_days: int = Field(default=7, ge=1, le=30)

class SubscriptionUpdate(BaseModel):
    plan: str = Field(..., pattern="^(basic|professional|enterprise|custom)$")
    features: List[str] = Field(default_factory=list)
    limits: Dict[str, int] = Field(default_factory=dict)

@router.get("/{tenant_id}")
async def get_tenant(
    tenant_id: str,
    current_user: User = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get tenant information"""
    try:
        # Check if user belongs to this tenant or is system admin
        if current_user.tenant_id != tenant_id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Access denied to this tenant")
        
        query = """
        SELECT t.id, t.name, t.display_name, t.domain, t.contact_email,
               t.subscription_plan, t.subscription_features, t.subscription_limits,
               t.settings, t.branding, t.is_active, t.created_at, t.updated_at,
               COUNT(u.id) as user_count,
               COUNT(s.site_id) as site_count
        FROM tenants t
        LEFT JOIN users u ON t.id = u.tenant_id AND u.is_active = true
        LEFT JOIN sites s ON t.id = s.tenant_id
        WHERE t.id = $1
        GROUP BY t.id, t.name, t.display_name, t.domain, t.contact_email,
                 t.subscription_plan, t.subscription_features, t.subscription_limits,
                 t.settings, t.branding, t.is_active, t.created_at, t.updated_at
        """
        
        row = await conn.fetchrow(query, tenant_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Tenant not found")
        
        tenant_data = {
            "id": row['id'],
            "name": row['name'],
            "display_name": row['display_name'],
            "domain": row['domain'],
            "contact_email": row['contact_email'],
            "is_active": row['is_active'],
            "created_at": row['created_at'].isoformat(),
            "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None,
            "subscription": {
                "plan": row['subscription_plan'],
                "features": row['subscription_features'] or [],
                "limits": row['subscription_limits'] or {}
            },
            "settings": row['settings'] or {
                "timezone": "UTC",
                "date_format": "YYYY-MM-DD",
                "currency": "USD",
                "language": "en"
            },
            "branding": row['branding'] or {
                "company_name": row['display_name'],
                "primary_color": "#3B82F6",
                "secondary_color": "#6B7280"
            },
            "statistics": {
                "user_count": row['user_count'],
                "site_count": row['site_count']
            }
        }
        
        logger.info(f"Retrieved tenant data for {tenant_id}")
        return tenant_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tenant")

@router.put("/{tenant_id}")
async def update_tenant(
    tenant_id: str,
    tenant_update: TenantUpdate,
    current_user: User = Depends(require_permission("tenant.manage")),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Update tenant information"""
    try:
        # Check if user belongs to this tenant or is system admin
        if current_user.tenant_id != tenant_id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Access denied to this tenant")
        
        # Build update query dynamically
        update_fields = []
        params = [tenant_id]
        param_count = 1
        
        if tenant_update.display_name is not None:
            param_count += 1
            update_fields.append(f"display_name = ${param_count}")
            params.append(tenant_update.display_name)
        
        if tenant_update.contact_email is not None:
            param_count += 1
            update_fields.append(f"contact_email = ${param_count}")
            params.append(tenant_update.contact_email)
        
        if tenant_update.settings is not None:
            param_count += 1
            update_fields.append(f"settings = ${param_count}")
            params.append(json.dumps(tenant_update.settings))
        
        if tenant_update.branding is not None:
            param_count += 1
            update_fields.append(f"branding = ${param_count}")
            params.append(json.dumps(tenant_update.branding))
        
        if tenant_update.is_active is not None:
            param_count += 1
            update_fields.append(f"is_active = ${param_count}")
            params.append(tenant_update.is_active)
        
        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        # Add updated_at
        param_count += 1
        update_fields.append(f"updated_at = ${param_count}")
        params.append(datetime.utcnow())
        
        query = f"""
        UPDATE tenants 
        SET {', '.join(update_fields)}
        WHERE id = $1
        RETURNING id, display_name, updated_at
        """
        
        row = await conn.fetchrow(query, *params)
        
        if not row:
            raise HTTPException(status_code=404, detail="Tenant not found")
        
        logger.info(f"Updated tenant {tenant_id}")
        
        return {
            "id": row['id'],
            "display_name": row['display_name'],
            "updated_at": row['updated_at'].isoformat(),
            "message": "Tenant updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update tenant")

@router.get("/{tenant_id}/users")
async def get_tenant_users(
    tenant_id: str,
    role: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(require_permission("users.view")),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get users for a tenant"""
    try:
        # Check if user belongs to this tenant or is system admin
        if current_user.tenant_id != tenant_id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Access denied to this tenant")
        
        query = """
        SELECT u.id, u.email, u.full_name, u.role, u.permissions, u.is_active,
               u.last_login, u.created_at, u.updated_at,
               COUNT(*) OVER() as total_count
        FROM users u
        WHERE u.tenant_id = $1
        """
        
        params = [tenant_id]
        
        if role:
            query += f" AND u.role = ${len(params) + 1}"
            params.append(role)
        
        if is_active is not None:
            query += f" AND u.is_active = ${len(params) + 1}"
            params.append(is_active)
        
        query += f" ORDER BY u.created_at DESC LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
        params.extend([limit, offset])
        
        rows = await conn.fetch(query, *params)
        
        users = []
        total_count = 0
        
        for row in rows:
            if total_count == 0:
                total_count = row['total_count']
            
            user_data = {
                "id": row['id'],
                "email": row['email'],
                "full_name": row['full_name'],
                "role": row['role'],
                "permissions": row['permissions'] or [],
                "is_active": row['is_active'],
                "last_login": row['last_login'].isoformat() if row['last_login'] else None,
                "created_at": row['created_at'].isoformat(),
                "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None
            }
            users.append(user_data)
        
        logger.info(f"Retrieved {len(users)} users for tenant {tenant_id}")
        
        return {
            "users": users,
            "total_count": total_count,
            "page": offset // limit + 1,
            "page_size": limit,
            "has_more": (offset + limit) < total_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving users for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tenant users")

@router.post("/{tenant_id}/users/invite")
async def invite_user_to_tenant(
    tenant_id: str,
    invite_data: TenantUserInvite,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission("users.invite")),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Invite a user to join the tenant"""
    try:
        # Check if user belongs to this tenant or is system admin
        if current_user.tenant_id != tenant_id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Access denied to this tenant")
        
        # Check if user already exists
        existing_user = await conn.fetchrow("""
            SELECT id, is_active FROM users 
            WHERE email = $1 AND tenant_id = $2
        """, invite_data.email, tenant_id)
        
        if existing_user:
            if existing_user['is_active']:
                raise HTTPException(status_code=400, detail="User already exists and is active")
            else:
                # Reactivate existing user
                await conn.execute("""
                    UPDATE users SET is_active = true, role = $3, permissions = $4, updated_at = NOW()
                    WHERE id = $1
                """, existing_user['id'], invite_data.role, invite_data.permissions)
                
                logger.info(f"Reactivated user {invite_data.email} for tenant {tenant_id}")
                return {"message": "User reactivated successfully"}
        
        # Create user invitation
        expires_at = datetime.utcnow() + timedelta(days=invite_data.expires_in_days)
        
        invitation_id = await conn.fetchval("""
            INSERT INTO user_invitations (
                email, tenant_id, role, permissions, invited_by,
                expires_at, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
            RETURNING id
        """, 
        invite_data.email, tenant_id, invite_data.role, 
        invite_data.permissions, current_user.id, expires_at)
        
        # Add background task to send invitation email
        background_tasks.add_task(
            send_invitation_email, 
            invitation_id, invite_data.email, tenant_id, current_user.full_name
        )
        
        logger.info(f"Created invitation {invitation_id} for {invite_data.email}")
        
        return {
            "invitation_id": invitation_id,
            "email": invite_data.email,
            "expires_at": expires_at.isoformat(),
            "message": "Invitation sent successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inviting user to tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to invite user")

@router.get("/{tenant_id}/subscription")
async def get_tenant_subscription(
    tenant_id: str,
    current_user: User = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get tenant subscription details"""
    try:
        # Check if user belongs to this tenant or is system admin
        if current_user.tenant_id != tenant_id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Access denied to this tenant")
        
        query = """
        SELECT subscription_plan, subscription_features, subscription_limits,
               subscription_expires_at, subscription_updated_at
        FROM tenants
        WHERE id = $1
        """
        
        row = await conn.fetchrow(query, tenant_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Tenant not found")
        
        # Get usage statistics
        usage_stats = await conn.fetchrow("""
            SELECT 
                (SELECT COUNT(*) FROM users WHERE tenant_id = $1 AND is_active = true) as users_count,
                (SELECT COUNT(*) FROM sites WHERE tenant_id = $1) as sites_count,
                (SELECT COALESCE(SUM(
                    CASE 
                        WHEN metadata->>'storage_mb' IS NOT NULL 
                        THEN (metadata->>'storage_mb')::integer 
                        ELSE 0 
                    END
                ), 0) FROM sites WHERE tenant_id = $1) as storage_mb
        """, tenant_id)
        
        subscription_data = {
            "plan": row['subscription_plan'],
            "features": row['subscription_features'] or [],
            "limits": row['subscription_limits'] or {
                "sites": 10,
                "users": 5,
                "storage": 1000  # MB
            },
            "expires_at": row['subscription_expires_at'].isoformat() if row['subscription_expires_at'] else None,
            "updated_at": row['subscription_updated_at'].isoformat() if row['subscription_updated_at'] else None,
            "usage": {
                "sites": usage_stats['sites_count'],
                "users": usage_stats['users_count'],
                "storage": usage_stats['storage_mb']
            }
        }
        
        # Calculate usage percentages
        limits = subscription_data["limits"]
        usage = subscription_data["usage"]
        
        subscription_data["usage_percentages"] = {
            "sites": (usage["sites"] / limits["sites"]) * 100 if limits["sites"] > 0 else 0,
            "users": (usage["users"] / limits["users"]) * 100 if limits["users"] > 0 else 0,
            "storage": (usage["storage"] / limits["storage"]) * 100 if limits["storage"] > 0 else 0
        }
        
        logger.info(f"Retrieved subscription data for tenant {tenant_id}")
        return subscription_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving subscription for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve subscription")

@router.put("/{tenant_id}/subscription")
async def update_tenant_subscription(
    tenant_id: str,
    subscription_update: SubscriptionUpdate,
    current_user: User = Depends(require_permission("tenant.subscription")),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Update tenant subscription"""
    try:
        # Only system admins can update subscriptions
        if not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Only system administrators can update subscriptions")
        
        # Define plan features and limits
        plan_configs = {
            "basic": {
                "features": ["basic_monitoring", "basic_alerts", "basic_reporting"],
                "limits": {"sites": 10, "users": 5, "storage": 1000}
            },
            "professional": {
                "features": ["advanced_monitoring", "predictive_analytics", "advanced_reporting", "api_access"],
                "limits": {"sites": 50, "users": 20, "storage": 5000}
            },
            "enterprise": {
                "features": ["full_monitoring", "ai_analytics", "custom_reporting", "api_access", "white_label"],
                "limits": {"sites": 500, "users": 100, "storage": 25000}
            },
            "custom": {
                "features": subscription_update.features,
                "limits": subscription_update.limits
            }
        }
        
        plan_config = plan_configs.get(subscription_update.plan)
        if not plan_config:
            raise HTTPException(status_code=400, detail="Invalid subscription plan")
        
        # Update subscription
        await conn.execute("""
            UPDATE tenants 
            SET subscription_plan = $2,
                subscription_features = $3,
                subscription_limits = $4,
                subscription_updated_at = NOW()
            WHERE id = $1
        """, 
        tenant_id,
        subscription_update.plan,
        json.dumps(plan_config["features"]),
        json.dumps(plan_config["limits"]))
        
        logger.info(f"Updated subscription for tenant {tenant_id} to {subscription_update.plan}")
        
        return {
            "tenant_id": tenant_id,
            "plan": subscription_update.plan,
            "features": plan_config["features"],
            "limits": plan_config["limits"],
            "updated_at": datetime.utcnow().isoformat(),
            "message": "Subscription updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating subscription for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update subscription")

@router.get("/{tenant_id}/usage/summary")
async def get_tenant_usage_summary(
    tenant_id: str,
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Get tenant usage summary and statistics"""
    try:
        # Check if user belongs to this tenant or is system admin
        if current_user.tenant_id != tenant_id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Access denied to this tenant")
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get usage statistics
        usage_stats = await conn.fetchrow("""
            SELECT 
                COUNT(DISTINCT s.site_id) as total_sites,
                COUNT(DISTINCT u.id) as total_users,
                COUNT(DISTINCT nm.site_id) as active_sites_with_data,
                COUNT(nm.id) as total_metrics,
                COUNT(DISTINCT e.site_id) as sites_with_events,
                COUNT(e.id) as total_events,
                COUNT(DISTINCT ka.site_id) as sites_with_alerts,
                COUNT(ka.id) as total_alerts
            FROM tenants t
            LEFT JOIN sites s ON t.id = s.tenant_id
            LEFT JOIN users u ON t.id = u.tenant_id AND u.is_active = true
            LEFT JOIN network_metrics nm ON s.site_id = nm.site_id AND nm.timestamp >= $2
            LEFT JOIN events e ON s.site_id = e.site_id AND e.timestamp >= $2
            LEFT JOIN kpi_alerts ka ON s.site_id = ka.site_id AND ka.triggered_at >= $2
            WHERE t.id = $1
        """, tenant_id, start_date)
        
        # Get daily metrics count for trend
        daily_metrics = await conn.fetch("""
            SELECT DATE(nm.timestamp) as date, COUNT(*) as metrics_count
            FROM network_metrics nm
            JOIN sites s ON nm.site_id = s.site_id
            WHERE s.tenant_id = $1 AND nm.timestamp >= $2
            GROUP BY DATE(nm.timestamp)
            ORDER BY date DESC
            LIMIT 30
        """, tenant_id, start_date)
        
        # Get top metrics by volume
        top_metrics = await conn.fetch("""
            SELECT nm.metric_name, COUNT(*) as count
            FROM network_metrics nm
            JOIN sites s ON nm.site_id = s.site_id
            WHERE s.tenant_id = $1 AND nm.timestamp >= $2
            GROUP BY nm.metric_name
            ORDER BY count DESC
            LIMIT 10
        """, tenant_id, start_date)
        
        # Get subscription limits
        subscription = await conn.fetchrow("""
            SELECT subscription_limits FROM tenants WHERE id = $1
        """, tenant_id)
        
        limits = subscription['subscription_limits'] if subscription else {}
        
        usage_summary = {
            "period_days": days,
            "period_start": start_date.isoformat(),
            "period_end": datetime.utcnow().isoformat(),
            "current_usage": {
                "sites": usage_stats['total_sites'] or 0,
                "users": usage_stats['total_users'] or 0,
                "active_sites": usage_stats['active_sites_with_data'] or 0,
                "total_metrics": usage_stats['total_metrics'] or 0,
                "total_events": usage_stats['total_events'] or 0,
                "total_alerts": usage_stats['total_alerts'] or 0
            },
            "subscription_limits": limits,
            "usage_percentages": {
                "sites": (usage_stats['total_sites'] / limits.get('sites', 1)) * 100 if limits.get('sites') else 0,
                "users": (usage_stats['total_users'] / limits.get('users', 1)) * 100 if limits.get('users') else 0
            },
            "daily_metrics_trend": [
                {
                    "date": row['date'].isoformat(),
                    "metrics_count": row['metrics_count']
                }
                for row in daily_metrics
            ],
            "top_metrics": [
                {
                    "metric_name": row['metric_name'],
                    "count": row['count']
                }
                for row in top_metrics
            ]
        }
        
        logger.info(f"Generated usage summary for tenant {tenant_id}")
        return usage_summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating usage summary for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate usage summary")

@router.delete("/{tenant_id}/users/{user_id}")
async def remove_user_from_tenant(
    tenant_id: str,
    user_id: str,
    current_user: User = Depends(require_permission("users.remove")),
    conn: asyncpg.Connection = Depends(get_connection)
):
    """Remove a user from tenant (deactivate)"""
    try:
        # Check if user belongs to this tenant or is system admin
        if current_user.tenant_id != tenant_id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Access denied to this tenant")
        
        # Verify user exists and belongs to tenant
        user_exists = await conn.fetchrow("""
            SELECT id, email, is_active FROM users 
            WHERE id = $1 AND tenant_id = $2
        """, user_id, tenant_id)
        
        if not user_exists:
            raise HTTPException(status_code=404, detail="User not found in this tenant")
        
        if user_id == current_user.id:
            raise HTTPException(status_code=400, detail="Cannot remove yourself")
        
        # Deactivate user
        await conn.execute("""
            UPDATE users 
            SET is_active = false, updated_at = NOW()
            WHERE id = $1
        """, user_id)
        
        logger.info(f"Deactivated user {user_id} from tenant {tenant_id}")
        
        return {
            "user_id": user_id,
            "email": user_exists['email'],
            "message": "User removed from tenant successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing user {user_id} from tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove user from tenant")


# Background task functions
async def send_invitation_email(invitation_id: int, email: str, tenant_id: str, invited_by: str):
    """Send invitation email to new user"""
    try:
        # This would integrate with email service
        logger.info(f"Sending invitation email to {email} for tenant {tenant_id}")
        # Simulate email sending
        await asyncio.sleep(2)
        logger.info(f"Invitation email sent to {email}")
        
    except Exception as e:
        logger.error(f"Error sending invitation email to {email}: {e}")