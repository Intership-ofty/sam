"""
Authentication API endpoints for frontend applications
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from typing import Dict, Any, Optional
import logging

from core.auth import get_current_user, keycloak_openid, keycloak_admin
from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/auth",
    tags=["Authentication"]
)

@router.get("/login")
async def login():
    """Redirect to Keycloak login"""
    if not keycloak_openid:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    
    # Generate Keycloak login URL
    login_url = keycloak_openid.auth_url(
        redirect_uri=f"{settings.ALLOWED_ORIGINS[0]}/auth/callback"
    )
    
    return {"login_url": login_url}

@router.get("/callback")
async def auth_callback(code: str, state: Optional[str] = None):
    """Handle OAuth callback from Keycloak"""
    if not keycloak_openid:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    
    try:
        # Exchange code for tokens
        token = keycloak_openid.token(code)
        
        # Get user info
        user_info = keycloak_openid.userinfo(token['access_token'])
        
        return {
            "access_token": token['access_token'],
            "refresh_token": token.get('refresh_token'),
            "user": {
                "id": user_info.get('sub'),
                "name": user_info.get('name'),
                "email": user_info.get('email'),
                "roles": user_info.get('realm_access', {}).get('roles', [])
            }
        }
        
    except Exception as e:
        logger.error(f"Authentication callback error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authentication failed"
        )

@router.get("/me")
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user information"""
    return {
        "user": current_user,
        "authenticated": True
    }

@router.post("/refresh")
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    if not keycloak_openid:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    
    try:
        token = keycloak_openid.refresh_token(refresh_token)
        
        return {
            "access_token": token['access_token'],
            "refresh_token": token.get('refresh_token')
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token refresh failed"
        )

@router.post("/logout")
async def logout():
    """Logout user"""
    return {"message": "Logged out successfully"}

@router.get("/config")
async def get_auth_config():
    """Get authentication configuration for frontend"""
    return {
        "auth_mode": settings.AUTH_MODE,
        "keycloak_url": settings.KEYCLOAK_SERVER_URL if hasattr(settings, 'KEYCLOAK_SERVER_URL') else None,
        "realm": settings.KEYCLOAK_REALM if hasattr(settings, 'KEYCLOAK_REALM') else None,
        "client_id": settings.KEYCLOAK_CLIENT_ID if hasattr(settings, 'KEYCLOAK_CLIENT_ID') else None
    }
