"""
Authentication and authorization using Keycloak
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import jwt
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from keycloak import KeycloakOpenID, KeycloakAdmin
from keycloak.exceptions import KeycloakError

from .config import settings
from .cache import CacheManager, CacheKeys

logger = logging.getLogger(__name__)

# Global Keycloak clients
keycloak_openid: Optional[KeycloakOpenID] = None
keycloak_admin: Optional[KeycloakAdmin] = None

# HTTP Bearer security
security = HTTPBearer()


async def init_auth():
    """Initialize Keycloak authentication with retry logic"""
    global keycloak_openid, keycloak_admin
    
    import socket
    import time
    
    # Wait for Keycloak to be available
    max_retries = 30
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Test socket connection first
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('keycloak', 8080))
            sock.close()
            
            if result == 0:
                logger.info("Keycloak socket is available, initializing...")
                break
            else:
                logger.info(f"Keycloak not ready, attempt {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
        except Exception as e:
            logger.info(f"Socket test failed, attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(retry_delay)
    else:
        raise RuntimeError("Keycloak not available after maximum retries")
    
    try:
        # Initialize OpenID client
        keycloak_openid = KeycloakOpenID(
            server_url=settings.KEYCLOAK_SERVER_URL,
            client_id=settings.KEYCLOAK_CLIENT_ID,
            realm_name=settings.KEYCLOAK_REALM,
            client_secret_key=settings.KEYCLOAK_CLIENT_SECRET,
            verify=True
        )
        
        # Initialize Admin client (optional)
        if settings.KEYCLOAK_CLIENT_SECRET:
            keycloak_admin = KeycloakAdmin(
                server_url=settings.KEYCLOAK_SERVER_URL,
                client_id=settings.KEYCLOAK_CLIENT_ID,
                client_secret_key=settings.KEYCLOAK_CLIENT_SECRET,
                realm_name=settings.KEYCLOAK_REALM,
                verify=True
            )
        
        # Test connection with retry
        await test_keycloak_connection()
        
        logger.info("Keycloak authentication initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Keycloak: {e}")
        raise


async def test_keycloak_connection():
    """Test Keycloak connectivity"""
    try:
        if keycloak_openid:
            # Get well-known configuration
            config = keycloak_openid.well_known()
            logger.info("Keycloak connection test successful")
            return config
        else:
            raise RuntimeError("Keycloak OpenID client not initialized")
            
    except Exception as e:
        logger.error(f"Keycloak connection test failed: {e}")
        raise


class TokenValidator:
    """JWT token validation"""
    
    @staticmethod
    async def validate_token(token: str) -> Dict[str, Any]:
        """Validate JWT token and return user info"""
        if not keycloak_openid:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service not available"
            )
        
        try:
            # Get token info from Keycloak
            token_info = keycloak_openid.introspect(token)
            
            if not token_info.get('active'):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token is not active"
                )
            
            # Decode token for additional information
            public_key = f"-----BEGIN PUBLIC KEY-----\n{keycloak_openid.public_key()}\n-----END PUBLIC KEY-----"
            decoded_token = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience="account"
            )
            
            # Combine introspect info with decoded token
            user_info = {
                **token_info,
                **decoded_token,
                'validated_at': datetime.utcnow().isoformat()
            }
            
            return user_info
            
        except KeycloakError as e:
            logger.error(f"Keycloak validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        except jwt.InvalidTokenError as e:
            logger.error(f"JWT validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format"
            )
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication error"
            )
    
    @staticmethod
    async def get_user_roles(token_info: Dict[str, Any]) -> List[str]:
        """Extract user roles from token"""
        roles = []
        
        # Realm roles
        realm_access = token_info.get('realm_access', {})
        roles.extend(realm_access.get('roles', []))
        
        # Client roles
        resource_access = token_info.get('resource_access', {})
        client_access = resource_access.get(settings.KEYCLOAK_CLIENT_ID, {})
        roles.extend(client_access.get('roles', []))
        
        return list(set(roles))  # Remove duplicates
    
    @staticmethod
    async def get_user_permissions(roles: List[str]) -> List[str]:
        """Map roles to permissions"""
        permissions = []
        
        # Role to permission mapping
        role_permissions = {
            'admin': [
                'sites:read', 'sites:write', 'sites:delete',
                'metrics:read', 'metrics:write',
                'kpis:read', 'kpis:write',
                'events:read', 'events:write', 'events:resolve',
                'reports:read', 'reports:write',
                'users:read', 'users:write',
                'system:admin'
            ],
            'operator': [
                'sites:read', 'sites:write',
                'metrics:read',
                'kpis:read',
                'events:read', 'events:resolve',
                'reports:read'
            ],
            'viewer': [
                'sites:read',
                'metrics:read',
                'kpis:read',
                'events:read',
                'reports:read'
            ],
            'client': [
                'sites:read',
                'metrics:read',
                'kpis:read',
                'reports:read'
            ]
        }
        
        for role in roles:
            permissions.extend(role_permissions.get(role, []))
        
        return list(set(permissions))  # Remove duplicates


class User:
    """User model for authentication"""
    
    def __init__(self, token_info: Dict[str, Any]):
        self.sub = token_info.get('sub')
        self.username = token_info.get('preferred_username')
        self.email = token_info.get('email')
        self.first_name = token_info.get('given_name')
        self.last_name = token_info.get('family_name')
        self.tenant_id = token_info.get('tenant_id')  # Custom claim
        self.roles = []
        self.permissions = []
        self.token_info = token_info
    
    async def load_roles_and_permissions(self):
        """Load user roles and permissions"""
        self.roles = await TokenValidator.get_user_roles(self.token_info)
        self.permissions = await TokenValidator.get_user_permissions(self.roles)
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role"""
        return role in self.roles
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions
    
    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.has_role('admin')
    
    def is_tenant_user(self, tenant_id: str) -> bool:
        """Check if user belongs to specific tenant"""
        return self.tenant_id == tenant_id or self.is_admin()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        return {
            'sub': self.sub,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'tenant_id': self.tenant_id,
            'roles': self.roles,
            'permissions': self.permissions
        }


# Authentication dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    cache: CacheManager = Depends(lambda: CacheManager())
) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    
    # Check cache first
    cache_key = CacheKeys.user_session(token[:20])  # Use token prefix as key
    cached_user_info = await cache.get(cache_key)
    
    if cached_user_info:
        user = User(cached_user_info)
        await user.load_roles_and_permissions()
        return user
    
    # Validate token
    token_info = await TokenValidator.validate_token(token)
    
    # Create user object
    user = User(token_info)
    await user.load_roles_and_permissions()
    
    # Cache user info for 5 minutes
    await cache.set(cache_key, token_info, expire=300)
    
    return user


async def get_optional_user(
    request: Request,
    cache: CacheManager = Depends(lambda: CacheManager())
) -> Optional[User]:
    """Get current user if authenticated, otherwise None"""
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
        
        token = auth_header.split(' ')[1]
        credentials = HTTPAuthorizationCredentials(
            scheme='Bearer',
            credentials=token
        )
        
        return await get_current_user(credentials, cache)
    except:
        return None


def require_permission(permission: str):
    """Decorator to require specific permission"""
    def dependency(current_user: User = Depends(get_current_user)):
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
        return current_user
    
    return dependency


def require_role(role: str):
    """Decorator to require specific role"""
    def dependency(current_user: User = Depends(get_current_user)):
        if not current_user.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role}"
            )
        return current_user
    
    return dependency


def require_tenant_access(tenant_id_param: str = 'tenant_id'):
    """Decorator to require access to specific tenant"""
    def dependency(
        request: Request,
        current_user: User = Depends(get_current_user)
    ):
        # Get tenant_id from path parameters
        tenant_id = request.path_params.get(tenant_id_param)
        
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID required"
            )
        
        if not current_user.is_tenant_user(tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant"
            )
        
        return current_user
    
    return dependency


class RateLimiter:
    """Rate limiting for API endpoints"""
    
    @staticmethod
    async def check_rate_limit(
        identifier: str,
        limit: int = None,
        window: int = 60,
        cache: CacheManager = None
    ) -> bool:
        """Check if rate limit is exceeded"""
        if not cache:
            cache = CacheManager()
        
        limit = limit or settings.RATE_LIMIT_REQUESTS_PER_MINUTE
        
        # Get current count
        key = CacheKeys.rate_limit(identifier)
        current_count = await cache.get(key) or 0
        
        if current_count >= limit:
            return False
        
        # Increment counter
        await cache.increment(key)
        await cache.expire(key, window)
        
        return True


async def rate_limit_dependency(
    request: Request,
    current_user: User = Depends(get_current_user),
    cache: CacheManager = Depends(lambda: CacheManager())
):
    """Rate limiting dependency"""
    identifier = f"{current_user.sub}:{request.client.host}"
    
    if not await RateLimiter.check_rate_limit(identifier, cache=cache):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    return current_user


# Administrative functions
class UserManager:
    """User management utilities"""
    
    @staticmethod
    async def create_user(
        username: str,
        email: str,
        password: str,
        first_name: str = None,
        last_name: str = None,
        tenant_id: str = None,
        roles: List[str] = None
    ) -> Dict[str, Any]:
        """Create new user in Keycloak"""
        if not keycloak_admin:
            raise RuntimeError("Keycloak admin client not available")
        
        try:
            user_data = {
                "username": username,
                "email": email,
                "enabled": True,
                "credentials": [{
                    "type": "password",
                    "value": password,
                    "temporary": False
                }]
            }
            
            if first_name:
                user_data["firstName"] = first_name
            if last_name:
                user_data["lastName"] = last_name
            
            # Add custom attributes
            if tenant_id:
                user_data["attributes"] = {"tenant_id": [tenant_id]}
            
            # Create user
            user_id = keycloak_admin.create_user(user_data)
            
            # Assign roles
            if roles:
                for role in roles:
                    try:
                        role_obj = keycloak_admin.get_realm_role(role)
                        keycloak_admin.assign_realm_roles(user_id, [role_obj])
                    except KeycloakError:
                        logger.warning(f"Role {role} not found")
            
            return {
                "user_id": user_id,
                "username": username,
                "email": email,
                "created": True
            }
            
        except KeycloakError as e:
            logger.error(f"Failed to create user: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create user: {e}"
            )
    
    @staticmethod
    async def get_user_info(user_id: str) -> Dict[str, Any]:
        """Get user information"""
        if not keycloak_admin:
            raise RuntimeError("Keycloak admin client not available")
        
        try:
            return keycloak_admin.get_user(user_id)
        except KeycloakError as e:
            logger.error(f"Failed to get user info: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
    
    @staticmethod
    async def update_user(user_id: str, user_data: Dict[str, Any]) -> bool:
        """Update user information"""
        if not keycloak_admin:
            raise RuntimeError("Keycloak admin client not available")
        
        try:
            keycloak_admin.update_user(user_id, user_data)
            return True
        except KeycloakError as e:
            logger.error(f"Failed to update user: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to update user: {e}"
            )
    
    @staticmethod
    async def delete_user(user_id: str) -> bool:
        """Delete user"""
        if not keycloak_admin:
            raise RuntimeError("Keycloak admin client not available")
        
        try:
            keycloak_admin.delete_user(user_id)
            return True
        except KeycloakError as e:
            logger.error(f"Failed to delete user: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to delete user: {e}"
            )


# Dependency for FastAPI
def get_user_manager() -> UserManager:
    """Get user manager instance"""
    return UserManager()