# ADR-003: Multi-tenant SaaS Architecture

## Status
Accepted

## Context
The Towerco AIOps platform needs to serve multiple telecom operators (tenants) with complete data isolation, customizable features, and scalable resource allocation. Each tenant requires:

- Complete data isolation for security and compliance
- Customizable branding and UI theming
- Flexible subscription plans with feature gates
- Independent scaling and performance isolation  
- Tenant-specific configurations and business rules
- Cost-efficient resource sharing where appropriate

The platform must support both small regional operators and large multinational telecom companies on the same infrastructure.

## Decision
We will implement a **Hybrid Multi-tenant Architecture** combining shared infrastructure with logical data isolation and tenant-specific customization.

### Architecture Components:
1. **Shared Application Layer** - Single codebase serving all tenants
2. **Logical Data Isolation** - Row-level security with tenant_id filtering
3. **Tenant Context Injection** - Middleware enforces tenant boundaries
4. **Configurable Features** - Feature flags and subscription-based access control
5. **Tenant Branding** - Dynamic theming and white-labeling capabilities

## Alternatives Considered

### 1. Single-Tenant Architecture (Separate Instances)
- **Pros**: Complete isolation, customization, compliance
- **Cons**: High operational overhead, poor resource utilization, scaling complexity

### 2. Shared Database with Shared Schema
- **Pros**: Simple implementation, efficient resource usage
- **Cons**: Poor isolation, security risks, difficult customization

### 3. Database-per-Tenant
- **Pros**: Strong isolation, independent backup/restore
- **Cons**: High operational overhead, connection pool limits, difficult cross-tenant analytics

### 4. Microservices with Tenant Routing
- **Pros**: Service-level isolation, independent scaling
- **Cons**: Complex orchestration, network overhead, higher latency

## Consequences

### Positive
- **Cost Efficiency**: Shared infrastructure reduces per-tenant costs
- **Scalability**: Horizontal scaling serves multiple tenants efficiently
- **Operational Efficiency**: Single deployment and maintenance process
- **Feature Velocity**: New features available to all tenants simultaneously
- **Security**: Proper isolation with row-level security
- **Compliance**: Tenant data never mixed, audit trails maintained

### Negative
- **Complexity**: Multi-tenant code is more complex than single-tenant
- **Testing Complexity**: Must test tenant isolation thoroughly
- **Performance Isolation**: One tenant's load can affect others
- **Customization Limits**: Deep customization is more difficult
- **Migration Risk**: Tenant migration between environments is complex

## Implementation

### Tenant Identification Strategy

#### 1. Domain-based Routing
```python
# Tenant identified by subdomain
tenant_a.towerco-aiops.com -> tenant_id: "tenant_a"
tenant_b.towerco-aiops.com -> tenant_id: "tenant_b"
```

#### 2. JWT Token-based Context
```python
# Tenant ID embedded in JWT claims
{
    "sub": "user_123",
    "tenant_id": "tenant_a",
    "role": "admin",
    "permissions": ["sites.read", "incidents.manage"]
}
```

### Data Isolation Pattern

#### Database Schema with Row-Level Security
```sql
-- All tenant-specific tables include tenant_id
CREATE TABLE sites (
    site_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL,
    name VARCHAR(100) NOT NULL,
    location JSONB,
    configuration JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Row-level security policy
CREATE POLICY tenant_isolation ON sites
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant_id'));

-- Enable RLS
ALTER TABLE sites ENABLE ROW LEVEL SECURITY;
```

#### Middleware for Tenant Context
```python
@app.middleware("http")
async def tenant_context_middleware(request: Request, call_next):
    # Extract tenant from subdomain or header
    tenant_id = extract_tenant_from_request(request)
    
    if not tenant_id:
        raise HTTPException(401, "Tenant not identified")
    
    # Set tenant context for database queries
    request.state.tenant_id = tenant_id
    
    # Set database session variable
    async with get_connection() as conn:
        await conn.execute(
            "SET app.current_tenant_id = $1", tenant_id
        )
    
    response = await call_next(request)
    return response
```

### Feature Flag System

#### Subscription-based Feature Gates
```python
class FeatureManager:
    def __init__(self, tenant_subscription: Dict):
        self.features = tenant_subscription.get('features', [])
        self.limits = tenant_subscription.get('limits', {})
    
    def has_feature(self, feature: str) -> bool:
        return feature in self.features
    
    def within_limit(self, resource: str, current_usage: int) -> bool:
        limit = self.limits.get(resource, float('inf'))
        return current_usage < limit
    
    def get_limit(self, resource: str) -> int:
        return self.limits.get(resource, 0)

# Usage in API endpoints
@router.get("/advanced-analytics")
async def get_advanced_analytics(
    current_user: User = Depends(get_current_user),
    features: FeatureManager = Depends(get_tenant_features)
):
    if not features.has_feature("advanced_analytics"):
        raise HTTPException(403, "Feature not available in current plan")
    
    return await analytics_service.get_advanced_data()
```

### Tenant Configuration System

#### Tenant Settings Schema
```python
@dataclass
class TenantSettings:
    # Branding and UI
    company_name: str
    logo_url: Optional[str]
    primary_color: str = "#3B82F6"
    secondary_color: str = "#6B7280"
    custom_css: Optional[str] = None
    
    # Localization
    timezone: str = "UTC"
    date_format: str = "YYYY-MM-DD"
    currency: str = "USD"
    language: str = "en"
    
    # Business Rules
    sla_targets: Dict[str, float]
    alert_thresholds: Dict[str, float]
    escalation_rules: List[Dict]
    
    # Integration Settings
    smtp_settings: Optional[Dict]
    webhook_endpoints: List[str]
    api_keys: Dict[str, str]
```

#### Dynamic Configuration Loading
```python
class TenantConfigService:
    async def get_tenant_config(self, tenant_id: str) -> TenantSettings:
        # Try cache first
        cached = await redis.get(f"tenant_config:{tenant_id}")
        if cached:
            return TenantSettings(**json.loads(cached))
        
        # Load from database
        config = await db.fetch_tenant_config(tenant_id)
        settings = TenantSettings(**config)
        
        # Cache for 1 hour
        await redis.setex(
            f"tenant_config:{tenant_id}", 
            3600, 
            json.dumps(asdict(settings))
        )
        
        return settings
```

### Resource Isolation and Scaling

#### Connection Pool Management
```python
# Per-tenant connection pool limits
TENANT_POOL_CONFIG = {
    "enterprise": {"min_size": 5, "max_size": 20},
    "professional": {"min_size": 2, "max_size": 10}, 
    "basic": {"min_size": 1, "max_size": 5}
}

async def get_tenant_connection(tenant_id: str):
    tenant_plan = await get_tenant_plan(tenant_id)
    pool_config = TENANT_POOL_CONFIG[tenant_plan]
    
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=pool_config["min_size"],
        max_size=pool_config["max_size"],
        command_timeout=60
    )
    
    return pool
```

#### Rate Limiting per Tenant
```python
class TenantRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def check_rate_limit(self, tenant_id: str, endpoint: str) -> bool:
        tenant_plan = await get_tenant_plan(tenant_id)
        limit = RATE_LIMITS[tenant_plan][endpoint]
        
        key = f"rate_limit:{tenant_id}:{endpoint}"
        current = await self.redis.incr(key)
        
        if current == 1:
            await self.redis.expire(key, 3600)  # 1 hour window
            
        return current <= limit
```

### Tenant Lifecycle Management

#### Tenant Provisioning
```python
async def provision_tenant(tenant_data: TenantCreate) -> Tenant:
    async with db.transaction():
        # Create tenant record
        tenant = await db.create_tenant(tenant_data)
        
        # Initialize tenant-specific data
        await db.create_tenant_schema(tenant.id)
        await db.create_default_settings(tenant.id)
        await db.create_admin_user(tenant.id, tenant_data.admin_email)
        
        # Setup tenant-specific resources
        await setup_tenant_monitoring(tenant.id)
        await create_tenant_dashboards(tenant.id)
        
        return tenant
```

#### Tenant Migration
```python
async def migrate_tenant_data(
    source_tenant: str, 
    target_tenant: str,
    data_types: List[str]
) -> MigrationResult:
    migration_id = str(uuid.uuid4())
    
    try:
        for data_type in data_types:
            await migrate_tenant_table(
                source_tenant, target_tenant, data_type
            )
        
        await log_migration_success(migration_id)
        return MigrationResult(success=True, id=migration_id)
        
    except Exception as e:
        await rollback_migration(migration_id)
        raise MigrationError(f"Migration failed: {e}")
```

### Security Implementation

#### Tenant Data Validation
```python
def validate_tenant_access(required_tenant: str):
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            current_tenant = request.state.tenant_id
            
            if current_tenant != required_tenant:
                logger.warning(
                    f"Cross-tenant access attempt: {current_tenant} -> {required_tenant}"
                )
                raise HTTPException(403, "Access denied")
                
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
```

#### Audit Logging
```python
async def log_tenant_activity(
    tenant_id: str,
    user_id: str, 
    action: str,
    resource: str,
    details: Dict[str, Any]
):
    await db.execute("""
        INSERT INTO tenant_audit_log (
            tenant_id, user_id, action, resource, 
            details, ip_address, timestamp
        ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
    """, tenant_id, user_id, action, resource, 
         json.dumps(details), get_client_ip())
```

## Monitoring and Operations

### Tenant-specific Metrics
- Resource usage per tenant
- API call patterns and rate limiting
- Feature usage analytics
- Performance metrics per tenant
- Cost allocation and billing data

### Health Monitoring
- Cross-tenant data leakage detection
- Performance impact monitoring
- Resource utilization alerts
- Tenant-specific SLA monitoring

## Migration Path

### Phase 1: Foundation (Completed)
- ✅ Tenant identification and routing
- ✅ Database row-level security
- ✅ Basic feature flags

### Phase 2: Advanced Features (Completed)
- ✅ Tenant branding and theming
- ✅ Subscription management
- ✅ Resource isolation

### Phase 3: Enterprise Features (Completed)
- ✅ Advanced security controls
- ✅ Audit logging and compliance
- ✅ Tenant lifecycle management

## Review Date
January 2025

## References
- [Multi-Tenant SaaS Architecture](https://docs.aws.amazon.com/wellarchitected/latest/saas-lens/saas-lens.html)
- [PostgreSQL Row Level Security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [SaaS Tenant Isolation Strategies](https://docs.microsoft.com/en-us/azure/architecture/guide/multitenant/considerations/tenant-isolation)