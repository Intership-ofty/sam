# ADR-002: Polystore Data Architecture

## Status
Accepted

## Context
The Towerco AIOps platform has diverse data storage requirements that cannot be efficiently served by a single database technology:

- **Time-series data**: High-volume metrics and KPIs requiring temporal queries
- **Transactional data**: User management, tenants, configurations requiring ACID properties  
- **Cache data**: Session data, real-time computations requiring sub-millisecond access
- **File storage**: Reports, logs, binary data requiring object storage
- **Search data**: Full-text search across incidents, logs, and documentation
- **Analytics data**: Historical data for business intelligence and ML training

Each data type has different consistency, performance, and scalability requirements that favor different storage technologies.

## Decision
We will implement a **Polystore Data Architecture** using multiple specialized databases, each optimized for specific data patterns and access requirements.

### Data Store Selection:
1. **TimescaleDB** - Time-series and metrics data
2. **PostgreSQL** - Transactional and relational data  
3. **Redis** - Caching and session data
4. **MinIO** - Object storage for files and reports
5. **Elasticsearch** - Search and log analytics (optional, via observability stack)

## Alternatives Considered

### 1. Single Database (PostgreSQL only)
- **Pros**: Simple, single source of truth, ACID transactions
- **Cons**: Poor time-series performance, scaling bottlenecks, not optimized for all use cases

### 2. Traditional Data Warehouse (Snowflake/BigQuery)
- **Pros**: Excellent analytics capabilities, managed service
- **Cons**: High cost, vendor lock-in, not suitable for real-time operations

### 3. NoSQL-First (MongoDB/Cassandra)
- **Pros**: Horizontal scaling, flexible schema
- **Cons**: No ACID guarantees, complex queries, not optimized for time-series

### 4. NewSQL Databases (CockroachDB/TiDB)
- **Pros**: ACID + horizontal scaling, SQL compatibility
- **Cons**: Complex setup, limited time-series optimization, higher latency

## Consequences

### Positive
- **Optimal Performance**: Each database optimized for its specific use case
- **Scalability**: Independent scaling of different data stores
- **Technology Fit**: Right tool for the right job
- **Cost Efficiency**: Only pay for capabilities you need
- **Fault Isolation**: Failure in one store doesn't affect others
- **Evolution**: Can upgrade/replace individual stores without affecting others

### Negative
- **Complexity**: Multiple databases to manage and monitor
- **Data Consistency**: No global transactions across stores
- **Integration Complexity**: Data synchronization between stores
- **Operational Overhead**: Multiple backup, monitoring, and maintenance procedures
- **Development Complexity**: Different query languages and patterns

## Implementation

### Data Store Responsibilities

#### TimescaleDB (Time-series data)
```sql
-- KPI metrics and measurements
kpi_metrics (timestamp, tenant_id, site_id, metric_name, value, metadata)

-- Raw sensor data
sensor_data (timestamp, site_id, sensor_type, reading, unit)

-- Performance metrics
performance_metrics (timestamp, component_id, cpu, memory, network, disk)
```

#### PostgreSQL (Transactional data)
```sql
-- Core business entities
tenants (id, name, settings, subscription, created_at)
users (id, tenant_id, email, role, permissions) 
sites (site_id, tenant_id, name, location, configuration)
noc_incidents (id, tenant_id, title, status, created_at)

-- Configuration and metadata
escalation_rules, optimization_tasks, business_insights
```

#### Redis (Cache and session data)
```python
# Session management
f"session:{user_id}" -> session_data

# Real-time calculations
f"kpi_cache:{site_id}:{metric}" -> calculated_value

# Rate limiting
f"rate_limit:{user_id}" -> request_count

# Temporary data
f"temp_analysis:{task_id}" -> analysis_results
```

#### MinIO (Object storage)
```
/tenant/{tenant_id}/reports/{report_id}.pdf
/tenant/{tenant_id}/exports/{export_id}.xlsx
/system/backups/{date}/database_backup.sql
/logs/{service}/{date}/application.log
```

### Data Synchronization Strategy

#### 1. Event-Driven Synchronization
- Primary data changes trigger events
- Secondary stores subscribe to relevant events
- Eventually consistent across stores

#### 2. Materialized Views
- TimescaleDB continuous aggregates for real-time metrics
- PostgreSQL materialized views for complex analytics
- Automatic refresh based on data changes

#### 3. ETL Pipelines
- Batch synchronization for analytical data
- Scheduled aggregations and reports
- Data validation and quality checks

### Data Access Patterns

#### Application Layer
```python
# Service layer abstracts data access
class MetricsService:
    def __init__(self, timescale_db, redis_cache):
        self.timescale = timescale_db
        self.cache = redis_cache
    
    async def get_realtime_kpi(self, site_id: str):
        # Try cache first
        cached = await self.cache.get(f"kpi:{site_id}")
        if cached:
            return cached
        
        # Fallback to database
        result = await self.timescale.fetch_latest_kpi(site_id)
        await self.cache.setex(f"kpi:{site_id}", 60, result)
        return result
```

#### Repository Pattern
- Each data store has dedicated repository classes
- Repositories handle store-specific optimizations
- Business logic remains database-agnostic

### Backup and Recovery Strategy

#### Per-Store Backup
- **TimescaleDB**: Continuous WAL archiving + daily full backups
- **PostgreSQL**: pg_dump + WAL archiving
- **Redis**: RDB snapshots + AOF for durability  
- **MinIO**: Cross-region replication + versioning

#### Cross-Store Consistency
- Event log serves as distributed transaction log
- Point-in-time recovery coordination across stores
- Automated consistency checks and reconciliation

## Technology Specifications

### TimescaleDB Configuration
```yaml
# Optimized for time-series workloads
shared_preload_libraries: 'timescaledb'
max_connections: 200
shared_buffers: 2GB
effective_cache_size: 6GB
work_mem: 50MB
timescaledb.max_background_workers: 8
```

### Redis Configuration  
```yaml
# Optimized for caching workload
maxmemory: 4gb
maxmemory-policy: allkeys-lru
save: 900 1 300 10 60 10000
appendonly: yes
appendfsync: everysec
```

### MinIO Configuration
```yaml
# Object storage configuration
MINIO_ROOT_USER: admin
MINIO_ROOT_PASSWORD: secure_password  
MINIO_STORAGE_CLASS_STANDARD: EC:4
MINIO_BROWSER: on
```

## Monitoring and Operations

### Health Checks
- Individual database health monitoring
- Cross-store data consistency checks
- Performance metrics per data store
- Automated failover procedures

### Performance Monitoring
- Query performance per database type
- Cache hit ratios and efficiency
- Storage utilization and growth trends
- Replication lag monitoring

## Migration Strategy

### Phase 1: Core Setup (Completed)
- ✅ TimescaleDB for metrics data
- ✅ PostgreSQL for transactional data  
- ✅ Redis for caching
- ✅ MinIO for object storage

### Phase 2: Optimization (Completed)
- ✅ Data partitioning and indexing
- ✅ Cache warming strategies
- ✅ Cross-store synchronization

### Phase 3: Advanced Features (Completed)
- ✅ Automated backup and recovery
- ✅ Performance tuning and monitoring
- ✅ Data lifecycle management

## Review Date
January 2025

## References
- [TimescaleDB Best Practices](https://docs.timescale.com/timescaledb/latest/how-to-guides/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Redis Best Practices](https://redis.io/docs/manual/clients-guide/)
- [MinIO Deployment Guide](https://docs.min.io/minio/baremetal/)