"""
Database connection and session management
"""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from .config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()

# Global variables
engine = None
AsyncSessionLocal = None
db_pool: Optional[asyncpg.Pool] = None


async def init_db():
    """Initialize database connections"""
    global engine, AsyncSessionLocal, db_pool
    
    try:
        # Create SQLAlchemy async engine
        engine_options = {
            "echo": settings.is_development,
        }
        if settings.is_development:
            engine_options["poolclass"] = NullPool
        else:
            engine_options["pool_size"] = settings.DB_POOL_SIZE
            engine_options["max_overflow"] = settings.DB_POOL_MAX_OVERFLOW
            engine_options["pool_timeout"] = settings.DB_POOL_TIMEOUT

        engine = create_async_engine(
            settings.DATABASE_URL,
            **engine_options
        )
        
        # Create session factory
        AsyncSessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        # Create asyncpg pool for direct queries
        db_pool = await asyncpg.create_pool(
            settings.DATABASE_URL.replace('+asyncpg', ''),
            min_size=5,
            max_size=settings.DB_POOL_SIZE,
            command_timeout=30,
        )
        
        # Test connections
        await test_db_connection()
        
        logger.info("Database connections initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def test_db_connection():
    """Test database connectivity"""
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        if db_pool:
            async with db_pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                assert result == 1
                
        logger.info("Database connection test successful")
        
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        raise


async def close_db():
    """Close database connections"""
    global engine, db_pool
    
    try:
        if engine:
            await engine.dispose()
            logger.info("SQLAlchemy engine disposed")
            
        if db_pool:
            await db_pool.close()
            logger.info("AsyncPG pool closed")
            
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for dependency injection"""
    if not AsyncSessionLocal:
        raise RuntimeError("Database not initialized")
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_pool() -> Optional[asyncpg.Pool]:
    """Get asyncpg pool for direct queries"""
    return db_pool


@asynccontextmanager
async def get_db_connection():
    """Get direct database connection"""
    if not db_pool:
        raise RuntimeError("Database pool not initialized")
    
    async with db_pool.acquire() as connection:
        try:
            yield connection
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise


class DatabaseManager:
    """Database management utilities"""
    
    @staticmethod
    async def execute_query(query: str, *args, **kwargs):
        """Execute a query with asyncpg"""
        async with get_db_connection() as conn:
            return await conn.fetch(query, *args, **kwargs)
    
    @staticmethod
    async def execute_query_one(query: str, *args, **kwargs):
        """Execute a query and return one result"""
        async with get_db_connection() as conn:
            return await conn.fetchrow(query, *args, **kwargs)
    
    @staticmethod
    async def execute_query_scalar(query: str, *args, **kwargs):
        """Execute a query and return scalar value"""
        async with get_db_connection() as conn:
            return await conn.fetchval(query, *args, **kwargs)
    
    @staticmethod
    async def execute_command(query: str, *args, **kwargs):
        """Execute a command (INSERT, UPDATE, DELETE)"""
        async with get_db_connection() as conn:
            return await conn.execute(query, *args, **kwargs)
    
    @staticmethod
    async def get_site_health_score(site_id: str) -> float:
        """Get calculated health score for a site"""
        query = "SELECT get_site_health_score($1)"
        result = await DatabaseManager.execute_query_scalar(query, site_id)
        return float(result) if result is not None else 0.0
    
    @staticmethod
    async def calculate_nur(site_id: str, start_time, end_time) -> float:
        """Calculate Network Unavailability Rate for a site"""
        query = "SELECT calculate_nur($1, $2, $3)"
        result = await DatabaseManager.execute_query_scalar(
            query, site_id, start_time, end_time
        )
        return float(result) if result is not None else 0.0
    
    @staticmethod
    async def get_site_metrics_summary(site_id: str, hours: int = 24):
        """Get metrics summary for a site in the last N hours"""
        query = """
        SELECT 
            metric_name,
            AVG(metric_value) as avg_value,
            MIN(metric_value) as min_value,
            MAX(metric_value) as max_value,
            AVG(quality_score) as avg_quality
        FROM network_metrics 
        WHERE site_id = $1 
        AND time >= NOW() - INTERVAL '%s hours'
        GROUP BY metric_name
        ORDER BY metric_name
        """ % hours
        
        return await DatabaseManager.execute_query(query, site_id)
    
    @staticmethod
    async def get_active_events(site_id: str = None, severity: str = None):
        """Get active events, optionally filtered by site and/or severity"""
        where_clauses = ["resolved = FALSE"]
        params = []
        param_count = 0
        
        if site_id:
            param_count += 1
            where_clauses.append(f"site_id = ${param_count}")
            params.append(site_id)
        
        if severity:
            param_count += 1
            where_clauses.append(f"severity = ${param_count}")
            params.append(severity)
        
        query = f"""
        SELECT 
            event_id,
            time,
            site_id,
            tenant_id,
            event_type,
            severity,
            title,
            description,
            source_system,
            acknowledged,
            correlation_id
        FROM events 
        WHERE {' AND '.join(where_clauses)}
        ORDER BY time DESC
        LIMIT 100
        """
        
        return await DatabaseManager.execute_query(query, *params)
    
    @staticmethod
    async def get_kpi_latest_values(site_id: str = None, kpi_name: str = None):
        """Get latest KPI values"""
        where_clauses = []
        params = []
        param_count = 0
        
        if site_id:
            param_count += 1
            where_clauses.append(f"kv.site_id = ${param_count}")
            params.append(site_id)
        
        if kpi_name:
            param_count += 1
            where_clauses.append(f"kd.kpi_name = ${param_count}")
            params.append(kpi_name)
        
        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        query = f"""
        SELECT DISTINCT ON (kd.kpi_id, kv.site_id)
            kd.kpi_name,
            kd.kpi_category,
            kd.unit,
            kd.target_value,
            kv.site_id,
            kv.tenant_id,
            kv.kpi_value,
            kv.deviation_pct,
            kv.quality_score,
            kv.trend,
            kv.time
        FROM kpi_values kv
        JOIN kpi_definitions kd ON kv.kpi_id = kd.kpi_id
        {where_clause}
        ORDER BY kd.kpi_id, kv.site_id, kv.time DESC
        """
        
        return await DatabaseManager.execute_query(query, *params)
    
    @staticmethod
    async def insert_network_metric(
        site_id: str,
        tenant_id: str,
        technology: str,
        metric_name: str,
        metric_value: float,
        unit: str = None,
        quality_score: float = None,
        metadata: dict = None
    ):
        """Insert network metric"""
        query = """
        INSERT INTO network_metrics 
        (time, site_id, tenant_id, technology, metric_name, metric_value, unit, quality_score, metadata)
        VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8)
        """
        
        await DatabaseManager.execute_command(
            query, site_id, tenant_id, technology, metric_name, 
            metric_value, unit, quality_score, metadata
        )
    
    @staticmethod
    async def insert_energy_metric(
        site_id: str,
        tenant_id: str,
        energy_type: str,
        metric_name: str,
        metric_value: float,
        unit: str = None,
        efficiency_score: float = None,
        metadata: dict = None
    ):
        """Insert energy metric"""
        query = """
        INSERT INTO energy_metrics 
        (time, site_id, tenant_id, energy_type, metric_name, metric_value, unit, efficiency_score, metadata)
        VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8)
        """
        
        await DatabaseManager.execute_command(
            query, site_id, tenant_id, energy_type, metric_name,
            metric_value, unit, efficiency_score, metadata
        )
    
    @staticmethod
    async def insert_event(
        site_id: str,
        tenant_id: str,
        event_type: str,
        severity: str,
        source_system: str,
        title: str,
        description: str = None,
        impact_assessment: str = None,
        correlation_id: str = None,
        metadata: dict = None
    ):
        """Insert event"""
        query = """
        INSERT INTO events 
        (time, site_id, tenant_id, event_type, severity, source_system, 
         title, description, impact_assessment, correlation_id, metadata)
        VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING event_id
        """
        
        result = await DatabaseManager.execute_query_scalar(
            query, site_id, tenant_id, event_type, severity, source_system,
            title, description, impact_assessment, correlation_id, metadata
        )
        
        return result


# Dependency for FastAPI
def get_database_manager() -> DatabaseManager:
    """Get database manager instance"""
    return DatabaseManager()