"""
Sites management API endpoints
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field, validator

from core.database import DatabaseManager, get_database_manager
from core.cache import CacheManager, CacheKeys, get_cache_manager
from core.auth import get_current_user, User, require_permission, require_tenant_access
from core.config import TECHNOLOGY_TYPES, SITE_TYPES, ENERGY_TYPES

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class SiteCreate(BaseModel):
    """Site creation model"""
    site_code: str = Field(..., min_length=2, max_length=50)
    site_name: str = Field(..., min_length=2, max_length=200)
    site_type: str = Field(..., description="Site type (BTS, NODEBS, etc.)")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    address: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    tenant_id: UUID
    technology: Optional[Dict[str, Any]] = None
    energy_config: Optional[Dict[str, Any]] = None
    
    @validator('site_type')
    def validate_site_type(cls, v):
        if v not in SITE_TYPES:
            raise ValueError(f'Site type must be one of: {SITE_TYPES}')
        return v


class SiteUpdate(BaseModel):
    """Site update model"""
    site_name: Optional[str] = Field(None, min_length=2, max_length=200)
    site_type: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    address: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    technology: Optional[Dict[str, Any]] = None
    energy_config: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(None, pattern='^(active|inactive|maintenance)$')
    
    @validator('site_type')
    def validate_site_type(cls, v):
        if v and v not in SITE_TYPES:
            raise ValueError(f'Site type must be one of: {SITE_TYPES}')
        return v


class Site(BaseModel):
    """Site response model"""
    site_id: UUID
    site_code: str
    site_name: str
    site_type: str
    latitude: Optional[float]
    longitude: Optional[float]
    address: Optional[str]
    region: Optional[str]
    country: Optional[str]
    tenant_id: UUID
    technology: Optional[Dict[str, Any]]
    energy_config: Optional[Dict[str, Any]]
    status: str
    created_at: datetime
    updated_at: datetime
    health_score: Optional[float] = None


class SiteList(BaseModel):
    """Site list response model"""
    sites: List[Site]
    total: int
    page: int
    page_size: int
    total_pages: int


class SiteHealth(BaseModel):
    """Site health information"""
    site_id: UUID
    site_code: str
    health_score: float
    status: str
    last_update: datetime
    metrics_summary: Dict[str, Any]
    active_events: int
    critical_events: int


@router.get("/", response_model=SiteList)
async def get_sites(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
    tenant_id: Optional[UUID] = Query(None),
    region: Optional[str] = Query(None),
    site_type: Optional[str] = Query(None),
    technology: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database_manager),
    cache: CacheManager = Depends(get_cache_manager)
):
    """
    Get list of sites with filtering and pagination
    
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 50, max: 1000)
    - **tenant_id**: Filter by tenant
    - **region**: Filter by region
    - **site_type**: Filter by site type
    - **technology**: Filter by technology (2G, 3G, 4G, 5G)
    - **status**: Filter by status (active, inactive, maintenance)
    - **search**: Search in site code or name
    """
    
    # Check cache first
    cache_key = f"sites:list:{page}:{page_size}:{tenant_id}:{region}:{site_type}:{technology}:{status}:{search}"
    cached_result = await cache.get(cache_key)
    if cached_result:
        return cached_result
    
    try:
        # Build WHERE clauses
        where_clauses = []
        params = []
        param_count = 0
        
        # Tenant filtering (users can only see their tenant's sites unless admin)
        if not current_user.is_admin():
            param_count += 1
            where_clauses.append(f"tenant_id = ${param_count}")
            params.append(str(current_user.tenant_id))
        elif tenant_id:
            param_count += 1
            where_clauses.append(f"tenant_id = ${param_count}")
            params.append(str(tenant_id))
        
        # Region filter
        if region:
            param_count += 1
            where_clauses.append(f"region = ${param_count}")
            params.append(region)
        
        # Site type filter
        if site_type:
            param_count += 1
            where_clauses.append(f"site_type = ${param_count}")
            params.append(site_type)
        
        # Status filter
        if status:
            param_count += 1
            where_clauses.append(f"status = ${param_count}")
            params.append(status)
        
        # Technology filter (JSON field)
        if technology:
            param_count += 1
            where_clauses.append(f"technology ? ${param_count}")
            params.append(technology)
        
        # Search filter
        if search:
            param_count += 1
            where_clauses.append(f"(site_code ILIKE ${param_count} OR site_name ILIKE ${param_count})")
            params.append(f"%{search}%")
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        # Get total count
        count_query = f"SELECT COUNT(*) FROM sites {where_clause}"
        total = await db.execute_query_scalar(count_query, *params)
        
        # Calculate pagination
        offset = (page - 1) * page_size
        total_pages = (total + page_size - 1) // page_size
        
        # Get sites
        sites_query = f"""
        SELECT 
            site_id, site_code, site_name, site_type, latitude, longitude,
            address, region, country, tenant_id, technology, energy_config,
            status, created_at, updated_at
        FROM sites 
        {where_clause}
        ORDER BY created_at DESC
        LIMIT {page_size} OFFSET {offset}
        """
        
        sites_data = await db.execute_query(sites_query, *params)
        
        # Convert to Site objects with health scores
        sites = []
        for site_row in sites_data:
            site_dict = dict(site_row)
            
            # Get health score
            health_score = await db.get_site_health_score(str(site_dict['site_id']))
            site_dict['health_score'] = health_score
            
            sites.append(Site(**site_dict))
        
        result = SiteList(
            sites=sites,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
        # Cache for 2 minutes
        await cache.set(cache_key, result.dict(), expire=120)
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching sites: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch sites"
        )


@router.post("/", response_model=Site, status_code=status.HTTP_201_CREATED)
async def create_site(
    site_data: SiteCreate,
    current_user: User = Depends(require_permission("sites:write")),
    db: DatabaseManager = Depends(get_database_manager),
    cache: CacheManager = Depends(get_cache_manager)
):
    """
    Create a new site
    
    Requires 'sites:write' permission
    """
    
    # Check tenant access
    if not current_user.is_tenant_user(str(site_data.tenant_id)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to tenant"
        )
    
    try:
        # Check if site code already exists
        existing_site = await db.execute_query_scalar(
            "SELECT site_id FROM sites WHERE site_code = $1",
            site_data.site_code
        )
        
        if existing_site:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Site with code '{site_data.site_code}' already exists"
            )
        
        # Generate site ID
        site_id = str(uuid4())
        
        # Insert site
        insert_query = """
        INSERT INTO sites (
            site_id, site_code, site_name, site_type, latitude, longitude,
            address, region, country, tenant_id, technology, energy_config
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING *
        """
        
        site_row = await db.execute_query_one(
            insert_query,
            site_id, site_data.site_code, site_data.site_name, site_data.site_type,
            site_data.latitude, site_data.longitude, site_data.address,
            site_data.region, site_data.country, str(site_data.tenant_id),
            site_data.technology, site_data.energy_config
        )
        
        # Clear cache
        await cache.delete(f"sites:list:*")
        
        # Convert to Site model
        site_dict = dict(site_row)
        site_dict['health_score'] = 1.0  # New sites start with perfect health
        
        logger.info(f"Site created: {site_data.site_code} by {current_user.username}")
        
        return Site(**site_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating site: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create site"
        )


@router.get("/{site_id}", response_model=Site)
async def get_site(
    site_id: UUID,
    current_user: User = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database_manager),
    cache: CacheManager = Depends(get_cache_manager)
):
    """
    Get site details by ID
    """
    
    # Check cache first
    cache_key = f"site:{site_id}"
    cached_site = await cache.get(cache_key)
    if cached_site:
        return Site(**cached_site)
    
    try:
        # Get site
        site_query = """
        SELECT 
            site_id, site_code, site_name, site_type, latitude, longitude,
            address, region, country, tenant_id, technology, energy_config,
            status, created_at, updated_at
        FROM sites 
        WHERE site_id = $1
        """
        
        site_row = await db.execute_query_one(site_query, str(site_id))
        
        if not site_row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Site not found"
            )
        
        site_dict = dict(site_row)
        
        # Check tenant access
        if not current_user.is_tenant_user(str(site_dict['tenant_id'])):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to site"
            )
        
        # Get health score
        health_score = await db.get_site_health_score(str(site_id))
        site_dict['health_score'] = health_score
        
        site = Site(**site_dict)
        
        # Cache for 5 minutes
        await cache.set(cache_key, site.dict(), expire=300)
        
        return site
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching site {site_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch site"
        )


@router.put("/{site_id}", response_model=Site)
async def update_site(
    site_id: UUID,
    site_data: SiteUpdate,
    current_user: User = Depends(require_permission("sites:write")),
    db: DatabaseManager = Depends(get_database_manager),
    cache: CacheManager = Depends(get_cache_manager)
):
    """
    Update site information
    
    Requires 'sites:write' permission
    """
    
    try:
        # Get existing site
        existing_site = await db.execute_query_one(
            "SELECT * FROM sites WHERE site_id = $1",
            str(site_id)
        )
        
        if not existing_site:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Site not found"
            )
        
        # Check tenant access
        if not current_user.is_tenant_user(str(existing_site['tenant_id'])):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to site"
            )
        
        # Build update fields
        update_fields = []
        params = []
        param_count = 0
        
        update_data = site_data.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if value is not None:
                param_count += 1
                update_fields.append(f"{field} = ${param_count}")
                params.append(value)
        
        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        # Add updated_at
        param_count += 1
        update_fields.append(f"updated_at = ${param_count}")
        params.append(datetime.utcnow())
        
        # Add site_id for WHERE clause
        param_count += 1
        params.append(str(site_id))
        
        # Update site
        update_query = f"""
        UPDATE sites 
        SET {', '.join(update_fields)}
        WHERE site_id = ${param_count}
        RETURNING *
        """
        
        site_row = await db.execute_query_one(update_query, *params)
        
        # Clear cache
        await cache.delete(f"site:{site_id}")
        await cache.delete(f"sites:list:*")
        
        # Convert to Site model
        site_dict = dict(site_row)
        health_score = await db.get_site_health_score(str(site_id))
        site_dict['health_score'] = health_score
        
        logger.info(f"Site updated: {site_id} by {current_user.username}")
        
        return Site(**site_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating site {site_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update site"
        )


@router.delete("/{site_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_site(
    site_id: UUID,
    current_user: User = Depends(require_permission("sites:delete")),
    db: DatabaseManager = Depends(get_database_manager),
    cache: CacheManager = Depends(get_cache_manager)
):
    """
    Delete site
    
    Requires 'sites:delete' permission
    """
    
    try:
        # Get existing site
        existing_site = await db.execute_query_one(
            "SELECT * FROM sites WHERE site_id = $1",
            str(site_id)
        )
        
        if not existing_site:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Site not found"
            )
        
        # Check tenant access
        if not current_user.is_tenant_user(str(existing_site['tenant_id'])):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to site"
            )
        
        # Delete site (soft delete by updating status)
        await db.execute_command(
            "UPDATE sites SET status = 'deleted', updated_at = NOW() WHERE site_id = $1",
            str(site_id)
        )
        
        # Clear cache
        await cache.delete(f"site:{site_id}")
        await cache.delete(f"sites:list:*")
        
        logger.info(f"Site deleted: {site_id} by {current_user.username}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting site {site_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete site"
        )


@router.get("/{site_id}/health", response_model=SiteHealth)
async def get_site_health(
    site_id: UUID,
    current_user: User = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database_manager),
    cache: CacheManager = Depends(get_cache_manager)
):
    """
    Get detailed site health information
    """
    
    # Check cache first
    cache_key = CacheKeys.site_health(str(site_id))
    cached_health = await cache.get(cache_key)
    if cached_health:
        return SiteHealth(**cached_health)
    
    try:
        # Get site basic info
        site_query = "SELECT site_code, tenant_id, status FROM sites WHERE site_id = $1"
        site_info = await db.execute_query_one(site_query, str(site_id))
        
        if not site_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Site not found"
            )
        
        # Check tenant access
        if not current_user.is_tenant_user(str(site_info['tenant_id'])):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to site"
            )
        
        # Get health score
        health_score = await db.get_site_health_score(str(site_id))
        
        # Get metrics summary
        metrics_summary = await db.get_site_metrics_summary(str(site_id))
        
        # Get event counts
        events = await db.get_active_events(str(site_id))
        active_events = len(events)
        critical_events = len([e for e in events if e.get('severity') == 'CRITICAL'])
        
        health_info = SiteHealth(
            site_id=site_id,
            site_code=site_info['site_code'],
            health_score=health_score,
            status=site_info['status'],
            last_update=datetime.utcnow(),
            metrics_summary={row['metric_name']: {
                'avg_value': float(row['avg_value']) if row['avg_value'] else None,
                'min_value': float(row['min_value']) if row['min_value'] else None,
                'max_value': float(row['max_value']) if row['max_value'] else None,
                'avg_quality': float(row['avg_quality']) if row['avg_quality'] else None
            } for row in metrics_summary},
            active_events=active_events,
            critical_events=critical_events
        )
        
        # Cache for 1 minute
        await cache.set(cache_key, health_info.dict(), expire=60)
        
        return health_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching site health {site_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch site health"
        )