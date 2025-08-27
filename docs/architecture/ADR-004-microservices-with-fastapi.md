# ADR-004: Microservices Architecture with FastAPI

## Status
Accepted

## Context
The Towerco AIOps platform has diverse functional requirements that benefit from service separation:

- **Data Ingestion**: High-throughput event processing with different scaling needs
- **KPI Calculation**: CPU-intensive computations that benefit from dedicated resources
- **AIOps Engine**: ML-heavy workloads requiring specialized infrastructure
- **NOC Operations**: Real-time incident management with low-latency requirements
- **Business Intelligence**: Analytics workloads with different performance characteristics
- **Client Portal**: User-facing API with high availability requirements

Each service has different scaling patterns, technology requirements, and operational characteristics that favor a microservices approach.

## Decision
We will implement a **Microservices Architecture** using **FastAPI** as the primary framework for all HTTP-based services, with specialized workers for background processing.

### Service Decomposition Strategy:
1. **API Gateway Service** - Request routing, authentication, rate limiting
2. **Core Services** - Business logic APIs (KPIs, Sites, Incidents, etc.)
3. **Background Workers** - Processing services (AIOps, BI, Optimization)
4. **Specialized Services** - Real-time services (Notifications, WebSocket)

## Alternatives Considered

### 1. Monolithic Architecture
- **Pros**: Simple deployment, easy debugging, no network latency
- **Cons**: Single point of failure, difficult scaling, technology lock-in

### 2. Serverless Functions (AWS Lambda/Azure Functions)  
- **Pros**: Automatic scaling, pay-per-use, no server management
- **Cons**: Vendor lock-in, cold starts, limited execution time, complex state management

### 3. Service Mesh Architecture (Istio/Linkerd)
- **Pros**: Advanced traffic management, security, observability
- **Cons**: High complexity, operational overhead, learning curve

### 4. Event-Driven Microservices Only
- **Pros**: Loose coupling, high throughput, fault tolerance
- **Cons**: No synchronous APIs, complex debugging, eventual consistency only

## Consequences

### Positive
- **Independent Scaling**: Each service scales based on its specific load patterns
- **Technology Diversity**: Can use optimal technology for each service
- **Fault Isolation**: Service failures don't cascade to entire system
- **Team Autonomy**: Different teams can own different services
- **Deployment Independence**: Deploy and update services independently
- **Performance Optimization**: Optimize each service for its specific workload

### Negative
- **Network Complexity**: Inter-service communication adds latency and failure points
- **Operational Overhead**: Multiple services to deploy, monitor, and maintain
- **Data Consistency**: Managing transactions across service boundaries
- **Testing Complexity**: Integration testing across multiple services
- **Debugging Difficulty**: Tracing requests across multiple services

## Implementation

### Service Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │────│  Load Balancer  │────│     Traefik     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │
          ├─── Core API Services (FastAPI)
          │    ├── KPI Service (:8001)
          │    ├── Sites Service (:8002) 
          │    ├── NOC Service (:8003)
          │    ├── BI Service (:8004)
          │    └── Notifications Service (:8005)
          │
          ├─── Background Workers (AsyncIO)
          │    ├── Data Ingestor
          │    ├── KPI Calculator
          │    ├── AIOps Engine
          │    └── BI Engine
          │
          └─── Supporting Services
               ├── WebSocket Service (:8006)
               └── File Service (:8007)
```

### FastAPI Service Template

```python
# Standard FastAPI service structure
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncpg
from contextlib import asynccontextmanager

# Service configuration
SERVICE_NAME = "kpi-service"
VERSION = "1.0.0"
PREFIX = "/api/v1"

# Database connection
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL)
    logger.info(f"{SERVICE_NAME} started")
    
    yield
    
    # Shutdown  
    await db_pool.close()
    logger.info(f"{SERVICE_NAME} shutdown")

# Create FastAPI app
app = FastAPI(
    title=f"Towerco AIOps {SERVICE_NAME}",
    description=f"Microservice for {SERVICE_NAME} functionality",
    version=VERSION,
    docs_url=f"{PREFIX}/docs",
    redoc_url=f"{PREFIX}/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "service": SERVICE_NAME,
        "status": "healthy",
        "version": VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }

# Include routers
from .routers import kpis
app.include_router(kpis.router, prefix=PREFIX)
```

### Service Communication Patterns

#### 1. Synchronous HTTP Communication
```python
# Service-to-service HTTP calls
class SitesServiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = httpx.AsyncClient(timeout=10.0)
    
    async def get_site_info(self, site_id: str) -> Dict:
        response = await self.session.get(
            f"{self.base_url}/sites/{site_id}",
            headers={"Authorization": f"Bearer {get_service_token()}"}
        )
        response.raise_for_status()
        return response.json()
    
    async def update_site_status(self, site_id: str, status: str):
        await self.session.put(
            f"{self.base_url}/sites/{site_id}/status",
            json={"status": status},
            headers={"Authorization": f"Bearer {get_service_token()}"}
        )
```

#### 2. Event-based Asynchronous Communication  
```python
# Publishing events for async communication
class EventPublisher:
    def __init__(self, kafka_client):
        self.kafka = kafka_client
    
    async def publish_kpi_calculated(self, kpi_data: Dict):
        event = {
            "event_type": "kpi.calculated",
            "tenant_id": kpi_data["tenant_id"],
            "site_id": kpi_data["site_id"], 
            "timestamp": datetime.utcnow().isoformat(),
            "payload": kpi_data
        }
        
        await self.kafka.send(
            "towerco.kpis.calculated",
            key=kpi_data["site_id"],
            value=json.dumps(event)
        )

# Consuming events
class EventConsumer:
    async def handle_kpi_calculated(self, event: Dict):
        # Update dashboards, trigger alerts, etc.
        await self.dashboard_service.update_kpi_display(event["payload"])
        await self.alert_service.check_thresholds(event["payload"])
```

### Service Discovery and Configuration

#### Environment-based Service Discovery
```python
# Service registry configuration
SERVICES = {
    "kpi-service": {
        "url": os.getenv("KPI_SERVICE_URL", "http://kpi-service:8001"),
        "health_endpoint": "/health",
        "timeout": 10
    },
    "sites-service": {
        "url": os.getenv("SITES_SERVICE_URL", "http://sites-service:8002"), 
        "health_endpoint": "/health",
        "timeout": 5
    },
    "noc-service": {
        "url": os.getenv("NOC_SERVICE_URL", "http://noc-service:8003"),
        "health_endpoint": "/health", 
        "timeout": 15
    }
}

class ServiceRegistry:
    def __init__(self):
        self.services = SERVICES
        self._health_cache = {}
    
    def get_service_url(self, service_name: str) -> str:
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        return self.services[service_name]["url"]
    
    async def check_service_health(self, service_name: str) -> bool:
        cache_key = f"health:{service_name}"
        
        # Check cache (30 second TTL)
        if cache_key in self._health_cache:
            cached_time, cached_result = self._health_cache[cache_key]
            if time.time() - cached_time < 30:
                return cached_result
        
        # Perform health check
        try:
            service_config = self.services[service_name]
            health_url = f"{service_config['url']}{service_config['health_endpoint']}"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    health_url, 
                    timeout=service_config["timeout"]
                )
                is_healthy = response.status_code == 200
                
            # Cache result
            self._health_cache[cache_key] = (time.time(), is_healthy)
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            return False
```

### Error Handling and Resilience

#### Circuit Breaker Pattern
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            # Success - reset circuit breaker
            self.failure_count = 0
            self.state = "closed"
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e
```

#### Retry with Exponential Backoff
```python
async def retry_with_backoff(
    func, 
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
):
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            delay = min(
                base_delay * (exponential_base ** attempt),
                max_delay
            )
            
            logger.warning(
                f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
            )
            await asyncio.sleep(delay)
```

### Service Monitoring and Observability

#### Structured Logging
```python
import structlog

logger = structlog.get_logger()

@app.middleware("http")  
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(
        "request_started",
        method=request.method,
        url=str(request.url),
        user_agent=request.headers.get("user-agent"),
        correlation_id=request.headers.get("x-correlation-id")
    )
    
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(
        "request_completed",
        status_code=response.status_code,
        duration_ms=duration * 1000,
        correlation_id=request.headers.get("x-correlation-id")
    )
    
    return response
```

#### Metrics Collection
```python
from prometheus_client import Counter, Histogram, generate_latest

# Service metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code', 'service']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint', 'service']
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
        service=SERVICE_NAME
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path, 
        service=SERVICE_NAME
    ).observe(time.time() - start_time)
    
    return response

@app.get("/metrics")
async def get_metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
```

### Deployment Strategy

#### Docker Configuration
```dockerfile
# Dockerfile for FastAPI services
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose Service Definition
```yaml
# Service definition in docker-compose.yml
  kpi-service:
    build: 
      context: ./backend
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://towerco:secure_password@timescaledb:5432/towerco_aiops
      - REDIS_URL=redis://redis:6379
      - SERVICE_NAME=kpi-service
      - LOG_LEVEL=INFO
    ports:
      - "8001:8000"
    depends_on:
      - timescaledb
      - redis
    networks:
      - towerco-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Technology Justification

### Why FastAPI?
1. **High Performance**: Comparable to NodeJS and Go
2. **Modern Python**: Full type hints and async/await support
3. **Automatic Documentation**: OpenAPI/Swagger generation
4. **Data Validation**: Pydantic models with automatic validation
5. **Developer Experience**: Excellent tooling and IDE support
6. **Community**: Large ecosystem and active development

### Service Granularity Principles
1. **Business Capability**: Each service owns a complete business function
2. **Data Ownership**: Services own their data and expose it via APIs
3. **Team Boundaries**: Services can be developed by independent teams
4. **Scaling Requirements**: Services with different scaling patterns are separated
5. **Technology Fit**: Services can use optimal technology for their domain

## Migration Strategy

### Phase 1: Core Services (Completed)
- ✅ API Gateway with Traefik
- ✅ Core business services (KPI, Sites, NOC, BI)
- ✅ Service communication patterns

### Phase 2: Background Services (Completed) 
- ✅ Background worker services
- ✅ Event-driven communication
- ✅ Service discovery and health checks

### Phase 3: Advanced Features (Completed)
- ✅ Circuit breakers and resilience patterns
- ✅ Comprehensive monitoring and observability  
- ✅ Performance optimization

## Review Date
January 2025

## References
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Microservices Patterns](https://microservices.io/patterns/)
- [Building Microservices](https://www.oreilly.com/library/view/building-microservices/9781491950340/)