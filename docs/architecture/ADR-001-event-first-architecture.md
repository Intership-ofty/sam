# ADR-001: Event-First Architecture with Kafka

## Status
Accepted

## Context
The Towerco AIOps platform needs to handle high-volume, real-time data from thousands of telecom sites with strict latency requirements. Traditional request-response architectures would create bottlenecks and single points of failure. We need an architecture that can:

- Handle 10,000+ events per second per site
- Provide real-time processing with sub-second latency
- Support horizontal scaling
- Enable event replay and audit capabilities
- Allow multiple consumers of the same events
- Provide fault tolerance and durability

## Decision
We will adopt an **Event-First Architecture** using **Apache Kafka** (specifically Redpanda for better performance) as the central event streaming platform.

Key architectural principles:
1. All significant state changes are captured as events
2. Events are the primary means of communication between services
3. Event streams are the source of truth
4. Services react to events rather than making direct API calls
5. Event sourcing patterns for critical business entities

## Alternatives Considered

### 1. Traditional REST API Architecture
- **Pros**: Simple, well-understood, easy debugging
- **Cons**: Poor scalability, tight coupling, no event history, synchronous bottlenecks

### 2. Message Queues (RabbitMQ, Amazon SQS)
- **Pros**: Good for simple async messaging, mature tooling
- **Cons**: Limited throughput, no event replay, not designed for streaming

### 3. Database-based Events (PostgreSQL LISTEN/NOTIFY)
- **Pros**: Simple integration, ACID guarantees
- **Cons**: Limited scalability, single point of failure, no horizontal scaling

### 4. Apache Pulsar
- **Pros**: Multi-tenancy, geo-replication, good performance
- **Cons**: Complex setup, less mature ecosystem, higher operational overhead

## Consequences

### Positive
- **High Throughput**: Can handle millions of events per day
- **Real-time Processing**: Sub-second event processing and correlation
- **Scalability**: Horizontal scaling of producers and consumers
- **Fault Tolerance**: Replication and partition tolerance
- **Event History**: Complete audit trail and replay capabilities
- **Loose Coupling**: Services communicate via events, reducing dependencies
- **Multiple Consumers**: Same events can power analytics, alerts, and dashboards

### Negative
- **Complexity**: More complex than traditional architectures
- **Eventual Consistency**: No strong consistency guarantees across services
- **Operational Overhead**: Requires Kafka/Redpanda cluster management
- **Learning Curve**: Team needs to understand event-driven patterns
- **Debugging**: More complex debugging across multiple event streams

## Implementation

### Event Schema Strategy
```python
# Base event schema
{
    "event_id": "uuid",
    "event_type": "string",
    "tenant_id": "string", 
    "site_id": "string",
    "timestamp": "datetime",
    "version": "string",
    "payload": "object",
    "metadata": "object"
}
```

### Topic Design
- **Site Events**: `towerco.sites.{site_id}.events`
- **KPI Events**: `towerco.kpis.{tenant_id}.metrics`
- **Alert Events**: `towerco.alerts.{tenant_id}.notifications`
- **Incident Events**: `towerco.incidents.{tenant_id}.lifecycle`

### Consumer Groups
- **AIOps Engine**: Processes events for anomaly detection and RCA
- **KPI Calculator**: Computes real-time metrics and aggregations
- **Notification Service**: Sends alerts and notifications
- **Analytics Service**: Stores events for historical analysis
- **Dashboard Service**: Updates real-time dashboards

### Event Processing Patterns
1. **Event Sourcing**: Critical entities (incidents, optimizations) store events
2. **CQRS**: Separate read/write models with event synchronization
3. **Saga Pattern**: Distributed transactions using event choreography
4. **Event Replay**: Ability to replay events for debugging and recovery

### Technology Stack
- **Event Broker**: Redpanda (Kafka-compatible with better performance)
- **Schema Registry**: Confluent Schema Registry for event schema management
- **Processing**: Python asyncio for lightweight event processing
- **Storage**: Events stored in Kafka topics + TimescaleDB for analytics

## Implementation Timeline

### Phase 1 (Completed)
- ✅ Kafka/Redpanda cluster setup
- ✅ Basic event schemas and topics
- ✅ Core event producers and consumers

### Phase 2 (Completed)  
- ✅ Event processing workers (KPI, AIOps, etc.)
- ✅ Event correlation and aggregation
- ✅ Real-time dashboard updates

### Phase 3 (Completed)
- ✅ Event sourcing for critical entities
- ✅ Event replay capabilities
- ✅ Advanced event processing patterns

## Monitoring and Observability
- Kafka/Redpanda metrics in Prometheus
- Event processing latency monitoring
- Consumer lag alerting
- Event schema evolution tracking
- Dead letter queue monitoring

## Review Date
January 2025

## References
- [Building Event-Driven Microservices](https://www.oreilly.com/library/view/building-event-driven-microservices/9781492057888/)
- [Redpanda Documentation](https://docs.redpanda.com/)
- [Event Sourcing Pattern](https://microservices.io/patterns/data/event-sourcing.html)