# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records for the Towerco AIOps platform. Each ADR documents a significant architectural decision, the context that led to it, the decision made, and the consequences.

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](ADR-001-event-first-architecture.md) | Event-First Architecture with Kafka | Accepted | 2024-01-15 |
| [ADR-002](ADR-002-polystore-data-architecture.md) | Polystore Data Architecture | Accepted | 2024-01-15 |
| [ADR-003](ADR-003-multi-tenant-saas-architecture.md) | Multi-tenant SaaS Architecture | Accepted | 2024-01-15 |
| [ADR-004](ADR-004-microservices-with-fastapi.md) | Microservices Architecture with FastAPI | Accepted | 2024-01-15 |
| [ADR-005](ADR-005-react-typescript-frontend.md) | React TypeScript for Frontend | Accepted | 2024-01-15 |
| [ADR-006](ADR-006-ml-frameworks-selection.md) | Machine Learning Frameworks Selection | Accepted | 2024-01-15 |
| [ADR-007](ADR-007-observability-stack.md) | Observability Stack Selection | Accepted | 2024-01-15 |
| [ADR-008](ADR-008-authentication-oauth2-keycloak.md) | OAuth2 with Keycloak for Authentication | Accepted | 2024-01-15 |
| [ADR-009](ADR-009-docker-deployment-strategy.md) | Docker-Only Deployment Strategy | Accepted | 2024-01-15 |
| [ADR-010](ADR-010-real-time-notifications.md) | WebSocket Real-time Notifications | Accepted | 2024-01-15 |

## ADR Template

When creating new ADRs, use the following template:

```markdown
# ADR-XXX: [Title]

## Status
[Proposed | Accepted | Rejected | Deprecated | Superseded by ADR-XXX]

## Context
[Description of the problem and context that led to this decision]

## Decision
[The decision that was made]

## Alternatives Considered
[Other options that were considered]

## Consequences
[Positive and negative consequences of the decision]

## Implementation
[How the decision will be implemented]

## Review Date
[When this decision should be reviewed]
```

## Decision Process

1. **Identify** architectural decisions that need documentation
2. **Research** alternatives and gather context
3. **Discuss** with stakeholders and team members
4. **Document** the decision using the ADR template
5. **Review** and get approval from architecture committee
6. **Implement** the decision
7. **Monitor** consequences and update as needed

## References

- [Architecture Decision Records](https://adr.github.io/)
- [ADR Tools](https://github.com/npryce/adr-tools)
- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)


## Recommandations

- to start docker -> `docker compose up -d --scale data-ingestor=2`
