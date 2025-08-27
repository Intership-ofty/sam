# SAM: Microservices-ready layout

This restructures the project to run as multiple containers and prepares Kubernetes deployment.

## What changed

- **Split images** for: `backend`, `workers`, `client-portal`, `frontend` (+ optional `observability`).
- **Removed central `docker-compose.yml`** in favor of modular files under `deploy/compose/`.
- **Added Kubernetes manifests** with Kustomize overlays (`deploy/k8s/`).
- **Added a sample Helm chart** for `backend` (`deploy/helm/backend`). Duplicate per service as needed.
- **Added Makefile** with build/push/compose/k8s shortcuts.

## Local with Compose

```bash
cp deploy/compose/common.env deploy/compose/.env   # adjust if needed
make compose-up
```

```bash
# Depuis la racine du projet
docker compose -f deploy/compose/compose.backend.yml `
               -f deploy/compose/compose.workers.yml `
               -f deploy/compose/compose.client-portal.yml `
               -f deploy/compose/compose.frontend.yml `
               -f deploy/compose/compose.observability.yml `
               up --build
```

Open:
- Backend: http://localhost:8000
- Client-portal: http://localhost:5173
- Frontend: http://localhost:8080
- Grafana: http://localhost:3000 (admin/admin on first start)
- Prometheus: http://localhost:9090

## Build & Push

```bash
REGISTRY=ghcr.io/you TAG=0.1.0 make build push
```

## Kubernetes with Kustomize

```bash
# Dev namespace
kubectl apply -k deploy/k8s/overlays/dev
# Prod namespace
kubectl apply -k deploy/k8s/overlays/prod
```

## Kubernetes with Helm (example, backend)

```bash
cd deploy/helm/backend
helm install backend . --namespace sam --create-namespace   --set image.repository=$REGISTRY/sam-backend --set image.tag=$TAG
```

> For observability on Kubernetes, prefer official charts:
> - Prometheus: `helm repo add prometheus-community https://prometheus-community.github.io/helm-charts`
> - Grafana: `helm repo add grafana https://grafana.github.io/helm-charts`

## Notes

- The `client-portal` Dockerfile creates a minimal `package.json` if missing; replace it with the real one when available.
- `workers/Dockerfile` lets you choose the worker via `WORKER=ingestor` (etc.). Scale by creating multiple Deployments with different `WORKER` env values.
- Move any secrets to a proper secret store (Kubernetes Secrets, External Secrets, etc.).
