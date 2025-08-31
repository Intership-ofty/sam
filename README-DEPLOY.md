# SAM: Microservices-ready layout

This restructures the project to run as multiple containers and prepares Kubernetes deployment.

## What changed

- **Split images** for: `backend`, `workers`, `client-portal`, `frontend` (+ optional `observability`).
- **Removed central `docker-compose.yml`** in favor of modular files under `deploy/compose/`.
- **Added Kubernetes manifests** with Kustomize overlays (`deploy/k8s/`).
- **Added a sample Helm chart** for `backend` (`deploy/helm/backend`). Duplicate per service as needed.
- **Added Makefile** with build/push/compose/k8s shortcuts.

## Local Development with Docker Compose

This section guides you on how to set up and run the SAM application locally using Docker Compose.

### 1. Environment Setup

First, copy the common environment variables file:

```bash
cp deploy/compose/common.env deploy/compose/.env   # adjust if needed
```

### 2. Starting the Services

To build and start all application services, navigate to the root directory of the project and execute the following command:

```bash
docker compose -f deploy/compose/compose.backend.yml \
               -f deploy/compose/compose.workers.yml \
               -f deploy/compose/compose.client-portal.yml \
               -f deploy/compose/compose.frontend.yml \
               -f deploy/compose/compose.observability.yml \
               up --build
```

This command will:
- Build the Docker images for `backend`, `workers`, `client-portal`, and `frontend`.
- Start all defined services, including `backend`, `workers`, `client-portal`, `frontend`, `prometheus`, `grafana`, `redis`, `zookeeper`, `kafka`, and `postgres`.

To stop all running services, use:

```bash
docker compose -f deploy/compose/compose.backend.yml \
               -f deploy/compose/compose.workers.yml \
               -f deploy/compose/compose.client-portal.yml \
               -f deploy/compose/compose.frontend.yml \
               -f deploy/compose/compose.observability.yml \
               down
```

### 3. Accessing the Application Frontends

The SAM application provides two main frontend interfaces: the `client-portal` and the `frontend` (main application).

- **Client Portal:** Accessible at `http://localhost:5173`. Upon access, you will be redirected to the `/dashboard` route.
- **Main Frontend Application:** Accessible at `http://localhost:8080`. Upon access, you will be redirected to the `/dashboard.html` page.

### 4. Accessing Other Services

- **Backend API:** `http://localhost:8000`
- **Grafana:** `http://localhost:3001` (admin/admin on first start)
- **Prometheus:** `http://localhost:9090`

### 5. Troubleshooting Common Issues

- **`port is already allocated` error:**
  This error indicates that a port required by one of the Docker services (e.g., Grafana on `3001`) is already in use on your host machine.
  - **Solution:** Identify and stop the conflicting application, or modify the port mapping in the respective `deploy/compose/*.yml` file to use an available port.

- **`ERR_CONNECTION_REFUSED` or "This site is inaccessible":**
  - Ensure all Docker containers are running. Use `docker ps` in your terminal to check their status.
  - Verify that no other applications on your host machine are using the same ports as the Docker services.
  - Check the Docker container logs for specific error messages using `docker logs <container_id_or_name>`.

- **"Welcome to nginx!" page instead of the application:**
  This indicates that Nginx is running but cannot find the application's static files.
  - **Solution:** Verify the `COPY` commands in the respective `Dockerfile` (e.g., `frontend/Dockerfile`, `client-portal/Dockerfile`) ensure that the built application files are correctly copied to `/usr/share/nginx/html`. Also, check the Nginx configuration files (`deploy/nginx/frontend.conf`, `deploy/nginx/client-portal.conf`) for correct `root` and `try_files` directives.

- **Redirection Issues:**
  - Ensure the Nginx configuration files (`deploy/nginx/frontend.conf` and `deploy/nginx/client-portal.conf`) are correctly configured with `try_files` and `return` directives pointing to the correct entry points of your applications.

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