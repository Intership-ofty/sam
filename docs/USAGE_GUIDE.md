# SAM Project: Comprehensive Usage Guide

This document provides a comprehensive guide to understanding, setting up, running, and using the SAM (Service Assurance Management) application.

## 1. Project Overview

The SAM project is a microservices-based application designed for monitoring and managing various aspects of a telecommunications tower infrastructure. It provides functionalities for:
- **Data Ingestion:** Collecting data from various sources (IoT, network, ITSM).
- **Anomaly Detection:** Identifying unusual patterns in collected data.
- **KPI Monitoring:** Tracking key performance indicators.
- **Incident Management:** Handling and correlating incidents.
- **SLA Management:** Monitoring Service Level Agreement compliance.
- **Business Intelligence:** Providing insights and analytics.
- **Real-time Notifications:** Alerting users about critical events.

The application is built with a modular architecture, leveraging technologies like:
- **Backend:** Python (FastAPI)
- **Frontends:** React (TypeScript)
- **Messaging:** Kafka
- **Database:** PostgreSQL (TimescaleDB for time-series data)
- **Monitoring & Observability:** Prometheus, Grafana, Loki, Tempo
- **Containerization:** Docker
- **Orchestration:** Docker Compose, Kubernetes (Kustomize, Helm)

## 2. Setting Up the Development Environment

To set up your local development environment, you need to have the following installed:
- **Git:** For cloning the repository.
- **Docker Desktop:** Includes Docker Engine and Docker Compose.
- **Node.js & npm (or yarn):** For frontend development (if you plan to modify frontend code).
- **Python & pip:** For backend and worker development (if you plan to modify backend/worker code).

### 2.1 Cloning the Repository

First, clone the SAM project repository to your local machine:

```bash
git clone <repository_url>
cd sam
```

### 2.2 Environment Variables

Copy the common environment variables file. You can adjust these variables if needed (e.g., database credentials, API base URLs).

```bash
cp deploy/compose/common.env deploy/compose/.env
```

## 3. Running the Application

The easiest way to run the entire SAM application locally is by using Docker Compose.

### 3.1 Starting All Services

Navigate to the root directory of the `sam` project and execute the following command:

```bash
docker compose -f deploy/compose/compose.backend.yml \
               -f deploy/compose/compose.workers.yml \
               -f deploy/compose/compose.client-portal.yml \
               -f deploy/compose/compose.frontend.yml \
               -f deploy/compose/compose.observability.yml \
               up --build
```

This command will:
- Build the Docker images for all services (`backend`, `workers`, `client-portal`, `frontend`).
- Start all defined services, including databases, messaging queues, and monitoring tools.

### 3.2 Stopping All Services

To stop all running services and remove their containers, networks, and volumes (except named volumes), use:

```bash
docker compose -f deploy/compose/compose.backend.yml \
               -f deploy/compose/compose.workers.yml \
               -f deploy/compose/compose.client-portal.yml \
               -f deploy/compose/compose.frontend.yml \
               -f deploy/compose/compose.observability.yml \
               down
```

## 4. Using the Application Services

Once all services are up and running, you can access them via your web browser or API clients.

### 4.1 Frontend Applications

The SAM project provides two distinct frontend applications:

#### 4.1.1 Client Portal

The `client-portal` is designed for end-users to view their specific site data, SLA compliance, and notifications.

- **Access URL:** `http://localhost:5173`
- **Initial Redirection:** Upon accessing the URL, you will be automatically redirected to the `/dashboard` route within the client portal.

#### 4.1.2 Main Frontend Application

The main `frontend` application provides a comprehensive overview and management interface for administrators and operators.

- **Access URL:** `http://localhost:8080`
- **Initial Redirection:** Upon accessing the URL, you will be automatically redirected to the `/dashboard.html` page.

### 4.2 Backend API

The backend API serves data to the frontends and can also be accessed directly for development or integration purposes.

- **Base URL:** `http://localhost:8000`
- **API Documentation (Swagger UI):** Typically available at `http://localhost:8000/docs`
- **Redoc Documentation:** Typically available at `http://localhost:8000/redoc`

### 4.3 Observability Tools

- **Grafana:** A powerful dashboarding and visualization tool for your metrics and logs.
  - **Access URL:** `http://localhost:3001`
  - **Default Credentials:** `admin` / `admin` (you will be prompted to change this on first login).
- **Prometheus:** A monitoring system with a flexible query language (PromQL).
  - **Access URL:** `http://localhost:9090`

## 5. Troubleshooting Common Issues

This section provides solutions to common problems you might encounter.

### 5.1 Docker Compose Issues

-   **`port is already allocated` error:**
    This error occurs when a port required by one of the Docker services (e.g., Grafana on `3001`, Frontend on `8080`, Client Portal on `5173`) is already in use by another application on your host machine.
    -   **Solution:**
        1.  Identify and stop the conflicting application.
        2.  Alternatively, modify the port mapping in the respective `deploy/compose/*.yml` file to use an available port. For example, change `- "3001:3000"` to `- "3002:3000"` for Grafana.

-   **`ERR_CONNECTION_REFUSED` or "This site is inaccessible":**
    This indicates that your browser could not connect to the web server.
    -   **Solution:**
        1.  **Check Docker Container Status:** Ensure all Docker containers are running. Open your terminal or command prompt and run `docker ps`. All expected containers should have a `Status` of `Up`.
        2.  **Verify Port Availability:** Confirm that no other applications on your host machine are using the same ports as the Docker services (e.g., `8080`, `5173`, `8000`, `3001`, `9090`).
        3.  **Review Container Logs:** Check the Docker container logs for specific error messages during startup or runtime. Use `docker logs <container_id_or_name>` (e.g., `docker logs compose-client-portal-1`).

### 5.2 Frontend Application Issues

-   **"Welcome to nginx!" page instead of the application:**
    This means the Nginx web server inside the container is running, but it cannot find the application's static files (HTML, CSS, JavaScript).
    -   **Solution:**
        1.  **Verify `COPY` Commands in Dockerfile:** Ensure that the `COPY` commands in the respective `Dockerfile` (e.g., `frontend/Dockerfile`, `client-portal/Dockerfile`) are correctly copying the built application files to the Nginx web root (`/usr/share/nginx/html`).
        2.  **Check Nginx Configuration:** Review the Nginx configuration files (`deploy/nginx/frontend.conf`, `deploy/nginx/client-portal.conf`) for correct `root` and `try_files` directives. The `root` directive should point to `/usr/share/nginx/html`, and `try_files` should correctly point to your application's main HTML file (e.g., `index.html` or `dashboard.html`).

-   **Redirection Issues:**
    If you are not being redirected to the expected `/dashboard` or `/dashboard.html` routes, or if internal routing within the single-page application is not working:
    -   **Solution:**
        1.  **Nginx Configuration:** Double-check the `location /` block in your Nginx configuration files (`frontend.conf`, `client-portal.conf`). Ensure `try_files $uri $uri/ /index.html;` (or `/dashboard.html`) is correctly set for SPA routing.
        2.  **Application Routing:** Verify that the React application's internal routing (e.g., using `react-router-dom`) is correctly configured to handle the `/dashboard` route.

## 6. Development Workflow

This section outlines the typical development workflow for contributing to the SAM project.

### 6.1 Backend Development (Python/FastAPI)

1.  **Install Dependencies:**
    ```bash
    cd backend
    pip install -r requirements.txt
    ```
2.  **Run Backend Locally:**
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    (Ensure your database and other dependencies are running via Docker Compose or locally).
3.  **Testing:**
    ```bash
    pytest
    ```

### 6.2 Frontend Development (React/TypeScript)

1.  **Install Dependencies:**
    ```bash
    cd client-portal # or frontend
    npm install # or yarn install
    ```
2.  **Run Frontend Locally (Development Server):**
    ```bash
    npm run dev # or yarn dev
    ```
    This will typically start a development server on `http://localhost:5173` (for client-portal) or another port.
3.  **Building for Production:**
    ```bash
    npm run build # or yarn build
    ```

### 6.3 Worker Development (Python)

1.  **Install Dependencies:**
    ```bash
    cd workers
    pip install -r requirements.txt
    ```
2.  **Run Specific Worker Locally:**
    ```bash
    python -m workers.<worker_name> # e.g., python -m workers.data_ingestor
    ```
    (Ensure Kafka, database, and other dependencies are running).

## 7. Contributing

We welcome contributions to the SAM project! Please refer to the `CONTRIBUTING.md` file (if available) for guidelines on submitting issues, pull requests, and coding standards.

---
