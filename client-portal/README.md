# Client Portal (Towerco AIOps)
Minimal React + Vite + TypeScript SPA ready to serve behind Traefik.

## Dev
```bash
npm ci
npm run dev
# http://localhost:5173  (calls to /api are proxied to http://localhost/api)
```

## Build
```bash
npm run build
```

## Docker
```bash
docker build -t client-portal:latest .
docker run --rm -p 8080:80 client-portal:latest
# open http://localhost:8080
```

## Compose snippet (Traefik static port 80)
```yaml
services:
  client-portal:
    build: ./client-portal
    depends_on: [ backend ]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.client.rule=PathPrefix(`/`)"
      - "traefik.http.routers.client.entrypoints=web"
      - "traefik.http.services.client.loadbalancer.server.port=80"
```

If you prefer a dedicated host:
```yaml
      - "traefik.http.routers.client.rule=Host(`client.localhost`)"
```
and add `127.0.0.1 client.localhost` to your hosts file.
