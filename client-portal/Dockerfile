# Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY client-portal ./
# Ensure a package.json exists (in case the repo didn't include it) - you can replace this with your own
# If you already have package.json/lockfile, they will override these defaults.
RUN if [ ! -f package.json ]; then       echo '{ "name":"client-portal","version":"0.1.0","private":true, "type":"module", "scripts": { "dev": "vite", "build": "vite build", "preview": "vite preview --port 4173" }, "dependencies": { "react": "^18.2.0", "react-dom": "^18.2.0" }, "devDependencies": { "typescript": "^5.0.0", "vite": "^5.2.0", "@types/react": "^18.0.0", "@types/react-dom": "^18.0.0" } }' > package.json;     fi
RUN npm ci || npm i
RUN npm run build

# Runtime stage
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY deploy/nginx/default.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3   CMD wget -qO- http://localhost/ || exit 1
