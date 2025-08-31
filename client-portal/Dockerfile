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
FROM node:20-alpine
WORKDIR /app
COPY --from=build /app/package.json ./package.json
# If package-lock.json exists, copy it
COPY --from=build /app/package-lock.json* ./
# We install only production dependencies
RUN npm install --production
COPY --from=build /app/dist ./dist
EXPOSE 3000
CMD ["npx", "serve", "-s", "dist", "-l", "3000"]
