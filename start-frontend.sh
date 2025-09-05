#!/bin/bash

echo "Starting Towerco AIOps Platform Frontend..."
echo

echo "1. Starting Backend Services..."
cd deploy/compose
docker compose -f compose.backend.yml up -d &

echo
echo "2. Waiting for backend to be ready..."
sleep 30

echo
echo "3. Starting Frontend Servers..."
cd ../../frontend

echo "Starting HTTP server on port 8090..."
python3 -m http.server 8090 &

echo
echo "4. Starting Client Portal..."
cd ../client-portal
npm run dev &

echo
echo "========================================"
echo "Frontend Services Started!"
echo "========================================"
echo
echo "Frontend HTML: http://localhost:8090"
echo "Client Portal: http://localhost:5173"
echo "Backend API: http://localhost:8000"
echo "Keycloak: http://localhost:8080"
echo
echo "Press Ctrl+C to stop all services"
echo

# Wait for user input
read -p "Press Enter to stop all services..."

# Stop all services
echo "Stopping services..."
docker compose -f deploy/compose/compose.backend.yml down
pkill -f "python3 -m http.server"
pkill -f "npm run dev"
