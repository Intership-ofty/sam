#!/bin/bash

echo "Starting Towerco AIOps Data Workers..."
echo

echo "1. Starting Backend Services..."
cd deploy/compose
docker compose -f compose.backend.yml up -d &

echo
echo "2. Waiting for backend to be ready..."
sleep 30

echo
echo "3. Testing data connections..."
cd ../../scripts
python3 test_data_connections.py all

echo
echo "4. Starting Data Workers..."
python3 start_data_workers.py &

echo
echo "========================================"
echo "Data Workers Started!"
echo "========================================"
echo
echo "Backend API: http://localhost:8000"
echo "Kafka: localhost:9092"
echo
echo "Press Ctrl+C to stop all services"
echo

# Wait for user input
read -p "Press Enter to stop all services..."

# Stop all services
echo "Stopping services..."
docker compose -f deploy/compose/compose.backend.yml down
pkill -f "start_data_workers.py"
