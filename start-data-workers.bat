@echo off
echo Starting Towerco AIOps Data Workers...
echo.

echo 1. Starting Backend Services...
cd deploy\compose
start "Backend Services" cmd /k "docker compose -f compose.backend.yml up"

echo.
echo 2. Waiting for backend to be ready...
timeout /t 30 /nobreak

echo.
echo 3. Testing data connections...
cd ..\..\scripts
python test_data_connections.py all

echo.
echo 4. Starting Data Workers...
python start_data_workers.py

echo.
echo ========================================
echo Data Workers Started!
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo Kafka: localhost:9092
echo.
echo Press any key to exit...
pause > nul
