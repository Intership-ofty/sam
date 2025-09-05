@echo off
echo Starting Towerco AIOps Platform Frontend...
echo.

echo 1. Starting Backend Services...
cd deploy\compose
start "Backend Services" cmd /k "docker compose -f compose.backend.yml up"

echo.
echo 2. Waiting for backend to be ready...
timeout /t 30 /nobreak

echo.
echo 3. Starting Frontend Servers...
cd ..\..\frontend

echo Starting HTTP server on port 8090...
start "Frontend Server" cmd /k "python -m http.server 8090"

echo.
echo 4. Starting Client Portal...
cd ..\client-portal
start "Client Portal" cmd /k "npm run dev"

echo.
echo ========================================
echo Frontend Services Started!
echo ========================================
echo.
echo Frontend HTML: http://localhost:8090
echo Client Portal: http://localhost:5173
echo Backend API: http://localhost:8000
echo Keycloak: http://localhost:8080
echo.
echo Press any key to exit...
pause > nul
