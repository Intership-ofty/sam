@echo off
echo Testing Client Portal Build...
echo.

echo 1. Installing dependencies...
cd client-portal
call npm install

echo.
echo 2. Running TypeScript check...
call npx tsc --noEmit

echo.
echo 3. Building the project...
call npm run build

echo.
echo ========================================
echo Client Portal Build Test Complete!
echo ========================================
echo.
pause
