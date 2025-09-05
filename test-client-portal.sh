#!/bin/bash

echo "Testing Client Portal Build..."
echo

echo "1. Installing dependencies..."
cd client-portal
npm install

echo
echo "2. Running TypeScript check..."
npx tsc --noEmit

echo
echo "3. Building the project..."
npm run build

echo
echo "========================================"
echo "Client Portal Build Test Complete!"
echo "========================================"
echo
