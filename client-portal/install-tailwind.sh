#!/bin/bash

echo "Installing Tailwind CSS dependencies..."

# Install Tailwind CSS and dependencies
npm install -D tailwindcss@latest postcss@latest autoprefixer@latest

# Initialize Tailwind config
npx tailwindcss init -p

echo "Tailwind CSS installation completed!"
echo "You can now run 'npm run build' to build the project with Tailwind CSS."
