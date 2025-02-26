#!/bin/bash

# Kill any existing instances
pkill -f "python.*app_sqlite.py"

# Start the server
cd "$(dirname "$0")"

# Get the IP address
IP_ADDRESS=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)

echo "Starting server..."
echo "Local URL: http://localhost:8000"
echo "Network URL: http://$IP_ADDRESS:8000"

nohup /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 app_sqlite.py --host 0.0.0.0 > logs/app.log 2> logs/error.log &

# Wait a few seconds
sleep 3

# Check if the server is running
if curl -s -I http://localhost:8000 > /dev/null 2>&1; then
    echo "Server started successfully!"
    echo "Share this URL with others on your network: http://$IP_ADDRESS:8000"
else
    echo "Failed to start server. Check logs/error.log for details."
    exit 1
fi
