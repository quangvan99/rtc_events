#!/bin/bash
# Start the Face Recognition WebRTC Stream containers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Face Stream containers..."

# Check if containers are already running
if docker compose ps --status running 2>/dev/null | grep -q "face-stream"; then
    echo "Container is already running. Use ./rebuild.sh to restart."
    exit 0
fi

# Start containers in detached mode
docker compose up -d

echo "Container started successfully!"
echo "Use ./log.sh to view logs"
echo "Use ./stop.sh to stop the container"
