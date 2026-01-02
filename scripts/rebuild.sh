#!/bin/bash
# Rebuild and restart the Face Recognition WebRTC Stream containers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Rebuilding and restarting Face Stream containers..."

# Stop existing containers
docker compose down

# Rebuild images (no cache if --no-cache flag is passed)
if [[ "$1" == "--no-cache" ]]; then
    echo "Building without cache..."
    docker compose build --no-cache
else
    echo "Building with cache..."
    docker compose build
fi

# Start containers
docker compose up -d

echo "Rebuild complete!"
echo "Use ./log.sh to view logs"
