#!/bin/bash
# Stop the Face Recognition WebRTC Stream containers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping Face Stream containers..."

docker compose down

echo "Containers stopped successfully!"
