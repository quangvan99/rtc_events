#!/bin/bash
# Extract face features from images in face-stream container

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

docker compose exec face-stream python3 /app/extract/extract.py "$@"
