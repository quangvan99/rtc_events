#!/bin/bash
# Full test with video recording verification
# Run inside Docker container: face-stream
#
# REQUIREMENTS:
#   - Docker container must be started with --device /dev/video0:/dev/video0 (or --privileged)
#   - NVIDIA GPU and drivers properly configured
#
# Example docker run:
#   docker run --gpus all --privileged -v /home/mq/disk2T/quangnv/face:/home/mq/disk2T/quangnv/face ...

DOCKER_CONTAINER="face-stream"
PROJECT_DIR="/home/mq/disk2T/quangnv/face"

echo "=== Full Test with Video Recording ==="

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER}$"; then
    echo "ERROR: Container '$DOCKER_CONTAINER' is not running!"
    exit 1
fi

# Execute all commands inside Docker container
docker exec -w "$PROJECT_DIR" "$DOCKER_CONTAINER" bash -c "
    echo '=== Clearing cache ==='

    # Kill existing processes
    pkill -9 -f 'python.*test_multi' 2>/dev/null || true
    sleep 2

    # Clear Python cache only (NOT GStreamer cache - it's needed for registry)
    find . -name '*.pyc' -delete 2>/dev/null || true
    find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
    rm -rf ~/.cache/gstreamer-1.0/registry.x86_64.bin
    echo '- Python cache: cleared'

    # Clear output files
    rm -f ./data/output_*.avi
    echo '- Output files: cleared'
"
