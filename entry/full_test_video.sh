#!/bin/bash
# Full test with video recording verification
# Run outside Docker - this script manages the Docker container

DOCKER_CONTAINER="face-stream"
PROJECT_DIR="/home/mq/disk2T/quangnv/face"
MAX_RETRIES=30
PORT=8083

echo "=== Full Test with Video Recording ==="

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER}$"; then
    echo "ERROR: Container '$DOCKER_CONTAINER' is not running!"
    exit 1
fi

# Kill any existing pipeline processes inside Docker
echo "=== Cleaning up existing processes ==="
docker exec -w "$PROJECT_DIR" "$DOCKER_CONTAINER" bash -c "
    pkill -9 -f 'python.*test_multi' 2>/dev/null || true
    pkill -9 -f 'entry/test_multi_branch' 2>/dev/null || true
    sleep 2
" 2>/dev/null

# Kill any process using port 8083 on host
if lsof -i :$PORT 2>/dev/null; then
    echo "=== Killing process on port $PORT ==="
    lsof -ti :$PORT | xargs -r kill -9 2>/dev/null || true
    sleep 2
fi

# Remove old output files
rm -f $PROJECT_DIR/data/face/output_*.avi 2>/dev/null || true

# Function to check if server is healthy
check_health() {
    curl -s --connect-timeout 3 http://localhost:$PORT/api/health 2>/dev/null | grep -q "healthy"
}

# Function to start pipeline
start_pipeline() {
    docker exec -d -w "$PROJECT_DIR" "$DOCKER_CONTAINER" bash -c "
        cd $PROJECT_DIR
        nohup python3 entry/test_multi_branch_video.py > /tmp/full_test.log 2>&1 &
        echo \$!
    "
}

# Function to get pipeline PID
get_pipeline_pid() {
    docker exec -w "$PROJECT_DIR" "$DOCKER_CONTAINER" pgrep -f 'python.*test_multi' 2>/dev/null | head -1
}

# Start pipeline with retry logic
echo "=== Starting pipeline with retry logic ==="
for attempt in $(seq 1 $MAX_RETRIES); do
    echo "Attempt $attempt/$MAX_RETRIES..."

    # Kill any existing pipeline
    docker exec -w "$PROJECT_DIR" "$DOCKER_CONTAINER" bash -c "pkill -9 -f 'python.*test_multi' 2>/dev/null || true"
    sleep 3

    # Start fresh pipeline
    start_pipeline
    echo "  Waiting for pipeline to initialize..."

    # Wait for server to be ready (up to 30 seconds)
    for i in $(seq 1 30); do
        if check_health; then
            echo "  Server ready after ${i}s"
            break
        fi
        sleep 1
    done

    if ! check_health; then
        echo "  Pipeline failed to start, checking logs..."
        docker exec "$DOCKER_CONTAINER" tail -20 /tmp/full_test.log 2>/dev/null | grep -i error || true
        continue
    fi

    # Try to add first camera
    echo "  Adding cam1..."
    curl -s -X POST http://localhost:$PORT/api/cameras \
        -H "Content-Type: application/json" \
        -d '{"camera_id": "cam1", "uri": "rtsp://192.168.6.14:8554/testface", "branches": ["recognition", "detection"]}' > /dev/null

    sleep 5

    if check_health; then
        echo "  SUCCESS! Pipeline stable after adding cam1"
        break
    else
        echo "  Crashed after adding cam1, retrying..."
    fi
done

if ! check_health; then
    echo "ERROR: Failed to stabilize pipeline after $MAX_RETRIES attempts"
    echo "=== Last 30 lines of log ==="
    docker exec "$DOCKER_CONTAINER" tail -30 /tmp/full_test.log 2>/dev/null || echo "Could not read log"
    exit 1
fi

echo ""
echo "=== Pipeline running, continuing with test steps ==="

echo ""
echo "=== Step 2: Add cam2 ==="
curl -s -X POST http://localhost:$PORT/api/cameras -H "Content-Type: application/json" \
     -d '{"camera_id": "cam2", "uri": "rtsp://192.168.6.14:8554/test", "branches": ["recognition", "detection"]}'
echo ""
sleep 5

echo "=== Step 3: Remove cam2 from detection ==="
curl -s -X DELETE http://localhost:$PORT/api/cameras/cam2/branches/detection
echo ""
sleep 5

echo "=== Step 4: Add cam3 ==="
curl -s -X POST http://localhost:$PORT/api/cameras -H "Content-Type: application/json" \
     -d '{"camera_id": "cam3", "uri": "rtsp://192.168.6.14:8554/test", "branches": ["recognition", "detection"]}'
echo ""
sleep 5

echo ""
echo "=== Camera Status ==="
curl -s http://localhost:$PORT/api/cameras | python3 -m json.tool 2>/dev/null || curl -s http://localhost:$PORT/api/cameras

echo ""
echo "=== Branch Status ==="
curl -s http://localhost:$PORT/api/branches | python3 -m json.tool 2>/dev/null || curl -s http://localhost:$PORT/api/branches

echo ""
echo "=== Wait 10s for video processing ==="
sleep 10

echo "=== Video Files ==="
ls -lh "$PROJECT_DIR/data/face/output_*.avi" 2>/dev/null || echo "No output files"

echo ""
echo "=== Log (last 30 lines) ==="
docker exec "$DOCKER_CONTAINER" cat /tmp/full_test.log 2>/dev/null | grep -E "SourceIDMapper|Pad added|linked|FPS|CONFIRMED|ERROR" | tail -30

echo ""
echo "=== Pipeline Process ==="
docker exec -w "$PROJECT_DIR" "$DOCKER_CONTAINER" ps aux | grep python || echo "No Python processes"

echo ""
echo "=== Remove all cameras (keep pipeline running) ==="
curl -s -X POST http://localhost:$PORT/api/pipeline/kill
echo ""

echo ""
echo "=== Stop the entire pipeline (shutdown application) ==="
curl -s -X POST http://localhost:$PORT/api/pipeline/stop
echo ""

echo ""
echo "=== Test Complete ==="
