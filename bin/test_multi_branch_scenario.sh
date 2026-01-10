#!/bin/bash
# Test script for multi-branch pipeline with camera add/remove operations

set -e

API_PORT=8083
BASE_URL="http://localhost:$API_PORT"

echo "=============================================="
echo "Multi-Branch Pipeline Test"
echo "=============================================="

# Kill any existing test processes
echo "[Setup] Killing any existing pipeline..."
curl -s -X POST "$BASE_URL/api/pipeline/kill" || true
sleep 1

# Wait for port to be available
echo "[Setup] Waiting for port $API_PORT to be available..."
for i in {1..10}; do
    if ! curl -s "$BASE_URL/api/health" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Start the test pipeline in background
echo "[Setup] Starting pipeline..."
cd /home/mq/disk2T/quangnv/face
python3 bin/test_multi_branch_video.py --config configs/multi-branch.yaml --port $API_PORT &
PIPELINE_PID=$!

# Wait for API to be ready
echo "[Setup] Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s "$BASE_URL/api/health" > /dev/null 2>&1; then
        echo "[Setup] API is ready!"
        break
    fi
    if ! kill -0 $PIPELINE_PID 2>/dev/null; then
        echo "[ERROR] Pipeline process died unexpectedly"
        exit 1
    fi
    sleep 1
done

echo ""
echo "=============================================="
echo "Running Test Scenario"
echo "=============================================="
echo ""

# Step 1: Add cam1 to recognition + detection
echo "[Step 1] Adding cam1 to recognition + detection..."
curl -s -X POST "$BASE_URL/api/cameras" \
    -H "Content-Type: application/json" \
    -d '{"camera_id": "cam1", "uri": "rtsp://192.168.6.14:8554/testface", "branches": ["recognition", "detection"]}'
echo ""
sleep 2

# Check cameras
echo "[Status] Cameras after step 1:"
curl -s "$BASE_URL/api/cameras" | python3 -m json.tool 2>/dev/null || echo "(no cameras or JSON parsing failed)"
echo ""

# Step 2: Add cam2 to recognition + detection
echo "[Step 2] Adding cam2 to recognition + detection..."
curl -s -X POST "$BASE_URL/api/cameras" \
    -H "Content-Type: application/json" \
    -d '{"camera_id": "cam2", "uri": "rtsp://192.168.6.14:8554/test", "branches": ["recognition", "detection"]}'
echo ""
sleep 3

# Check cameras
echo "[Status] Cameras after step 2:"
curl -s "$BASE_URL/api/cameras" | python3 -m json.tool 2>/dev/null || echo "(no cameras or JSON parsing failed)"
echo ""

# Step 3: Remove cam2 from detection branch (user's original step)
echo "[Step 3] Removing cam2 from detection branch..."
curl -s -X DELETE "$BASE_URL/api/cameras/cam2/branches/detection"
echo ""
sleep 2

# Check cameras
echo "[Status] Cameras after step 3:"
curl -s "$BASE_URL/api/cameras" | python3 -m json.tool 2>/dev/null || echo "(no cameras or JSON parsing failed)"
echo ""

# Step 4: Add cam3 to recognition + detection
echo "[Step 4] Adding cam3 to recognition + detection..."
curl -s -X POST "$BASE_URL/api/cameras" \
    -H "Content-Type: application/json" \
    -d '{"camera_id": "cam3", "uri": "rtsp://192.168.6.14:8554/test", "branches": ["recognition", "detection"]}'
echo ""
sleep 3

# Check cameras
echo "[Status] Cameras after step 4:"
curl -s "$BASE_URL/api/cameras" | python3 -m json.tool 2>/dev/null || echo "(no cameras or JSON parsing failed)"
echo ""

# Step 5: Remove cam1 from recognition
echo "[Step 5] Removing cam1 from recognition branch..."
curl -s -X DELETE "$BASE_URL/api/cameras/cam1/branches/recognition"
echo ""
sleep 2

# Check cameras
echo "[Status] Cameras after step 5:"
curl -s "$BASE_URL/api/cameras" | python3 -m json.tool 2>/dev/null || echo "(no cameras or JSON parsing failed)"
echo ""

# Step 6: Remove cam1 entirely
echo "[Step 6] Removing cam1 entirely..."
curl -s -X DELETE "$BASE_URL/api/cameras/cam1"
echo ""
sleep 2

# Check cameras
echo "[Status] Cameras after step 6:"
curl -s "$BASE_URL/api/cameras" | python3 -m json.tool 2>/dev/null || echo "(no cameras or JSON parsing failed)"
echo ""

echo "=============================================="
echo "Test completed!"
echo "=============================================="

# Keep pipeline running for inspection
echo "[Info] Pipeline is still running. Press Ctrl+C to stop."
echo "[Info] Or run: curl -X POST '$BASE_URL/api/pipeline/kill' to stop"

# Wait for user interrupt or kill signal
wait $PIPELINE_PID 2>/dev/null || true
