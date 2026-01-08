#!/usr/bin/env bash
set -e

APP_DIR="/home/jetson/FACE"
HTTP_PORT=8000
SIGNAL_PORT=8555
PY=python3

cd "$APP_DIR"
mkdir -p logs

echo "=============================="
echo " FACE BACKEND START SCRIPT"
echo "=============================="

echo "[1] Cleanup old processes..."
pkill -f "python3 -m http.server ${HTTP_PORT}" || true
pkill -f "python3 bin/run_signaling.py" || true
pkill -f "python3 bin/run_face_webrtc.py" || true
sleep 1

echo "[2] Starting HTTP server..."
nohup $PY -m http.server ${HTTP_PORT} > logs/http.log 2>&1 &
sleep 2
if ! curl -s "http://127.0.0.1:${HTTP_PORT}/view.html" >/dev/null; then
  echo "❌ HTTP server failed (cannot fetch /view.html). Check file path."
  exit 1
fi
echo "✅ HTTP server OK"

echo "[3] Starting signaling server..."
nohup $PY bin/run_signaling.py > logs/signaling.log 2>&1 &
sleep 2
if ! ss -lnt | grep -q ":${SIGNAL_PORT}"; then
  echo "❌ Signaling server not listening on ${SIGNAL_PORT}"
  exit 1
fi
echo "✅ Signaling server OK"

echo "[4] Waiting for browser(receiver) to connect to signaling..."
ok=0
for i in $(seq 1 90); do
  # đợi có TCP ESTABLISHED tới :8555 (browser mở view.html sẽ connect ws)
  if ss -nt 2>/dev/null | grep -E ":${SIGNAL_PORT}\b" | grep -q ESTAB; then
    ok=1
    break
  fi
  sleep 1
done

if [ "$ok" -ne 1 ]; then
  echo "⚠️  Browser not connected after 90s. Start stream anyway."
else
  echo "✅ Browser connected."
fi
sleep 30
echo "[5] Starting stream pipeline..."
nohup $PY bin/run_face_webrtc.py > logs/stream.log 2>&1 &

echo "=============================="
echo " ✅ FACE BACKEND STARTED"
echo "=============================="
