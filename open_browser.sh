#!/usr/bin/env bash
set -e

SIGNAL_PORT=8555
HTTP_PORT=8000
URL="http://127.0.0.1:${HTTP_PORT}/view.html"

# Đợi signaling thật sự LISTEN (tối đa 90s)
for i in $(seq 1 90); do
  if ss -lnt 2>/dev/null | grep -q ":${SIGNAL_PORT}"; then
    break
  fi
  sleep 1
done

# Đợi HTTP server (tối đa 60s)
for i in $(seq 1 60); do
  if curl -s "http://127.0.0.1:${HTTP_PORT}/view.html" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# Mở Chromium kiosk
chromium-browser \
  --kiosk \
  --autoplay-policy=no-user-gesture-required \
  --disable-infobars \
  --noerrdialogs \
  "$URL" >/dev/null 2>&1 &
