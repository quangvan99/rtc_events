#!/bin/bash
# Wrapper: suppress GStreamer debug logs
# Set GST_DEBUG=1 or higher to enable debug output
export GST_DEBUG="${GST_DEBUG:-0}"
exec python3 "$(dirname "$0")/run_face_webrtc.py" "$@"
