#!/usr/bin/env python3
"""
Camera CRUD Test with WebRTC Output

Test file for dynamic add/remove cameras via REST API with WebRTC streaming.
Signaling server and web viewer should be running before this script.

Prerequisites:
    1. Start signaling: python bin/run_signaling.py
    2. Open browser: view.html

Usage:
    python bin/test_camera_crud_webrtc.py

Then use these curl commands to add/remove cameras:

Add camera:
    curl -X POST 'http://localhost:9000/api/v1/stream/add' \
      -H 'Content-Type: application/json' \
      -d '{"value":{"camera_id":"cam1","camera_url":"file:///path/to/video.mp4","change":"camera_add"}}'

    curl -X POST 'http://localhost:9000/api/v1/stream/add' \
      -H 'Content-Type: application/json' \
      -d '{"value":{"camera_id":"cam1","camera_url":"rtsp://admin:pass@192.168.1.100/stream","change":"camera_add"}}'

Remove camera:
    curl -X POST 'http://localhost:9000/api/v1/stream/remove' \
      -H 'Content-Type: application/json' \
      -d '{"value":{"camera_id":"cam1","change":"camera_remove"}}'

List cameras (via Python API - not REST):
    # See CameraManager.list() for active cameras
"""

# Suppress GStreamer debug logs
import os as _os
if "GST_DEBUG" not in _os.environ:
    _os.environ["GST_DEBUG"] = "*:2"
del _os

import os
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import load_config, PipelineBuilder
from core.source_mapper import SourceIDMapper
from core.camera_manager import CameraManager
from sinks.webrtc import WebRTCAdapter
from apps.face import FaceDatabase, TrackerManager
from apps.face.probes import FaceProbes


# Default config path
CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "multi-camera.yaml")


async def main():
    config = load_config(CONFIG_PATH)
    cam_cfg = config.get("camera", {})
    webrtc_cfg = config.get("sink", {}).get("webrtc", {})
    rec = config.get("recognition", {})

    loop = asyncio.get_event_loop()

    # Create source mapper for camera_id <-> source_id tracking
    source_mapper = SourceIDMapper()

    # Create camera manager (for listing cameras, etc.)
    camera_mgr = CameraManager(
        host=cam_cfg.get("rest_host", "localhost"),
        port=cam_cfg.get("rest_port", 9000),
        mapper=source_mapper
    )

    # Create WebRTC sink
    ws_url = webrtc_cfg.get("ws_url", "ws://192.168.6.16:8555")
    peer_id = webrtc_cfg.get("peer_id", "1")

    webrtc_sink = WebRTCAdapter(
        ws_url=ws_url,
        peer_id=peer_id,
        stun_server=webrtc_cfg.get("stun_server", ""),
        loop=loop,
    )

    # Create face recognition components
    db = FaceDatabase(rec.get("features_json", ""))
    tracker_mgr = TrackerManager(rec)
    probes = FaceProbes(rec, db, tracker_mgr, webrtc_sink, source_mapper)

    # Build pipeline with face recognition probes
    builder = PipelineBuilder(config, webrtc_sink)
    builder.register_probe("tracker_probe", probes.tracker_probe)
    builder.register_probe("sgie_probe", probes.sgie_probe)
    builder.register_probe("fps_probe", probes.fps_probe)

    rest_host = cam_cfg.get("rest_host", "localhost")
    rest_port = cam_cfg.get("rest_port", 9000)

    print("=" * 70)
    print("Camera CRUD Test with WebRTC Output")
    print("=" * 70)
    print(f"Config: {CONFIG_PATH}")
    print(f"WebRTC: {ws_url} (peer_id={peer_id})")
    print(f"REST API: http://{rest_host}:{rest_port}")
    print("=" * 70)
    print()
    print("ADD CAMERA:")
    print(f"  curl -X POST 'http://{rest_host}:{rest_port}/api/v1/stream/add' \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"value\":{\"camera_id\":\"cam1\",\"camera_url\":\"file:///path/to/video.mp4\",\"change\":\"camera_add\"}}'")
    print()
    print("REMOVE CAMERA:")
    print(f"  curl -X POST 'http://{rest_host}:{rest_port}/api/v1/stream/remove' \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"value\":{\"camera_id\":\"cam1\",\"change\":\"camera_remove\"}}'")
    print()
    print("=" * 70)
    print("Press Ctrl+C to stop")
    print("=" * 70)

    async def setup():
        """Setup WebRTC connection before pipeline starts"""
        await webrtc_sink.connect()
        asyncio.create_task(webrtc_sink.handle_signaling())
        await asyncio.sleep(2)
        webrtc_sink.start()
        print("\n[WebRTC] Ready - waiting for cameras...")

    try:
        await builder.run_async(setup)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await webrtc_sink.disconnect()
        print("Done")


if __name__ == "__main__":
    asyncio.run(main())
