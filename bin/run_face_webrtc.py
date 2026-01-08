#!/usr/bin/env python3
# Suppress GStreamer FIXME logs - MUST be before any imports
import os as _os
if "GST_DEBUG" not in _os.environ:
    _os.environ["GST_DEBUG"] = "*:2"
del _os

"""
Face Recognition Pipeline - WebRTC Mode

Usage:
    python bin/run_face_webrtc.py

Requires signaling server: python bin/run_signaling.py
"""

import os
import sys

import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.face import FaceDatabase, TrackerManager
from apps.face.probes import FaceProbes
from sinks.webrtc import WebRTCAdapter
from core import load_config, PipelineBuilder


CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "face-recognition.yaml")


async def main():
    config = load_config(CONFIG_PATH)
    rec = config.get("recognition", {})
    webrtc = config.get("sink", {}).get("webrtc", {})

    loop = asyncio.get_event_loop()

    # Create WebRTC sink
    webrtc_sink = WebRTCAdapter(
        ws_url=webrtc.get("ws_url", "ws://192.168.6.112:8555"),
        peer_id=webrtc.get("peer_id", "1"),
        stun_server=webrtc.get("stun_server", ""),
        loop=loop,
    )

    # Create components
    db = FaceDatabase(rec.get("features_json", ""))
    tracker_mgr = TrackerManager(rec)
    probes = FaceProbes(rec, db, tracker_mgr, webrtc_sink)

    # Build pipeline
    builder = PipelineBuilder(config, webrtc_sink)
    builder.register_probe("tracker_probe", probes.tracker_probe)
    builder.register_probe("sgie_probe", probes.sgie_probe)
    builder.register_probe("fps_probe", probes.fps_probe)

    print("=" * 70)
    print("DeepStream Face Recognition with WebRTC")
    print("=" * 70)
    print(f"Config: {CONFIG_PATH}")
    print(f"WebRTC: {webrtc.get('ws_url')}")
    print(f"Features: {len(db.names)} registered faces")
    print("=" * 70)

    async def setup():
        await webrtc_sink.connect()
        asyncio.create_task(webrtc_sink.handle_signaling())
        await asyncio.sleep(2)
        webrtc_sink.start()

    try:
        await builder.run_async(setup)
    finally:
        await webrtc_sink.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
