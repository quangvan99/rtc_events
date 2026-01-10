#!/usr/bin/env python3
"""
Simple test script that uses camera_api.py with proper GStreamer integration.
"""

import sys
import os
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from core.tee_fanout_builder import TeeFanoutPipelineBuilder
from core.multibranch_camera_manager import MultibranchCameraManager
from core.config import load_config
from sinks.filesink_adapter import FilesinkAdapter
from api.camera_api import CameraAPIServer

DEFAULT_CONFIG = "configs/multi-branch.yaml"
DEFAULT_OUTPUT_DIR = "/home/mq/disk2T/quangnv/face/data"


def main():
    print("=" * 60)
    print("Multi-Branch Pipeline Test with CameraAPIServer")
    print("=" * 60)

    config = load_config(DEFAULT_CONFIG)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    branches_cfg = config.get("pipeline", {}).get("branches", {})
    sinks = {}
    for branch_name in branches_cfg.keys():
        sinks[branch_name] = FilesinkAdapter(location=f"{DEFAULT_OUTPUT_DIR}/output_{branch_name}.avi")

    print(f"\n[Setup] Building pipeline with {len(branches_cfg)} branches...")
    for branch_name, branch_cfg in branches_cfg.items():
        elements = branch_cfg.get("elements", [])
        element_types = [e.get("type", "") for e in elements if e.get("name")]
        print(f"  - {branch_name}: {', '.join(element_types) if element_types else 'configured in YAML'}")

    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    manager = MultibranchCameraManager(pipeline, builder.branches)

    stop_event = threading.Event()

    def on_shutdown(signum, frame):
        print(f"\n[Shutdown] Received signal, stopping...")
        stop_event.set()

    import signal
    signal.signal(signal.SIGINT, on_shutdown)
    signal.signal(signal.SIGTERM, on_shutdown)

    api_server = CameraAPIServer(manager, host="0.0.0.0", port=8083, shutdown_event=stop_event)

    for sink in sinks.values():
        sink.start()

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to start pipeline!")
        return 1

    print(f"\n[Running] Pipeline started!")
    print(f"[API] Server at http://localhost:8083")

    async def run_server():
        runner = await api_server.start()
        return runner

    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    runner = loop.run_until_complete(run_server())

    print("\nCURL Commands to test:")
    print("1. Add cam1: curl -X POST http://localhost:8083/api/cameras -H 'Content-Type: application/json' -d '{\"camera_id\": \"cam1\", \"uri\": \"rtsp://192.168.6.14:8554/test\", \"branches\": [\"recognition\"]}'")
    print("2. List: curl http://localhost:8083/api/cameras")
    print("3. Stop: curl -X POST http://localhost:8083/api/pipeline/stop")

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

    print("\n[Shutdown] Stopping pipeline...")
    pipeline.set_state(Gst.State.NULL)
    for sink in sinks.values():
        sink.stop()
    loop.run_until_complete(runner.cleanup())
    print("[Done]")


if __name__ == "__main__":
    main()
