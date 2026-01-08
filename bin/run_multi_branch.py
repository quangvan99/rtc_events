#!/usr/bin/env python3
"""
Multi-Branch Pipeline Runner

Runs multi-branch pipeline with REST API for camera management.

Usage:
    python bin/run_multi_branch.py --config configs/multi-branch.yaml

API Endpoints (default port 8080):
    POST   /api/cameras                 - Add camera
    DELETE /api/cameras/{id}            - Remove camera
    GET    /api/cameras                 - List cameras
    GET    /api/branches                - List branches
    GET    /api/health                  - Health check

Example:
    curl -X POST http://localhost:8080/api/cameras \\
         -H "Content-Type: application/json" \\
         -d '{"camera_id": "cam1", "uri": "file:///video.mp4", "branches": ["recognition"]}'
"""

import argparse
import asyncio
import signal
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from core.config import load_config
from core.tee_fanout_builder import TeeFanoutPipelineBuilder
from core.multibranch_camera_manager import MultibranchCameraManager
from api.camera_api import CameraAPIServer
from sinks.fakesink_adapter import FakesinkAdapter


async def main(config_path: str, api_port: int):
    """Main entry point"""
    print(f"[Runner] Loading config: {config_path}")
    config = load_config(config_path)

    # Create sinks for each branch
    branch_names = list(config["pipeline"]["branches"].keys())
    sinks = {name: FakesinkAdapter(sync=False) for name in branch_names}

    print(f"[Runner] Building pipeline with branches: {branch_names}")
    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    # Create camera manager
    manager = MultibranchCameraManager(pipeline, builder.branches)

    # Create API server
    api = CameraAPIServer(manager, port=api_port)

    # Setup shutdown
    stop_event = asyncio.Event()

    def on_shutdown(sig):
        print(f"\n[Runner] Received {sig}, shutting down...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: on_shutdown(s))

    # Start API server
    runner = await api.start()

    # Start pipeline
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[Runner] ERROR: Failed to start pipeline")
        return 1

    print(f"[Runner] Pipeline PLAYING")
    print(f"[Runner] API server: http://0.0.0.0:{api_port}")
    print(f"[Runner] Use Ctrl+C to stop")

    # Wait for shutdown
    await stop_event.wait()

    # Cleanup
    print("[Runner] Stopping pipeline...")
    pipeline.set_state(Gst.State.NULL)
    await runner.cleanup()

    print("[Runner] Done")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Branch Pipeline Runner")
    parser.add_argument("--config", "-c", default="configs/multi-branch.yaml", help="Config file path")
    parser.add_argument("--port", "-p", type=int, default=8080, help="API server port")
    args = parser.parse_args()

    Gst.init(None)
    sys.exit(asyncio.run(main(args.config, args.port)))
