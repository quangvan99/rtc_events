#!/usr/bin/env python3
"""
Simple synchronous test with direct camera management.
Uses threading for HTTP server and direct calls for GStreamer operations.
"""

import sys
import os
import threading
import time
import http.server
import socketserver
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from core.tee_fanout_builder import TeeFanoutPipelineBuilder
from core.multibranch_camera_manager import MultibranchCameraManager
from core.config import load_config
from sinks.filesink_adapter import FilesinkAdapter

DEFAULT_CONFIG = "configs/multi-branch.yaml"
DEFAULT_OUTPUT_DIR = "/home/mq/disk2T/quangnv/face/data"


def main():
    print("=" * 60)
    print("Multi-Branch Pipeline - Synchronous Test")
    print("=" * 60)

    config = load_config(DEFAULT_CONFIG)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    branches_cfg = config.get("pipeline", {}).get("branches", {})
    sinks = {}
    for branch_name in branches_cfg.keys():
        sinks[branch_name] = FilesinkAdapter(location=f"{DEFAULT_OUTPUT_DIR}/output_{branch_name}.avi")

    print(f"\n[Setup] Building pipeline with {len(branches_cfg)} branches...")

    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    manager = MultibranchCameraManager(pipeline, builder.branches)

    print("\n[Running] Pipeline starting...")

    for sink in sinks.values():
        sink.start()

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to start pipeline!")
        return 1

    print("[Running] Pipeline started!")

    print("\n=== Test: Add cameras synchronously ===")

    print("\n[1] Adding cam1 to recognition...")
    result1 = manager.add_camera("cam1", "rtsp://192.168.6.14:8554/test", ["recognition"])
    print(f"    Result: {result1}")
    time.sleep(2)

    print("\n[2] Adding cam2 to detection...")
    result2 = manager.add_camera("cam2", "rtsp://192.168.6.14:8554/test", ["detection"])
    print(f"    Result: {result2}")
    time.sleep(2)

    print("\n[3] Listing cameras...")
    cameras = manager.list_cameras()
    print(f"    Cameras: {cameras}")

    print("\n[4] Listing branches...")
    for name, info in manager.branches.items():
        cams = [cid for cid, cam in manager._cameras.items() if name in cam.branch_queues]
        print(f"    {name}: {cams}")

    print("\n[5] Removing cam2 from detection...")
    result3 = manager.remove_camera("cam2")
    print(f"    Result: {result3}")
    time.sleep(1)

    print("\n[6] Adding cam2 to both branches...")
    result4 = manager.add_camera_to_branch("cam2", "recognition")
    print(f"    Result: {result4}")
    time.sleep(2)

    print("\n[7] Final state...")
    cameras = manager.list_cameras()
    print(f"    Cameras: {cameras}")

    print("\n=== Test Complete ===")

    print("\n[Shutdown] Stopping pipeline...")
    pipeline.set_state(Gst.State.NULL)
    for sink in sinks.values():
        sink.stop()
    print("[Done]")


if __name__ == "__main__":
    main()
