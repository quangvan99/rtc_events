#!/usr/bin/env python3
"""
Multi-Branch Pipeline Test with Face Recognition

Usage:
    python bin/test_multi_branch_video.py
"""

import argparse
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.tee_fanout_builder import TeeFanoutPipelineBuilder
from src.multibranch_camera_manager import MultibranchCameraManager
from src.config import load_config
from apps.face.setup import setup_face_recognition, register_face_probes
from api.camera_api import CameraAPIServer
from api.shutdown import stop_event, setup_signal_handlers, wait_for_shutdown

DEFAULT_CONFIG = "configs/multi-branch.yaml"


def main():
    print("=" * 60)
    print("Multi-Branch Pipeline with Face Recognition")
    print("=" * 60)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)

    api_config = config.get("camera_api", {})
    api_host = api_config.get("host", "0.0.0.0")
    api_port = api_config.get("port", 8083)

    branches_cfg = config.get("pipeline", {}).get("branches", {})
    print(f"\n[Setup] Building pipeline with {len(branches_cfg)} branches...")

    builder = TeeFanoutPipelineBuilder(config)

    rec_config = config.get("recognition", {})
    recognition_sink = builder.branch_sinks.get("recognition", builder.branch_sinks.get(list(builder.branch_sinks.keys())[0]))
    probes = setup_face_recognition(rec_config, recognition_sink)
    register_face_probes(builder, probes)

    pipeline = builder.build()

    manager = MultibranchCameraManager(pipeline, builder.branches)

    setup_signal_handlers()

    for sink in builder.branch_sinks.values():
        sink.start()

    print("\n[Pipeline] Setting to PLAYING...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to start pipeline!")
        return 1

    print("[Pipeline] Started!")

    api = CameraAPIServer(manager=manager, host=api_host, port=api_port, shutdown_event=stop_event)
    api.start()

    print(f"\n" + "=" * 60)
    print(f"[API] Server at http://localhost:{api_port}")
    print(f"[Faces] {len(probes.db.names)} registered")
    print("=" * 60)

    wait_for_shutdown()

    print("\n[Shutdown] Stopping...")
    api.stop()
    pipeline.set_state(Gst.State.NULL)
    for sink in builder.branch_sinks.values():
        sink.stop()
    print("[Done]")


if __name__ == "__main__":
    main()
