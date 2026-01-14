#!/usr/bin/env python3
"""
Multi-Branch Pipeline - Clean Architecture with Auto-Discovery

STABILITY FIX (2026-01-13):
- Pipeline starts in READY state, not PLAYING
- Only transitions to PLAYING after first camera is added
- This prevents nvstreammux crash with empty sources

Usage:
    python entry/test_multi_branch_video.py
    python entry/test_multi_branch_video.py --config configs/multi-branch.yaml
"""

import argparse
import logging
import sys

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# Import directly from modules (no __init__.py)
from src.pipeline_builder import PipelineBuilder
from src.camera_manager import MultibranchCameraManager
from src.common import load_config
from api.camera_api import CameraAPIServer
from api.shutdown import setup_signal_handlers, wait_for_shutdown


def main():
    print("=" * 60)
    print("Multi-Branch Pipeline with Auto-Discovery")
    print("=" * 60)

    parser = argparse.ArgumentParser(
        description="Run multi-branch DeepStream pipeline with auto-discovered processors"
    )
    parser.add_argument(
        "--config",
        default="configs/multi-branch.yaml",
        help="Path to pipeline configuration YAML"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"[Config] Loaded: {args.config}")

    # Note: plate_recognition OCR now runs in separate subprocess (multiprocessing)
    # It's initialized automatically in PlateRecognitionProcessor.setup()

    builder = PipelineBuilder(config)

    # Build pipeline
    pipeline = builder.build()

    # Create camera manager for dynamic camera control
    manager = MultibranchCameraManager(pipeline, builder.branches)

    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    # Start sinks
    for sink in builder.branch_sinks.values():
        sink.start()

    # STABILITY FIX: Start in READY state, not PLAYING
    # This prevents crashes from empty nvstreammux
    # Pipeline will transition to PLAYING when first camera is added
    print("\n[Pipeline] Setting to READY (will PLAY after first camera added)...")
    ret = pipeline.set_state(Gst.State.READY)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to set pipeline to READY!")
        return 1

    # Wait for READY state
    ret, _, _ = pipeline.get_state(5 * Gst.SECOND)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Pipeline failed to reach READY state!")
        return 1

    print("[Pipeline] READY - waiting for cameras...")

    # Start processors after pipeline is ready
    builder.start_processors()

    # Start camera API server
    api = CameraAPIServer(config.get("camera_api", {}), manager)
    api.start()

    # Wait for shutdown signal
    wait_for_shutdown()

    # Graceful shutdown sequence
    print("\n[Shutdown] Stopping...")
    api.stop()
    builder.stop_processors()
    pipeline.set_state(Gst.State.NULL)
    for sink in builder.branch_sinks.values():
        sink.stop()

    # Note: OCR worker process cleanup is handled by PlateRecognitionProcessor.on_stop()

    print("[Done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
