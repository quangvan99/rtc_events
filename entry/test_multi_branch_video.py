#!/usr/bin/env python3
"""
Multi-Branch Pipeline - Clean Architecture with Auto-Discovery

This entry point demonstrates the cleanest usage pattern:
- ProcessorRegistry auto-discovers processors from apps/
- PipelineBuilder auto-creates processors for configured branches
- No explicit processor imports or instantiation needed

Usage:
    python entry/test_multi_branch_video.py
    python entry/test_multi_branch_video.py --config configs/multi-branch.yaml
"""

import argparse
import sys

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

    # Set pipeline to PLAYING state
    print("\n[Pipeline] Setting to PLAYING...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to start pipeline!")
        return 1

    # Start processors after pipeline is playing
    builder.start_processors()

    # Start camera API server
    api = CameraAPIServer(config.get("camera_api", {}), manager)
    api.start()

    # Display processor-specific information
    # _display_processor_info(builder)

    # Wait for shutdown signal
    wait_for_shutdown()

    # Graceful shutdown sequence
    print("\n[Shutdown] Stopping...")
    api.stop()
    builder.stop_processors()
    pipeline.set_state(Gst.State.NULL)
    for sink in builder.branch_sinks.values():
        sink.stop()
    
    print("[Done]")
    return 0


def _display_processor_info(builder: PipelineBuilder) -> None:
    """Display information from processors after startup."""
    print("\n" + "=" * 60)
    print(f"[Processors] Active: {list(builder.processors.keys())}")
    
    # Face recognition processor info
    face_proc = builder.get_processor("recognition")
    if face_proc and hasattr(face_proc, 'database') and face_proc.database:
        print(f"[Face Recognition] {len(face_proc.database.names)} faces registered")
    
    # Detection processor info
    detection_proc = builder.get_processor("detection")
    if detection_proc:
        print("[Detection] Processor active")
    
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
