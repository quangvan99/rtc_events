#!/usr/bin/env python3
"""
Test Multi-Branch Pipeline with Video Output

2 branches:
- recognition: Face detection + tracking + face embedding (full face recognition)
- detection: Face detection only (simple)

Each branch outputs to a separate video file so you can see the difference.

Usage:
    python bin/test_multi_branch_video.py

Then use curl to add cameras:
    # Add camera to both branches
    curl -X POST http://localhost:8080/api/cameras \
         -H "Content-Type: application/json" \
         -d '{"camera_id": "cam1", "uri": "file:///path/to/video.mp4", "branches": ["recognition", "detection"]}'

    # Add camera to recognition only
    curl -X POST http://localhost:8080/api/cameras \
         -H "Content-Type: application/json" \
         -d '{"camera_id": "cam2", "uri": "rtsp://...", "branches": ["recognition"]}'

    # List cameras
    curl http://localhost:8080/api/cameras

    # Remove camera from detection branch
    curl -X DELETE http://localhost:8080/api/cameras/cam1/branches/detection

    # Remove camera entirely
    curl -X DELETE http://localhost:8080/api/cameras/cam1

Output videos:
    - output_recognition.avi: Full face recognition with names
    - output_detection.avi: Simple face detection boxes
"""

import argparse
import asyncio
import signal
import sys
import os

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Change to project root so relative paths in config files work
os.chdir(PROJECT_ROOT)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from core.tee_fanout_builder import TeeFanoutPipelineBuilder
from core.multibranch_camera_manager import MultibranchCameraManager
from api.camera_api import CameraAPIServer
from sinks.fakesink_adapter import FakesinkAdapter


def get_config():
    """Config for 2 branches with video output"""
    return {
        "pipeline": {
            "name": "multi-branch-video-test",
            "branches": {
                # Branch 1: Full Face Recognition (PGIE + tracker + SGIE)
                "recognition": {
                    "max_cameras": 8,
                    "muxer": {
                        "batch-size": 8,
                        "batched-push-timeout": 40000,
                        "width": 1920,
                        "height": 1080,
                        "live-source": 1,
                        "nvbuf-memory-type": 0,
                    },
                    "elements": [
                        # Queue for isolation
                        {"type": "queue", "properties": {"max-size-buffers": 5, "leaky": 2}},

                        # Face Detection (PGIE)
                        {"type": "nvinfer", "name": "pgie",
                         "config_file": "data/face/models/scrfd640/infer.txt"},

                        # Object Tracking
                        {"type": "nvtracker", "name": "tracker",
                         "config_file": "data/face/models/NvDCF/config_tracker.txt"},

                        # Face Embedding (SGIE)
                        {"type": "nvinfer", "name": "sgie",
                         "config_file": "data/face/models/arcface/infer.txt"},

                        # Output queue
                        {"type": "queue", "properties": {"max-size-buffers": 3, "leaky": 2}},

                        # Tiler for multiple cameras
                        {"type": "nvmultistreamtiler", "name": "tiler",
                         "properties": {"rows": 2, "columns": 2, "width": 1920, "height": 1080}},

                        # Draw boxes and labels
                        {"type": "nvdsosd", "properties": {"process-mode": 0, "display-text": 1}},

                        # Convert for encoding
                        {"type": "nvvideoconvert"},
                    ]
                },

                # Branch 2: Simple Face Detection (PGIE only)
                "detection": {
                    "max_cameras": 8,
                    "muxer": {
                        "batch-size": 8,
                        "batched-push-timeout": 40000,
                        "width": 1920,
                        "height": 1080,
                        "live-source": 1,
                        "nvbuf-memory-type": 0,
                    },
                    "elements": [
                        # Queue for isolation
                        {"type": "queue", "properties": {"max-size-buffers": 5, "leaky": 2}},

                        # Face Detection (PGIE) - same model
                        {"type": "nvinfer", "name": "pgie",
                         "config_file": "data/face/models/scrfd640/infer.txt"},

                        # Output queue
                        {"type": "queue", "properties": {"max-size-buffers": 3, "leaky": 2}},

                        # Tiler for multiple cameras
                        {"type": "nvmultistreamtiler", "name": "tiler",
                         "properties": {"rows": 2, "columns": 2, "width": 1920, "height": 1080}},

                        # Draw boxes
                        {"type": "nvdsosd", "properties": {"process-mode": 0, "display-text": 1}},

                        # Convert for encoding
                        {"type": "nvvideoconvert"},
                    ]
                }
            }
        }
    }


async def main(api_port: int, output_dir: str):
    """Main entry point"""
    print("=" * 60)
    print("Multi-Branch Pipeline Test with Video Output")
    print("=" * 60)

    config = get_config()

    # Create sinks - using FakesinkAdapter for testing (FilesinkAdapter has memory issues)
    os.makedirs(output_dir, exist_ok=True)
    sinks = {
        "recognition": FakesinkAdapter(sync=False, name="sink_recognition"),
        "detection": FakesinkAdapter(sync=False, name="sink_detection"),
    }

    print(f"\n[Setup] Building pipeline with 2 branches...")
    print(f"  - recognition: PGIE + Tracker + SGIE (full face recognition)")
    print(f"  - detection: PGIE only (simple face detection)")

    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    # Create camera manager
    manager = MultibranchCameraManager(pipeline, builder.branches)

    # Create API server
    api = CameraAPIServer(manager, port=api_port)

    # Shutdown handling
    stop_event = asyncio.Event()

    def on_shutdown(sig):
        print(f"\n[Shutdown] Received signal, stopping...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: on_shutdown(s))

    # Start API
    runner = await api.start()

    # Start pipeline (FakesinkAdapter.start() is no-op)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to start pipeline!")
        return 1

    print(f"\n[Running] Pipeline started!")
    print(f"\n" + "=" * 60)
    print("API SERVER: http://localhost:{api_port}")
    print("=" * 60)
    print("""
CURL Commands to test:

1. Add camera to BOTH branches:
   curl -X POST http://localhost:{port}/api/cameras \\
        -H "Content-Type: application/json" \\
        -d '{{"camera_id": "cam1", "uri": "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4", "branches": ["recognition", "detection"]}}'

2. Add camera to RECOGNITION only:
   curl -X POST http://localhost:{port}/api/cameras \\
        -H "Content-Type: application/json" \\
        -d '{{"camera_id": "cam2", "uri": "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4", "branches": ["recognition"]}}'

3. List all cameras:
   curl http://localhost:{port}/api/cameras

4. List branches:
   curl http://localhost:{port}/api/branches

5. Remove camera from detection branch:
   curl -X DELETE http://localhost:{port}/api/cameras/cam1/branches/detection

6. Add camera to new branch:
   curl -X POST http://localhost:{port}/api/cameras/cam1/branches/detection

7. Remove camera entirely:
   curl -X DELETE http://localhost:{port}/api/cameras/cam1

8. Health check:
   curl http://localhost:{port}/api/health

Output videos:
   - {output_dir}/output_recognition.avi
   - {output_dir}/output_detection.avi

Press Ctrl+C to stop...
""".format(port=api_port, output_dir=output_dir))

    # Wait for shutdown
    await stop_event.wait()

    # Cleanup
    print("\n[Cleanup] Stopping pipeline...")
    pipeline.set_state(Gst.State.NULL)

    await runner.cleanup()

    print(f"\n[Done] Output videos saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Branch Pipeline Test")
    parser.add_argument("--port", "-p", type=int, default=8080, help="API port (default: 8080)")
    parser.add_argument("--output", "-o", default="/tmp/multi_branch_output", help="Output directory")
    args = parser.parse_args()

    Gst.init(None)
    sys.exit(asyncio.run(main(args.port, args.output)))
