#!/usr/bin/env python3
"""
Test Multi-Branch Pipeline with Video Output

2 branches:
- recognition: Face detection + tracking + face embedding (full face recognition)
- detection: Face detection only (simple)

Each branch outputs to a separate video file so you can see the difference.

Usage:
    python bin/test_multi_branch_video.py
    python bin/test_multi_branch_video.py --config configs/multi-branch.yaml

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

    # Kill all cameras (remove all)
    curl -X POST http://localhost:8080/api/pipeline/kill

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
from core.config import load_config
from api.camera_api import CameraAPIServer
from sinks.filesink_adapter import FilesinkAdapter

# Default paths
DEFAULT_CONFIG = "configs/multi-branch.yaml"
DEFAULT_OUTPUT_DIR = "/home/mq/disk2T/quangnv/face/data"


async def main(config_path: str, api_port: int, output_dir: str):
    """Main entry point"""
    print("=" * 60)
    print("Multi-Branch Pipeline Test with Video Output")
    print("=" * 60)

    print(f"\n[Config] Loading: {config_path}")
    config = load_config(config_path)

    # Create video output sinks for each branch defined in config
    os.makedirs(output_dir, exist_ok=True)
    branches_cfg = config.get("pipeline", {}).get("branches", {})
    sinks = {}
    for branch_name in branches_cfg.keys():
        sinks[branch_name] = FilesinkAdapter(location=f"{output_dir}/output_{branch_name}.avi")

    print(f"\n[Setup] Building pipeline with {len(branches_cfg)} branches...")
    for branch_name, branch_cfg in branches_cfg.items():
        elements = branch_cfg.get("elements", [])
        element_types = [e.get("type", "") for e in elements if e.get("name")]
        print(f"  - {branch_name}: {', '.join(element_types) if element_types else 'configured in YAML'}")

    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    # Create camera manager
    manager = MultibranchCameraManager(pipeline, builder.branches)

    # Shutdown handling (must be created before API server)
    stop_event = asyncio.Event()

    def on_shutdown(sig):
        print(f"\n[Shutdown] Received signal, stopping...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: on_shutdown(s))

    # Create API server with shutdown event
    api = CameraAPIServer(manager, port=api_port, shutdown_event=stop_event)

    # Start API
    runner = await api.start()

    # Start sinks
    for sink in sinks.values():
        sink.start()

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to start pipeline!")
        return 1

    print(f"\n[Running] Pipeline started!")
    print(f"\n" + "=" * 60)
    print(f"API SERVER: http://localhost:{api_port}")
    print("=" * 60)
    print("""
CURL Commands to test:

1. Add camera to BOTH branches:
   curl -X POST http://localhost:{port}/api/cameras \\
        -H "Content-Type: application/json" \\
        -d '{{"camera_id": "cam1", "uri": "rtsp://192.168.6.14:8554/test", "branches": ["recognition", "detection"]}}'

2. Add camera to RECOGNITION only:
   curl -X POST http://localhost:{port}/api/cameras \\
        -H "Content-Type: application/json" \\
        -d '{{"camera_id": "cam2", "uri": "rtsp://192.168.6.14:8554/test", "branches": ["recognition"]}}'

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

8. KILL ALL cameras (remove all):
   curl -X POST http://localhost:{port}/api/pipeline/kill

9. STOP pipeline (shutdown entire application):
   curl -X POST http://localhost:{port}/api/pipeline/stop

10. Health check:
   curl http://localhost:{port}/api/health

Output videos:
{output_videos}

Press Ctrl+C to stop...
""".format(
        port=api_port,
        output_dir=output_dir,
        output_videos="\n".join(f"   - {output_dir}/output_{b}.avi" for b in branches_cfg.keys())
    ))

    # Wait for shutdown
    await stop_event.wait()

    # Cleanup
    print("\n[Cleanup] Stopping pipeline...")
    pipeline.set_state(Gst.State.NULL)

    for sink in sinks.values():
        sink.stop()

    await runner.cleanup()

    print(f"\n[Done] Output videos saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Branch Pipeline Test")
    parser.add_argument("--config", "-c", default=DEFAULT_CONFIG, help=f"Config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--port", "-p", type=int, default=8083, help="API port (default: 8083)")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    Gst.init(None)
    sys.exit(asyncio.run(main(args.config, args.port, args.output)))
