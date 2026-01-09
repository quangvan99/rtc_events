#!/usr/bin/env python3
"""
Test Multi-Branch Pipeline with Video Output

Uses YAML config from configs/multi-branch.yaml.
Runs automated test scenario:
  1. Add camera1 to recognition + detection branches
  2. Wait 2s, add camera2 to recognition + detection
  3. Wait 3s, remove camera1 from detection branch
  4. Wait for video to complete

Usage:
    python bin/test_multi_branch_video.py

Output:
    - /tmp/multi_branch_output/output_recognition.avi
    - /tmp/multi_branch_output/output_detection.avi
"""

import argparse
import asyncio
import os
import signal
import sys

import yaml

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from core.tee_fanout_builder import TeeFanoutPipelineBuilder
from core.multibranch_camera_manager import MultibranchCameraManager
from api.camera_api import CameraAPIServer
from sinks.filesink_adapter import FilesinkAdapter


def load_yaml_config(config_path: str) -> dict:
    """Load YAML config with environment variable substitution"""
    with open(config_path, 'r') as f:
        content = f.read()

    # Simple env var substitution: ${VAR:default}
    import re
    def replace_env(match):
        var_expr = match.group(1)
        if ':' in var_expr:
            var_name, default = var_expr.split(':', 1)
        else:
            var_name, default = var_expr, ''
        return os.environ.get(var_name, default)

    content = re.sub(r'\$\{([^}]+)\}', replace_env, content)
    return yaml.safe_load(content)


async def run_test_scenario(manager: MultibranchCameraManager, camera1_uri: str, camera2_uri: str):
    """
    Automated test scenario:
      1. Add camera1 to recognition + detection
      2. Wait 2s, add camera2 to recognition + detection
      3. Wait 3s, remove camera1 from detection
    """
    print("\n" + "=" * 60)
    print("AUTOMATED TEST SCENARIO")
    print("=" * 60)

    # Step 1: Add camera1 to both branches
    print("\n[Step 1] Adding cam1 to recognition + detection...")
    result = manager.add_camera("cam1", camera1_uri, ["recognition", "detection"])
    print(f"  Result: {result}")

    # Step 2: Wait 2s, then add camera2
    print("\n[Step 2] Waiting 2 seconds...")
    await asyncio.sleep(2)
    print("  Adding cam2 to recognition + detection...")
    result = manager.add_camera("cam2", camera2_uri, ["recognition", "detection"])
    print(f"  Result: {result}")

    # Step 3: Wait 3s, then remove camera1 from detection
    print("\n[Step 3] Waiting 3 seconds...")
    await asyncio.sleep(3)
    print("  Removing cam1 from detection branch...")
    result = manager.remove_camera_from_branch("cam1", "detection")
    print(f"  Result: {result}")

    print("\n" + "=" * 60)
    print("TEST SCENARIO COMPLETE - Recording continues...")
    print("=" * 60)
    print("\nCurrent state:")
    print(f"  - cam1: recognition only")
    print(f"  - cam2: recognition + detection")
    print("\nPress Ctrl+C to stop recording.\n")


async def main(
    config_path: str,
    output_dir: str,
    api_port: int,
    camera1_uri: str,
    camera2_uri: str,
    auto_test: bool
):
    """Main entry point"""
    print("=" * 60)
    print("Multi-Branch Pipeline Test with Video Output")
    print("=" * 60)

    # Load config
    print(f"\n[Config] Loading: {config_path}")
    config = load_yaml_config(config_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create video output sinks
    sinks = {
        "recognition": FilesinkAdapter(location=f"{output_dir}/output_recognition.avi"),
        "detection": FilesinkAdapter(location=f"{output_dir}/output_detection.avi"),
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

    # Run automated test scenario
    if auto_test:
        asyncio.create_task(run_test_scenario(manager, camera1_uri, camera2_uri))
    else:
        print(f"""
Manual mode - Use curl to add cameras:

1. Add camera to BOTH branches:
   curl -X POST http://localhost:{api_port}/api/cameras \\
        -H "Content-Type: application/json" \\
        -d '{{"camera_id": "cam1", "uri": "{camera1_uri}", "branches": ["recognition", "detection"]}}'

2. List cameras:
   curl http://localhost:{api_port}/api/cameras

3. Remove camera from detection:
   curl -X DELETE http://localhost:{api_port}/api/cameras/cam1/branches/detection

Output videos:
   - {output_dir}/output_recognition.avi
   - {output_dir}/output_detection.avi

Press Ctrl+C to stop...
""")

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
    parser = argparse.ArgumentParser(description="Multi-Branch Pipeline Test with Video Output")
    parser.add_argument("--config", "-c", default="configs/multi-branch.yaml",
                        help="YAML config file (default: configs/multi-branch.yaml)")
    parser.add_argument("--port", "-p", type=int, default=8080,
                        help="API port (default: 8080)")
    parser.add_argument("--output", "-o", default="/tmp/multi_branch_output",
                        help="Output directory (default: /tmp/multi_branch_output)")
    parser.add_argument("--camera1", default="file:///home/mq/disk2T/quangnv/face/data/videos/faceQuangnv3.mp4",
                        help="Camera 1 URI")
    parser.add_argument("--camera2", default="file:///home/mq/disk2T/quangnv/face/data/videos/faceQuangnv4.mp4",
                        help="Camera 2 URI")
    parser.add_argument("--manual", action="store_true",
                        help="Manual mode - don't run automated test scenario")

    args = parser.parse_args()

    Gst.init(None)
    sys.exit(asyncio.run(main(
        config_path=args.config,
        output_dir=args.output,
        api_port=args.port,
        camera1_uri=args.camera1,
        camera2_uri=args.camera2,
        auto_test=not args.manual
    )))
