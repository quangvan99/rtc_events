#!/usr/bin/env python3
"""
End-to-End Test for Multi-Camera Dynamic CRUD

Tests:
1. Start pipeline with no cameras
2. Add camera 1 (video file)
3. Wait and process some frames
4. Add camera 2 (another video)
5. Wait and process
6. Remove camera 1
7. Wait and process
8. Remove camera 2
9. Stop pipeline

Output: MP4 file showing the CRUD process
"""

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import load_config, PipelineBuilder
from core.source_mapper import SourceIDMapper
from core.camera_manager import CameraManager, Camera
from sinks import FilesinkAdapter

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst


# Test configuration
VIDEO_DIR = "/home/mq/disk2T/quangnv/face/data/videos"
OUTPUT_FILE = "/home/mq/disk2T/quangnv/face/data/output_crud_test.avi"
CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "multi-camera.yaml")


def create_test_config():
    """Create minimal config for testing without face recognition"""
    return {
        "pipeline": {
            "name": "test-multi-camera",
            "source": {
                "type": "nvmultiurisrcbin",
                "name": "multi_src",
                "properties": {
                    "uri-list": "",
                    "sensor-id-list": "",
                    "max-batch-size": 4,
                    "ip-address": "0.0.0.0",
                    "port": 9000,
                    "drop-pipeline-eos": True,
                    "live-source": 0,  # 0 for file sources
                    "file-loop": True,  # Loop video files for longer test
                    "width": 640,
                    "height": 480,
                    "batched-push-timeout": 33333,
                }
            },
            "elements": [
                {"type": "nvvideoconvert", "properties": {"compute-hw": 1}},
                {"type": "queue", "properties": {"max-size-buffers": 3, "leaky": 2}},
                {
                    "type": "nvmultistreamtiler",
                    "name": "tiler",
                    "properties": {
                        "rows": 1,
                        "columns": 2,
                        "width": 1280,
                        "height": 480,
                    }
                },
                {"type": "nvvideoconvert"},
                {
                    "type": "nvdsosd",
                    "properties": {"process-mode": 0, "display-text": 1}
                },
                {"type": "nvvideoconvert"},
            ]
        },
        "camera": {
            "rest_host": "localhost",
            "rest_port": 9000,
        }
    }


class CRUDTester:
    """Manages CRUD test sequence"""

    def __init__(self, camera_mgr: CameraManager):
        self.camera_mgr = camera_mgr
        self.running = True

    def run_test_sequence(self):
        """Run CRUD test in separate thread"""
        print("\n" + "=" * 60)
        print("Starting CRUD Test Sequence")
        print("=" * 60)

        time.sleep(2)  # Wait for pipeline to start

        # Test 1: Add first camera
        print("\n[TEST 1] Adding camera 1 (test.mp4)...")
        cam1 = Camera(
            camera_id="cam_test_1",
            url=f"file://{VIDEO_DIR}/test.mp4",
            name="Test Camera 1"
        )
        if self.camera_mgr.add(cam1):
            print("[SUCCESS] Camera 1 added - should see video on LEFT")
        else:
            print("[FAILED] Camera 1 add failed")

        time.sleep(4)  # Process for 4 seconds - see camera 1 only

        # Test 2: Add second camera
        print("\n[TEST 2] Adding camera 2 (change_id.mp4)...")
        cam2 = Camera(
            camera_id="cam_test_2",
            url=f"file://{VIDEO_DIR}/change_id.mp4",
            name="Test Camera 2"
        )
        if self.camera_mgr.add(cam2):
            print("[SUCCESS] Camera 2 added - should see BOTH videos side by side")
        else:
            print("[FAILED] Camera 2 add failed")

        time.sleep(4)  # Process for 4 seconds with both cameras

        # List cameras
        print("\n[STATUS] Active cameras:")
        for cam in self.camera_mgr.list():
            print(f"  - {cam.camera_id}: {cam.url}")

        # Test 3: Remove first camera via direct REST call
        print("\n[TEST 3] Removing camera 1 via REST API...")
        import urllib.request
        import json
        payload = json.dumps({
            "value": {
                "camera_id": "cam_test_1",
                "change": "camera_remove"
            }
        }).encode('utf-8')
        req = urllib.request.Request(
            "http://localhost:9000/api/v1/stream/remove",
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                print(f"[SUCCESS] Camera 1 removed - LEFT side should be BLACK now")
                self.camera_mgr.mapper.remove("cam_test_1")
        except Exception as e:
            print(f"[FAILED] Remove error: {e}")

        time.sleep(4)  # Process for 4 seconds with only camera 2

        # Test 4: Remove second camera
        print("\n[TEST 4] Removing camera 2 via REST API...")
        payload = json.dumps({
            "value": {
                "camera_id": "cam_test_2",
                "change": "camera_remove"
            }
        }).encode('utf-8')
        req = urllib.request.Request(
            "http://localhost:9000/api/v1/stream/remove",
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                print(f"[SUCCESS] Camera 2 removed - screen should be BLACK")
                self.camera_mgr.mapper.remove("cam_test_2")
        except Exception as e:
            print(f"[FAILED] Remove error: {e}")

        time.sleep(2)  # Final wait - black screen

        print("\n" + "=" * 60)
        print("CRUD Test Complete!")
        print(f"Output saved to: {OUTPUT_FILE}")
        print("=" * 60)

        self.running = False


def main():
    Gst.init(None)

    print("=" * 60)
    print("Multi-Camera CRUD End-to-End Test")
    print("=" * 60)
    print(f"Video source: {VIDEO_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 60)

    # Create config
    config = create_test_config()

    # Create source mapper and camera manager
    source_mapper = SourceIDMapper()
    camera_mgr = CameraManager(
        host="localhost",
        port=9000,
        mapper=source_mapper
    )

    # Create filesink for output
    sink = FilesinkAdapter(location=OUTPUT_FILE, bitrate=4000000)

    # Build pipeline
    builder = PipelineBuilder(config, sink)
    pipeline = builder.build()

    # Create tester
    tester = CRUDTester(camera_mgr)

    # Start test thread
    test_thread = threading.Thread(target=tester.run_test_sequence, daemon=True)
    test_thread.start()

    # Run pipeline
    loop = GLib.MainLoop()

    def on_message(bus, message):
        if message.type == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err.message}")
            print(f"Debug: {debug}")
            loop.quit()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)

    # Timeout to stop after test completes
    def check_done():
        if not tester.running:
            print("\nTest sequence complete, stopping pipeline...")
            pipeline.send_event(Gst.Event.new_eos())
            return False
        return True

    GLib.timeout_add(1000, check_done)

    try:
        print("\nStarting pipeline...")
        pipeline.set_state(Gst.State.PLAYING)
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print(f"\nOutput file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
