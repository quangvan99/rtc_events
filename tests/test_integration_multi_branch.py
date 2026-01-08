#!/usr/bin/env python3
"""
Integration Tests - Phase 4 Testing & Validation

End-to-end tests for multi-branch pipeline with camera management.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from core.tee_fanout_builder import TeeFanoutPipelineBuilder
from core.multibranch_camera_manager import MultibranchCameraManager
from api.camera_api import CameraAPIServer
from sinks.fakesink_adapter import FakesinkAdapter

TEST_VIDEO = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"


def get_full_config():
    """Full config with all components"""
    return {
        "pipeline": {
            "name": "integration-test",
            "branches": {
                "recognition": {
                    "max_cameras": 8,
                    "muxer": {"batch-size": 8, "batched-push-timeout": 40000, "width": 1920, "height": 1080, "live-source": 1},
                    "elements": [
                        {"type": "queue", "properties": {"max-size-buffers": 5, "leaky": 2}},
                        {"type": "nvvideoconvert"},
                        {"type": "nvdsosd", "properties": {"process-mode": 0}},
                    ]
                },
                "detection": {
                    "max_cameras": 8,
                    "muxer": {"batch-size": 8, "batched-push-timeout": 40000, "width": 1920, "height": 1080, "live-source": 1},
                    "elements": [
                        {"type": "queue", "properties": {"max-size-buffers": 5, "leaky": 2}},
                        {"type": "nvvideoconvert"},
                        {"type": "nvdsosd", "properties": {"process-mode": 0}},
                    ]
                }
            }
        }
    }


def test_full_pipeline_lifecycle():
    """Test complete pipeline lifecycle: build -> add cameras -> remove -> cleanup"""
    print("\n=== Integration Test: Full Pipeline Lifecycle ===")

    # 1. Build pipeline
    config = get_full_config()
    sinks = {"recognition": FakesinkAdapter(), "detection": FakesinkAdapter()}
    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    # 2. Create manager and API
    manager = MultibranchCameraManager(pipeline, builder.branches)
    api = CameraAPIServer(manager, port=8888)

    # 3. Set pipeline to READY
    ret = pipeline.set_state(Gst.State.READY)
    assert ret != Gst.StateChangeReturn.FAILURE, "Pipeline should reach READY"
    print("  [1/6] Pipeline READY")

    # 4. Add camera to both branches
    result = manager.add_camera("cam_int_1", TEST_VIDEO, ["recognition", "detection"])
    assert result == True, "Should add camera"
    assert manager.count() == 1
    print("  [2/6] Camera added to both branches")

    # 5. Add second camera to one branch
    result = manager.add_camera("cam_int_2", TEST_VIDEO, ["recognition"])
    assert result == True
    assert manager.count() == 2
    print("  [3/6] Second camera added")

    # 6. Remove first camera from detection only
    result = manager.remove_camera_from_branch("cam_int_1", "detection")
    assert result == True
    assert manager.get_camera_branches("cam_int_1") == ["recognition"]
    print("  [4/6] Camera removed from detection branch")

    # 7. Remove second camera entirely
    result = manager.remove_camera("cam_int_2")
    assert result == True
    assert manager.count() == 1
    print("  [5/6] Second camera removed entirely")

    # 8. Cleanup
    pipeline.set_state(Gst.State.NULL)
    print("  [6/6] Pipeline stopped")

    print("PASS: Full pipeline lifecycle")
    return True


def test_api_integration():
    """Test API server integration with manager"""
    print("\n=== Integration Test: API Integration ===")

    config = get_full_config()
    sinks = {"recognition": FakesinkAdapter(), "detection": FakesinkAdapter()}
    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    manager = MultibranchCameraManager(pipeline, builder.branches)
    api = CameraAPIServer(manager, port=8889)

    # Verify API has access to manager
    assert api.manager is manager
    assert api.manager.branches == builder.branches
    print("  [1/2] API connected to manager")

    # Verify routes are setup
    route_paths = [str(r.resource) for r in api.app.router.routes()]
    assert any("/api/cameras" in p for p in route_paths)
    assert any("/api/health" in p for p in route_paths)
    print("  [2/2] API routes configured")

    pipeline.set_state(Gst.State.NULL)
    print("PASS: API integration")
    return True


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("Phase 4: Integration Tests")
    print("=" * 60)

    tests = [
        test_full_pipeline_lifecycle,
        test_api_integration,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"FAIL: {test_fn.__name__} - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
