"""
Test MultibranchCameraManager - Phase 2 Camera Management

Tests:
1. Add camera to single branch
2. Add camera to multiple branches
3. Remove camera from one branch (keep others)
4. Remove camera entirely
5. List cameras with branches
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from core.tee_fanout_builder import TeeFanoutPipelineBuilder, BranchInfo
from core.multibranch_camera_manager import MultibranchCameraManager
from core.camera_bin import CameraBin
from sinks.fakesink_adapter import FakesinkAdapter


# Test video file (use sample video or test pattern)
TEST_VIDEO = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
# Fallback to videotestsrc if file not found
USE_TEST_PATTERN = not os.path.exists("/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4")


def get_test_config():
    """Return minimal test config with 2 branches"""
    return {
        "pipeline": {
            "name": "test-camera-manager",
            "branches": {
                "recognition": {
                    "max_cameras": 4,
                    "muxer": {
                        "batch-size": 4,
                        "batched-push-timeout": 40000,
                        "width": 1920,
                        "height": 1080,
                        "live-source": 1,
                    },
                    "elements": [
                        {"type": "queue", "properties": {"max-size-buffers": 5, "leaky": 2}},
                        {"type": "nvvideoconvert"},
                        {"type": "nvdsosd", "properties": {"process-mode": 0}},
                    ]
                },
                "detection": {
                    "max_cameras": 4,
                    "muxer": {
                        "batch-size": 4,
                        "batched-push-timeout": 40000,
                        "width": 1920,
                        "height": 1080,
                        "live-source": 1,
                    },
                    "elements": [
                        {"type": "queue", "properties": {"max-size-buffers": 5, "leaky": 2}},
                        {"type": "nvvideoconvert"},
                        {"type": "nvdsosd", "properties": {"process-mode": 0}},
                    ]
                }
            }
        }
    }


def create_test_pipeline():
    """Create test pipeline with 2 branches"""
    config = get_test_config()
    sinks = {
        "recognition": FakesinkAdapter(sync=False),
        "detection": FakesinkAdapter(sync=False)
    }
    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()
    return pipeline, builder.branches


def test_manager_initialization():
    """Test MultibranchCameraManager initialization"""
    print("\n=== Test: Manager Initialization ===")

    pipeline, branches = create_test_pipeline()
    manager = MultibranchCameraManager(pipeline, branches)

    assert manager.pipeline is not None
    assert len(manager.branches) == 2
    assert manager.count() == 0
    assert manager.list_cameras() == {}

    print("  Pipeline: OK")
    print(f"  Branches: {list(manager.branches.keys())}")
    print(f"  Camera count: {manager.count()}")

    pipeline.set_state(Gst.State.NULL)
    print("PASS: Manager initialized correctly")
    return True


def test_add_camera_single_branch():
    """Test adding camera to single branch"""
    print("\n=== Test: Add Camera to Single Branch ===")

    pipeline, branches = create_test_pipeline()
    manager = MultibranchCameraManager(pipeline, branches)

    # Set pipeline to READY first
    pipeline.set_state(Gst.State.READY)

    # Add camera to recognition branch only
    result = manager.add_camera("cam1", TEST_VIDEO, ["recognition"])

    assert result == True, "add_camera should return True"
    assert manager.count() == 1
    assert manager.has_camera("cam1")

    cam_info = manager.list_cameras()
    assert "cam1" in cam_info
    assert cam_info["cam1"]["branches"] == ["recognition"]

    print(f"  Camera added: cam1")
    print(f"  Branches: {cam_info['cam1']['branches']}")
    print(f"  Source ID: {cam_info['cam1']['source_id']}")

    pipeline.set_state(Gst.State.NULL)
    print("PASS: Camera added to single branch")
    return True


def test_add_camera_multiple_branches():
    """Test adding camera to multiple branches"""
    print("\n=== Test: Add Camera to Multiple Branches ===")

    pipeline, branches = create_test_pipeline()
    manager = MultibranchCameraManager(pipeline, branches)

    pipeline.set_state(Gst.State.READY)

    # Add camera to both branches
    result = manager.add_camera("cam2", TEST_VIDEO, ["recognition", "detection"])

    assert result == True
    assert manager.count() == 1

    cam_info = manager.list_cameras()
    assert "cam2" in cam_info
    assert set(cam_info["cam2"]["branches"]) == {"recognition", "detection"}

    print(f"  Camera added: cam2")
    print(f"  Branches: {cam_info['cam2']['branches']}")

    pipeline.set_state(Gst.State.NULL)
    print("PASS: Camera added to multiple branches")
    return True


def test_add_duplicate_camera():
    """Test adding duplicate camera fails"""
    print("\n=== Test: Add Duplicate Camera ===")

    pipeline, branches = create_test_pipeline()
    manager = MultibranchCameraManager(pipeline, branches)

    pipeline.set_state(Gst.State.READY)

    result1 = manager.add_camera("cam_dup", TEST_VIDEO, ["recognition"])
    assert result1 == True

    # Try to add same camera again
    result2 = manager.add_camera("cam_dup", TEST_VIDEO, ["detection"])
    assert result2 == False, "Adding duplicate camera should fail"

    assert manager.count() == 1

    pipeline.set_state(Gst.State.NULL)
    print("PASS: Duplicate camera rejected correctly")
    return True


def test_add_camera_to_existing():
    """Test adding existing camera to additional branch"""
    print("\n=== Test: Add Existing Camera to New Branch ===")

    pipeline, branches = create_test_pipeline()
    manager = MultibranchCameraManager(pipeline, branches)

    pipeline.set_state(Gst.State.READY)

    # Add camera to one branch
    manager.add_camera("cam3", TEST_VIDEO, ["recognition"])

    cam_info = manager.list_cameras()
    assert cam_info["cam3"]["branches"] == ["recognition"]

    # Add same camera to another branch
    result = manager.add_camera_to_branch("cam3", "detection")
    assert result == True

    cam_info = manager.list_cameras()
    assert set(cam_info["cam3"]["branches"]) == {"recognition", "detection"}

    print(f"  Camera: cam3")
    print(f"  Branches after add: {cam_info['cam3']['branches']}")

    pipeline.set_state(Gst.State.NULL)
    print("PASS: Camera added to additional branch")
    return True


def test_remove_camera_from_branch():
    """Test removing camera from specific branch"""
    print("\n=== Test: Remove Camera from Branch ===")

    pipeline, branches = create_test_pipeline()
    manager = MultibranchCameraManager(pipeline, branches)

    pipeline.set_state(Gst.State.READY)

    # Add camera to both branches
    manager.add_camera("cam4", TEST_VIDEO, ["recognition", "detection"])

    cam_info = manager.list_cameras()
    assert set(cam_info["cam4"]["branches"]) == {"recognition", "detection"}

    # Remove from one branch
    result = manager.remove_camera_from_branch("cam4", "detection")
    assert result == True

    cam_info = manager.list_cameras()
    assert cam_info["cam4"]["branches"] == ["recognition"]
    assert manager.count() == 1  # Camera still exists

    print(f"  Camera: cam4")
    print(f"  Branches after remove: {cam_info['cam4']['branches']}")

    pipeline.set_state(Gst.State.NULL)
    print("PASS: Camera removed from specific branch")
    return True


def test_remove_camera_entirely():
    """Test removing camera from all branches"""
    print("\n=== Test: Remove Camera Entirely ===")

    pipeline, branches = create_test_pipeline()
    manager = MultibranchCameraManager(pipeline, branches)

    pipeline.set_state(Gst.State.READY)

    # Add cameras
    manager.add_camera("cam5", TEST_VIDEO, ["recognition", "detection"])
    manager.add_camera("cam6", TEST_VIDEO, ["recognition"])

    assert manager.count() == 2

    # Remove one camera entirely
    result = manager.remove_camera("cam5")
    assert result == True
    assert manager.count() == 1
    assert not manager.has_camera("cam5")
    assert manager.has_camera("cam6")

    print("  Removed: cam5")
    print(f"  Remaining cameras: {list(manager.list_cameras().keys())}")

    pipeline.set_state(Gst.State.NULL)
    print("PASS: Camera removed entirely")
    return True


def test_remove_nonexistent_camera():
    """Test removing non-existent camera"""
    print("\n=== Test: Remove Non-existent Camera ===")

    pipeline, branches = create_test_pipeline()
    manager = MultibranchCameraManager(pipeline, branches)

    pipeline.set_state(Gst.State.READY)

    result = manager.remove_camera("nonexistent")
    assert result == False

    pipeline.set_state(Gst.State.NULL)
    print("PASS: Non-existent camera removal handled correctly")
    return True


def test_get_camera_branches():
    """Test getting camera branch list"""
    print("\n=== Test: Get Camera Branches ===")

    pipeline, branches = create_test_pipeline()
    manager = MultibranchCameraManager(pipeline, branches)

    pipeline.set_state(Gst.State.READY)

    manager.add_camera("cam7", TEST_VIDEO, ["recognition", "detection"])

    branches_list = manager.get_camera_branches("cam7")
    assert set(branches_list) == {"recognition", "detection"}

    # Non-existent camera
    empty_list = manager.get_camera_branches("nonexistent")
    assert empty_list == []

    print(f"  cam7 branches: {branches_list}")

    pipeline.set_state(Gst.State.NULL)
    print("PASS: Get camera branches works correctly")
    return True


def run_all_tests():
    """Run all Phase 2 tests"""
    print("\n" + "=" * 60)
    print("Phase 2 Tests: Camera Management")
    print("=" * 60)

    if USE_TEST_PATTERN:
        print(f"\nNote: Using test video: {TEST_VIDEO}")
    else:
        print(f"\nNote: Test video not found, some tests may behave differently")

    tests = [
        test_manager_initialization,
        test_add_camera_single_branch,
        test_add_camera_multiple_branches,
        test_add_duplicate_camera,
        test_add_camera_to_existing,
        test_remove_camera_from_branch,
        test_remove_camera_entirely,
        test_remove_nonexistent_camera,
        test_get_camera_branches,
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
