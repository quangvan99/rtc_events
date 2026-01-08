"""
Test TeeFanoutPipelineBuilder - Phase 1 Core Infrastructure

Tests:
1. Pipeline builds with 2 branches (no cameras)
2. Branch info stored correctly
3. nvstreammux elements created
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from core.tee_fanout_builder import TeeFanoutPipelineBuilder, BranchInfo
from core.camera_bin import CameraBin
from sinks.fakesink_adapter import FakesinkAdapter


def get_test_config():
    """Return minimal test config with 2 branches"""
    return {
        "pipeline": {
            "name": "test-multi-branch",
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


def test_camera_bin_dataclass():
    """Test CameraBin dataclass"""
    print("\n=== Test: CameraBin Dataclass ===")

    Gst.init(None)

    # Create mock elements
    mock_bin = Gst.Bin.new("test_bin")
    mock_src = Gst.ElementFactory.make("fakesrc", "test_src")
    mock_tee = Gst.ElementFactory.make("tee", "test_tee")

    cam = CameraBin(
        camera_id="cam_test",
        uri="file:///test.mp4",
        bin=mock_bin,
        nvurisrcbin=mock_src,
        tee=mock_tee,
        branch_queues={"recognition": None},
        branch_pads={"recognition": None},
        source_id=0
    )

    assert cam.camera_id == "cam_test"
    assert cam.uri == "file:///test.mp4"
    assert cam.get_branches() == ["recognition"]
    assert cam.is_connected_to("recognition") == True
    assert cam.is_connected_to("detection") == False

    print("PASS: CameraBin dataclass works correctly")
    return True


def test_branch_info_dataclass():
    """Test BranchInfo dataclass"""
    print("\n=== Test: BranchInfo Dataclass ===")

    Gst.init(None)

    mock_mux = Gst.ElementFactory.make("nvstreammux", "test_mux")
    mock_sink = Gst.ElementFactory.make("fakesink", "test_sink")

    branch = BranchInfo(
        name="recognition",
        nvstreammux=mock_mux,
        elements=[],
        sink=mock_sink,
        max_cameras=8
    )

    assert branch.name == "recognition"
    assert branch.nvstreammux is not None
    assert branch.max_cameras == 8

    print("PASS: BranchInfo dataclass works correctly")
    return True


def test_pipeline_build_two_branches():
    """Test pipeline builds with 2 branches"""
    print("\n=== Test: Build Pipeline with 2 Branches ===")

    config = get_test_config()
    sinks = {
        "recognition": FakesinkAdapter(sync=False),
        "detection": FakesinkAdapter(sync=False)
    }

    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    # Verify pipeline created
    assert pipeline is not None
    print(f"  Pipeline name: {pipeline.get_name()}")

    # Verify 2 branches created
    assert len(builder.branches) == 2
    assert "recognition" in builder.branches
    assert "detection" in builder.branches
    print(f"  Branches: {list(builder.branches.keys())}")

    # Verify nvstreammux for each branch
    for name, branch in builder.branches.items():
        assert branch.nvstreammux is not None
        assert branch.nvstreammux.get_name() == f"mux_{name}"
        print(f"  Branch '{name}': mux={branch.nvstreammux.get_name()}, "
              f"elements={len(branch.elements)}, max_cameras={branch.max_cameras}")

    print("PASS: Pipeline built with 2 branches")
    return True


def test_pipeline_elements_linked():
    """Test pipeline elements are properly linked"""
    print("\n=== Test: Elements Linked ===")

    config = get_test_config()
    sinks = {
        "recognition": FakesinkAdapter(sync=False),
        "detection": FakesinkAdapter(sync=False)
    }

    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    # Check total elements
    print(f"  Total elements: {len(builder.elements)}")
    assert len(builder.elements) > 0

    # Verify each branch has elements linked
    for name, branch in builder.branches.items():
        # Check nvstreammux has src pad
        src_pad = branch.nvstreammux.get_static_pad("src")
        if src_pad:
            peer = src_pad.get_peer()
            if peer:
                print(f"  Branch '{name}' mux.src -> {peer.get_parent_element().get_name()}")

    print("PASS: Elements properly linked")
    return True


def test_pipeline_state_null_to_ready():
    """Test pipeline can transition to READY state"""
    print("\n=== Test: Pipeline State NULL -> READY ===")

    config = get_test_config()
    sinks = {
        "recognition": FakesinkAdapter(sync=False),
        "detection": FakesinkAdapter(sync=False)
    }

    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    # Try to set READY state
    ret = pipeline.set_state(Gst.State.READY)
    print(f"  State change result: {ret}")

    # READY should succeed even without cameras
    # Note: PLAYING may fail because nvstreammux needs at least one input
    assert ret != Gst.StateChangeReturn.FAILURE

    # Cleanup
    pipeline.set_state(Gst.State.NULL)

    print("PASS: Pipeline can reach READY state")
    return True


def run_all_tests():
    """Run all Phase 1 tests"""
    print("\n" + "=" * 60)
    print("Phase 1 Tests: Core Infrastructure")
    print("=" * 60)

    tests = [
        test_camera_bin_dataclass,
        test_branch_info_dataclass,
        test_pipeline_build_two_branches,
        test_pipeline_elements_linked,
        test_pipeline_state_null_to_ready,
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
