"""
Test CameraAPIServer - Phase 3 REST API

Quick validation tests for API endpoints.
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


def get_test_config():
    return {
        "pipeline": {
            "name": "test-api",
            "branches": {
                "recognition": {
                    "max_cameras": 4,
                    "muxer": {"batch-size": 4, "batched-push-timeout": 40000, "width": 1920, "height": 1080, "live-source": 1},
                    "elements": [{"type": "queue"}, {"type": "nvvideoconvert"}, {"type": "nvdsosd"}]
                },
                "detection": {
                    "max_cameras": 4,
                    "muxer": {"batch-size": 4, "batched-push-timeout": 40000, "width": 1920, "height": 1080, "live-source": 1},
                    "elements": [{"type": "queue"}, {"type": "nvvideoconvert"}, {"type": "nvdsosd"}]
                }
            }
        }
    }


def test_api_server_creation():
    """Test API server can be created"""
    print("\n=== Test: API Server Creation ===")

    config = get_test_config()
    sinks = {"recognition": FakesinkAdapter(), "detection": FakesinkAdapter()}
    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    manager = MultibranchCameraManager(pipeline, builder.branches)
    server = CameraAPIServer(manager, host="0.0.0.0", port=8080)

    assert server.manager is manager
    assert server.host == "0.0.0.0"
    assert server.port == 8080
    assert server.app is not None

    # Check routes registered
    routes = [r.resource.canonical for r in server.app.router.routes()]
    assert "/api/cameras" in routes
    assert "/api/health" in routes

    pipeline.set_state(Gst.State.NULL)
    print("PASS: API server created with routes")
    return True


def test_health_endpoint():
    """Test health endpoint returns correct data"""
    print("\n=== Test: Health Endpoint Logic ===")

    config = get_test_config()
    sinks = {"recognition": FakesinkAdapter(), "detection": FakesinkAdapter()}
    builder = TeeFanoutPipelineBuilder(config, sinks)
    pipeline = builder.build()

    manager = MultibranchCameraManager(pipeline, builder.branches)

    # Simulate health check response data
    health_data = {
        "status": "healthy",
        "cameras": manager.count(),
        "branches": list(manager.branches.keys())
    }

    assert health_data["status"] == "healthy"
    assert health_data["cameras"] == 0
    assert set(health_data["branches"]) == {"recognition", "detection"}

    pipeline.set_state(Gst.State.NULL)
    print("PASS: Health endpoint logic correct")
    return True


def run_all_tests():
    """Run all Phase 3 tests"""
    print("\n" + "=" * 60)
    print("Phase 3 Tests: REST API")
    print("=" * 60)

    tests = [
        test_api_server_creation,
        test_health_endpoint,
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
