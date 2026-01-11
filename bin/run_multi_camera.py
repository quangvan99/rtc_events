#!/usr/bin/env python3
"""
Multi-Camera Face Recognition Pipeline

Uses nvmultiurisrcbin for dynamic camera add/remove via REST API.

Usage:
    python bin/run_multi_camera.py

Environment Variables:
    FEATURES_JSON  - Path to face features JSON
    REST_HOST      - REST API bind address (default: 0.0.0.0)
    REST_PORT      - REST API port (default: 9000)
    MAX_CAMERAS    - Maximum cameras (default: 16)
    OUTPUT_MODE    - Output mode: fakesink, webrtc (default: fakesink)

Add camera at runtime:
    curl -X POST 'http://localhost:9000/api/v1/stream/add' \\
      -H 'Content-Type: application/json' \\
      -d '{"value":{"camera_id":"cam1","camera_url":"rtsp://...","change":"camera_add"}}'

Remove camera:
    curl -X POST 'http://localhost:9000/api/v1/stream/remove' \\
      -H 'Content-Type: application/json' \\
      -d '{"value":{"camera_id":"cam1","change":"camera_remove"}}'
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.face import FaceDatabase, TrackerManager
from apps.face.probes import FaceProbes
from sinks import FakesinkAdapter
from core import load_config, PipelineBuilder
from core.source_mapper import SourceIDMapper
from core.camera_manager import CameraManager, Camera


CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "multi-camera.yaml")


def main():
    config = load_config(CONFIG_PATH)
    rec = config.get("recognition", {})
    cam_cfg = config.get("camera", {})

    # Create source mapper for camera_id <-> source_id tracking
    source_mapper = SourceIDMapper()

    # Create camera manager (REST API wrapper)
    camera_mgr = CameraManager(
        host=cam_cfg.get("rest_host", "localhost"),
        port=cam_cfg.get("rest_port", 9000),
        mapper=source_mapper
    )

    # Create components
    sink = FakesinkAdapter(sync=False)
    db = FaceDatabase(rec.get("features_json", ""))
    tracker_mgr = TrackerManager(rec)

    # Pass source_mapper to probes for multi-camera support
    probes = FaceProbes(rec, db, tracker_mgr, sink, source_mapper=source_mapper)

    # Build pipeline
    builder = PipelineBuilder(config, sink)
    builder.register_probe("tracker_probe", probes.tracker_probe)
    builder.register_probe("sgie_probe", probes.sgie_probe)
    builder.register_probe("fps_probe", probes.fps_probe)

    rest_host = cam_cfg.get("rest_host", "localhost")
    rest_port = cam_cfg.get("rest_port", 9000)
    max_cameras = cam_cfg.get("max_cameras", 16)

    print("=" * 70)
    print("Multi-Camera Face Recognition (nvmultiurisrcbin)")
    print("=" * 70)
    print(f"Config: {CONFIG_PATH}")
    print(f"Features: {len(db.names)} registered faces")
    print(f"REST API: http://{rest_host}:{rest_port}")
    print(f"Max cameras: {max_cameras}")
    print("=" * 70)
    print("Add camera:")
    print(f"  curl -X POST 'http://{rest_host}:{rest_port}/api/v1/stream/add' \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"value\":{\"camera_id\":\"cam1\",\"camera_url\":\"rtsp://...\",\"change\":\"camera_add\"}}'")
    print("=" * 70)

    # Add initial cameras from config (optional)
    initial_cameras = config.get("cameras", [])
    for cam in initial_cameras:
        camera_mgr.add(Camera(
            camera_id=cam["id"],
            url=cam["url"],
            name=cam.get("name", "")
        ))

    builder.run()


if __name__ == "__main__":
    main()
