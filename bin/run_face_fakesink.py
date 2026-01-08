#!/usr/bin/env python3
"""
Face Recognition Pipeline - Fakesink Mode

Usage:
    python bin/run_face_fakesink.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.face import FaceDatabase, TrackerManager
from apps.face.probes import FaceProbes
from sinks import FakesinkAdapter
from core import load_config, PipelineBuilder


CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "face-recognition.yaml")


def main():
    config = load_config(CONFIG_PATH)
    rec = config.get("recognition", {})

    # Create components
    sink = FakesinkAdapter(sync=False)
    db = FaceDatabase(rec.get("features_json", ""))
    tracker_mgr = TrackerManager(rec)
    probes = FaceProbes(rec, db, tracker_mgr, sink)

    # Build pipeline
    builder = PipelineBuilder(config, sink)
    builder.register_probe("tracker_probe", probes.tracker_probe)
    builder.register_probe("sgie_probe", probes.sgie_probe)
    builder.register_probe("fps_probe", probes.fps_probe)

    print("=" * 70)
    print("DeepStream Face Recognition (fakesink mode)")
    print("=" * 70)
    print(f"Config: {CONFIG_PATH}")
    print(f"Features: {len(db.names)} registered faces")
    print("=" * 70)

    builder.run()


if __name__ == "__main__":
    main()
