"""Face recognition setup helper for multi-branch pipeline"""

from apps.face.database import FaceDatabase
from apps.face.tracker import TrackerManager
from apps.face.probes import FaceProbes


def setup_face_recognition(config: dict, sink) -> FaceProbes:
    """
    Setup face recognition components

    Args:
        config: Recognition config dict from YAML
        sink: Output sink adapter for recognition branch

    Returns:
        FaceProbes instance with registered callbacks
    """
    features_path = config.get("features_json", "data/face/features.json")

    print(f"[FaceDB] Loading {features_path}...")
    db = FaceDatabase(features_path)
    print(f"[FaceDB] Loaded {len(db.names)} faces")

    print("[Tracker] Initializing tracker...")
    tracker_mgr = TrackerManager(config)

    probes = FaceProbes(config, db, tracker_mgr, sink)
    return probes


def register_face_probes(builder, probes: FaceProbes):
    """Register face probes to pipeline builder"""
    builder.register_probe("tracker_probe", probes.tracker_probe)
    builder.register_probe("sgie_probe", probes.sgie_probe)
    builder.register_probe("fps_probe", probes.fps_probe)
