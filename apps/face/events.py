"""Event emission for face detection with multi-camera support"""

import time

from src.sinks.base_sink import BaseSink


class FaceEventEmitter:
    """Sends face detection events via sink

    Supports multi-camera mode with (source_id, object_id) tracking.
    """

    def __init__(self, sink: BaseSink, avatars: dict[str, str],
                 face_db=None, source_mapper=None):
        """
        Initialize FaceEventEmitter

        Args:
            sink: Sink adapter with send_event() method
            avatars: Dict mapping name -> avatar URL
            face_db: FaceDatabase instance for getting person info
            source_mapper: Optional SourceIDMapper for camera_id lookup
        """
        self.sink = sink
        self.avatars = avatars
        self.face_db = face_db
        self.source_mapper = source_mapper
        # Track (source_id, object_id) tuples for multi-camera
        self.sent_faces: set[tuple[int, int]] = set()

    def send_detection(self, source_id: int, object_id: int, name: str) -> None:
        """
        Send face detection event (once per source_id, object_id)

        Args:
            source_id: Camera source ID
            object_id: Tracker object ID
            name: Detected person name
        """
        key = (source_id, object_id)
        if key in self.sent_faces:
            return

        self.sent_faces.add(key)

        # Get camera_id from mapper
        camera_id = None
        if self.source_mapper:
            camera_id = self.source_mapper.get_camera_id(source_id)

        # Get person info from database
        info = None
        if self.face_db and hasattr(self.face_db, "get_person_info"):
            try:
                info = self.face_db.get_person_info(name)
            except Exception:
                info = None

        event = {
            "type": "face_detected",
            "camera_id": camera_id,
            "source_id": source_id,
            "name": name,
            "timestamp": time.strftime("%H:%M:%S"),
            "object_id": object_id,
            "person_id": (info.get("person_id") if info else None),
            "person_name": (info.get("person_name") if info else name),
            "company_name": (info.get("company_name") if info else None),
            "job_title": (info.get("job_title") if info else None),
            "avatar": self.avatars.get(name),
        }

        self.sink.send_event(event)

    def cleanup(self, removed: list[tuple[int, int]]) -> None:
        """
        Remove stale (source_id, object_id) tuples

        Args:
            removed: List of (source_id, object_id) tuples to remove
        """
        for key in removed:
            self.sent_faces.discard(key)
