"""Event emission for face detection"""

import time

from sinks.base_sink import BaseSink


class FaceEventEmitter:
    """Sends face detection events via sink"""

    def __init__(self, sink: BaseSink, avatars: dict[str, str], face_db=None):
        """
        Args:
            sink: Sink adapter with send_event() method
            avatars: Dict mapping name -> avatar URL
            face_db: FaceDatabase instance for getting person info
        """
        self.sink = sink
        self.avatars = avatars
        self.face_db = face_db
        self.sent_faces: set[int] = set()

    def send_detection(self, object_id: int, name: str) -> None:
        """
        Send face detection event (once per object_id)

        Args:
            object_id: Tracker object ID
            name: Detected person name
        """
        if object_id in self.sent_faces:
            return

        self.sent_faces.add(object_id)

        # Lấy thông tin từ mapping.csv (label có thể là ID_nguoi hoặc Ten_nguoi)
        info = None
        if self.face_db and hasattr(self.face_db, "get_person_info"):
            try:
                info = self.face_db.get_person_info(name)
            except Exception:
                info = None

        event = {
            "type": "face_detected",
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

    def cleanup(self, removed_ids: list[int]) -> None:
        """
        Remove stale object_ids

        Args:
            removed_ids: List of object IDs to remove from tracking
        """
        for oid in removed_ids:
            self.sent_faces.discard(oid)
