"""
Source ID Mapper - Bidirectional mapping camera_id <-> source_id

Maps user-defined camera_id (string) to DeepStream source_id (int).
"""

from __future__ import annotations
import threading


class SourceIDMapper:
    """
    Thread-safe bidirectional mapping camera_id <-> source_id
    
    Usage:
        mapper = SourceIDMapper()
        source_id = mapper.add("cam_01", "rtsp://...")
        camera_id = mapper.get_camera_id(source_id)
        mapper.remove("cam_01")
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cam_to_src: dict[str, int] = {}
        self._src_to_cam: dict[int, str] = {}
        self._next_id = 0
        self._freed: list[int] = []

    def add(self, camera_id: str, url: str = "") -> int:
        """Add camera, returns assigned source_id. Raises ValueError if exists."""
        with self._lock:
            if camera_id in self._cam_to_src:
                raise ValueError(f"Camera {camera_id} already exists")

            source_id = self._freed.pop(0) if self._freed else self._next_id
            if source_id == self._next_id:
                self._next_id += 1

            self._cam_to_src[camera_id] = source_id
            self._src_to_cam[source_id] = camera_id
            return source_id

    def remove(self, camera_id: str) -> int | None:
        """Remove camera, returns source_id or None."""
        with self._lock:
            source_id = self._cam_to_src.pop(camera_id, None)
            if source_id is not None:
                del self._src_to_cam[source_id]
                self._freed.append(source_id)
            return source_id

    def get_camera_id(self, source_id: int) -> str | None:
        """Get camera_id by source_id (thread-safe)."""
        with self._lock:
            return self._src_to_cam.get(source_id)

    def get_source_id(self, camera_id: str) -> int | None:
        """Get source_id by camera_id."""
        with self._lock:
            return self._cam_to_src.get(camera_id)

    def clear(self):
        """Clear all mappings."""
        with self._lock:
            self._cam_to_src.clear()
            self._src_to_cam.clear()
            self._freed.clear()
            self._next_id = 0
