"""
Source ID Mapper - Thread-safe bidirectional mapping between camera_id and source_id

Maps user-defined camera_id (string) to DeepStream source_id (int).
Handles source_id reuse after camera removal.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraInfo:
    """Camera metadata with source_id mapping"""
    camera_id: str
    source_id: int
    url: str
    name: str = ""
    added_at: float = 0.0

    def __post_init__(self):
        if self.added_at == 0.0:
            self.added_at = time.time()


class SourceIDMapper:
    """Thread-safe bidirectional mapping camera_id <-> source_id

    Usage:
        mapper = SourceIDMapper()
        source_id = mapper.add("cam_01", "rtsp://...", "Entrance")
        camera_id = mapper.get_camera_id(source_id)  # In probe
        mapper.remove("cam_01")
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._by_camera_id: dict[str, CameraInfo] = {}
        self._by_source_id: dict[int, str] = {}
        self._next_source_id = 0
        self._freed_ids: list[int] = []

    def add(self, camera_id: str, url: str, name: str = "") -> int:
        """
        Add camera to mapping

        Args:
            camera_id: Unique camera identifier (user-defined)
            url: Camera URL (rtsp://, file://, etc.)
            name: Optional display name

        Returns:
            Assigned source_id

        Raises:
            ValueError: If camera_id already exists
        """
        with self._lock:
            if camera_id in self._by_camera_id:
                raise ValueError(f"Camera {camera_id} already exists")

            # Reuse freed ID or allocate new
            if self._freed_ids:
                source_id = self._freed_ids.pop(0)
            else:
                source_id = self._next_source_id
                self._next_source_id += 1

            info = CameraInfo(camera_id, source_id, url, name)
            self._by_camera_id[camera_id] = info
            self._by_source_id[source_id] = camera_id

            print(f"[SourceIDMapper] Added {camera_id} -> source_id={source_id}")
            return source_id

    def remove(self, camera_id: str) -> Optional[int]:
        """
        Remove camera from mapping

        Args:
            camera_id: Camera identifier to remove

        Returns:
            Removed source_id, or None if not found
        """
        with self._lock:
            info = self._by_camera_id.pop(camera_id, None)
            if info is None:
                return None

            del self._by_source_id[info.source_id]
            self._freed_ids.append(info.source_id)

            print(f"[SourceIDMapper] Removed {camera_id} (source_id={info.source_id})")
            return info.source_id

    def get_by_camera_id(self, camera_id: str) -> Optional[CameraInfo]:
        """
        Get camera info by camera_id

        Args:
            camera_id: Camera identifier

        Returns:
            CameraInfo if found, None otherwise
        """
        with self._lock:
            return self._by_camera_id.get(camera_id)

    def get_camera_id(self, source_id: int) -> Optional[str]:
        """
        Get camera_id by source_id (for probes)

        Args:
            source_id: DeepStream source ID

        Returns:
            camera_id if found, None otherwise
        """
        with self._lock:
            return self._by_source_id.get(source_id)

    def get_camera_id_unsafe(self, source_id: int) -> Optional[str]:
        """
        Get camera_id without lock (use only in probe hot path)

        Safe if add/remove operations are infrequent and probe
        just needs read access. May return stale data briefly after remove.

        Args:
            source_id: DeepStream source ID

        Returns:
            camera_id if found, None otherwise
        """
        return self._by_source_id.get(source_id)

    def list_cameras(self) -> list[CameraInfo]:
        """
        List all active cameras

        Returns:
            List of CameraInfo for all active cameras
        """
        with self._lock:
            return list(self._by_camera_id.values())

    def count(self) -> int:
        """
        Get number of active cameras

        Returns:
            Number of active cameras
        """
        with self._lock:
            return len(self._by_camera_id)

    def has_camera(self, camera_id: str) -> bool:
        """
        Check if camera exists

        Args:
            camera_id: Camera identifier

        Returns:
            True if camera exists
        """
        with self._lock:
            return camera_id in self._by_camera_id

    def get_source_id(self, camera_id: str) -> Optional[int]:
        """
        Get source_id by camera_id

        Args:
            camera_id: Camera identifier

        Returns:
            source_id if found, None otherwise
        """
        with self._lock:
            info = self._by_camera_id.get(camera_id)
            return info.source_id if info else None
