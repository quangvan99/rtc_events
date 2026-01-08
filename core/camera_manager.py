"""
Camera Manager - REST API wrapper for nvmultiurisrcbin dynamic camera management

Provides clean Python API for add/remove cameras at runtime.
Integrates with SourceIDMapper for camera_id <-> source_id mapping.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

from core.source_mapper import SourceIDMapper, CameraInfo

logger = logging.getLogger(__name__)


@dataclass
class Camera:
    """Camera configuration for add operation"""
    camera_id: str
    url: str
    name: str = ""


class CameraManager:
    """REST API wrapper for nvmultiurisrcbin dynamic camera management

    Usage:
        manager = CameraManager(host="localhost", port=9000)
        manager.add(Camera("cam_01", "rtsp://...", "Entrance"))
        manager.remove("cam_01")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        mapper: Optional[SourceIDMapper] = None,
        timeout: float = 5.0
    ):
        """
        Initialize CameraManager

        Args:
            host: nvmultiurisrcbin REST API host
            port: nvmultiurisrcbin REST API port
            mapper: SourceIDMapper instance (created if None)
            timeout: REST request timeout in seconds
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/api/v1/stream"
        self.mapper = mapper or SourceIDMapper()
        self.timeout = timeout

    def _post_json(self, url: str, payload: dict) -> tuple[bool, str]:
        """Send POST request with JSON payload using urllib"""
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return True, resp.read().decode('utf-8')
        except urllib.error.HTTPError as e:
            return False, f"HTTP {e.code}: {e.reason}"
        except urllib.error.URLError as e:
            return False, str(e.reason)
        except Exception as e:
            return False, str(e)

    def add(self, camera: Camera) -> bool:
        """
        Add camera to pipeline

        Args:
            camera: Camera configuration

        Returns:
            True if successful, False otherwise
        """
        # Allocate source_id in mapper
        try:
            source_id = self.mapper.add(camera.camera_id, camera.url, camera.name)
        except ValueError as e:
            logger.error(f"Failed to add camera: {e}")
            return False

        # Build REST payload
        payload = {
            "value": {
                "camera_id": camera.camera_id,
                "camera_url": camera.url,
                "change": "camera_add"
            }
        }

        ok, msg = self._post_json(f"{self.base_url}/add", payload)

        if ok:
            logger.info(f"Camera added: {camera.camera_id} (source_id={source_id})")
            return True
        else:
            logger.error(f"REST API error: {msg}")
            self.mapper.remove(camera.camera_id)  # Rollback
            return False

    def remove(self, camera_id: str) -> bool:
        """
        Remove camera from pipeline

        Args:
            camera_id: Camera identifier to remove

        Returns:
            True if successful, False otherwise
        """
        info = self.mapper.get_by_camera_id(camera_id)
        if not info:
            logger.warning(f"Camera not found: {camera_id}")
            return False

        payload = {
            "value": {
                "camera_id": camera_id,
                "camera_url": info.url,  # Required by nvmultiurisrcbin REST API
                "change": "camera_remove"
            }
        }

        ok, msg = self._post_json(f"{self.base_url}/remove", payload)

        if ok:
            self.mapper.remove(camera_id)
            logger.info(f"Camera removed: {camera_id}")
            return True
        else:
            logger.error(f"REST API error: {msg}")
            return False

    def list(self) -> list[CameraInfo]:
        """
        List active cameras

        Returns:
            List of CameraInfo for all active cameras
        """
        return self.mapper.list_cameras()

    def get(self, camera_id: str) -> Optional[CameraInfo]:
        """
        Get camera info by camera_id

        Args:
            camera_id: Camera identifier

        Returns:
            CameraInfo if found, None otherwise
        """
        return self.mapper.get_by_camera_id(camera_id)

    def get_camera_id(self, source_id: int) -> Optional[str]:
        """
        Get camera_id by source_id (for probes)

        Args:
            source_id: DeepStream source ID

        Returns:
            camera_id if found, None otherwise
        """
        return self.mapper.get_camera_id(source_id)

    def get_camera_id_unsafe(self, source_id: int) -> Optional[str]:
        """
        Get camera_id without lock (use only in probe hot path)

        Args:
            source_id: DeepStream source ID

        Returns:
            camera_id if found, None otherwise
        """
        return self.mapper.get_camera_id_unsafe(source_id)

    def count(self) -> int:
        """
        Get number of active cameras

        Returns:
            Number of active cameras
        """
        return self.mapper.count()

    def has_camera(self, camera_id: str) -> bool:
        """
        Check if camera exists

        Args:
            camera_id: Camera identifier

        Returns:
            True if camera exists
        """
        return self.mapper.has_camera(camera_id)

    # Batch operations

    def add_batch(self, cameras: list[Camera]) -> dict[str, bool]:
        """
        Add multiple cameras

        Args:
            cameras: List of Camera configurations

        Returns:
            Dict mapping camera_id to success status
        """
        results = {}
        for cam in cameras:
            results[cam.camera_id] = self.add(cam)
        return results

    def remove_all(self) -> int:
        """
        Remove all cameras

        Returns:
            Number of cameras removed
        """
        removed = 0
        for info in list(self.mapper.list_cameras()):
            if self.remove(info.camera_id):
                removed += 1
        return removed

    # Health check

    def is_healthy(self) -> bool:
        """
        Check if REST API is reachable

        Returns:
            True if REST API responds, False otherwise
        """
        try:
            req = urllib.request.Request(f"http://{self.host}:{self.port}/")
            with urllib.request.urlopen(req, timeout=2.0):
                return True
        except Exception:
            return False
