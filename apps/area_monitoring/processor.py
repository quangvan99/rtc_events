from typing import Any, Callable, Dict, List, Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import pyds

from src.processor_registry import ProcessorRegistry
from src.sinks.base_sink import BaseSink
from src.common import get_batch_meta, fps_probe_factory, BatchIterator
from apps.area_monitoring.roi_config_generator import get_analytics_config_manager

# -----------------------------------------------------------------------------
# Main Processor
# -----------------------------------------------------------------------------

@ProcessorRegistry.register("area_monitoring")
class AreaMonitoringProcessor:
    def __init__(self, source_mapper=None):
        self._config: Dict[str, Any] = {}
        self._sink: Optional[BaseSink] = None
        self._analytics_manager = None

    @property
    def name(self) -> str:
        return "area_monitoring"

    def setup(self, config: Dict[str, Any], sink: BaseSink) -> None:
        """Initialize area monitoring components"""
        self._config = config
        self._sink = sink

        # Initialize analytics config manager
        self._analytics_manager = get_analytics_config_manager("area_monitoring")

        # Generate initial config from yaml cameras
        params = config.get("params", {})
        cameras = params.get("cameras", {})

        # Set dimensions from first camera
        if cameras:
            first_cam = next(iter(cameras.values()))
            self._analytics_manager._config_width = first_cam.get("config_width", 1920)
            self._analytics_manager._config_height = first_cam.get("config_height", 1080)

        # Pre-register cameras from config (static setup)
        for idx, (camera_id, camera_config) in enumerate(cameras.items()):
            self._analytics_manager.add_camera(camera_id, idx, camera_config)

    def get_probes(self) -> Dict[str, Callable]:
        """Return probe callbacks"""
        params = self._config.get("params", {})
        return {
            "area_fps_probe": fps_probe_factory(
                name="AreaMonitoring",
                log_interval=params.get("log_interval", 1.0),
                stats_interval=params.get("stats_interval", 10.0),
            ),
        }

    def get_analytics_config_path(self) -> str:
        """Get path to nvdsanalytics config file."""
        if self._analytics_manager:
            return self._analytics_manager.config_path
        return "data/roi/area_monitoring_analytics.txt"

    def add_camera_roi(self, camera_id: str, source_id: int, camera_config: Dict[str, Any]) -> str:
        """
        Add camera ROI to analytics config (called when camera is added dynamically).

        Args:
            camera_id: Camera identifier
            source_id: Source ID from pipeline
            camera_config: Camera config with roi_filtering

        Returns:
            Path to updated config file
        """
        if self._analytics_manager:
            return self._analytics_manager.add_camera(camera_id, source_id, camera_config)
        return ""

    def remove_camera_roi(self, camera_id: str) -> None:
        """Remove camera ROI from analytics config."""
        if self._analytics_manager:
            self._analytics_manager.remove_camera(camera_id)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any) -> None:
        """Called when pipeline is built"""
        print(f"[AreaMonitoringProcessor] Pipeline built, branch: {branch_info.name}")

    def on_start(self) -> None:
        """Called when pipeline starts"""
        print("[AreaMonitoringProcessor] Started")

    def on_stop(self) -> None:
        """Called when pipeline stops"""
        pass
