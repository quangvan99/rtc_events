"""
ROI Config Generator for nvdsanalytics
Generates unified config file with multiple stream sections for all cameras.

nvdsanalytics uses single config file with [roi-filtering-stream-{source_id}] sections.
Each camera's ROI maps to its source_id in the pipeline.
"""

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Default paths
DEFAULT_ROI_DIR = "data/roi"
DEFAULT_CONFIG_FILE = "analytics_config.txt"


def polygon_to_nvds_format(polygon: List[List[int]]) -> str:
    """
    Convert polygon points to nvdsanalytics format.
    Input: [[100, 100], [1000, 100], [1000, 1000], [100, 1000]]
    Output: "100;100;1000;100;1000;1000;100;1000"
    """
    return ";".join(str(coord) for point in polygon for coord in point)


def generate_stream_section(camera_config: Dict[str, Any], source_id: int) -> List[str]:
    """
    Generate ROI filtering section for a single camera/stream.

    Args:
        camera_config: Camera config with roi_filtering list
        source_id: Source ID assigned by pipeline (sink_X pad number)

    Returns:
        List of config lines for this stream
    """
    roi_list = camera_config.get("roi_filtering", [])
    if not roi_list:
        return []

    lines = [
        f"[roi-filtering-stream-{source_id}]",
        "enable=1",
    ]

    for roi in roi_list:
        roi_id = roi.get("roi_id", "roi1")
        polygon = roi.get("polygon", [])
        class_ids = roi.get("class_id", [0])
        inverse = roi.get("inverse", False)

        # ROI polygon
        polygon_str = polygon_to_nvds_format(polygon)
        lines.append(f"roi-RF-{roi_id}={polygon_str}")

        # Inverse ROI setting
        lines.append(f"inverse-roi={1 if inverse else 0}")

        # Class IDs
        class_id_str = ";".join(str(c) for c in class_ids)
        lines.append(f"class-id={class_id_str}")

    lines.append("")
    return lines


class AnalyticsConfigManager:
    """
    Manages nvdsanalytics config file with multiple camera streams.
    Thread-safe for dynamic camera add/remove.
    """

    def __init__(
        self,
        output_dir: str = DEFAULT_ROI_DIR,
        config_filename: str = DEFAULT_CONFIG_FILE,
        config_width: int = 1920,
        config_height: int = 1080
    ):
        self._output_dir = output_dir
        self._config_filename = config_filename
        self._config_width = config_width
        self._config_height = config_height
        self._cameras: Dict[str, Dict] = {}  # camera_id -> {source_id, config}
        self._lock = threading.Lock()

        Path(output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def config_path(self) -> str:
        return os.path.join(self._output_dir, self._config_filename)

    def _write_config(self) -> None:
        """Write current config to file (called with lock held)."""
        lines = [
            "[property]",
            "enable=1",
            f"config-width={self._config_width}",
            f"config-height={self._config_height}",
            "osd-mode=2",
            "display-font-size=12",
            "",
        ]

        # Add sections for each camera sorted by source_id
        for camera_id, data in sorted(self._cameras.items(), key=lambda x: x[1]["source_id"]):
            section = generate_stream_section(data["config"], data["source_id"])
            lines.extend(section)

        with open(self.config_path, "w") as f:
            f.write("\n".join(lines))

        print(f"[AnalyticsConfig] Updated: {self.config_path} ({len(self._cameras)} cameras)")

    def add_camera(self, camera_id: str, source_id: int, camera_config: Dict[str, Any]) -> str:
        """
        Add camera ROI to analytics config.

        Args:
            camera_id: Camera identifier
            source_id: Source ID from pipeline (mux sink_{source_id})
            camera_config: Camera config containing roi_filtering

        Returns:
            Path to config file
        """
        with self._lock:
            self._cameras[camera_id] = {
                "source_id": source_id,
                "config": camera_config
            }
            self._write_config()
            return self.config_path

    def remove_camera(self, camera_id: str) -> None:
        """Remove camera from analytics config."""
        with self._lock:
            if camera_id in self._cameras:
                del self._cameras[camera_id]
                self._write_config()

    def update_camera(self, camera_id: str, camera_config: Dict[str, Any]) -> None:
        """Update camera ROI config."""
        with self._lock:
            if camera_id in self._cameras:
                self._cameras[camera_id]["config"] = camera_config
                self._write_config()

    def get_camera_source_id(self, camera_id: str) -> Optional[int]:
        """Get source_id for a camera."""
        with self._lock:
            if camera_id in self._cameras:
                return self._cameras[camera_id]["source_id"]
            return None

    def initialize_from_yaml(self, config_path: str) -> str:
        """
        Initialize config from YAML file (for static camera setup).

        Args:
            config_path: Path to area_monitoring config.yaml

        Returns:
            Path to generated config file
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        params = config.get("params", {})
        cameras = params.get("cameras", {})

        # Update dimensions from first camera if available
        if cameras:
            first_cam = next(iter(cameras.values()))
            self._config_width = first_cam.get("config_width", self._config_width)
            self._config_height = first_cam.get("config_height", self._config_height)

        with self._lock:
            self._cameras.clear()
            for idx, (cam_id, cam_cfg) in enumerate(cameras.items()):
                self._cameras[cam_id] = {
                    "source_id": idx,
                    "config": cam_cfg
                }
            self._write_config()

        return self.config_path


# Singleton instance for branch
_manager_instances: Dict[str, AnalyticsConfigManager] = {}


def get_analytics_config_manager(branch_name: str = "area_monitoring") -> AnalyticsConfigManager:
    """Get or create AnalyticsConfigManager for a branch."""
    if branch_name not in _manager_instances:
        _manager_instances[branch_name] = AnalyticsConfigManager(
            config_filename=f"{branch_name}_analytics.txt"
        )
    return _manager_instances[branch_name]


# Legacy functions for backward compatibility
def generate_roi_config_content(camera_config: Dict[str, Any], stream_id: int = 0) -> str:
    """Generate config content for single camera (legacy)."""
    width = camera_config.get("config_width", 1920)
    height = camera_config.get("config_height", 1080)

    lines = [
        "[property]",
        "enable=1",
        f"config-width={width}",
        f"config-height={height}",
        "osd-mode=2",
        "display-font-size=12",
        "",
    ]
    lines.extend(generate_stream_section(camera_config, stream_id))
    return "\n".join(lines)


def generate_roi_config_file(
    camera_id: str,
    camera_config: Dict[str, Any],
    output_dir: str = DEFAULT_ROI_DIR,
    stream_id: int = 0
) -> str:
    """Generate config file for single camera (legacy)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    content = generate_roi_config_content(camera_config, stream_id)
    output_path = os.path.join(output_dir, f"{camera_id}.txt")
    with open(output_path, "w") as f:
        f.write(content)
    print(f"[ROI Config] Generated: {output_path}")
    return output_path


if __name__ == "__main__":
    # Test with config.yaml
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    manager = get_analytics_config_manager("area_monitoring")
    path = manager.initialize_from_yaml(config_path)
    print(f"\nGenerated: {path}")

    # Show content
    with open(path) as f:
        print(f.read())
