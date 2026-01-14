"""
License Plate Detection Processor

All-in-one module for plate detection with keypoint visualization.
Auto-registered with ProcessorRegistry using @register decorator.
"""

import time
from ctypes import c_float, sizeof
from typing import Callable, Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.processor_registry import ProcessorRegistry
from src.sinks.base_sink import BaseSink
from src.common import BatchIterator, get_batch_meta, fps_probe_factory
import pyds

# =============================================================================
# Constants
# =============================================================================

# OSD Colors (RGBA normalized 0.0-1.0)
COLOR_VALID = (0.0, 1.0, 0.0, 1.0)      # Green
COLOR_INVALID = (1.0, 0.5, 0.0, 1.0)    # Orange

# Keypoint visualization colors (RGBA normalized for DeepStream OSD)
KEYPOINT_COLORS_RGBA = [
    (0.0, 1.0, 0.0, 1.0),    # Green - top-left
    (1.0, 1.0, 0.0, 1.0),    # Yellow - top-right
    (1.0, 0.0, 0.0, 1.0),    # Red - bottom-right
    (1.0, 0.0, 1.0, 1.0),    # Magenta - bottom-left
]
KEYPOINT_RADIUS = 8

# Display settings
BORDER_WIDTH = 3

# Streammux dimensions (should match pipeline config)
STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080


# =============================================================================
# Display Functions
# =============================================================================

def update_display(obj_meta, is_valid: bool = True) -> None:
    """Update OSD display for detected plate."""
    rect = obj_meta.rect_params

    # Border color by validity
    r, g, b, a = COLOR_VALID if is_valid else COLOR_INVALID
    rect.border_color.red, rect.border_color.green = r, g
    rect.border_color.blue, rect.border_color.alpha = b, a
    rect.border_width = BORDER_WIDTH


def draw_keypoints_osd(display_meta, keypoints: np.ndarray) -> int:
    """Draw keypoints as circles using DeepStream OSD.

    Args:
        display_meta: NvDsDisplayMeta object
        keypoints: Array of 4 corner points [[x,y], ...]

    Returns:
        Number of circles added
    """
    num_circles = display_meta.num_circles
    max_circles = 16  # DeepStream limit per display_meta

    for i, point in enumerate(keypoints):
        if num_circles >= max_circles:
            break

        x, y = int(point[0]), int(point[1])
        color = KEYPOINT_COLORS_RGBA[i % len(KEYPOINT_COLORS_RGBA)]

        circle = display_meta.circle_params[num_circles]
        circle.xc = x
        circle.yc = y
        circle.radius = KEYPOINT_RADIUS
        circle.circle_color.red = color[0]
        circle.circle_color.green = color[1]
        circle.circle_color.blue = color[2]
        circle.circle_color.alpha = color[3]
        circle.has_bg_color = 1
        circle.bg_color.red = color[0]
        circle.bg_color.green = color[1]
        circle.bg_color.blue = color[2]
        circle.bg_color.alpha = color[3]

        num_circles += 1

    display_meta.num_circles = num_circles
    return len(keypoints)


# =============================================================================
# Post-Processing Functions
# =============================================================================

def check_plate_square(plate_img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """Detect 2-line plates based on aspect ratio."""
    height, width = plate_img.shape[:2]
    if width == 0:
        return None, None

    scale = width / height

    if scale < 2:  # 2-line plate (square-ish)
        mid_height = height // 2
        top_plate = plate_img[:mid_height, :]
        bottom_plate = plate_img[mid_height:, :]
        bottom_plate = cv2.resize(bottom_plate, (top_plate.shape[1], top_plate.shape[0]))
        horizontal = cv2.hconcat([top_plate, bottom_plate])
        return horizontal, [top_plate, bottom_plate]
    return None, None


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def warp_plate(frame: np.ndarray, points: np.ndarray) -> Optional[np.ndarray]:
    """Perspective warp plate region to rectangle."""
    try:
        points = order_points(points.astype(np.float32))

        width_a = np.linalg.norm(points[0] - points[1])
        width_b = np.linalg.norm(points[2] - points[3])
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(points[0] - points[3])
        height_b = np.linalg.norm(points[1] - points[2])
        max_height = int(max(height_a, height_b))

        if max_width < 10 or max_height < 10:
            return None

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(points, dst)
        warped = cv2.warpPerspective(frame, matrix, (max_width, max_height))

        return warped
    except Exception:
        return None


# =============================================================================
# Frame Extraction Helper
# =============================================================================

def extract_frame(gst_buffer, frame_meta) -> Optional[np.ndarray]:
    """Extract OpenCV frame from GStreamer buffer."""
    try:
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame_copy = np.array(n_frame, copy=True, order='C')
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
        pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        return frame_copy
    except Exception as e:
        print(f"[Plate] Error extracting frame: {e}")
        return None


def extract_keypoints(obj_meta, point_threshold: float = 0.5) -> Optional[np.ndarray]:
    """Extract 4 corner keypoints from mask metadata."""
    try:
        num_points = int(obj_meta.mask_params.size / (sizeof(c_float) * 2))
        print(f"[DEBUG extract_keypoints] mask_size={obj_meta.mask_params.size}, num_points={num_points}")
        if num_points < 4:
            print(f"[DEBUG] num_points < 4, returning None")
            return None

        data = obj_meta.mask_params.get_mask_array()
        if data is None or len(data) < 9:
            print(f"[DEBUG] data is None or len(data) < 9: data={data}, returning None")
            return None

        gain = min(
            obj_meta.mask_params.width / STREAMMUX_WIDTH,
            obj_meta.mask_params.height / STREAMMUX_HEIGHT
        )
        pad_x = (obj_meta.mask_params.width - STREAMMUX_WIDTH * gain) * 0.5

        bbox = [
            obj_meta.rect_params.left,
            obj_meta.rect_params.top,
            obj_meta.rect_params.left + obj_meta.rect_params.width,
            obj_meta.rect_params.top + obj_meta.rect_params.height
        ]

        points = []
        for i in range(4):
            x = (data[i * 2] - pad_x) / gain
            y = data[i * 2 + 1] / gain
            x = np.clip(x, bbox[0], bbox[2])
            y = np.clip(y, bbox[1], bbox[3])
            points.append([x, y])

        confidence = data[-1] if len(data) > 8 else 1.0
        if confidence < point_threshold:
            return None

        kps = np.array(points)
        pnt0 = np.maximum(kps[0], [bbox[0], bbox[1]])
        pnt1 = np.array([np.minimum(kps[1][0], bbox[2]), np.maximum(kps[1][1], bbox[1])])
        pnt2 = np.minimum(kps[3], [bbox[2], bbox[3]])
        pnt3 = np.array([np.maximum(kps[2][0], bbox[0]), np.minimum(kps[2][1], bbox[3])])

        return np.array([pnt0, pnt1, pnt2, pnt3], dtype=np.float32)

    except Exception as e:
        print(f"[Plate] Error extracting keypoints: {e}")
        return None


# =============================================================================
# Main Processor
# =============================================================================

@ProcessorRegistry.register("plate_recognition")
class PlateRecognitionProcessor:
    """
    License plate detection processor with keypoint visualization.

    Pipeline: PGIE (YOLOv8-pose) -> Probe -> OSD

    Config params (from branch YAML):
        params:
            point_confidence_threshold: 0.5
            log_interval: 1.0
            stats_interval: 10.0
            visualize_keypoints: true
    """

    def __init__(self, source_mapper=None):
        self._config: Dict[str, Any] = {}
        self._sink: Optional[BaseSink] = None
        self._source_mapper = source_mapper

        # Stats
        self._total_plates = 0
        self._total_frames = 0

    @property
    def name(self) -> str:
        return "plate_recognition"

    def setup(self, config: Dict[str, Any], sink: BaseSink) -> None:
        """Initialize plate detection components."""
        self._config = config
        self._sink = sink
        params = config.get("params", {})

        print(f"[PlateRecognitionProcessor] Setup complete "
              f"(point_thresh={params.get('point_confidence_threshold', 0.5)}, "
              f"visualize_keypoints={params.get('visualize_keypoints', True)})")

    def _get_stats(self) -> dict:
        """Return stats dict for FPSMonitor."""
        return {
            "plates": self._total_plates,
            "frames": self._total_frames,
        }

    def get_probes(self) -> Dict[str, Callable]:
        """Return probe callbacks."""
        params = self._config.get("params", {})
        return {
            "pgie_plate_probe": self._pgie_probe,
            "plate_fps_probe": fps_probe_factory(
                name="Plate",
                log_interval=params.get("log_interval", 1.0),
                stats_interval=params.get("stats_interval", 10.0),
                stats_callback=self._get_stats,
            ),
        }

    # -------------------------------------------------------------------------
    # Probe Callbacks
    # -------------------------------------------------------------------------

    def _pgie_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """Main probe: Extract keypoints, visualize with OSD, update display."""
        batch = get_batch_meta(info.get_buffer())
        for frame, obj in BatchIterator(batch):
            keypoints = extract_keypoints(obj, point_threshold=self._config.get("params", {}).get("point_confidence_threshold", 0.5))
            if keypoints is not None:
                print("=====check")
                surface = pyds.get_nvds_buf_surface(hash(info.get_buffer()), frame.batch_id)
                frame_image = np.array(surface, copy=True, order='C')                
                # frame = extract_frame(info.get_buffer(), frame)
                print("===================", frame_image.shape)
        return Gst.PadProbeReturn.OK

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any) -> None:
        print(f"[PlateRecognitionProcessor] Pipeline built, branch: {branch_info.name}")

    def on_start(self) -> None:
        print("[PlateRecognitionProcessor] Started")

    def on_stop(self) -> None:
        print(f"[PlateRecognitionProcessor] Stopped - plates={self._total_plates}, frames={self._total_frames}")

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Return processor statistics."""
        return {
            "total_plates": self._total_plates,
            "total_frames": self._total_frames,
        }

    def set_source_mapper(self, source_mapper) -> None:
        """Set source mapper for camera ID resolution."""
        self._source_mapper = source_mapper
