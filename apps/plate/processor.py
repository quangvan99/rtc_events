"""
License Plate Detection Processor

All-in-one module for plate detection with keypoint visualization.
Auto-registered with ProcessorRegistry using @register decorator.
"""

import multiprocessing as mp
import os
import threading
import time
import uuid
from ctypes import c_float, sizeof
from queue import Empty
from typing import Callable, Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.processor_registry import ProcessorRegistry
from src.sinks.base_sink import BaseSink
from src.common import BatchIterator, get_batch_meta, fps_probe_factory
from apps.plate.plate_ocr_mp import PlateOCRWorker
import pyds


# =============================================================================
# Display Functions
# =============================================================================

def update_display(
    obj_meta,
    display_meta,
    keypoints: Optional[np.ndarray],
    params: Dict[str, Any],
    plate_text: str = "",
    confidence: float = 0.0,
    is_valid: bool = True
) -> None:
    """Update OSD display for detected plate with border, keypoints, and text.

    Args:
        obj_meta: Object metadata for bounding box
        display_meta: NvDsDisplayMeta for drawing keypoints and text
        keypoints: Array of 4 corner points [[x,y], ...] or None
        params: Config params (colors, radius, etc.)
        plate_text: Recognized plate text to display
        confidence: OCR confidence score
        is_valid: Whether plate is valid (affects border color)
    """
    # Update border color
    rect = obj_meta.rect_params
    color = params.get("color_valid", [0.0, 1.0, 0.0, 1.0]) if is_valid else params.get("color_invalid", [1.0, 0.5, 0.0, 1.0])
    r, g, b, a = color
    rect.border_color.red, rect.border_color.green = r, g
    rect.border_color.blue, rect.border_color.alpha = b, a
    rect.border_width = params.get("border_width", 3)

    # Display plate text above bounding box
    if plate_text and display_meta is not None:
        num_labels = display_meta.num_labels
        max_labels = 16  # DeepStream limit

        if num_labels < max_labels:
            text_params = display_meta.text_params[num_labels]

            # Position text above bounding box
            text_params.x_offset = int(rect.left)
            text_params.y_offset = max(0, int(rect.top) - 25)

            # Format: "PLATE_TEXT (XX%)"
            text_params.display_text = f"{plate_text} ({confidence:.02%})"

            # Font settings
            text_params.font_params.font_name = "Serif"
            text_params.font_params.font_size = params.get("text_font_size", 12)
            text_params.font_params.font_color.red = 1.0
            text_params.font_params.font_color.green = 1.0
            text_params.font_params.font_color.blue = 1.0
            text_params.font_params.font_color.alpha = 1.0

            # Background
            text_params.set_bg_clr = 1
            text_params.text_bg_clr.red = 0.0
            text_params.text_bg_clr.green = 0.0
            text_params.text_bg_clr.blue = 0.0
            text_params.text_bg_clr.alpha = 0.7

            display_meta.num_labels = num_labels + 1

    # Draw keypoints if provided and visualization enabled
    if keypoints is not None and params.get("visualize_keypoints", True) and display_meta is not None:
        num_circles = display_meta.num_circles
        max_circles = 16  # DeepStream limit per display_meta

        keypoint_colors = params.get("keypoint_colors", [
            [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]
        ])
        keypoint_radius = params.get("keypoint_radius", 8)

        for i, point in enumerate(keypoints):
            if num_circles >= max_circles:
                break

            x, y = int(point[0]), int(point[1])
            kp_color = keypoint_colors[i % len(keypoint_colors)]

            circle = display_meta.circle_params[num_circles]
            circle.xc = x
            circle.yc = y
            circle.radius = keypoint_radius
            circle.circle_color.red = kp_color[0]
            circle.circle_color.green = kp_color[1]
            circle.circle_color.blue = kp_color[2]
            circle.circle_color.alpha = kp_color[3]
            circle.has_bg_color = 1
            circle.bg_color.red = kp_color[0]
            circle.bg_color.green = kp_color[1]
            circle.bg_color.blue = kp_color[2]
            circle.bg_color.alpha = kp_color[3]

            num_circles += 1

        display_meta.num_circles = num_circles


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


def extract_keypoints(obj_meta, muxer_width: int, muxer_height: int, point_threshold: float = 0.5) -> Optional[np.ndarray]:
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
            obj_meta.mask_params.width / muxer_width,
            obj_meta.mask_params.height / muxer_height
        )
        pad_x = (obj_meta.mask_params.width - muxer_width * gain) * 0.5
        pad_y = (obj_meta.mask_params.height - muxer_height * gain) * 0.5

        bbox = [
            obj_meta.rect_params.left,
            obj_meta.rect_params.top,
            obj_meta.rect_params.left + obj_meta.rect_params.width,
            obj_meta.rect_params.top + obj_meta.rect_params.height
        ]

        points = []
        for i in range(4):
            x = (data[i * 2] - pad_x) / gain
            y = (data[i * 2 + 1] - pad_y)/ gain
            x = np.clip(x, bbox[0], bbox[2])
            y = np.clip(y, bbox[1], bbox[3])
            points.append([x, y])
        # print(points)
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
        self._total_ocr_results = 0

        # Muxer dimensions (from config)
        self._muxer_width = 1920
        self._muxer_height = 1080

        # OCR multiprocessing
        self._input_queue: Optional[mp.Queue] = None
        self._output_queue: Optional[mp.Queue] = None
        self._ocr_worker: Optional[PlateOCRWorker] = None
        self._result_thread: Optional[threading.Thread] = None
        self._stop_result_thread = threading.Event()

        # OCR result cache: tracking_id -> (plate_text, confidence)
        self._ocr_cache: Dict[int, Tuple[str, float]] = {}
        self._ocr_cache_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "plate_recognition"

    def setup(self, config: Dict[str, Any], sink: BaseSink) -> None:
        """Initialize plate detection components."""
        self._config = config
        self._sink = sink
        params = config.get("params", {})

        # Muxer dimensions from config
        muxer = config.get("muxer", {})
        self._muxer_width = muxer.get("width", 1920)
        self._muxer_height = muxer.get("height", 1080)

        # OCR configuration (all params from config.yaml)
        engine_path = params["ocr_engine_path"]
        dict_path = params["ocr_dict_path"]
        queue_size = params.get("ocr_queue_size", 100)
        input_shape = tuple(params.get("ocr_input_shape", [2, 3, 48, 640]))

        # Create queues using spawn context for CUDA compatibility
        spawn_ctx = mp.get_context("spawn")
        self._input_queue = spawn_ctx.Queue(maxsize=queue_size)
        self._output_queue = spawn_ctx.Queue(maxsize=queue_size)

        # Create OCR worker
        self._ocr_worker = PlateOCRWorker(
            input_queue=self._input_queue,
            output_queue=self._output_queue,
            engine_path=engine_path,
            dict_path=dict_path,
            input_shape=input_shape,
        )

        print(f"[PlateRecognitionProcessor] Setup complete "
              f"(point_thresh={params.get('point_confidence_threshold', 0.5)}, "
              f"visualize_keypoints={params.get('visualize_keypoints', True)}, "
              f"ocr_engine={os.path.basename(engine_path)})")

    def _get_stats(self) -> dict:
        """Return stats dict for FPSMonitor."""
        return {
            "plates": self._total_plates,
            "frames": self._total_frames,
            "ocr_results": self._total_ocr_results,
        }

    def get_probes(self) -> Dict[str, Callable]:
        """Return probe callbacks."""
        params = self._config.get("params", {})
        return {
            "tracker_plate_probe": self.tracker_plate_probe,
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

    def tracker_plate_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """Main probe: Extract keypoints, visualize with OSD, send to OCR."""
        gst_buffer = info.get_buffer()
        batch = get_batch_meta(gst_buffer)
        params = self._config.get("params", {})
        point_threshold = params.get("point_confidence_threshold", 0.5)
        ocr_threshold = params.get("ocr_confidence_threshold", 0.9)

        images = {}
        for frame, obj in BatchIterator(batch):
            tracking_id = obj.object_id  # Get tracker ID

            keypoints = extract_keypoints(
                obj,
                muxer_width=self._muxer_width,
                muxer_height=self._muxer_height,
                point_threshold=point_threshold
            )

            # Check if we have cached OCR result for this tracking ID
            plate_text = ""
            confidence = 0.0
            with self._ocr_cache_lock:
                if tracking_id in self._ocr_cache:
                    plate_text, confidence = self._ocr_cache[tracking_id]

            # Only update display if we have plate text with sufficient confidence
            if plate_text and confidence >= ocr_threshold:
                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch)
                update_display(
                    obj, display_meta, keypoints, params,
                    plate_text=plate_text,
                    confidence=confidence,
                    is_valid=True
                )
                pyds.nvds_add_display_meta_to_frame(frame, display_meta)

            if keypoints is not None and frame.batch_id not in images:
                images[frame.batch_id] = extract_frame(gst_buffer, frame)

            if keypoints is not None and images.get(frame.batch_id) is not None:
                frame_align = warp_plate(images[frame.batch_id], keypoints)
                if frame_align is None:
                    continue

                check_result, img_list = check_plate_square(frame_align)
                processed_images = [frame_align]
                is_two_line = False
                if check_result is not None:
                    processed_images = img_list
                    is_two_line = True

                # Send to OCR queue with tracking_id
                if self._input_queue is not None:
                    self._total_plates += 1
                    try:
                        self._input_queue.put_nowait({
                            "id": f"{tracking_id}",
                            "tracking_id": tracking_id,
                            "images": processed_images,  # Send all parts together
                            "is_two_line": is_two_line,
                        })
                    except Exception as e:
                        print(f"[Plate] Queue full, skipping OCR: {e}")

        return Gst.PadProbeReturn.OK

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def _process_ocr_results(self) -> None:
        """Background thread to process OCR results from output_queue."""
        print("[PlateRecognitionProcessor] OCR result thread started")
        ocr_threshold = self._config.get("params", {}).get("ocr_confidence_threshold", 0.9)

        while not self._stop_result_thread.is_set():
            try:
                if self._output_queue is None:
                    time.sleep(0.1)
                    continue

                result = self._output_queue.get(timeout=0.1)
                self._total_ocr_results += 1

                text = result.get("text", "")
                confidence = result.get("confidence", 0.0)
                error = result.get("error")
                request_id = result.get("id", "unknown")
                tracking_id = result.get("tracking_id")

                if error:
                    print(f"[Plate OCR] Error {request_id}: {error}")
                elif text and tracking_id is not None:
                    # Cache result if confidence meets threshold
                    if confidence >= ocr_threshold:
                        with self._ocr_cache_lock:
                            # Update cache with new result (or keep better one)
                            if tracking_id not in self._ocr_cache or confidence > self._ocr_cache[tracking_id][1]:
                                self._ocr_cache[tracking_id] = (text, confidence)
                        print(f"[Plate OCR] ID:{tracking_id} {text} ({confidence:.2%}) [cached]")
                    else:
                        print(f"[Plate OCR] ID:{tracking_id} {text} ({confidence:.2%}) [below threshold]")

            except Empty:
                continue
            except Exception as e:
                print(f"[Plate OCR] Result thread error: {e}")
                continue

        print("[PlateRecognitionProcessor] OCR result thread stopped")

    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any) -> None:
        print(f"[PlateRecognitionProcessor] Pipeline built, branch: {branch_info.name}")

    def on_start(self) -> None:
        """Start OCR worker and result processing thread."""
        # Start OCR worker process
        if self._ocr_worker:
            self._ocr_worker.start()
            print("[PlateRecognitionProcessor] OCR worker started")

        # Start result processing thread
        self._stop_result_thread.clear()
        self._result_thread = threading.Thread(
            target=self._process_ocr_results,
            daemon=True,
            name="PlateOCRResultThread"
        )
        self._result_thread.start()

        print("[PlateRecognitionProcessor] Started")

    def on_stop(self) -> None:
        """Stop OCR worker and result processing thread."""
        # Stop result thread
        self._stop_result_thread.set()
        if self._result_thread and self._result_thread.is_alive():
            self._result_thread.join(timeout=2.0)

        # Stop OCR worker
        if self._ocr_worker:
            self._ocr_worker.stop()
            self._ocr_worker.join(timeout=5.0)
            print("[PlateRecognitionProcessor] OCR worker stopped")

        print(f"[PlateRecognitionProcessor] Stopped - plates={self._total_plates}, "
              f"frames={self._total_frames}, ocr_results={self._total_ocr_results}")

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Return processor statistics."""
        return {
            "total_plates": self._total_plates,
            "total_frames": self._total_frames,
            "total_ocr_results": self._total_ocr_results,
        }

    def set_source_mapper(self, source_mapper) -> None:
        """Set source mapper for camera ID resolution."""
        self._source_mapper = source_mapper