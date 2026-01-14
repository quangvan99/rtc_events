"""
License Plate Recognition Processor

All-in-one module for plate detection and OCR using multiprocessing.
Auto-registered with ProcessorRegistry using @register decorator.

The OCR engine runs in a separate process (spawn) to avoid CUDA conflicts
with GStreamer/DeepStream.
"""

import time
from ctypes import c_float, sizeof
from typing import Callable, Dict, Any, List, Optional, Set

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.processor_registry import ProcessorRegistry
from src.sinks.base_sink import BaseSink
from src.common import BatchIterator, get_batch_meta, fps_probe_factory

# Import OCR worker and helper functions from ocr module
from apps.plate.ocr import (
    OCRWorkerProcess,
    check_plate_square,
    check_format_plate,
    check_format_plate_append,
)


# =============================================================================
# Constants
# =============================================================================

# OSD Colors (RGBA)
COLOR_VALID = (0.0, 1.0, 0.0, 1.0)      # Green
COLOR_INVALID = (1.0, 0.5, 0.0, 1.0)    # Orange
COLOR_TEXT = (1.0, 1.0, 1.0, 1.0)       # White
COLOR_TEXT_BG = (0.0, 0.0, 0.0, 0.7)    # Black transparent

# Display settings
BORDER_WIDTH = 3
FONT_SIZE = 14
FONT_NAME = "Serif"

# Streammux dimensions (should match pipeline config)
STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080


# =============================================================================
# Display Functions
# =============================================================================

def update_display(obj_meta, plate_text: str) -> None:
    """Update OSD display for detected plate."""
    is_valid = plate_text and not plate_text.startswith("INVALID") and plate_text != "??????"
    rect = obj_meta.rect_params

    # Border color by validity
    r, g, b, a = COLOR_VALID if is_valid else COLOR_INVALID
    rect.border_color.red, rect.border_color.green = r, g
    rect.border_color.blue, rect.border_color.alpha = b, a
    rect.border_width = BORDER_WIDTH

    # Display text
    text = obj_meta.text_params
    text.display_text = plate_text or "??????"
    text.x_offset = int(rect.left)
    text.y_offset = max(0, int(rect.top) - 25)
    text.font_params.font_name = FONT_NAME
    text.font_params.font_size = FONT_SIZE

    # Text color
    r, g, b, a = COLOR_TEXT
    text.font_params.font_color.red, text.font_params.font_color.green = r, g
    text.font_params.font_color.blue, text.font_params.font_color.alpha = b, a

    # Text background
    text.set_bg_clr = 1
    r, g, b, a = COLOR_TEXT_BG
    text.text_bg_clr.red, text.text_bg_clr.green = r, g
    text.text_bg_clr.blue, text.text_bg_clr.alpha = b, a


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
    """Extract full frame from GStreamer buffer.

    Requires nvbuf-memory-type: 3 (CUDA_UNIFIED) for dGPU in muxer config.
    """
    import pyds
    try:
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame_copy = np.array(n_frame, copy=True, order='C')
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
        pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        return frame_copy
    except Exception as e:
        print(f"[Plate] Error extracting frame: {e}", flush=True)
        try:
            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        except:
            pass
        return None


def extract_keypoints(obj_meta, point_threshold: float = 0.5) -> Optional[np.ndarray]:
    """Extract 4 corner keypoints from mask metadata.

    Returns ordered points: top-left, top-right, bottom-right, bottom-left
    """
    try:
        from ctypes import c_float, sizeof
        mask_size = obj_meta.mask_params.size
        num_points = int(mask_size / (sizeof(c_float) * 2))

        if num_points < 4:
            return None

        try:
            data = obj_meta.mask_params.get_mask_array()
        except Exception:
            return None

        if data is None or len(data) < 9:
            return None

        gain = min(
            obj_meta.mask_params.width / STREAMMUX_WIDTH,
            obj_meta.mask_params.height / STREAMMUX_HEIGHT
        )
        pad_x = (obj_meta.mask_params.width - STREAMMUX_WIDTH * gain) * 0.5
        pad_y = (obj_meta.mask_params.height - STREAMMUX_HEIGHT * gain) * 0.5

        bbox = [
            obj_meta.rect_params.left,
            obj_meta.rect_params.top,
            obj_meta.rect_params.left + obj_meta.rect_params.width,
            obj_meta.rect_params.top + obj_meta.rect_params.height
        ]

        points = []
        for i in range(4):
            x = (data[i * 2] - pad_x) / gain
            y = (data[i * 2 + 1] - pad_y) / gain
            x = np.clip(x, bbox[0], bbox[2])
            y = np.clip(y, bbox[1], bbox[3])
            points.append([x, y])

        confidence = data[-1] if len(data) > 8 else 1.0
        if confidence < point_threshold:
            return None

        # Normalize points within bbox
        kps = np.array(points)
        pnt0 = np.maximum(kps[0], [bbox[0], bbox[1]])
        pnt1 = np.array([np.minimum(kps[1][0], bbox[2]), np.maximum(kps[1][1], bbox[1])])
        pnt2 = np.minimum(kps[3], [bbox[2], bbox[3]])
        pnt3 = np.array([np.maximum(kps[2][0], bbox[0]), np.minimum(kps[2][1], bbox[3])])

        return np.array([pnt0, pnt1, pnt2, pnt3], dtype=np.float32)

    except Exception as e:
        print(f"[Plate] Error extracting keypoints: {e}")
        return None


def warp_plate(frame: np.ndarray, points: np.ndarray) -> Optional[np.ndarray]:
    """Perspective warp plate region to rectangle using 4 corner points."""
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


def crop_plate_bbox(frame: np.ndarray, obj_meta) -> Optional[np.ndarray]:
    """Fallback: crop plate using bounding box (no warp)."""
    try:
        rect = obj_meta.rect_params
        left, top = int(rect.left), int(rect.top)
        width, height = int(rect.width), int(rect.height)

        if width < 10 or height < 10:
            return None

        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, left)
        y1 = max(0, top)
        x2 = min(frame_w, left + width)
        y2 = min(frame_h, top + height)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()
    except Exception:
        return None


# =============================================================================
# Main Processor
# =============================================================================

@ProcessorRegistry.register("plate_recognition")
class PlateRecognitionProcessor:
    """
    License plate detection and OCR processor.

    Pipeline: PGIE (YOLOv8-pose) -> Tracker -> Probe (OCR) -> OSD

    Config params (from branch YAML):
        params:
            point_confidence_threshold: 0.5
            ocr_confidence_threshold: 0.9
            log_interval: 1.0
            stats_interval: 10.0
    """

    def __init__(self, source_mapper=None):
        self._config: Dict[str, Any] = {}
        self._sink: Optional[BaseSink] = None
        self._source_mapper = source_mapper

        # OCR worker (multiprocessing)
        self._ocr_worker: Optional[OCRWorkerProcess] = None

        # Tracking state
        self._processed_ids: Set[int] = set()
        self._ocr_cache: Dict[int, str] = {}

        # Stats
        self._total_plates = 0
        self._total_frames = 0
        self._ocr_submitted = 0
        self._ocr_dropped = 0

    @property
    def name(self) -> str:
        return "plate_recognition"

    def setup(self, config: Dict[str, Any], sink: BaseSink) -> None:
        """Initialize plate recognition components."""
        self._config = config
        self._sink = sink
        params = config.get("params", {})

        # Get OCR config
        engine_path = params.get(
            "ocr_engine_path",
            "data/license_plate/ocr/ppocr_dummy.engine"
        )
        dict_path = params.get(
            "ocr_dict_path",
            "data/license_plate/ocr/licence_plate_dict.txt"
        )
        ocr_threshold = params.get("ocr_confidence_threshold", 0.9)
        queue_maxsize = params.get("ocr_queue_maxsize", 10)

        # Initialize OCR worker process
        self._ocr_worker = OCRWorkerProcess(
            engine_path=engine_path,
            dict_path=dict_path,
            ocr_threshold=ocr_threshold,
            maxsize=queue_maxsize,
        )

        print(f"[PlateRecognitionProcessor] Setup complete "
              f"(point_thresh={params.get('point_confidence_threshold', 0.5)}, "
              f"ocr_thresh={ocr_threshold}, queue_maxsize={queue_maxsize})")

    def _get_stats(self) -> dict:
        """Return stats dict for FPSMonitor."""
        return {
            "plates": self._total_plates,
            "frames": self._total_frames,
            "cached": len(self._ocr_cache),
            "ocr_submitted": self._ocr_submitted,
            "ocr_dropped": self._ocr_dropped,
        }

    def get_probes(self) -> Dict[str, Callable]:
        """Return probe callbacks."""
        params = self._config.get("params", {})
        return {
            "tracker_plate_probe": self._tracker_probe,
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

    def _tracker_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """Main probe: Extract plates with perspective warp, submit to OCR worker.

        Flow:
        1. Poll OCR results from worker (updates cache)
        2. For new plates:
           a. Extract full frame (lazy load)
           b. Extract keypoints -> perspective warp (or fallback to bbox crop)
           c. Submit to OCR worker
        3. Display cached OCR results
        """
        import pyds

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch:
            return Gst.PadProbeReturn.OK

        params = self._config.get("params", {})
        point_threshold = params.get("point_confidence_threshold", 0.5)

        # 1. Poll OCR results from worker (non-blocking)
        self._poll_ocr_results()

        # 2. Process each frame
        l_frame = batch.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                frame_image = None  # Lazy load

                l_obj = frame_meta.obj_meta_list
                while l_obj:
                    try:
                        obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                        self._total_frames += 1
                        object_id = obj.object_id

                        # Process new objects only
                        if object_id not in self._processed_ids:
                            self._total_plates += 1
                            self._processed_ids.add(object_id)

                            # Mark as pending for now - frame extraction disabled
                            self._ocr_cache[object_id] = "DETECTED"
                            print(f"[Plate] id={object_id}, conf={obj.confidence:.2f}", flush=True)

                        # 3. Update display (from cache)
                        cached = self._ocr_cache.get(object_id, "??????")
                        update_display(obj, cached)

                        l_obj = l_obj.next
                    except StopIteration:
                        break

                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _poll_ocr_results(self) -> None:
        """Poll OCR results from worker and update cache."""
        if self._ocr_worker is None:
            return

        results = self._ocr_worker.poll_results()
        for object_id, text in results:
            if text is not None:
                self._ocr_cache[object_id] = text
                print(f"[Plate] ID:{object_id} -> {text}")

                # Send event if sink configured
                if self._sink:
                    self._send_event_simple(object_id, text)
            else:
                # OCR failed or returned None
                self._ocr_cache[object_id] = "INVALID_OCR"
                print(f"[Plate] ID:{object_id} -> OCR returned None (no text detected)")

    def _submit_to_worker(self, object_id: int, plate_img: np.ndarray) -> None:
        """Submit plate image to OCR worker queue.

        Args:
            object_id: Tracking ID for the plate
            plate_img: Warped plate image (worker handles 2-line detection)
        """
        if self._ocr_worker is None:
            return

        success = self._ocr_worker.submit(object_id, plate_img)
        if success:
            self._ocr_submitted += 1
        else:
            self._ocr_dropped += 1
            print(f"[Plate] Queue full, dropped OCR for id={object_id}")

    def _send_event_simple(self, object_id: int, plate_text: str) -> None:
        """Send plate detection event to sink."""
        event = {
            "type": "plate_detected",
            "plate_text": plate_text,
            "object_id": object_id,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        try:
            self._sink.send_event(event)
        except Exception as e:
            print(f"[Plate] Error sending event: {e}")

    def _send_event(self, frame_meta, object_id: int, plate_text: str) -> None:
        """Send plate detection event to sink."""
        camera_id = None
        if self._source_mapper:
            camera_id = self._source_mapper.get_camera_id(frame_meta.source_id)

        event = {
            "type": "plate_detected",
            "camera_id": camera_id or f"source_{frame_meta.source_id}",
            "plate_text": plate_text,
            "object_id": object_id,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        if self._sink:
            try:
                self._sink.send_event(event)
            except Exception as e:
                print(f"[Plate] Error sending event: {e}")

        print(f"[Plate] ID:{object_id} -> {plate_text}")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any) -> None:
        print(f"[PlateRecognitionProcessor] Pipeline built, branch: {branch_info.name}")

    def on_start(self) -> None:
        print("[PlateRecognitionProcessor] Started")

    def on_stop(self) -> None:
        """Stop processor and shutdown OCR worker."""
        # Shutdown OCR worker
        if self._ocr_worker is not None:
            print(f"[PlateRecognitionProcessor] Stopping - "
                  f"plates={self._total_plates}, frames={self._total_frames}, "
                  f"ocr_submitted={self._ocr_submitted}, ocr_dropped={self._ocr_dropped}")
            self._ocr_worker.shutdown()
            self._ocr_worker = None
        else:
            print(f"[PlateRecognitionProcessor] Stopped - "
                  f"plates={self._total_plates}, frames={self._total_frames}")

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Return processor statistics."""
        return {
            "total_plates": self._total_plates,
            "total_frames": self._total_frames,
            "cached_ocr": len(self._ocr_cache),
            "processed_ids": len(self._processed_ids),
            "ocr_submitted": self._ocr_submitted,
            "ocr_dropped": self._ocr_dropped,
        }

    def set_source_mapper(self, source_mapper) -> None:
        """Set source mapper for camera ID resolution."""
        self._source_mapper = source_mapper

    def cleanup_stale(self, active_ids: Set[int]) -> None:
        """Remove objects that are no longer tracked."""
        stale = self._processed_ids - active_ids
        for oid in stale:
            self._processed_ids.discard(oid)
            self._ocr_cache.pop(oid, None)
