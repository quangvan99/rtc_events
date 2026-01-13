"""
License Plate Recognition Processor

All-in-one module for plate detection and OCR.
Auto-registered with ProcessorRegistry using @register decorator.

IMPORTANT: OCR engine MUST be preloaded BEFORE Gst.init() to avoid CUDA conflicts:
    from apps.plate import OCREngineHolder
    OCREngineHolder.preload()
"""

import gc
import math
import re
import threading
import time
from ctypes import c_float, sizeof
from typing import Callable, Dict, Any, List, Optional, Set, Tuple

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.processor_registry import ProcessorRegistry
from src.sinks.base_sink import BaseSink
from src.common import BatchIterator, get_batch_meta, fps_probe_factory


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


# =============================================================================
# CTC Label Decode
# =============================================================================

class BaseRecLabelDecode:
    """Convert between text-label and text-index."""

    def __init__(self, character_dict_path: str = None, use_space_char: bool = False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = list("0123456789abcdefghijklmnopqrstuvwxyz")
            dict_character = self.character_str
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {char: i for i, char in enumerate(dict_character)}
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def get_ignored_tokens(self):
        return [0]  # CTC blank token

    def decode(self, text_index, text_prob=None, is_remove_duplicate: bool = False):
        """Convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)

            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]

            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)

            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)
            if self.reverse:
                text = self._pred_reverse(text)

            result_list.append((text, float(np.mean(conf_list))))

        return result_list

    def _pred_reverse(self, pred):
        """Reverse prediction for RTL languages."""
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)
        return "".join(pred_re[::-1])


class CTCLabelDecode(BaseRecLabelDecode):
    """CTC Label Decode for PaddleOCR recognition models."""

    def __init__(self, character_dict_path: str = None, use_space_char: bool = False, **kwargs):
        super().__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, **kwargs):
        """Decode CTC predictions to text."""
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]

        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)

        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        """Add CTC blank token at index 0."""
        return ["blank"] + dict_character


# =============================================================================
# TensorRT OCR Engine
# =============================================================================

class TRTOCREngine:
    """TensorRT inference engine for PaddleOCR text recognition."""

    INPUT_HEIGHT = 48
    INPUT_WIDTH = 320
    INPUT_CHANNELS = 3
    OUTPUT_SHAPE = (1, 40, 42)

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self._lock = threading.Lock()

        # Initialize CUDA
        cuda.init()
        self.ctx = cuda.Device(0).make_context()

        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        trt.init_libnvinfer_plugins(self.logger, "")

        # Load engine
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Setup I/O buffers
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            buffer = {"name": name, "host": host_mem, "device": device_mem, "shape": shape}

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(buffer)
            else:
                self.outputs.append(buffer)

        self.ctx.pop()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for OCR inference."""
        h, w = image.shape[:2]
        ratio = w / float(h)

        if math.ceil(self.INPUT_HEIGHT * ratio) > self.INPUT_WIDTH:
            resized_w = self.INPUT_WIDTH
        else:
            resized_w = int(math.ceil(self.INPUT_HEIGHT * ratio))

        resized = cv2.resize(image, (resized_w, self.INPUT_HEIGHT))
        resized = resized.astype(np.float32)
        resized = resized.transpose((2, 0, 1)) / 255.0
        resized = (resized - 0.5) / 0.5

        padded = np.zeros(
            (self.INPUT_CHANNELS, self.INPUT_HEIGHT, self.INPUT_WIDTH),
            dtype=np.float32
        )
        padded[:, :, :resized_w] = resized

        return padded

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed image."""
        if image.ndim == 3 and image.shape[2] == 3:
            image = self.preprocess(image)

        with self._lock:
            self.ctx.push()
            try:
                np.copyto(self.inputs[0]["host"], image.ravel())
                cuda.memcpy_htod_async(
                    self.inputs[0]["device"],
                    self.inputs[0]["host"],
                    self.stream,
                )

                for buf in self.inputs + self.outputs:
                    self.context.set_tensor_address(buf["name"], int(buf["device"]))

                self.context.execute_async_v3(stream_handle=self.stream.handle)

                for out in self.outputs:
                    cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

                self.stream.synchronize()

                result = self.outputs[0]["host"].copy()
                return result.reshape(self.OUTPUT_SHAPE)
            finally:
                self.ctx.pop()

    def destroy(self):
        """Clean up CUDA resources."""
        if hasattr(self, '_destroyed'):
            return
        self._destroyed = True

        try:
            self.ctx.push()
            try:
                if hasattr(self, 'context'):
                    del self.context
                if hasattr(self, 'engine'):
                    del self.engine
                if hasattr(self, 'runtime'):
                    del self.runtime
                if hasattr(self, 'stream'):
                    del self.stream
                for buf in self.inputs + self.outputs:
                    if 'device' in buf:
                        buf["device"].free()
                gc.collect()
            finally:
                self.ctx.pop()
                self.ctx.detach()
        except Exception:
            pass


# =============================================================================
# OCR Engine Singleton Holder
# =============================================================================

class OCREngineHolder:
    """Singleton holder for the OCR TensorRT engine.

    MUST be preloaded BEFORE Gst.init() to avoid CUDA context conflicts.
    """

    _engine: Optional[TRTOCREngine] = None
    _decoder: Optional[CTCLabelDecode] = None
    _lock = threading.Lock()
    _initialized = False

    DEFAULT_ENGINE_PATH = "data/license_plate/ocr/ppocr_dummy.engine"
    DEFAULT_DICT_PATH = "data/license_plate/ocr/licence_plate_dict.txt"

    @classmethod
    def preload(cls, engine_path: str = None, dict_path: str = None) -> None:
        """Preload OCR engine before DeepStream initialization."""
        with cls._lock:
            if cls._initialized:
                raise RuntimeError("OCR engine already initialized")

            engine_path = engine_path or cls.DEFAULT_ENGINE_PATH
            dict_path = dict_path or cls.DEFAULT_DICT_PATH

            cls._engine = TRTOCREngine(engine_path)
            cls._decoder = CTCLabelDecode(
                character_dict_path=dict_path,
                use_space_char=True,
            )
            cls._initialized = True

    @classmethod
    def get_engine(cls) -> TRTOCREngine:
        """Get the OCR engine instance."""
        if not cls._initialized or cls._engine is None:
            raise RuntimeError(
                "OCR engine not initialized. Call OCREngineHolder.preload() "
                "BEFORE Gst.init()"
            )
        return cls._engine

    @classmethod
    def get_decoder(cls) -> CTCLabelDecode:
        """Get the CTC decoder instance."""
        if not cls._initialized or cls._decoder is None:
            raise RuntimeError(
                "OCR decoder not initialized. Call OCREngineHolder.preload() "
                "BEFORE Gst.init()"
            )
        return cls._decoder

    @classmethod
    def recognize(cls, image) -> tuple:
        """Convenience method to run full OCR pipeline with GIL protection."""
        import ctypes

        # Get Python C API for GIL management
        pythonapi = ctypes.pythonapi
        PyGILState_Ensure = pythonapi.PyGILState_Ensure
        PyGILState_Release = pythonapi.PyGILState_Release
        PyGILState_Ensure.restype = ctypes.c_int

        engine = cls.get_engine()
        decoder = cls.get_decoder()

        # Acquire GIL before TensorRT/pybind11 operations
        gil_state = PyGILState_Ensure()
        try:
            output = engine.infer(image)
            results = decoder(output)
        finally:
            PyGILState_Release(gil_state)

        if results and len(results) > 0:
            return results[0]
        return ("", 0.0)

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if engine is initialized."""
        return cls._initialized

    @classmethod
    def shutdown(cls) -> None:
        """Clean up engine resources."""
        with cls._lock:
            if cls._engine is not None:
                cls._engine.destroy()
                cls._engine = None
            cls._decoder = None
            cls._initialized = False


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


def check_format_plate(plate_text: str) -> bool:
    """Check if license plate matches standard Vietnamese formats."""
    patterns = [
        r"^[0-9]{2}[A-Z]{1}[0-9]{4}$",
        r"^[0-9]{2}[A-Z]{1}[0-9]{5}$",
        r"^[0-9]{2}[A-Z]{1}[0-9]{6}$",
        r"^[0-9]{2}[A-Z]{1}[0-9]{3}\.[0-9]{2}$",
    ]
    return any(re.match(p, plate_text) for p in patterns)


def check_format_plate_append(plate_text: str) -> Optional[str]:
    """Check and append markers for non-standard formats."""
    patterns = [
        (r"^[0-9]{2}[ABCDEFGHIJKLMNOPQRSTUVWXYZĐ]{1,2}([0-9]{4,5}|[0-9]{6})$", "##"),
        (r"^[0-9]{4,5}[A-Z]{2}[0-9]{2,3}$", "#"),
        (r"^[A-Z]{0,1}[0-9]{6,7}$", ""),
    ]

    for pattern, suffix in patterns:
        if re.match(pattern, plate_text):
            return plate_text + suffix
    return None


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
    import pyds
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
        # DEBUG
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

        # Tracking state
        self._processed_ids: Set[int] = set()
        self._ocr_cache: Dict[int, str] = {}

        # Stats
        self._total_plates = 0
        self._total_frames = 0

    @property
    def name(self) -> str:
        return "plate_recognition"

    def setup(self, config: Dict[str, Any], sink: BaseSink) -> None:
        """Initialize plate recognition components."""
        self._config = config
        self._sink = sink
        params = config.get("params", {})

        if not OCREngineHolder.is_initialized():
            raise RuntimeError(
                "OCR engine not initialized! "
                "Call OCREngineHolder.preload() BEFORE Gst.init()"
            )

        print(f"[PlateRecognitionProcessor] Setup complete "
              f"(point_thresh={params.get('point_confidence_threshold', 0.5)}, "
              f"ocr_thresh={params.get('ocr_confidence_threshold', 0.9)})")

    def _get_stats(self) -> dict:
        """Return stats dict for FPSMonitor."""
        return {
            "plates": self._total_plates,
            "frames": self._total_frames,
            "cached": len(self._ocr_cache),
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
        """Main probe: Extract plates, run OCR, update display."""
        gst_buffer = info.get_buffer()
        batch = get_batch_meta(gst_buffer)
        if not batch:
            return Gst.PadProbeReturn.OK

        params = self._config.get("params", {})
        point_threshold = params.get("point_confidence_threshold", 0.5)
        ocr_threshold = params.get("ocr_confidence_threshold", 0.9)

        # DEBUG: Count objects in this batch per frame
        frame_image_cache = {}  # Cache frame per source_id
        batch_iter = BatchIterator(batch)  # Create once, reuse
        frame_count = 0
        total_objs = 0
        for frame in batch_iter.frames():
            frame_count += 1
            obj_count = 0
            for obj in batch_iter.objects(frame):
                obj_count += 1
                total_objs += 1
                self._total_frames += 1
                object_id = obj.object_id

                # Process new objects only
                if object_id not in self._processed_ids:
                    # Log detection info
                    print(f"[Plate] Detected plate id={object_id}, conf={obj.confidence:.2f}")
                    self._total_plates += 1
                    self._processed_ids.add(object_id)

                    # NOTE: Frame extraction via pyds.get_nvds_buf_surface() causes deadlock
                    # when called from tracker probe. Needs queue-based async processing.
                    # For now, mark as detected without OCR

                # Update display (from cache)
                cached = self._ocr_cache.get(object_id, "??????")
                update_display(obj, cached)

        return Gst.PadProbeReturn.OK

    def _process_plate(self, obj_meta, frame_image: np.ndarray,
                       point_threshold: float, ocr_threshold: float) -> Optional[str]:
        """Process license plate: extract points -> warp -> OCR -> validate."""
        keypoints = extract_keypoints(obj_meta, point_threshold)
        if keypoints is None:
            return None

        plate_img = warp_plate(frame_image, keypoints)
        if plate_img is None:
            return None

        # Check if 2-line plate
        _, img_list = check_plate_square(plate_img)
        images_to_ocr = img_list if img_list is not None else [plate_img]

        # Run OCR on each region
        results = []
        for img in images_to_ocr:
            text, conf = OCREngineHolder.recognize(img)
            if conf >= ocr_threshold:
                results.append(text)

        if not results:
            return None

        ocr_result = "".join(results)

        # Validate format
        if not check_format_plate(ocr_result):
            fixed = check_format_plate_append(ocr_result)
            if fixed is not None:
                ocr_result = fixed

        return ocr_result

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
        print(f"[PlateRecognitionProcessor] Stopped - plates={self._total_plates}, frames={self._total_frames}")

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Return processor statistics."""
        return {
            "total_plates": self._total_plates,
            "total_frames": self._total_frames,
            "cached_ocr": len(self._ocr_cache),
            "processed_ids": len(self._processed_ids),
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
