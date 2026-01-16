"""
Fire Detection Processor - Smoke/Fire Detection with Conditional Helmet Detection

Pipeline: PGIE (fire/smoke/person) → nvtracker → [optional SGIE helmet] → OSD
- PGIE detects: person (0), fire (81), smoke (82)
- SGIE runs on person bbox when camera has hat:True
- Visualization: Red (fire), Gray (smoke), Green (person), Yellow (helmet violation)

Auto-registered with ProcessorRegistry using @register decorator.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.processor_registry import ProcessorRegistry
from src.sinks.base_sink import BaseSink
from src.common import BatchIterator, get_batch_meta, fps_probe_factory, IntervalRunner


# =============================================================================
# Constants
# =============================================================================

# Class IDs from PGIE (config_pgie_fire.txt)
# Note: Update these based on actual model labels
# Default: person=0, fire=81, smoke=82 (user-specified)
# Can be overridden in config.yaml params
CLASS_PERSON = 0
CLASS_FIRE = 81
CLASS_SMOKE = 82

# SGIE unique ID (for filtering classifier metadata)
SGIE_HELMET_ID = 2

# OSD Colors (RGBA)
COLOR_FIRE = (1.0, 0.0, 0.0, 1.0)              # Red
COLOR_SMOKE = (0.5, 0.5, 0.5, 1.0)             # Gray
COLOR_PERSON = (0.0, 1.0, 0.0, 1.0)            # Green
COLOR_HELMET_OK = (0.0, 0.8, 0.0, 1.0)         # Dark Green (wearing helmet)
COLOR_HELMET_VIOLATION = (1.0, 1.0, 0.0, 1.0)  # Yellow (no helmet)
COLOR_TEXT = (1.0, 1.0, 1.0, 1.0)              # White
COLOR_TEXT_BG = (0.0, 0.0, 0.0, 0.7)           # Black transparent

# Display settings
BORDER_WIDTH_CRITICAL = 5  # Fire/smoke
BORDER_WIDTH_NORMAL = 3    # Person/helmet
FONT_SIZE = 14
FONT_NAME = "Serif"

# Skip SGIE marker
SKIP_SGIE_COMPONENT_ID = -1


# =============================================================================
# Helper Functions
# =============================================================================

def update_display(obj_meta, class_id: int, confidence: float, label: str,
                   color: tuple = None, border_width: int = None) -> None:
    """Update OSD display for detected object."""
    rect = obj_meta.rect_params

    # Determine color and border based on class
    if color is None:
        if class_id == CLASS_FIRE:
            color = COLOR_FIRE
            border_width = BORDER_WIDTH_CRITICAL
        elif class_id == CLASS_SMOKE:
            color = COLOR_SMOKE
            border_width = BORDER_WIDTH_CRITICAL
        elif class_id == CLASS_PERSON:
            color = COLOR_PERSON
            border_width = BORDER_WIDTH_NORMAL
        else:
            color = COLOR_HELMET_VIOLATION
            border_width = BORDER_WIDTH_NORMAL

    if border_width is None:
        border_width = BORDER_WIDTH_NORMAL

    # Set border color
    r, g, b, a = color
    rect.border_color.red = r
    rect.border_color.green = g
    rect.border_color.blue = b
    rect.border_color.alpha = a
    rect.border_width = border_width

    # Set text label
    # text = obj_meta.text_params
    # text.display_text = f"{label} {int(confidence * 100)}%"
    # text.x_offset = int(rect.left)
    # text.y_offset = max(0, int(rect.top) - 25)
    # text.font_params.font_name = FONT_NAME
    # text.font_params.font_size = FONT_SIZE

    # # Text color
    # r, g, b, a = COLOR_TEXT
    # text.font_params.font_color.red = r
    # text.font_params.font_color.green = g
    # text.font_params.font_color.blue = b
    # text.font_params.font_color.alpha = a

    # # Text background
    # text.set_bg_clr = 1
    # r, g, b, a = COLOR_TEXT_BG
    # text.text_bg_clr.red = r
    # text.text_bg_clr.green = g
    # text.text_bg_clr.blue = b
    # text.text_bg_clr.alpha = a

def hide_display(obj_meta) -> None:
    """Hide OSD display for detected object."""
    rect = obj_meta.rect_params
    rect.border_width = 0
    rect.border_color.alpha = 0.0
    text = obj_meta.text_params
    text.display_text = ""
    text.font_params.font_size = 0
    text.font_params.font_color.alpha = 0.0
    text.set_bg_clr = 0
# =============================================================================
# Event Tracking (prevent duplicate events)
# =============================================================================

@dataclass
class EventSet:
    """Track sent events with frame-based storage and auto-cleanup."""
    max_age: int = 30
    _events: Dict[tuple, int] = field(default_factory=dict)

    def add(self, key: tuple, frame: int) -> bool:
        """Add event. Returns True if newly added."""
        if key not in self._events:
            self._events[key] = frame
            return True
        return False

    def contains(self, key: tuple) -> bool:
        return key in self._events

    def cleanup(self, current_frame: int) -> int:
        """Remove stale entries. Returns count removed."""
        to_remove = [k for k, f in self._events.items() if current_frame - f > self.max_age]
        for k in to_remove:
            del self._events[k]
        return len(to_remove)


# =============================================================================
# Main Processor
# =============================================================================

@ProcessorRegistry.register("fire_detection")
class FireDetectionProcessor:
    """
    Fire/Smoke detection processor with conditional helmet detection.

    Features:
    - PGIE: Detect person (0), fire (81), smoke (82)
    - SGIE: Detect helmet on person bbox (conditional on hat:True per camera)
    - Visualization: Color-coded bboxes
    - Events: fire_detected, smoke_detected, helmet_violation
    """

    def __init__(self, source_mapper=None):
        self._config: Dict[str, Any] = {}
        self._sink: Optional[BaseSink] = None
        self._source_mapper = source_mapper
        self._sent_events: Optional[EventSet] = None
        self._cleanup_runner: Optional[IntervalRunner] = None
        self._frame_count = 0
        # Internal camera mapping: source_id -> camera_id
        self._camera_map: Dict[int, str] = {}

    @property
    def name(self) -> str:
        return "fire_detection"

    def setup(self, config: Dict[str, Any], sink: BaseSink) -> None:
        """Initialize fire detection components."""
        self._config = config
        self._sink = sink
        params = config.get("params", {})

        # Initialize event tracking
        self._sent_events = EventSet(max_age=params.get("max_age", 30))

        # Pre-register cameras from config (static setup like area_monitoring)
        # Maps source_id (index) -> camera_id for cameras defined in config
        cameras = params.get("cameras", {})
        for idx, camera_id in enumerate(cameras.keys()):
            self._camera_map[idx] = camera_id

        print(f"[FireDetectionProcessor] Setup complete")
        print(f"  - Cameras config: {list(cameras.keys())}")
        print(f"  - Camera map: {self._camera_map}")

    def get_probes(self) -> Dict[str, Callable]:
        """Return probe callbacks."""
        params = self._config.get("params", {})
        return {
            "tracker_probe": self._tracker_probe,
            "sgie_probe": self._sgie_probe,
            "fire_fps_probe": fps_probe_factory(
                name="FireDetection",
                log_interval=params.get("log_interval", 1.0),
                stats_interval=params.get("stats_interval", 10.0),
            ),
        }

    def _get_camera_config(self, camera_id: str) -> dict:
        """Get camera-specific config."""
        cameras = self._config.get("params", {}).get("cameras", {})
        return cameras.get(camera_id, {})

    def _get_camera_id(self, source_id: int) -> Optional[str]:
        """Get camera_id from source_id using internal mapping."""
        return self._camera_map.get(source_id)

    # -------------------------------------------------------------------------
    # Camera Management (called by CameraManager when cameras are added/removed)
    # -------------------------------------------------------------------------

    def add_camera(self, camera_id: str, source_id: int, camera_config: Dict[str, Any] = None) -> None:
        """
        Register camera mapping when camera is added to this branch.

        Args:
            camera_id: Camera identifier
            source_id: Source ID assigned by nvstreammux
            camera_config: Optional camera-specific config
        """
        self._camera_map[source_id] = camera_id
        print(f"[FireDetectionProcessor] Camera added: {camera_id} -> source_id={source_id}")

    def remove_camera(self, camera_id: str) -> None:
        """Remove camera mapping when camera is removed."""
        to_remove = [sid for sid, cid in self._camera_map.items() if cid == camera_id]
        for sid in to_remove:
            del self._camera_map[sid]
        if to_remove:
            print(f"[FireDetectionProcessor] Camera removed: {camera_id}")

    # -------------------------------------------------------------------------
    # Tracker Probe - Filter objects before SGIE + visualize fire/smoke
    # -------------------------------------------------------------------------

    def _tracker_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """
        Process tracker output:
        1. Visualize fire/smoke if fire_smoke:True
        2. Filter objects for SGIE (only person + hat:True)
        """
        batch = get_batch_meta(info.get_buffer())
        if not batch:
            return Gst.PadProbeReturn.OK

        self._frame_count += 1
        detected_count = 0

        for frame, obj in BatchIterator(batch):
            # Get camera config using internal mapping
            camera_id = self._get_camera_id(frame.source_id)
            camera_cfg = self._get_camera_config(camera_id) if camera_id else {}

            class_id = obj.class_id
            confidence = obj.confidence
            fire_smoke_enabled = camera_cfg.get("fire_smoke", True)
            hat_enabled = camera_cfg.get("hat", False)

            # Handle fire detection
            if class_id == CLASS_FIRE:
                if fire_smoke_enabled:
                    update_display(obj, class_id, confidence, "FIRE")
                    self._send_fire_event(frame.source_id, camera_id, obj, frame.frame_num)
                obj.unique_component_id = SKIP_SGIE_COMPONENT_ID  # Never send to SGIE

            # Handle smoke detection
            elif class_id == CLASS_SMOKE:
                if fire_smoke_enabled:
                    update_display(obj, class_id, confidence, "SMOKE")
                    self._send_smoke_event(frame.source_id, camera_id, obj, frame.frame_num)
                obj.unique_component_id = SKIP_SGIE_COMPONENT_ID  # Never send to SGIE

            # Handle person detection
            elif class_id == CLASS_PERSON:
                hide_display(obj)
                # update_display(obj, class_id, confidence, "PERSON")
                # Only send to SGIE if hat detection enabled for this camera
                if not hat_enabled:
                    obj.unique_component_id = SKIP_SGIE_COMPONENT_ID

            # Other classes - skip SGIE
            else:
                hide_display(obj)
                obj.unique_component_id = SKIP_SGIE_COMPONENT_ID

            detected_count += 1

        # Log detection stats periodically
        if self._frame_count % 100 == 0:
            if detected_count > 0:
                # Log sample camera mapping
                sample_cam = self._get_camera_id(0)
                sample_cfg = self._get_camera_config(sample_cam) if sample_cam else {}
                print(f"[FireDetection] Frame {self._frame_count}: {detected_count} obj, cam={sample_cam}, cfg={sample_cfg}")
            if self._sent_events:
                self._sent_events.cleanup(self._frame_count)

        return Gst.PadProbeReturn.OK

    # -------------------------------------------------------------------------
    # SGIE Probe - Process helmet detection results
    # -------------------------------------------------------------------------

    def _sgie_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """Process SGIE helmet detection results."""
        batch = get_batch_meta(info.get_buffer())
        if not batch:
            return Gst.PadProbeReturn.OK

        for frame, obj in BatchIterator(batch):
            # Only process persons (SGIE operates on person class)
            if obj.class_id != CLASS_PERSON:
                continue

            # Get camera_id using internal mapping
            camera_id = self._get_camera_id(frame.source_id)

            # Check SGIE classifier results
            helmet_detected = False
            helmet_confidence = 0.0
            helmet_class = -1

            classifier_meta_list = obj.classifier_meta_list
            while classifier_meta_list:
                try:
                    import pyds
                    classifier_meta = pyds.NvDsClassifierMeta.cast(classifier_meta_list.data)

                    # Check if this is our helmet SGIE
                    if classifier_meta.unique_component_id == SGIE_HELMET_ID:
                        label_info_list = classifier_meta.label_info_list
                        while label_info_list:
                            label_info = pyds.NvDsLabelInfo.cast(label_info_list.data)
                            helmet_class = label_info.result_class_id
                            helmet_confidence = label_info.result_prob
                            helmet_detected = True
                            label_info_list = label_info_list.next

                    classifier_meta_list = classifier_meta_list.next
                except Exception:
                    break

            # Update display based on helmet detection
            if helmet_detected:
                # Class mapping from data/hat/best_hat_v8m-trt.txt:
                # 0=None (no helmet), 1=Others, 2=White, 3=Yellow
                if helmet_class == 0:  # None = no helmet
                    update_display(obj, -1, helmet_confidence, "NO HELMET",
                                   color=COLOR_HELMET_VIOLATION, border_width=BORDER_WIDTH_NORMAL)
                    self._send_helmet_event(frame.source_id, camera_id, obj,
                                            helmet_class, helmet_confidence, frame.frame_num)
                else:  # Has helmet (Others, White, Yellow)
                    update_display(obj, -1, helmet_confidence, "HELMET OK",
                                   color=COLOR_HELMET_OK, border_width=BORDER_WIDTH_NORMAL)

        return Gst.PadProbeReturn.OK

    # -------------------------------------------------------------------------
    # Event Emission
    # -------------------------------------------------------------------------

    def _send_fire_event(self, source_id: int, camera_id: str, obj_meta, frame: int) -> None:
        """Emit fire detection event."""
        key = (source_id, obj_meta.object_id, "fire")
        if not self._sent_events.add(key, frame):
            return

        rect = obj_meta.rect_params
        event = {
            "type": "fire_detected",
            "camera_id": camera_id,
            "source_id": source_id,
            "confidence": obj_meta.confidence,
            "bbox": [rect.left, rect.top, rect.width, rect.height],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "object_id": obj_meta.object_id,
        }
        if self._sink:
            self._sink.send_event(event)
        print(f"[FireDetectionProcessor] FIRE detected: camera={camera_id}, conf={obj_meta.confidence:.2f}")

    def _send_smoke_event(self, source_id: int, camera_id: str, obj_meta, frame: int) -> None:
        """Emit smoke detection event."""
        key = (source_id, obj_meta.object_id, "smoke")
        if not self._sent_events.add(key, frame):
            return

        rect = obj_meta.rect_params
        event = {
            "type": "smoke_detected",
            "camera_id": camera_id,
            "source_id": source_id,
            "confidence": obj_meta.confidence,
            "bbox": [rect.left, rect.top, rect.width, rect.height],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "object_id": obj_meta.object_id,
        }
        if self._sink:
            self._sink.send_event(event)
        print(f"[FireDetectionProcessor] SMOKE detected: camera={camera_id}, conf={obj_meta.confidence:.2f}")

    def _send_helmet_event(self, source_id: int, camera_id: str, obj_meta,
                           helmet_class: int, confidence: float, frame: int) -> None:
        """Emit helmet violation event."""
        key = (source_id, obj_meta.object_id, "helmet")
        if not self._sent_events.add(key, frame):
            return

        rect = obj_meta.rect_params
        event = {
            "type": "helmet_violation",
            "camera_id": camera_id,
            "source_id": source_id,
            "person_id": obj_meta.object_id,
            "helmet_status": "not_wearing",
            "helmet_class": helmet_class,
            "confidence": confidence,
            "bbox": [rect.left, rect.top, rect.width, rect.height],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if self._sink:
            self._sink.send_event(event)
        print(f"[FireDetectionProcessor] HELMET VIOLATION: camera={camera_id}, person_id={obj_meta.object_id}")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any) -> None:
        """Called when pipeline is built."""
        print(f"[FireDetectionProcessor] Pipeline built, branch: {branch_info.name}")

    def on_start(self) -> None:
        """Called when pipeline starts."""
        print("[FireDetectionProcessor] Started")

    def on_stop(self) -> None:
        """Called when pipeline stops."""
        print("[FireDetectionProcessor] Stopped")

    def set_source_mapper(self, source_mapper) -> None:
        """Set source mapper for camera ID lookup (legacy, uses internal mapping now)."""
        self._source_mapper = source_mapper

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Return processor statistics."""
        return {
            "frame_count": self._frame_count,
            "events_tracked": len(self._sent_events._events) if self._sent_events else 0,
        }
