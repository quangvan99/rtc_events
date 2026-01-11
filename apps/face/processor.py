"""
Face Recognition Processor - Simplified Module

This module contains face recognition components with minimal class usage:
- FaceDatabase: Face feature storage and matching
- TrackedFace/TrackerManager: Identity tracking (dataclass + manager)
- FaceRecognitionProcessor: Main BranchProcessor with probes

Auto-registered with ProcessorRegistry using @register decorator.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional

import numpy as np
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.interfaces import BranchProcessor
from src.registry import ProcessorRegistry
from src.sinks.base_sink import BaseSink
from src.utils.extractors import BatchIterator, extract_embedding, get_batch_meta
from src.utils.metric import fps_probe_factory, IntervalRunner


# =============================================================================
# Constants
# =============================================================================

# OSD Colors (RGBA)
COLOR_CONFIRMED = (0.0, 1.0, 0.0, 1.0)  # Green
COLOR_UNKNOWN = (1.0, 0.5, 0.0, 1.0)    # Orange
COLOR_TEXT = (1.0, 1.0, 1.0, 1.0)       # White
COLOR_TEXT_BG = (0.0, 0.0, 0.0, 0.7)    # Black transparent

# Display settings
BORDER_WIDTH = 3
FONT_SIZE = 14
FONT_NAME = "Serif"

# Face-specific constants
SKIP_SGIE_COMPONENT_ID = 100


# =============================================================================
# Face-Specific Helper Functions
# =============================================================================

def should_skip_face(obj_meta, min_size: int = 50) -> bool:
    """Check if face should be skipped based on size (face-specific logic)"""
    rect = obj_meta.rect_params
    return rect.width < min_size or rect.height < min_size


def mark_skip_sgie(obj_meta) -> None:
    """Mark object to skip SGIE processing"""
    obj_meta.unique_component_id = SKIP_SGIE_COMPONENT_ID


# =============================================================================
# Display Functions (replaces FaceDisplay class)
# =============================================================================

def update_display(obj_meta, name: str, score: float, state: str = "confirmed") -> None:
    """Update OSD display for a detected face"""
    rect = obj_meta.rect_params
    face_w, face_h = int(rect.width), int(rect.height)

    # Border color by state
    r, g, b, a = COLOR_CONFIRMED if state == "confirmed" else COLOR_UNKNOWN
    rect.border_color.red, rect.border_color.green = r, g
    rect.border_color.blue, rect.border_color.alpha = b, a
    rect.border_width = BORDER_WIDTH

    # Display text
    display_text = f"{name} ({score:.2f}) [{face_w}x{face_h}]" if state == "confirmed" else f"[{face_w}x{face_h}]"

    text = obj_meta.text_params
    text.display_text = display_text
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
# Face Database
# =============================================================================

class FaceDatabase:
    """
    Manages registered face features for matching.
    
    Loads face embeddings from JSON and provides L2-distance matching.
    """

    def __init__(self, features_path: str):
        self.names: list[str] = []
        self.features_matrix: np.ndarray | None = None
        self.avatars: dict[str, str] = {}
        self._load(features_path)

    def _load(self, path: str) -> None:
        """Load pre-registered face features from JSON"""
        if not os.path.exists(path):
            print(f"Warning: Features file not found: {path}")
            return

        start = time.time()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = []
        for name, info in data.items():
            if "feature" not in info:
                continue

            feat = info["feature"]
            if isinstance(feat[0], list):
                feat = feat[0]
            vec = np.array(feat, dtype=np.float32)

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            self.names.append(name)
            features.append(vec)
            if "avatar" in info:
                self.avatars[name] = info["avatar"]

        if features:
            self.features_matrix = np.vstack(features).astype(np.float32)

        print(f"Loaded {len(self.names)} registered faces in {(time.time() - start) * 1000:.1f}ms")

    def match(self, embedding: np.ndarray) -> tuple[int, float]:
        """Match embedding against database. Returns (person_idx, distance)"""
        if self.features_matrix is None:
            return -1, float("inf")

        distances = np.linalg.norm(self.features_matrix - embedding, axis=1)
        idx = int(np.argmin(distances))
        return idx, float(distances[idx])


# =============================================================================
# Face Tracker
# =============================================================================

@dataclass
class TrackedFace:
    """
    Track a face and confirm identity via consecutive matches.
    Identity is confirmed when the same person matches min_streak times.
    """
    object_id: int
    l2_threshold: float = 1.0
    min_streak: int = 3
    skip_reid: int = 3
    reid_interval: int = 30

    label: str | None = None
    score: float = 0.0

    _person: int = -1
    _streak: int = 0
    _distances: list[float] = field(default_factory=list)

    last_sgie: int = 0
    age: int = 0

    def should_run_sgie(self, frame: int) -> bool:
        """Check if SGIE should run based on frame interval"""
        interval = self.reid_interval if self.label else self.skip_reid
        return (frame - self.last_sgie) >= interval

    def add_match(self, person: int, distance: float) -> bool:
        """Add match result. Returns True if identity confirmed."""
        if distance > self.l2_threshold:
            return False
        if person != self._person:
            self._person = person
            self._distances = [distance]
            self._streak = 1
            return False
        self._streak += 1
        self._distances.append(distance)
        return self._streak >= self.min_streak

    def confirm(self, name: str) -> bool:
        """Confirm identity. Returns True if first confirmation."""
        is_new = self.label is None
        self.label = name
        self.score = sum(self._distances) / len(self._distances) if self._distances else 0.0
        self._person, self._streak, self._distances = -1, 0, []
        if is_new:
            print(f"[CONFIRMED] id={self.object_id} -> {name} (score={self.score:.3f})")
        return is_new


class TrackerManager:
    """Manage tracked faces per camera (source_id namespace) with auto-cleanup."""

    def __init__(self, config: dict, max_age: int = 30, cleanup_interval: int = 10):
        self.config = config
        self.max_age = max_age
        self.cleanup_interval = cleanup_interval
        self._trackers: dict[int, dict[int, TrackedFace]] = {}
        self._last_cleanup = 0

    def get(self, source_id: int, oid: int) -> TrackedFace | None:
        return self._trackers.get(source_id, {}).get(oid)

    def get_or_create(self, source_id: int, oid: int, frame: int) -> TrackedFace:
        if source_id not in self._trackers:
            self._trackers[source_id] = {}
        cam_dict = self._trackers[source_id]
        if oid not in cam_dict:
            cam_dict[oid] = TrackedFace(
                object_id=oid,
                l2_threshold=self.config.get("l2_threshold", 1.0),
                min_streak=self.config.get("min_streak", 3),
                skip_reid=self.config.get("skip_reid", 3),
                reid_interval=self.config.get("reid_interval", 30),
                last_sgie=frame,
            )
        return cam_dict[oid]

    def cleanup(self, current_frame: int = 0) -> list[tuple[int, int]]:
        """Increment age and remove stale trackers. Returns removed list."""
        removed = []
        for source_id, cam_dict in self._trackers.items():
            to_remove = []
            for oid, t in cam_dict.items():
                t.age += 1
                if t.age > self.max_age:
                    to_remove.append(oid)
            for oid in to_remove:
                del cam_dict[oid]
                removed.append((source_id, oid))
        self._last_cleanup = current_frame
        return removed

    def auto_cleanup(self, current_frame: int) -> list[tuple[int, int]]:
        """Auto-cleanup based on frame interval. Returns removed list."""
        if current_frame - self._last_cleanup >= self.cleanup_interval:
            return self.cleanup(current_frame)
        return []

    def stats(self) -> tuple[int, int, int]:
        """Returns (total, confirmed, pending)"""
        total = confirmed = 0
        for cam_dict in self._trackers.values():
            for t in cam_dict.values():
                total += 1
                confirmed += 1 if t.label else 0
        return total, confirmed, total - confirmed


class EventSet:
    """Track sent events with frame-based storage and auto-cleanup."""

    def __init__(self, max_age: int = 30):
        self.max_age = max_age
        self._events: dict[tuple[int, int], int] = {}

    def add(self, key: tuple[int, int], frame: int) -> bool:
        """Add event. Returns True if newly added."""
        if key not in self._events:
            self._events[key] = frame
            return True
        return False

    def contains(self, key: tuple[int, int]) -> bool:
        """Check if key exists."""
        return key in self._events

    def discard(self, key: tuple[int, int]) -> None:
        """Remove key."""
        self._events.pop(key, None)

    def cleanup(self, current_frame: int) -> list[tuple[int, int]]:
        """Remove stale entries based on frame age. Returns removed list."""
        removed = []
        to_remove = [k for k, f in self._events.items() if current_frame - f > self.max_age]
        for k in to_remove:
            del self._events[k]
            removed.append(k)
        return removed

    def auto_cleanup(self, current_frame: int) -> list[tuple[int, int]]:
        """Auto-cleanup based on frame interval."""
        if current_frame % self.max_age == 0:
            return self.cleanup(current_frame)
        return []


# =============================================================================
# Main Processor (includes probes and event handling)
# =============================================================================

@ProcessorRegistry.register("recognition")
class FaceRecognitionProcessor(BranchProcessor):
    """
    Face recognition processor - all-in-one implementation.
    
    Combines:
    - Database loading/matching
    - Tracker management
    - Probe callbacks
    - Event emission
    - OSD display
    """

    def __init__(self, source_mapper=None):
        self._config: Dict[str, Any] = {}
        self._sink: Optional[BaseSink] = None
        self._source_mapper = source_mapper
        self._db: Optional[FaceDatabase] = None
        self._trackers: Optional[TrackerManager] = None
        self._sent_faces: Optional[EventSet] = None
        self._cleanup_runner: Optional[IntervalRunner] = None

    @property
    def name(self) -> str:
        return "recognition"

    def setup(self, config: Dict[str, Any], sink: BaseSink) -> None:
        """Initialize face recognition components"""
        self._config = config
        self._sink = sink
        params = config.get("params", {})
        features_path = params.get("features_json", "data/face/features.json")

        print(f"[FaceRecognitionProcessor] Loading {features_path}...")
        self._db = FaceDatabase(features_path)
        print(f"[FaceRecognitionProcessor] Loaded {len(self._db.names)} faces")
        
        self._trackers = TrackerManager(params)
        self._sent_faces = EventSet(max_age=params.get("max_age", 30))
        
        cleanup_interval = params.get("cleanup_interval", 10) * 1000
        self._cleanup_runner = IntervalRunner(cleanup_interval, self._cleanup)
        
        print("[FaceRecognitionProcessor] Setup complete")
    
    def _get_stats(self) -> dict:
        """Return stats dict for """
        if not self._trackers:
            return {"total": 0, "confirmed": 0, "pending": 0}
        total, confirmed, pending = self._trackers.stats()
        return {"total": total, "confirmed": confirmed, "pending": pending}

    def get_probes(self) -> Dict[str, Callable]:
        """Return probe callbacks"""
        if not self._db:
            raise RuntimeError("Processor not initialized. Call setup() first.")
        params = self._config.get("params", {})
        return {
            "tracker_probe": self._tracker_probe,
            "sgie_probe": self._sgie_probe,
            "fps_probe": fps_probe_factory(
                name="Recognition",
                log_interval=params.get("log_interval", 1.0),
                stats_interval=params.get("stats_interval", 10.0),
                stats_callback=self._get_stats,
            ),
        }

    # -------------------------------------------------------------------------
    # Probe Callbacks
    # -------------------------------------------------------------------------

    def _tracker_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """Decide whether to skip SGIE for each face"""
        batch = get_batch_meta(info.get_buffer())
        if not batch:
            return Gst.PadProbeReturn.OK

        min_face = self._config.get("params", {}).get("min_face_size", 50)

        for frame, obj in BatchIterator(batch):
            # Skip small faces
            if should_skip_face(obj, min_face):
                mark_skip_sgie(obj)
                continue
            # Skip if tracker says not ready for SGIE
            trk = self._trackers.get(frame.source_id, obj.object_id)
            if trk and not trk.should_run_sgie(frame.frame_num):
                mark_skip_sgie(obj)

        return Gst.PadProbeReturn.OK

    def _sgie_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """Process recognition results and update display"""
        batch = get_batch_meta(info.get_buffer())
        if not batch:
            return Gst.PadProbeReturn.OK

        for frame, obj in BatchIterator(batch):
            name, state, score = self._process_face(frame.source_id, obj, frame.frame_num)
            update_display(obj, name, score, state)

        return Gst.PadProbeReturn.OK

    def _cleanup(self, current_frame: int) -> None:
        """Cleanup stale trackers and events"""
        self._trackers.auto_cleanup(current_frame)
        self._sent_faces.auto_cleanup(current_frame)

    def _process_face(self, source_id: int, obj_meta, frame: int) -> tuple[str, str, float]:
        """Process face and return (name, state, score) for display"""
        oid = obj_meta.object_id
        trk = self._trackers.get_or_create(source_id, oid, frame)
        trk.age = 0

        emb = extract_embedding(obj_meta)
        if emb is not None:
            trk.last_sgie = frame
            person, dist = self._db.match(emb)
            if trk.add_match(person, dist):
                name = self._db.names[person]
                if trk.confirm(name):
                    self._send_event(source_id, oid, name, frame)

        return (trk.label, "confirmed", trk.score) if trk.label else ("", "unknown", 0.0)

    def _send_event(self, source_id: int, object_id: int, name: str, frame: int) -> None:
        """Send face detection event (once per face)"""
        key = (source_id, object_id)
        if not self._sent_faces.add(key, frame):
            return

        camera_id = self._source_mapper.get_camera_id(source_id) if self._source_mapper else None
        event = {
            "type": "face_detected",
            "camera_id": camera_id,
            "source_id": source_id,
            "name": name,
            "timestamp": time.strftime("%H:%M:%S"),
            "object_id": object_id,
            "avatar": self._db.avatars.get(name),
        }
        self._sink.send_event(event)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any) -> None:
        print(f"[FaceRecognitionProcessor] Pipeline built, branch: {branch_info.name}")

    def on_start(self) -> None:
        """Start cleanup timer when pipeline starts"""
        if self._cleanup_runner:
            self._cleanup_runner.start()
        print("[FaceRecognitionProcessor] Started")

    def on_stop(self) -> None:
        """Stop cleanup timer when pipeline stops"""
        if self._cleanup_runner:
            self._cleanup_runner.stop()
        print("[FaceRecognitionProcessor] Stopped")

    @property
    def database(self) -> Optional[FaceDatabase]:
        return self._db

    def set_source_mapper(self, source_mapper) -> None:
        self._source_mapper = source_mapper