"""Face recognition probe callbacks for multi-camera DeepStream pipeline"""

import ctypes
import time
from typing import Optional

import numpy as np
import pyds
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from apps.face.database import FaceDatabase
from apps.face.tracker import TrackerManager
from apps.face.display import FaceDisplay
from apps.face.events import FaceEventEmitter
from sinks.base_sink import BaseSink


# Skip SGIE for faces with this component ID
SKIP_SGIE_COMPONENT_ID = 100


class FaceProbes:
    """Face recognition probe callbacks for DeepStream pipeline

    Supports multi-camera mode with source_id-based tracker isolation.
    """

    def __init__(
        self,
        config: dict,
        database: FaceDatabase,
        tracker_mgr: TrackerManager,
        sink: BaseSink,
        source_mapper=None
    ):
        """
        Initialize FaceProbes

        Args:
            config: Recognition config dict
            database: FaceDatabase instance
            tracker_mgr: TrackerManager instance
            sink: Output sink adapter
            source_mapper: Optional SourceIDMapper for camera_id lookup
        """
        self.config = config
        self.db = database
        self.trackers = tracker_mgr
        self.display = FaceDisplay()
        self.source_mapper = source_mapper

        # Pass mapper to events for camera_id lookup
        self.events = FaceEventEmitter(
            sink, database.avatars, database,
            source_mapper=source_mapper
        )

        # FPS tracking
        self._fps_start = time.time()
        self._fps_count = 0
        self._stats_last = time.time()

    # =========================================================================
    # Core logic
    # =========================================================================

    def _extract_embedding(self, obj_meta) -> np.ndarray | None:
        """Extract 512-dim L2-normalized embedding from SGIE tensor"""
        l_user = obj_meta.obj_user_meta_list
        while l_user:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    layer = pyds.get_nvds_LayerInfo(tensor, 0)
                    if layer:
                        ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                        emb = np.ctypeslib.as_array(ptr, shape=(512,)).copy()
                        norm = np.linalg.norm(emb)
                        return emb / norm if norm > 0 else emb
                l_user = l_user.next
            except StopIteration:
                break
        return None

    def _process_face(self, source_id: int, obj_meta, frame: int) -> tuple[str, str, float]:
        """Process face and return (name, state, score) for display

        Args:
            source_id: Camera source ID from frame_meta
            obj_meta: NvDsObjectMeta for this face
            frame: Frame number

        Returns:
            Tuple of (name, state, score)
        """
        oid = obj_meta.object_id
        trk = self.trackers.get_or_create(source_id, oid, frame)
        trk.age = 0  # Face visible

        # Process embedding if SGIE ran
        emb = self._extract_embedding(obj_meta)
        if emb is not None:
            trk.last_sgie = frame
            person, dist = self.db.match(emb)
            if trk.add_match(person, dist):
                name = self.db.names[person]
                if trk.confirm(name):
                    # Include source_id in event
                    self.events.send_detection(source_id, oid, name)

        # Return display info
        if trk.label:
            return trk.label, "confirmed", trk.score
        return "", "unknown", 0.0

    # =========================================================================
    # Probe callbacks
    # =========================================================================

    def tracker_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """Decide whether to skip SGIE for each face"""
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch:
            return Gst.PadProbeReturn.OK

        min_face = self.config.get("min_face_size", 50)

        l_frame = batch.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                source_id = frame_meta.source_id  # Extract source_id
                frame = frame_meta.frame_num

                l_obj = frame_meta.obj_meta_list
                while l_obj:
                    try:
                        obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                        rect = obj.rect_params

                        # Skip: small face OR tracker says skip
                        skip = rect.width < min_face or rect.height < min_face
                        if not skip:
                            # Pass source_id to get()
                            trk = self.trackers.get(source_id, obj.object_id)
                            skip = trk and not trk.should_run_sgie(frame)

                        if skip:
                            obj.unique_component_id = SKIP_SGIE_COMPONENT_ID

                        l_obj = l_obj.next
                    except StopIteration:
                        break
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def sgie_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """Process face recognition results and update display"""
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch:
            return Gst.PadProbeReturn.OK

        l_frame = batch.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                source_id = frame_meta.source_id  # Extract source_id

                l_obj = frame_meta.obj_meta_list
                while l_obj:
                    try:
                        obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                        # Pass source_id to _process_face
                        name, state, score = self._process_face(
                            source_id, obj, frame_meta.frame_num
                        )
                        self.display.update(obj, name, score, state)
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def fps_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """FPS calculation, stats, and cleanup"""
        now = time.time()
        self._fps_count += 1

        # FPS every 1s
        if now - self._fps_start >= 1.0:
            print(f"[FPS] {self._fps_count / (now - self._fps_start):.1f}")
            self._fps_start, self._fps_count = now, 0

        # Stats every 10s
        if now - self._stats_last >= 10.0:
            total, confirmed, pending = self.trackers.stats()
            print(f"[STATS] total={total}, confirmed={confirmed}, pending={pending}")
            self._stats_last = now

        # Cleanup stale trackers - returns list of (source_id, oid)
        removed = self.trackers.cleanup()
        self.events.cleanup(removed)

        return Gst.PadProbeReturn.OK
