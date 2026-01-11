"""
MultibranchCameraManager - Dynamic camera management with tee fanout

Handles camera lifecycle: add/remove cameras to multiple branches at runtime.
Each camera decodes once via nvurisrcbin, then distributes via tee to branches.
Thread-safe operations for concurrent add/remove.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from src.tee_fanout_builder import BranchInfo
from src.source_mapper import SourceIDMapper

logger = logging.getLogger(__name__)

MIN_OP_DELAY = 2.0


@dataclass
class CameraBin:
    """Container for single camera's GStreamer elements

    Attributes:
        camera_id: Unique identifier for camera
        uri: Camera source URI (rtsp://, file://, etc.)
        bin: GstBin containing nvurisrcbin + tee
        nvurisrcbin: Source element for decoding
        tee: Fanout element for multi-branch distribution
        branch_queues: Mapping branch_name -> queue element
        branch_pads: Mapping branch_name -> tee src pad
        source_id: DeepStream source ID for this camera
        pad_added_handler: Signal handler ID for pad-added signal
        active_probes: List of active pad probe IDs
    """
    camera_id: str
    uri: str
    bin: Gst.Bin
    nvurisrcbin: Gst.Element
    tee: Gst.Element
    branch_queues: dict[str, Gst.Element] = field(default_factory=dict)
    branch_pads: dict[str, Gst.Pad] = field(default_factory=dict)
    source_id: int = 0
    pad_added_handler: int = 0
    active_probes: list = field(default_factory=list)

    def get_branches(self) -> list[str]:
        """Get list of branch names this camera is connected to"""
        return list(self.branch_queues.keys())

    def is_connected_to(self, branch_name: str) -> bool:
        """Check if camera is connected to specified branch"""
        return branch_name in self.branch_queues


class MultibranchCameraManager:
    """Manager for dynamic camera add/remove with tee fanout to multiple branches

    Usage:
        manager = MultibranchCameraManager(pipeline, branches)
        manager.add_camera("cam1", "rtsp://...", ["recognition", "detection"])
        manager.remove_camera_from_branch("cam1", "detection")
        manager.remove_camera("cam1")
    """

    def __init__(self, pipeline: Gst.Pipeline, branches: dict[str, BranchInfo]):
        """
        Initialize camera manager

        Args:
            pipeline: GStreamer pipeline to add camera bins to
            branches: Dict of branch_name -> BranchInfo (from TeeFanoutPipelineBuilder)
        """
        self.pipeline = pipeline
        self.branches = branches
        self._cameras: dict[str, CameraBin] = {}
        self._mapper = SourceIDMapper()
        self._lock = threading.Lock()
        self._op_lock = threading.Lock()  # Serialize camera operations
        self._ghost_pad_counter = 0  # For unique ghost pad naming
        self._last_op_time = 0.0  # Track last operation time

    def add_camera(self, camera_id: str, uri: str, branch_names: list[str]) -> bool:
        """
        Add camera to specified branches with tee fanout

        Single decode via nvurisrcbin, zero-copy fanout to branches via tee.

        Args:
            camera_id: Unique camera identifier
            uri: Camera source URI (rtsp://, file://, etc.)
            branch_names: List of branch names to connect camera to

        Returns:
            True if successful, False if camera_id exists or no valid branches
        """
        # Serialize operations and enforce minimum delay
        with self._op_lock:
            elapsed = time.time() - self._last_op_time
            if elapsed < MIN_OP_DELAY:
                time.sleep(MIN_OP_DELAY - elapsed)

            with self._lock:
                if camera_id in self._cameras:
                    logger.warning(f"Camera already exists: {camera_id}")
                    return False

                # Filter valid branches
                valid_branches = [b for b in branch_names if b in self.branches]
                if not valid_branches:
                    logger.error(f"No valid branches for camera {camera_id}: {branch_names}")
                    return False

                try:
                    # Allocate source_id
                    source_id = self._mapper.add(camera_id, uri)

                    # 1. Create camera bin
                    bin_elem = Gst.Bin.new(f"camera_bin_{camera_id}")

                    # Create nvurisrcbin for decoding
                    nvurisrcbin = Gst.ElementFactory.make("nvurisrcbin", f"src_{camera_id}")
                    if not nvurisrcbin:
                        raise RuntimeError("Cannot create nvurisrcbin")

                    nvurisrcbin.set_property("uri", uri)
                    nvurisrcbin.set_property("gpu-id", 0)
                    nvurisrcbin.set_property("cudadec-memtype", 0)  # NVBUF_MEM_DEFAULT
                    # Increase extra surfaces for multi-camera multi-branch to avoid buffer starvation
                    nvurisrcbin.set_property("num-extra-surfaces", 4)

                    # Create tee for fanout
                    tee = Gst.ElementFactory.make("tee", f"tee_{camera_id}")
                    if not tee:
                        raise RuntimeError("Cannot create tee")
                    tee.set_property("allow-not-linked", True)

                    bin_elem.add(nvurisrcbin)
                    bin_elem.add(tee)

                    # Fakesink for audio - needed to prevent not-linked errors
                    audio_fakesink = Gst.ElementFactory.make("fakesink", f"audio_fake_{camera_id}")
                    audio_fakesink.set_property("sync", False)
                    audio_fakesink.set_property("async", False)
                    bin_elem.add(audio_fakesink)

                    # 3. Create queues and link to branches FIRST
                    # This must be done BEFORE connecting pad-added callback to prevent race
                    branch_queues = {}
                    branch_pads = {}

                    for branch_name in valid_branches:
                        branch = self.branches[branch_name]

                        # Create queue for this branch connection
                        queue = Gst.ElementFactory.make("queue", f"q_{camera_id}_{branch_name}")
                        if not queue:
                            raise RuntimeError(f"Cannot create queue for {branch_name}")

                        queue.set_property("max-size-buffers", 30)
                        queue.set_property("max-size-bytes", 0)
                        queue.set_property("max-size-time", 0)
                        queue.set_property("leaky", 2)
                        bin_elem.add(queue)

                        # Create nvvideoconvert for buffer isolation
                        nvconv = Gst.ElementFactory.make("nvvideoconvert", f"nvconv_{camera_id}_{branch_name}")
                        if not nvconv:
                            raise RuntimeError(f"Cannot create nvvideoconvert for {branch_name}")
                        nvconv.set_property("nvbuf-memory-type", 0)
                        nvconv.set_property("copy-hw", 1)
                        bin_elem.add(nvconv)

                        # Request tee src pad and link: tee -> queue -> nvvideoconvert
                        tee_src = tee.request_pad_simple("src_%u")
                        if not tee_src:
                            raise RuntimeError("Cannot request tee src pad")

                        queue_sink = queue.get_static_pad("sink")
                        ret = tee_src.link(queue_sink)
                        if ret != Gst.PadLinkReturn.OK:
                            raise RuntimeError(f"Cannot link tee -> queue: {ret}")

                        if not queue.link(nvconv):
                            raise RuntimeError(f"Cannot link queue -> nvvideoconvert for {branch_name}")

                        branch_queues[branch_name] = nvconv
                        branch_pads[branch_name] = tee_src

                    # 4. Add bin to pipeline
                    self.pipeline.add(bin_elem)

                    # 5. Create ghost pads and link to nvstreammux
                    for branch_name, nvconv in branch_queues.items():
                        branch = self.branches[branch_name]
                        mux = branch.nvstreammux

                        mux_sink = mux.request_pad_simple(f"sink_{source_id}")
                        if not mux_sink:
                            raise RuntimeError(f"Cannot request mux sink pad for {branch_name}")

                        nvconv_src = nvconv.get_static_pad("src")
                        self._ghost_pad_counter += 1
                        ghost_name = f"src_{branch_name}_{self._ghost_pad_counter}"
                        ghost = Gst.GhostPad.new(ghost_name, nvconv_src)
                        bin_elem.add_pad(ghost)

                        ret = ghost.link(mux_sink)
                        if ret != Gst.PadLinkReturn.OK:
                            raise RuntimeError(f"Cannot link ghost -> mux: {ret}")

                        logger.info(f"[{camera_id}] Connected to branch '{branch_name}' (sink_{source_id})")

                    # 6. Connect pad-added callback with safety check
                    def on_pad_added(src, pad, data):
                        tee_elem, cam_id, fake_audio, manager = data
                        # Safety: check if camera still exists
                        if not manager.has_camera(cam_id):
                            print(f"[{cam_id}] Pad added after removal, ignoring")
                            return

                        pad_name = pad.get_name()
                        print(f"[{cam_id}] Pad added: {pad_name}")

                        try:
                            if pad_name.startswith("vsrc"):
                                tee_sink = tee_elem.get_static_pad("sink")
                                if tee_sink and not tee_sink.is_linked():
                                    ret = pad.link(tee_sink)
                                    print(f"[{cam_id}] nvurisrcbin video -> tee linked: {ret}")
                            elif pad_name.startswith("asrc"):
                                fake_sink = fake_audio.get_static_pad("sink")
                                if fake_sink and not fake_sink.is_linked():
                                    ret = pad.link(fake_sink)
                                    print(f"[{cam_id}] nvurisrcbin audio -> fakesink linked: {ret}")
                        except Exception as e:
                            print(f"[{cam_id}] Error in pad-added callback: {e}")

                    nvurisrcbin.connect("pad-added", on_pad_added, (tee, camera_id, audio_fakesink, self))

                    # 7. Sync state with pipeline
                    bin_elem.sync_state_with_parent()

                    # 8. Store camera info
                    self._cameras[camera_id] = CameraBin(
                        camera_id=camera_id,
                        uri=uri,
                        bin=bin_elem,
                        nvurisrcbin=nvurisrcbin,
                        tee=tee,
                        branch_queues=branch_queues,
                        branch_pads=branch_pads,
                        source_id=source_id
                    )

                    self._last_op_time = time.time()
                    logger.info(f"Camera added: {camera_id} -> {valid_branches} (source_id={source_id})")
                    return True

                except Exception as e:
                    logger.error(f"Failed to add camera {camera_id}: {e}")
                    # Cleanup on failure
                    self._mapper.remove(camera_id)
                    return False

    def remove_camera(self, camera_id: str) -> bool:
        """
        Remove camera from all branches

        Properly blocks pads, unlinks, and releases resources.

        Args:
            camera_id: Camera identifier to remove

        Returns:
            True if successful, False if camera not found
        """
        with self._op_lock:
            elapsed = time.time() - self._last_op_time
            if elapsed < MIN_OP_DELAY:
                time.sleep(MIN_OP_DELAY - elapsed)

            with self._lock:
                cam = self._cameras.get(camera_id)
                if not cam:
                    logger.warning(f"Camera not found: {camera_id}")
                    return False

                try:
                    # 1. Block all tee src pads to prevent data flow
                    for branch_name, tee_pad in cam.branch_pads.items():
                        tee_pad.add_probe(
                            Gst.PadProbeType.BLOCK_DOWNSTREAM,
                            lambda pad, info: Gst.PadProbeReturn.OK
                        )

                    # Wait for buffers to drain
                    time.sleep(0.3)

                    # 2. Unlink from all muxers and release pads
                    for branch_name in list(cam.branch_queues.keys()):
                        self._unlink_from_branch(cam, branch_name)

                    # 3. Set bin to NULL state and wait
                    cam.bin.set_state(Gst.State.NULL)
                    cam.bin.get_state(Gst.CLOCK_TIME_NONE)

                    # 4. Remove bin from pipeline
                    self.pipeline.remove(cam.bin)

                    # 5. Cleanup mapper
                    self._mapper.remove(camera_id)

                    # 6. Remove from tracking
                    del self._cameras[camera_id]

                    self._last_op_time = time.time()
                    logger.info(f"Camera removed: {camera_id}")
                    return True

                except Exception as e:
                    logger.error(f"Failed to remove camera {camera_id}: {e}")
                    return False

    def add_camera_to_branch(self, camera_id: str, branch_name: str) -> bool:
        """
        Add existing camera to additional branch

        Args:
            camera_id: Existing camera identifier
            branch_name: Branch to add camera to

        Returns:
            True if successful, False if camera not found or already connected
        """
        with self._op_lock:
            elapsed = time.time() - self._last_op_time
            if elapsed < MIN_OP_DELAY:
                time.sleep(MIN_OP_DELAY - elapsed)

            with self._lock:
                cam = self._cameras.get(camera_id)
                if not cam:
                    logger.warning(f"Camera not found: {camera_id}")
                    return False

                if branch_name in cam.branch_queues:
                    logger.warning(f"Camera {camera_id} already connected to {branch_name}")
                    return False

                branch = self.branches.get(branch_name)
                if not branch:
                    logger.error(f"Branch not found: {branch_name}")
                    return False

                try:
                    # 1. Create queue for branch connection
                    queue = Gst.ElementFactory.make("queue", f"q_{camera_id}_{branch_name}")
                    queue.set_property("max-size-buffers", 30)
                    queue.set_property("max-size-bytes", 0)
                    queue.set_property("max-size-time", 0)
                    queue.set_property("leaky", 2)
                    cam.bin.add(queue)

                    # 2. Create nvvideoconvert for buffer isolation
                    nvconv = Gst.ElementFactory.make("nvvideoconvert", f"nvconv_{camera_id}_{branch_name}")
                    nvconv.set_property("nvbuf-memory-type", 0)
                    nvconv.set_property("copy-hw", 1)
                    cam.bin.add(nvconv)

                    # 3. Request tee src pad and link: tee -> queue -> nvvideoconvert
                    tee_src = cam.tee.request_pad_simple("src_%u")
                    queue_sink = queue.get_static_pad("sink")
                    tee_src.link(queue_sink)
                    queue.link(nvconv)

                    # 4. Create ghost pad and link to muxer
                    mux_sink = branch.nvstreammux.request_pad_simple(f"sink_{cam.source_id}")
                    nvconv_src = nvconv.get_static_pad("src")

                    self._ghost_pad_counter += 1
                    ghost_name = f"src_{branch_name}_{self._ghost_pad_counter}"
                    ghost = Gst.GhostPad.new(ghost_name, nvconv_src)
                    cam.bin.add_pad(ghost)
                    ghost.link(mux_sink)

                    # 5. Sync state
                    queue.sync_state_with_parent()
                    nvconv.sync_state_with_parent()

                    # 6. Update tracking
                    cam.branch_queues[branch_name] = nvconv
                    cam.branch_pads[branch_name] = tee_src

                    self._last_op_time = time.time()
                    logger.info(f"Camera {camera_id} added to branch '{branch_name}'")
                    return True

                except Exception as e:
                    logger.error(f"Failed to add {camera_id} to {branch_name}: {e}")
                    return False

    def remove_camera_from_branch(self, camera_id: str, branch_name: str) -> bool:
        """
        Remove camera from specific branch (keep in other branches)

        Args:
            camera_id: Camera identifier
            branch_name: Branch to remove camera from

        Returns:
            True if successful, False if camera/branch not found
        """
        with self._op_lock:
            elapsed = time.time() - self._last_op_time
            if elapsed < MIN_OP_DELAY:
                time.sleep(MIN_OP_DELAY - elapsed)

            with self._lock:
                cam = self._cameras.get(camera_id)
                if not cam:
                    logger.warning(f"Camera not found: {camera_id}")
                    return False

                if branch_name not in cam.branch_queues:
                    logger.warning(f"Camera {camera_id} not connected to {branch_name}")
                    return False

                # Don't allow removing from last branch - use remove_camera() instead
                if len(cam.branch_queues) == 1:
                    logger.warning(f"Cannot remove {camera_id} from last branch. Use remove_camera()")
                    return False

                try:
                    # 1. Block tee src pad
                    tee_src = cam.branch_pads[branch_name]
                    tee_src.add_probe(
                        Gst.PadProbeType.BLOCK_DOWNSTREAM,
                        lambda pad, info: Gst.PadProbeReturn.OK
                    )

                    # Wait for buffers to drain
                    time.sleep(0.3)

                    # 2. Unlink and cleanup
                    self._unlink_from_branch(cam, branch_name)

                    self._last_op_time = time.time()
                    logger.info(f"Camera {camera_id} removed from branch '{branch_name}'")
                    return True

                except Exception as e:
                    logger.error(f"Failed to remove {camera_id} from {branch_name}: {e}")
                    return False

    def _unlink_from_branch(self, cam: CameraBin, branch_name: str) -> None:
        """
        Internal: Unlink camera from specific branch

        Releases muxer pad, removes ghost pad, removes queue and nvvideoconvert, releases tee pad.

        Element chain: tee -> queue -> nvvideoconvert -> ghost_pad -> muxer
        """
        if branch_name not in cam.branch_queues:
            return

        branch = self.branches.get(branch_name)
        if not branch:
            return

        tee_src = cam.branch_pads[branch_name]

        # Find and unlink ghost pad
        # CRITICAL: GstIterator requires proper iteration - Python for loop doesn't work
        ghost_pad = None
        iterator = cam.bin.iterate_pads()
        while True:
            ret, pad = iterator.next()
            if ret == Gst.IteratorResult.OK:
                if pad.get_name().startswith(f"src_{branch_name}"):
                    ghost_pad = pad
                    break
            elif ret == Gst.IteratorResult.DONE:
                break
            elif ret == Gst.IteratorResult.RESYNC:
                iterator.resync()
            else:
                break

        if ghost_pad:
            peer = ghost_pad.get_peer()
            if peer:
                ghost_pad.unlink(peer)
                # Release muxer sink pad
                branch.nvstreammux.release_request_pad(peer)
            cam.bin.remove_pad(ghost_pad)

        # Get queue and nvvideoconvert elements by name and remove them
        queue_name = f"q_{cam.camera_id}_{branch_name}"
        nvconv_name = f"nvconv_{cam.camera_id}_{branch_name}"
        queue = cam.bin.get_by_name(queue_name)
        nvconv = cam.bin.get_by_name(nvconv_name)

        # Remove in reverse order: nvvideoconvert -> queue
        if nvconv:
            nvconv.set_state(Gst.State.NULL)
            cam.bin.remove(nvconv)
        if queue:
            queue.set_state(Gst.State.NULL)
            cam.bin.remove(queue)

        # Release tee src pad
        cam.tee.release_request_pad(tee_src)

        # Update tracking
        del cam.branch_queues[branch_name]
        del cam.branch_pads[branch_name]

    def list_cameras(self) -> dict:
        """
        List all cameras with branch assignments

        Returns:
            Dict mapping camera_id to camera info
        """
        with self._lock:
            return {
                cam_id: {
                    "uri": cam.uri,
                    "source_id": cam.source_id,
                    "branches": list(cam.branch_queues.keys())
                }
                for cam_id, cam in self._cameras.items()
            }

    def get_camera(self, camera_id: str) -> Optional[CameraBin]:
        """Get camera info by ID"""
        with self._lock:
            return self._cameras.get(camera_id)

    def count(self) -> int:
        """Get number of active cameras"""
        with self._lock:
            return len(self._cameras)

    def has_camera(self, camera_id: str) -> bool:
        """Check if camera exists"""
        with self._lock:
            return camera_id in self._cameras

    def get_camera_branches(self, camera_id: str) -> list[str]:
        """Get list of branches camera is connected to"""
        with self._lock:
            cam = self._cameras.get(camera_id)
            return list(cam.branch_queues.keys()) if cam else []

    def kill_all(self) -> int:
        """
        Remove all cameras from all branches

        Returns:
            Number of cameras removed
        """
        with self._lock:
            for camera_id, cam in list(self._cameras.items()):
                try:
                    cam.bin.set_state(Gst.State.NULL)
                    self.pipeline.remove(cam.bin)
                except Exception as e:
                    logger.error(f"Error removing {camera_id}: {e}")
            count = len(self._cameras)
            self._cameras.clear()
            self._mapper.clear()
            logger.info(f"Kill all: {count} cameras removed")
            return count
