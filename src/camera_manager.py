"""Dynamic camera management with tee fanout to multiple branches.

STABILITY FIX (2026-01-13):
- PAUSE pipeline during camera add/remove for safe topology changes
- Incremental state sync: NULL -> READY -> PAUSED -> PLAYING with waits
- Reference: /home/mq/disk2T/duy/tks_prj/infra/deepstream
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.pipeline_builder import BranchInfo
from src.source_mapper import SourceIDMapper

logger = logging.getLogger(__name__)

STATE_CHANGE_TIMEOUT = 5 * Gst.SECOND


@dataclass
class CameraContext:
    """Context for camera addition operation."""
    camera_id: str
    uri: str
    source_id: int
    bin_elem: Gst.Bin
    tee: Gst.Element
    branches: list[str]
    is_first_camera: bool
    prev_state: Gst.State
    linked_state: dict
    pad_linked_event: Optional[threading.Event] = None


class MultibranchCameraManager:
    """Add/remove cameras to multiple branches at runtime."""

    def __init__(self, pipeline: Gst.Pipeline, branches: dict[str, BranchInfo], gpu_id: int = 0):
        self.pipeline = pipeline
        self.branches = branches
        self._gpu_id = gpu_id
        self._cameras: dict[str, dict] = {}
        self._mapper = SourceIDMapper()
        self._lock = threading.Lock()
        self._last_op = 0.0

    def _delay(self):
        """Wait 2s between operations."""
        wait = 2.0 - (time.time() - self._last_op)
        if wait > 0:
            time.sleep(wait)

    def _set_state_with_wait(self, element: Gst.Element, state: Gst.State, timeout=STATE_CHANGE_TIMEOUT) -> bool:
        """Set element state and wait for completion."""
        element.set_state(state)
        ret, _, _ = element.get_state(timeout)
        return ret != Gst.StateChangeReturn.FAILURE

    def _incremental_state_sync(self, element: Gst.Element, target_state: Gst.State) -> bool:
        """Sync element state incrementally: NULL -> READY -> PAUSED -> PLAYING."""
        name = element.get_name()
        element.set_state(Gst.State.NULL)

        states = [(Gst.State.READY, "READY"), (Gst.State.PAUSED, "PAUSED"), (Gst.State.PLAYING, "PLAYING")]
        for state, state_name in states:
            if target_state >= state:
                if not self._set_state_with_wait(element, state):
                    logger.error(f"[CAM-MANAGER] Failed to set {name} to {state_name}")
                    return False
                logger.debug(f"[CAM-MANAGER] {name} set to {state_name}")
        return True

    def _create_source(self, ctx: CameraContext) -> Gst.Element:
        """Create source element based on URI scheme."""
        uri = ctx.uri
        camera_id = ctx.camera_id
        is_rtsp = uri.startswith("rtsp://") or uri.startswith("rtsps://")
        is_file = uri.startswith("file://")

        if is_rtsp:
            return self._create_rtsp_source(ctx)
        elif is_file:
            return self._create_file_source(ctx, with_event=True)
        else:
            return self._create_file_source(ctx, with_event=False)

    def _create_rtsp_source(self, ctx: CameraContext) -> Gst.Element:
        """Create nvurisrcbin for RTSP sources."""
        source = Gst.ElementFactory.make("nvurisrcbin", f"nvurisrc_{ctx.camera_id}")
        source.set_property("uri", ctx.uri)
        source.set_property("gpu-id", self._gpu_id)
        source.set_property("disable-audio", True)
        source.set_property("source-id", ctx.source_id)
        source.set_property("cudadec-memtype", 0)
        source.set_property("num-extra-surfaces", 1)

        def on_pad_added(_s, pad, tee, linked):
            if linked["done"]:
                return
            if pad.get_name().startswith("vsrc_"):
                sink = tee.get_static_pad("sink")
                if not sink.is_linked():
                    if pad.link(sink) == Gst.PadLinkReturn.OK:
                        linked["done"] = True
                        logger.info(f"[CAM-MANAGER] nvurisrcbin linked for {ctx.camera_id}")

        source.connect("pad-added", on_pad_added, ctx.tee, ctx.linked_state)
        logger.info(f"[CAM-MANAGER] Using nvurisrcbin for: {ctx.camera_id}")
        return source

    def _create_file_source(self, ctx: CameraContext, with_event: bool) -> Gst.Element:
        """Create uridecodebin for file/HTTP sources."""
        source = Gst.ElementFactory.make("uridecodebin", f"uridecodebin_{ctx.camera_id}")
        source.set_property("uri", ctx.uri)

        if with_event:
            ctx.pad_linked_event = threading.Event()

        def on_pad_added(_s, pad, tee, linked, event):
            if linked["done"]:
                return
            caps = pad.get_current_caps() or pad.query_caps(None)
            if caps:
                struct = caps.get_structure(0)
                if struct and struct.get_name().startswith("video"):
                    sink = tee.get_static_pad("sink")
                    if sink and not sink.is_linked():
                        if pad.link(sink) == Gst.PadLinkReturn.OK:
                            linked["done"] = True
                            if event:
                                event.set()
                            logger.info(f"[CAM-MANAGER] uridecodebin linked for {ctx.camera_id}")

        source.connect("pad-added", on_pad_added, ctx.tee, ctx.linked_state, ctx.pad_linked_event)
        logger.info(f"[CAM-MANAGER] Using uridecodebin for: {ctx.camera_id}")
        return source

    def _setup_branch_drop_probes(self, ctx: CameraContext, branch_pads: dict) -> dict:
        """Add DROP probes on non-first branches for first camera startup."""
        drop_flags = {}
        if not ctx.is_first_camera or len(ctx.branches) <= 1:
            return drop_flags

        for idx, b in enumerate(ctx.branches):
            if idx > 0:  # Skip first branch
                drop_flag = [True]

                def make_probe(flag):
                    def probe_fn(p, info):
                        return Gst.PadProbeReturn.DROP if flag[0] else Gst.PadProbeReturn.OK
                    return probe_fn

                pad = branch_pads[b]
                probe_id = pad.add_probe(Gst.PadProbeType.BUFFER, make_probe(drop_flag))
                drop_flags[b] = (pad, probe_id, drop_flag)
                logger.debug(f"[CAM-MANAGER] Added DROP probe to branch {b}")

        return drop_flags

    def _start_first_camera(self, ctx: CameraContext, drop_flags: dict) -> bool:
        """Handle first camera startup with warmup probes."""
        logger.info(f"[CAM-MANAGER] First camera - transitioning to PLAYING")

        # Warmup probe to drop first 3 frames
        first_frame_event = threading.Event()
        frame_count = [0]

        def warmup_probe(pad, info):
            frame_count[0] += 1
            if frame_count[0] <= 3:
                return Gst.PadProbeReturn.DROP
            if frame_count[0] == 4:
                first_frame_event.set()
                logger.info(f"[CAM-MANAGER] First valid frame for {ctx.camera_id}")
            return Gst.PadProbeReturn.PASS

        tee_sink = ctx.tee.get_static_pad("sink")
        probe_id = tee_sink.add_probe(Gst.PadProbeType.BUFFER, warmup_probe)

        # PAUSED -> wait -> PLAYING
        self._set_state_with_wait(self.pipeline, Gst.State.PAUSED, 10 * Gst.SECOND)
        time.sleep(2.0)

        self._set_state_with_wait(self.pipeline, Gst.State.PLAYING)
        first_frame_event.wait(timeout=5.0)
        tee_sink.remove_probe(probe_id)

        # Enable remaining branches sequentially
        if drop_flags:
            time.sleep(2.0)
            logger.info(f"[CAM-MANAGER] Enabling remaining branches...")
            for idx, (b, (pad, pid, flag)) in enumerate(drop_flags.items()):
                logger.info(f"[CAM-MANAGER] Enabling branch: {b}")
                flag[0] = False
                if idx < len(drop_flags) - 1:
                    time.sleep(1.5)
            logger.info(f"[CAM-MANAGER] All branches enabled")

        return True

    def _resume_additional_camera(self, ctx: CameraContext, drop_flags: dict) -> bool:
        """Handle additional camera with pipeline resume."""
        logger.info(f"[CAM-MANAGER] Additional camera - resuming pipeline")
        is_file = ctx.uri.startswith("file://")

        if is_file:
            # Sync bin to PLAYING first, then resume pipeline
            ctx.bin_elem.set_state(Gst.State.PLAYING)
            ctx.bin_elem.get_state(2 * Gst.SECOND)

        self._set_state_with_wait(self.pipeline, Gst.State.PLAYING)

        if not is_file:
            self._incremental_state_sync(ctx.bin_elem, Gst.State.PLAYING)

        time.sleep(3.0)
        logger.info(f"[CAM-MANAGER] Additional camera stabilized")

        # Enable branches
        if drop_flags:
            for idx, (b, (pad, pid, flag)) in enumerate(drop_flags.items()):
                flag[0] = False
                if idx < len(drop_flags) - 1:
                    time.sleep(0.5)

        return True

    def add_camera(self, camera_id: str, uri: str, branch_names: list[str]) -> bool:
        """Add camera to branches with safe pipeline state management."""
        with self._lock:
            self._delay()
            if camera_id in self._cameras:
                return False

            branches = [b for b in branch_names if b in self.branches]
            if not branches:
                return False

            _, prev_state, _ = self.pipeline.get_state(0)
            is_first = len(self._cameras) == 0
            logger.info(f"[CAM-MANAGER] Adding {camera_id} - state: {prev_state.value_nick}, first: {is_first}")

            try:
                # Pause if playing
                if prev_state == Gst.State.PLAYING:
                    logger.info(f"[CAM-MANAGER] Pausing pipeline for safe addition")
                    if not self._set_state_with_wait(self.pipeline, Gst.State.PAUSED):
                        return False
                    time.sleep(0.5)

                # Create camera bin
                source_id = self._mapper.add(camera_id, uri)
                bin_elem = Gst.Bin.new(f"cam_{camera_id}")
                tee = Gst.ElementFactory.make("tee", f"tee_{camera_id}")
                tee.set_property("allow-not-linked", True)
                bin_elem.add(tee)

                ctx = CameraContext(
                    camera_id=camera_id, uri=uri, source_id=source_id,
                    bin_elem=bin_elem, tee=tee, branches=branches,
                    is_first_camera=is_first, prev_state=prev_state,
                    linked_state={"done": False}
                )

                # Create and add source
                source = self._create_source(ctx)
                bin_elem.add(source)

                # Ghost pad for external access
                ghost_src = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
                bin_elem.add_pad(ghost_src)
                self.pipeline.add(bin_elem)

                # Link branches
                branch_pads = {}
                for idx, b in enumerate(branches):
                    logger.info(f"[CAM-MANAGER] Linking branch {idx+1}/{len(branches)}: {b}")
                    pad = self._link_branch(bin_elem, tee, camera_id, source_id, b)
                    branch_pads[b] = pad

                drop_flags = self._setup_branch_drop_probes(ctx, branch_pads)

                # Sync bin state
                if is_first or prev_state == Gst.State.READY:
                    self._incremental_state_sync(bin_elem, Gst.State.READY)
                else:
                    self._incremental_state_sync(bin_elem, Gst.State.PAUSED)
                    if ctx.pad_linked_event:
                        ctx.pad_linked_event.wait(timeout=5.0)
                    else:
                        time.sleep(1.0)

                # Store camera
                self._cameras[camera_id] = {
                    "bin": bin_elem, "tee": tee, "source_id": source_id,
                    "uri": uri, "branch_pads": branch_pads
                }

                # Start/resume pipeline
                if is_first or prev_state == Gst.State.READY:
                    self._start_first_camera(ctx, drop_flags)
                else:
                    self._resume_additional_camera(ctx, drop_flags)

                self._last_op = time.time()
                logger.info(f"[CAM-MANAGER] Successfully added {camera_id} to: {branches}")
                return True

            except Exception as e:
                logger.error(f"add_camera failed: {e}", exc_info=True)
                try:
                    self._cameras.pop(camera_id, None)
                    self.pipeline.remove(bin_elem)
                except:
                    pass
                self._mapper.remove(camera_id)
                if prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PLAYING)
                return False

    def remove_camera(self, camera_id: str) -> bool:
        """Remove camera from all branches."""
        with self._lock:
            self._delay()
            cam = self._cameras.get(camera_id)
            if not cam:
                return False

            try:
                for pad in cam["branch_pads"].values():
                    pad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, lambda *_: Gst.PadProbeReturn.REMOVE)
                time.sleep(0.2)

                cam["bin"].set_state(Gst.State.NULL)
                cam["bin"].get_state(Gst.CLOCK_TIME_NONE)

                for b in list(cam["branch_pads"].keys()):
                    self._unlink_branch(cam, b, camera_id)

                self.pipeline.remove(cam["bin"])
                self._mapper.remove(camera_id)
                del self._cameras[camera_id]
                self._last_op = time.time()
                return True

            except Exception as e:
                logger.error(f"remove_camera failed: {e}")
                return False

    def add_camera_to_branch(self, camera_id: str, branch_name: str) -> bool:
        """Add existing camera to additional branch."""
        with self._lock:
            self._delay()
            cam = self._cameras.get(camera_id)
            if not cam or branch_name in cam["branch_pads"] or branch_name not in self.branches:
                return False

            _, prev_state, _ = self.pipeline.get_state(0)
            logger.info(f"[CAM-MANAGER] Adding {camera_id} to branch {branch_name}")

            try:
                if prev_state == Gst.State.PLAYING:
                    self._set_state_with_wait(self.pipeline, Gst.State.PAUSED)

                pad = self._link_branch(cam["bin"], cam["tee"], camera_id, cam["source_id"], branch_name, sync=True)
                cam["branch_pads"][branch_name] = pad

                if prev_state == Gst.State.PLAYING:
                    self._set_state_with_wait(self.pipeline, Gst.State.PLAYING)

                self._last_op = time.time()
                logger.info(f"[CAM-MANAGER] Successfully added {camera_id} to {branch_name}")
                return True
            except Exception as e:
                logger.error(f"add_camera_to_branch failed: {e}", exc_info=True)
                if prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PLAYING)
                return False

    def remove_camera_from_branch(self, camera_id: str, branch_name: str) -> bool:
        """Remove camera from single branch."""
        with self._lock:
            self._delay()
            cam = self._cameras.get(camera_id)
            if not cam:
                logger.warning(f"[CAM-MANAGER] Camera {camera_id} not found")
                return False
            if branch_name not in cam["branch_pads"]:
                logger.warning(f"[CAM-MANAGER] Camera {camera_id} not in branch {branch_name}")
                return False
            if len(cam["branch_pads"]) <= 1:
                logger.info(f"[CAM-MANAGER] Use remove_camera instead (only 1 branch)")
                return False

            logger.info(f"[CAM-MANAGER] Removing {camera_id} from {branch_name}")
            try:
                tee_pad = cam["branch_pads"].get(branch_name)
                blocked = threading.Event()
                probe_id = None

                if tee_pad:
                    def block_probe(pad, info):
                        blocked.set()
                        return Gst.PadProbeReturn.OK

                    probe_id = tee_pad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, block_probe)
                    blocked.wait(timeout=1.0)

                self._unlink_branch(cam, branch_name, camera_id, tee_pad, probe_id)
                self._last_op = time.time()
                logger.info(f"[CAM-MANAGER] Removed {camera_id} from {branch_name}")
                return True
            except Exception as e:
                logger.error(f"remove_camera_from_branch failed: {e}", exc_info=True)
                return False

    def _link_branch(self, bin_elem, tee, camera_id, source_id, branch_name, sync=False) -> Gst.Pad:
        """Link: tee -> queue -> nvstreammux."""
        b = self.branches[branch_name]

        q = Gst.ElementFactory.make("queue", f"q_{camera_id}_{branch_name}")
        q.set_property("max-size-buffers", 30)
        q.set_property("max-size-bytes", 0)
        q.set_property("max-size-time", 0)
        q.set_property("leaky", 2)
        bin_elem.add(q)

        tee_src = tee.request_pad_simple("src_%u")
        tee_src.link(q.get_static_pad("sink"))

        mux_sink = b.nvstreammux.request_pad_simple(f"sink_{source_id}")
        ghost = Gst.GhostPad.new(f"g_{branch_name}_{camera_id}", q.get_static_pad("src"))
        bin_elem.add_pad(ghost)
        ghost.link(mux_sink)

        if sync:
            _, parent_state, _ = bin_elem.get_state(0)
            self._incremental_state_sync(q, parent_state)

        return tee_src

    def _unlink_branch(self, cam: dict, branch_name: str, camera_id: str,
                       tee_pad: Optional[Gst.Pad] = None, probe_id: Optional[int] = None) -> None:
        """Unlink and cleanup branch elements."""
        b = self.branches.get(branch_name)
        if not b:
            return

        tee_pad = tee_pad or cam["branch_pads"].get(branch_name)
        q_name = f"q_{camera_id}_{branch_name}"
        ghost_name = f"g_{branch_name}_{camera_id}"

        # Find and unlink ghost pad from mux
        ghost_pad, mux_pad = None, None
        it = cam["bin"].iterate_pads()
        while True:
            ret, pad = it.next()
            if ret == Gst.IteratorResult.OK and pad.get_name() == ghost_name:
                ghost_pad = pad
                mux_pad = pad.get_peer()
                if mux_pad:
                    pad.unlink(mux_pad)
                break
            elif ret == Gst.IteratorResult.RESYNC:
                it.resync()
            elif ret != Gst.IteratorResult.OK:
                break

        # Unlink queue from tee
        q_elem = cam["bin"].get_by_name(q_name)
        if q_elem:
            q_sink = q_elem.get_static_pad("sink")
            if q_sink and q_sink.is_linked():
                peer = q_sink.get_peer()
                if peer:
                    peer.unlink(q_sink)

        # Remove probe
        if tee_pad and probe_id is not None:
            tee_pad.remove_probe(probe_id)

        time.sleep(0.1)

        # Set NULL and remove queue
        if q_elem:
            q_elem.set_state(Gst.State.NULL)
            q_elem.get_state(Gst.CLOCK_TIME_NONE)
            time.sleep(0.05)
            cam["bin"].remove(q_elem)

        # Remove ghost pad
        if ghost_pad:
            cam["bin"].remove_pad(ghost_pad)

        # Release pads
        if mux_pad:
            try:
                b.nvstreammux.release_request_pad(mux_pad)
            except Exception as e:
                logger.warning(f"release mux pad failed: {e}")

        if tee_pad:
            try:
                cam["tee"].release_request_pad(tee_pad)
            except Exception as e:
                logger.warning(f"release tee pad failed: {e}")

        cam["branch_pads"].pop(branch_name, None)
        logger.info(f"Unlinked {camera_id} from {branch_name}")

    # Query methods
    def list_cameras(self) -> dict:
        with self._lock:
            return {k: {"uri": v["uri"], "source_id": v["source_id"], "branches": list(v["branch_pads"].keys())}
                    for k, v in self._cameras.items()}

    def get_camera(self, camera_id: str) -> Optional[dict]:
        with self._lock:
            return self._cameras.get(camera_id)

    def has_camera(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._cameras

    def count(self) -> int:
        with self._lock:
            return len(self._cameras)

    def get_camera_branches(self, camera_id: str) -> list[str]:
        with self._lock:
            cam = self._cameras.get(camera_id)
            return list(cam["branch_pads"].keys()) if cam else []

    def get_mapper(self) -> SourceIDMapper:
        return self._mapper

    def kill_all(self) -> int:
        """Remove all cameras."""
        with self._lock:
            for cam in self._cameras.values():
                try:
                    cam["bin"].set_state(Gst.State.NULL)
                    cam["bin"].get_state(Gst.CLOCK_TIME_NONE)
                    self.pipeline.remove(cam["bin"])
                except:
                    pass
            n = len(self._cameras)
            self._cameras.clear()
            self._mapper.clear()
            self._last_op = time.time()
            return n
