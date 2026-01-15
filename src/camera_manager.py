"""Dynamic camera management with tee fanout to multiple branches.

STABILITY FIX (2026-01-13):
- PAUSE pipeline during camera add/remove for safe topology changes
- Incremental state sync: NULL → READY → PAUSED → PLAYING with waits
- Update batch size AFTER state sync completes
- Reference: /home/mq/disk2T/duy/tks_prj/infra/deepstream
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.pipeline_builder import BranchInfo
from src.source_mapper import SourceIDMapper

logger = logging.getLogger(__name__)

# State transition timeout (5 seconds per state)
STATE_CHANGE_TIMEOUT = 5 * Gst.SECOND


class MultibranchCameraManager:
    """Add/remove cameras to multiple branches at runtime."""

    def __init__(self, pipeline: Gst.Pipeline, branches: dict[str, BranchInfo], gpu_id: int = 0):
        self.pipeline = pipeline
        self.branches = branches
        self._gpu_id = gpu_id
        self._cameras: dict[str, dict] = {}  # camera_id -> {bin, tee, source_id, uri, branch_pads}
        self._mapper = SourceIDMapper()
        self._lock = threading.Lock()
        self._pad_counter = 0
        self._last_op = 0.0

    def _delay(self):
        """Wait 2s between operations."""
        wait = 2.0 - (time.time() - self._last_op)
        if wait > 0:
            time.sleep(wait)

    def _incremental_state_sync(self, element: Gst.Element, target_state: Gst.State) -> bool:
        """Sync element state incrementally: NULL → READY → PAUSED → PLAYING.

        Reference implementation pattern from duy/tks_prj/infra/deepstream.
        Each transition waits for completion before proceeding.
        """
        element_name = element.get_name()

        # Always start from NULL
        element.set_state(Gst.State.NULL)

        # Incremental transitions with explicit waits
        if target_state >= Gst.State.READY:
            element.set_state(Gst.State.READY)
            ret, _, _ = element.get_state(STATE_CHANGE_TIMEOUT)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error(f"[CAM-MANAGER] Failed to set {element_name} to READY")
                return False
            logger.debug(f"[CAM-MANAGER] {element_name} set to READY")

        if target_state >= Gst.State.PAUSED:
            element.set_state(Gst.State.PAUSED)
            ret, _, _ = element.get_state(STATE_CHANGE_TIMEOUT)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error(f"[CAM-MANAGER] Failed to set {element_name} to PAUSED")
                return False
            logger.debug(f"[CAM-MANAGER] {element_name} set to PAUSED")

        if target_state >= Gst.State.PLAYING:
            element.set_state(Gst.State.PLAYING)
            ret, _, _ = element.get_state(STATE_CHANGE_TIMEOUT)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error(f"[CAM-MANAGER] Failed to set {element_name} to PLAYING")
                return False
            logger.debug(f"[CAM-MANAGER] {element_name} set to PLAYING")

        return True

    def add_camera(self, camera_id: str, uri: str, branch_names: list[str]) -> bool:
        """Add camera to branches with safe pipeline state management.

        STABILITY FIX:
        - If pipeline is READY (no cameras yet), transition to PLAYING after adding
        - If pipeline is PLAYING, PAUSE during topology change to prevent race conditions
        Reference: duy/tks_prj/infra/deepstream pattern.
        """
        with self._lock:
            self._delay()
            if camera_id in self._cameras:
                return False

            branches = [b for b in branch_names if b in self.branches]
            if not branches:
                return False

            # Get current pipeline state
            _, prev_state, _ = self.pipeline.get_state(0)
            is_first_camera = len(self._cameras) == 0
            logger.info(f"[CAM-MANAGER] Adding {camera_id} - pipeline state: {prev_state.value_nick}, first_camera: {is_first_camera}")

            try:
                # STEP 1: ALWAYS pause pipeline before adding new camera
                # This prevents nvurisrcbin internal pad conflict when adding during PLAYING
                if prev_state == Gst.State.PLAYING:
                    logger.info(f"[CAM-MANAGER] Pausing pipeline for safe camera addition")
                    self.pipeline.set_state(Gst.State.PAUSED)
                    ret, _, _ = self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
                    if ret == Gst.StateChangeReturn.FAILURE:
                        logger.error(f"[CAM-MANAGER] Failed to pause pipeline")
                        return False
                    time.sleep(0.5)  # Brief stabilization

                # STEP 2: Create camera bin elements
                source_id = self._mapper.add(camera_id, uri)
                bin_elem = Gst.Bin.new(f"cam_{camera_id}")

                tee = Gst.ElementFactory.make("tee", f"tee_{camera_id}")
                tee.set_property("allow-not-linked", True)
                bin_elem.add(tee)

                # Track linked state to avoid duplicate connections
                linked_state = {"done": False}

                # Determine source type based on URI scheme
                is_rtsp = uri.startswith("rtsp://") or uri.startswith("rtsps://")
                is_file = uri.startswith("file://")

                # Use nvurisrcbin for RTSP sources (better DeepStream integration)
                if is_rtsp:
                    source = Gst.ElementFactory.make("nvurisrcbin", f"nvurisrc_{camera_id}")
                    source.set_property("uri", uri)
                    source.set_property("gpu-id", self._gpu_id)
                    source.set_property("disable-audio", True)
                    source.set_property("source-id", source_id)
                    source.set_property("cudadec-memtype", 0)  # Device memory
                    source.set_property("num-extra-surfaces", 1)
                    bin_elem.add(source)

                    # Connect pad-added for nvurisrcbin -> tee (video pads: vsrc_*)
                    def on_nvurisrc_pad_added(_s, pad, t):
                        if linked_state["done"]:
                            return
                        pad_name = pad.get_name()
                        if pad_name.startswith("vsrc_"):
                            sink = t.get_static_pad("sink")
                            if not sink.is_linked():
                                ret = pad.link(sink)
                                if ret == Gst.PadLinkReturn.OK:
                                    linked_state["done"] = True
                                    logger.info(f"[CAM-MANAGER] nvurisrcbin {pad_name} linked to tee for {camera_id}")
                                else:
                                    logger.warning(f"[CAM-MANAGER] Failed to link {pad_name} to tee: {ret}")

                    source.connect("pad-added", on_nvurisrc_pad_added, tee)
                    logger.info(f"[CAM-MANAGER] Using nvurisrcbin for RTSP source: {camera_id}")

                elif is_file:
                    # For file sources, use uridecodebin with proper state handling
                    # Create an event to signal when pad linking is complete
                    pad_linked_event = threading.Event()

                    source = Gst.ElementFactory.make("uridecodebin", f"uridecodebin_{camera_id}")
                    source.set_property("uri", uri)
                    bin_elem.add(source)

                    # Connect pad-added for uridecodebin -> tee (video pads only)
                    def on_uridecodebin_pad_added(_s, pad, t, cam_id, linked, event):
                        if linked["done"]:
                            return
                        caps = pad.get_current_caps()
                        if not caps:
                            caps = pad.query_caps(None)
                        if caps:
                            struct = caps.get_structure(0)
                            if struct and struct.get_name().startswith("video"):
                                sink = t.get_static_pad("sink")
                                if sink and not sink.is_linked():
                                    ret = pad.link(sink)
                                    if ret == Gst.PadLinkReturn.OK:
                                        linked["done"] = True
                                        event.set()  # Signal that pad is linked
                                        logger.info(f"[CAM-MANAGER] uridecodebin linked to tee for {cam_id}")
                                    else:
                                        logger.warning(f"[CAM-MANAGER] Failed to link uridecodebin to tee: {ret}")

                    source.connect("pad-added", on_uridecodebin_pad_added, tee, camera_id, linked_state, pad_linked_event)
                    logger.info(f"[CAM-MANAGER] Using uridecodebin for file source: {camera_id}")

                else:
                    # File/HTTP sources: use uridecodebin -> tee
                    source = Gst.ElementFactory.make("uridecodebin", f"uridecodebin_{camera_id}")
                    source.set_property("uri", uri)
                    bin_elem.add(source)

                    # Connect pad-added for uridecodebin -> tee (video pads only)
                    def on_uridecodebin_pad_added(_s, pad, t):
                        if linked_state["done"]:
                            return
                        caps = pad.get_current_caps()
                        if not caps:
                            caps = pad.query_caps(None)
                        if caps:
                            struct = caps.get_structure(0)
                            if struct and struct.get_name().startswith("video"):
                                sink = t.get_static_pad("sink")
                                if sink and not sink.is_linked():
                                    ret = pad.link(sink)
                                    if ret == Gst.PadLinkReturn.OK:
                                        linked_state["done"] = True
                                        logger.info(f"[CAM-MANAGER] uridecodebin linked to tee for {camera_id}")
                                    else:
                                        logger.warning(f"[CAM-MANAGER] Failed to link uridecodebin to tee: {ret}")

                    source.connect("pad-added", on_uridecodebin_pad_added, tee)
                    logger.info(f"[CAM-MANAGER] Using uridecodebin for file source: {camera_id}")

                # Create ghost pad from tee src for external access
                ghost_src = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
                bin_elem.add_pad(ghost_src)

                self.pipeline.add(bin_elem)

                # STEP 3: Link to branches FIRST (before state sync) - like reference implementation
                # Link while bin is in NULL state for safe topology change
                branch_pads = {}
                branch_drop_flags = {}  # Store drop flags for each branch (except first)

                for idx, b in enumerate(branches):
                    logger.info(f"[CAM-MANAGER] Linking branch {idx+1}/{len(branches)}: {b}")
                    pad = self._link_branch(bin_elem, tee, camera_id, source_id, b, sync=False)  # Don't sync yet
                    branch_pads[b] = pad

                    # For FIRST camera only: add controlled DROP probe on non-first branches
                    # This prevents race conditions during initial pipeline startup
                    # For additional cameras, pipeline is already stable - no DROP needed
                    if is_first_camera and len(branches) > 1 and idx > 0:
                        drop_flag = [True]

                        def make_probe_fn(flag):
                            def probe_fn(p, info):
                                if flag[0]:
                                    return Gst.PadProbeReturn.DROP
                                return Gst.PadProbeReturn.OK
                            return probe_fn

                        probe_id = pad.add_probe(Gst.PadProbeType.BUFFER, make_probe_fn(drop_flag))
                        branch_drop_flags[b] = (pad, probe_id, drop_flag)
                        logger.debug(f"[CAM-MANAGER] Added DROP probe to branch {b}")

                # STEP 4: Now sync camera bin state (with branches already linked)
                logger.info(f"[CAM-MANAGER] Syncing camera bin to pipeline state")

                if is_first_camera or prev_state == Gst.State.READY:
                    # First camera - sync to READY, then transition with pipeline
                    if not self._incremental_state_sync(bin_elem, Gst.State.READY):
                        logger.error(f"[CAM-MANAGER] Failed to sync camera bin to READY")
                elif prev_state == Gst.State.PLAYING:
                    # Additional camera - pipeline was paused in STEP 1
                    # Sync bin to PAUSED state (matching current pipeline state)
                    if not self._incremental_state_sync(bin_elem, Gst.State.PAUSED):
                        logger.warning(f"[CAM-MANAGER] Camera bin PAUSED sync warning")

                    # For file sources, wait for uridecodebin to link its pads BEFORE resuming pipeline
                    if is_file:
                        logger.info(f"[CAM-MANAGER] Waiting for uridecodebin pad linking...")
                        if pad_linked_event.wait(timeout=5.0):
                            logger.info(f"[CAM-MANAGER] uridecodebin pad linked successfully")
                        else:
                            logger.warning(f"[CAM-MANAGER] Timeout waiting for uridecodebin pad linking")
                    else:
                        time.sleep(1.0)  # Wait for decoder to initialize

                # STEP 5: Store camera info before state changes
                self._cameras[camera_id] = {
                    "bin": bin_elem, "tee": tee, "source_id": source_id,
                    "uri": uri, "branch_pads": branch_pads
                }

                # STEP 6: Transition pipeline and camera to PLAYING
                if is_first_camera or prev_state == Gst.State.READY:
                    # First camera - need to start entire pipeline
                    logger.info(f"[CAM-MANAGER] First camera - transitioning pipeline to PLAYING")

                    # Add blocking probe on tee to control first frame timing
                    first_frame_event = threading.Event()
                    frame_count = [0]

                    def tee_buffer_probe(pad, info):
                        frame_count[0] += 1
                        if frame_count[0] <= 3:  # Block first 3 frames for warmup
                            logger.debug(f"[CAM-MANAGER] Warmup frame {frame_count[0]}")
                            return Gst.PadProbeReturn.DROP
                        if frame_count[0] == 4:
                            first_frame_event.set()
                            logger.info(f"[CAM-MANAGER] First valid frame passed for {camera_id}")
                        return Gst.PadProbeReturn.PASS

                    tee_sink = tee.get_static_pad("sink")
                    probe_id = tee_sink.add_probe(Gst.PadProbeType.BUFFER, tee_buffer_probe)

                    # Set pipeline to PAUSED first (load all elements including branches)
                    logger.info(f"[CAM-MANAGER] Setting pipeline to PAUSED for safe preroll")
                    self.pipeline.set_state(Gst.State.PAUSED)
                    ret, _, _ = self.pipeline.get_state(10 * Gst.SECOND)  # Wait for PAUSED
                    if ret == Gst.StateChangeReturn.FAILURE:
                        logger.warning(f"[CAM-MANAGER] Pipeline PAUSED transition warning")

                    # Wait for nvurisrcbin to connect and negotiate caps
                    time.sleep(2.0)

                    # Now set to PLAYING
                    logger.info(f"[CAM-MANAGER] Setting pipeline to PLAYING")
                    self.pipeline.set_state(Gst.State.PLAYING)
                    ret, _, _ = self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
                    if ret == Gst.StateChangeReturn.FAILURE:
                        logger.warning(f"[CAM-MANAGER] Pipeline PLAYING transition returned FAILURE (may still work)")

                    # Wait for first valid frame to pass through
                    if first_frame_event.wait(timeout=5.0):
                        logger.info(f"[CAM-MANAGER] Pipeline stable with first frames")
                    else:
                        logger.warning(f"[CAM-MANAGER] Timeout waiting for first frame")

                    # Remove warmup probe after stable
                    tee_sink.remove_probe(probe_id)

                    # STEP 7: Release DROP probes on remaining branches SEQUENTIALLY
                    # Key fix: Let first branch's inference engine initialize before starting others
                    if branch_drop_flags:
                        # Wait for first branch to fully initialize (process several frames)
                        time.sleep(2.0)
                        logger.info(f"[CAM-MANAGER] Enabling remaining branches...")
                        for idx, (b, (pad, pid, drop_flag)) in enumerate(branch_drop_flags.items()):
                            logger.info(f"[CAM-MANAGER] Enabling branch: {b}")
                            # Toggle flag instead of removing probe (safer)
                            drop_flag[0] = False
                            # Wait between each branch to allow inference engine initialization
                            if idx < len(branch_drop_flags) - 1:
                                time.sleep(1.5)
                        logger.info(f"[CAM-MANAGER] All branches enabled")

                    ret, _, _ = self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
                    if ret == Gst.StateChangeReturn.FAILURE:
                        logger.warning(f"[CAM-MANAGER] Pipeline PLAYING transition returned FAILURE (may still work)")

                elif prev_state == Gst.State.PLAYING:
                    # Additional camera - pipeline was PAUSED in STEP 1
                    logger.info(f"[CAM-MANAGER] Additional camera - resuming pipeline")

                    # For file sources, pad linking already happened during PAUSED state sync
                    # Just resume pipeline - the camera bin will follow automatically as a child
                    if is_file:
                        # Sync camera bin to PLAYING BEFORE resuming pipeline
                        # This ensures all internal elements are ready
                        logger.info(f"[CAM-MANAGER] Syncing camera bin to PLAYING (file source)")
                        bin_elem.set_state(Gst.State.PLAYING)
                        ret, _, _ = bin_elem.get_state(2 * Gst.SECOND)
                        if ret == Gst.StateChangeReturn.FAILURE:
                            logger.warning(f"[CAM-MANAGER] Camera bin PLAYING sync warning")

                        # Now resume pipeline
                        logger.info(f"[CAM-MANAGER] Resuming pipeline to PLAYING")
                        self.pipeline.set_state(Gst.State.PLAYING)
                        ret, _, _ = self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
                        if ret == Gst.StateChangeReturn.FAILURE:
                            logger.warning(f"[CAM-MANAGER] Pipeline resume returned FAILURE (may still work)")
                    else:
                        # For RTSP sources, resume pipeline first then sync camera bin
                        logger.info(f"[CAM-MANAGER] Resuming pipeline to PLAYING")
                        self.pipeline.set_state(Gst.State.PLAYING)
                        ret, _, _ = self.pipeline.get_state(STATE_CHANGE_TIMEOUT)
                        if ret == Gst.StateChangeReturn.FAILURE:
                            logger.warning(f"[CAM-MANAGER] Pipeline resume returned FAILURE (may still work)")

                        # Sync camera bin to PLAYING
                        logger.info(f"[CAM-MANAGER] Syncing camera bin to PLAYING")
                        if not self._incremental_state_sync(bin_elem, Gst.State.PLAYING):
                            logger.warning(f"[CAM-MANAGER] Camera bin PLAYING sync warning")

                    # Wait for camera to connect and stabilize
                    time.sleep(3.0)
                    logger.info(f"[CAM-MANAGER] Additional camera stabilized")

                    # Release DROP probes on non-first branches for additional cameras
                    if branch_drop_flags:
                        logger.info(f"[CAM-MANAGER] Enabling remaining branches for additional camera...")
                        for idx, (b, (pad, pid, drop_flag)) in enumerate(branch_drop_flags.items()):
                            logger.info(f"[CAM-MANAGER] Enabling branch: {b}")
                            drop_flag[0] = False
                            if idx < len(branch_drop_flags) - 1:
                                time.sleep(0.5)
                        logger.info(f"[CAM-MANAGER] All branches enabled for additional camera")

                self._last_op = time.time()
                logger.info(f"[CAM-MANAGER] Successfully added {camera_id} to branches: {branches}")
                return True

            except Exception as e:
                logger.error(f"add_camera failed: {e}", exc_info=True)
                # Attempt cleanup
                try:
                    if camera_id in self._cameras:
                        del self._cameras[camera_id]
                    self.pipeline.remove(bin_elem)
                except:
                    pass
                self._mapper.remove(camera_id)
                # Restore pipeline state if needed
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
        """Add camera to additional branch with safe pipeline state management."""
        with self._lock:
            self._delay()
            cam = self._cameras.get(camera_id)
            if not cam or branch_name in cam["branch_pads"] or branch_name not in self.branches:
                return False

            # Get current pipeline state
            _, prev_state, _ = self.pipeline.get_state(0)
            logger.info(f"[CAM-MANAGER] Adding {camera_id} to branch {branch_name}")

            try:
                # PAUSE pipeline for safe topology change
                if prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PAUSED)
                    self.pipeline.get_state(STATE_CHANGE_TIMEOUT)

                # Link branch with element state sync
                pad = self._link_branch(cam["bin"], cam["tee"], camera_id, cam["source_id"], branch_name, True)
                cam["branch_pads"][branch_name] = pad

                # Resume pipeline
                if prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PLAYING)
                    self.pipeline.get_state(STATE_CHANGE_TIMEOUT)

                self._last_op = time.time()
                logger.info(f"[CAM-MANAGER] Successfully added {camera_id} to branch {branch_name}")
                return True
            except Exception as e:
                logger.error(f"add_camera_to_branch failed: {e}", exc_info=True)
                if prev_state == Gst.State.PLAYING:
                    self.pipeline.set_state(Gst.State.PLAYING)
                return False

    def remove_camera_from_branch(self, camera_id: str, branch_name: str) -> bool:
        """Remove camera from branch."""
        with self._lock:
            self._delay()
            cam = self._cameras.get(camera_id)
            if not cam:
                logger.warning(f"[CAM-MANAGER] remove_camera_from_branch: camera {camera_id} not found. Existing: {list(self._cameras.keys())}")
                return False
            if branch_name not in cam["branch_pads"]:
                logger.warning(f"[CAM-MANAGER] remove_camera_from_branch: camera {camera_id} not in branch {branch_name}. Branches: {list(cam['branch_pads'].keys())}")
                return False
            if len(cam["branch_pads"]) <= 1:
                logger.info(f"[CAM-MANAGER] remove_camera_from_branch: camera {camera_id} has only {len(cam['branch_pads'])} branch(s), use remove_camera instead")
                return False

            logger.info(f"[CAM-MANAGER] Starting to remove {camera_id} from branch {branch_name}")
            try:
                tee_pad = cam["branch_pads"].get(branch_name)

                # Step 1: Block the tee src pad to stop data flow
                blocked = threading.Event()
                probe_id = None

                def block_probe(pad, info):
                    blocked.set()
                    return Gst.PadProbeReturn.OK  # Keep blocking until we remove the probe

                if tee_pad:
                    logger.debug(f"[CAM-MANAGER] Adding block probe to tee pad for {camera_id}/{branch_name}")
                    probe_id = tee_pad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, block_probe)
                    # Wait for probe to trigger (data flow blocked)
                    blocked.wait(timeout=1.0)

                logger.debug(f"[CAM-MANAGER] Data flow blocked, proceeding with unlink")

                # Step 2: Unlink the branch elements (safe now that data is blocked)
                self._unlink_branch_safe(cam, branch_name, camera_id, tee_pad, probe_id)

                self._last_op = time.time()
                logger.info(f"[CAM-MANAGER] Successfully removed {camera_id} from branch {branch_name}")
                return True
            except Exception as e:
                logger.error(f"[CAM-MANAGER] remove_camera_from_branch failed: {e}", exc_info=True)
                return False

    def _link_branch(self, bin_elem, tee, camera_id, source_id, branch_name, sync=False) -> Gst.Pad:
        """Link: tee -> queue -> mux.

        When sync=True, elements are synced to parent state after creation.
        """
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

        # Sync element states if requested (for dynamic branch addition)
        if sync:
            _, parent_state, _ = bin_elem.get_state(0)
            self._incremental_state_sync(q, parent_state)

        return tee_src

    def _unlink_branch_safe(self, cam: dict, branch_name: str, camera_id: str,
                            tee_pad: Optional[Gst.Pad] = None, probe_id: Optional[int] = None) -> None:
        """Safely unlink and cleanup branch elements with proper synchronization.

        This method should be called when data flow is already blocked.
        """
        b = self.branches.get(branch_name)
        if not b:
            logger.warning(f"_unlink_branch_safe: branch {branch_name} not found")
            return

        # Get elements to remove
        elements = []
        elem = cam["bin"].get_by_name(f"q_{camera_id}_{branch_name}")
        if elem:
            elements.append(elem)

        # Step 1: Unlink from nvstreammux first (while data is blocked)
        ghost_pad = None
        mux_pad = None
        it = cam["bin"].iterate_pads()
        while True:
            ret, pad = it.next()
            # Look for ghost pad with this specific camera_id pattern
            if ret == Gst.IteratorResult.OK and pad.get_name() == f"g_{branch_name}_{camera_id}":
                ghost_pad = pad
                mux_pad = pad.get_peer()
                if mux_pad:
                    pad.unlink(mux_pad)
                break
            elif ret == Gst.IteratorResult.RESYNC:
                it.resync()
            elif ret != Gst.IteratorResult.OK:
                break

        # Step 2: Unlink tee from queue
        if tee_pad and elements:
            q_elem = cam["bin"].get_by_name(f"q_{camera_id}_{branch_name}")
            if q_elem:
                q_sink = q_elem.get_static_pad("sink")
                if q_sink and q_sink.is_linked():
                    peer = q_sink.get_peer()
                    if peer:
                        peer.unlink(q_sink)

        # Step 3: Remove probe (allow any pending data to flush)
        if tee_pad and probe_id is not None:
            tee_pad.remove_probe(probe_id)

        # Small delay to let any in-flight data clear
        time.sleep(0.1)

        # Step 4: Set elements to NULL state and remove them
        for elem in elements:
            elem.set_state(Gst.State.NULL)
            elem.get_state(Gst.CLOCK_TIME_NONE)

        time.sleep(0.05)

        for elem in elements:
            cam["bin"].remove(elem)

        # Step 5: Remove ghost pad from bin
        if ghost_pad:
            cam["bin"].remove_pad(ghost_pad)

        # Step 6: Release the nvstreammux sink pad
        if mux_pad:
            try:
                b.nvstreammux.release_request_pad(mux_pad)
            except Exception as e:
                logger.warning(f"release nvstreammux pad failed: {e}")

        # Step 7: Release the tee src pad
        if tee_pad:
            try:
                cam["tee"].release_request_pad(tee_pad)
            except Exception as e:
                logger.warning(f"release tee pad failed: {e}")

        cam["branch_pads"].pop(branch_name, None)
        logger.info(f"Safely unlinked {camera_id} from branch {branch_name}")

    def _unlink_branch(self, cam: dict, branch_name: str, camera_id: str) -> None:
        """Unlink and cleanup branch elements (legacy - for remove_camera)."""
        b = self.branches.get(branch_name)
        if not b:
            logger.warning(f"_unlink_branch: branch {branch_name} not found")
            return

        tee_pad = cam["branch_pads"].get(branch_name)

        elem = cam["bin"].get_by_name(f"q_{camera_id}_{branch_name}")
        if elem:
            elem.set_state(Gst.State.NULL)

        time.sleep(0.15)

        elem = cam["bin"].get_by_name(f"q_{camera_id}_{branch_name}")
        if elem:
            cam["bin"].remove(elem)

        it = cam["bin"].iterate_pads()
        while True:
            ret, pad = it.next()
            if ret == Gst.IteratorResult.OK and pad.get_name() == f"g_{branch_name}_{camera_id}":
                peer = pad.get_peer()
                if peer:
                    pad.unlink(peer)
                    b.nvstreammux.release_request_pad(peer)
                cam["bin"].remove_pad(pad)
                break
            elif ret == Gst.IteratorResult.RESYNC:
                it.resync()
            elif ret != Gst.IteratorResult.OK:
                break

        if tee_pad:
            try:
                cam["tee"].release_request_pad(tee_pad)
            except Exception as e:
                logger.warning(f"release_request_pad failed: {e}")

        cam["branch_pads"].pop(branch_name, None)
        logger.info(f"Unlinked {camera_id} from branch {branch_name}")

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
        """Get SourceIDMapper for probe lookups."""
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
