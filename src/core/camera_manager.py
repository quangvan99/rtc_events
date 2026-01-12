"""Dynamic camera management with tee fanout to multiple branches."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.core.pipeline_builder import BranchInfo
from src.core.source_mapper import SourceIDMapper

logger = logging.getLogger(__name__)


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

    def add_camera(self, camera_id: str, uri: str, branch_names: list[str]) -> bool:
        """Add camera to branches."""
        with self._lock:
            self._delay()
            if camera_id in self._cameras:
                return False

            branches = [b for b in branch_names if b in self.branches]
            if not branches:
                return False

            try:
                source_id = self._mapper.add(camera_id, uri)
                bin_elem = Gst.Bin.new(f"cam_{camera_id}")

                src = Gst.ElementFactory.make("nvurisrcbin", f"src_{camera_id}")
                src.set_property("uri", uri)
                src.set_property("gpu-id", self._gpu_id)
                src.set_property("cudadec-memtype", 0)
                src.set_property("num-extra-surfaces", 4)

                tee = Gst.ElementFactory.make("tee", f"tee_{camera_id}")
                tee.set_property("allow-not-linked", True)

                bin_elem.add(src)
                bin_elem.add(tee)
                self.pipeline.add(bin_elem)

                branch_pads = {}
                for b in branches:
                    branch_pads[b] = self._link_branch(bin_elem, tee, camera_id, source_id, b)

                def on_pad_added(_s, pad, data):
                    t, cid = data
                    if cid in self._cameras and pad.get_name().startswith("vsrc"):
                        sink = t.get_static_pad("sink")
                        if sink and not sink.is_linked():
                            pad.link(sink)

                src.connect("pad-added", on_pad_added, (tee, camera_id))
                bin_elem.sync_state_with_parent()

                self._cameras[camera_id] = {
                    "bin": bin_elem, "tee": tee, "source_id": source_id,
                    "uri": uri, "branch_pads": branch_pads
                }
                self._last_op = time.time()
                return True

            except Exception as e:
                logger.error(f"add_camera failed: {e}")
                self._mapper.remove(camera_id)
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

                for b in list(cam["branch_pads"].keys()):
                    self._unlink_branch(cam, b, camera_id)

                cam["bin"].set_state(Gst.State.NULL)
                cam["bin"].get_state(Gst.CLOCK_TIME_NONE)
                self.pipeline.remove(cam["bin"])
                self._mapper.remove(camera_id)
                del self._cameras[camera_id]
                self._last_op = time.time()
                return True

            except Exception as e:
                logger.error(f"remove_camera failed: {e}")
                return False

    def add_camera_to_branch(self, camera_id: str, branch_name: str) -> bool:
        """Add camera to additional branch."""
        with self._lock:
            self._delay()
            cam = self._cameras.get(camera_id)
            if not cam or branch_name in cam["branch_pads"] or branch_name not in self.branches:
                return False

            try:
                pad = self._link_branch(cam["bin"], cam["tee"], camera_id, cam["source_id"], branch_name, True)
                cam["branch_pads"][branch_name] = pad
                self._last_op = time.time()
                return True
            except Exception as e:
                logger.error(f"add_camera_to_branch failed: {e}")
                return False

    def remove_camera_from_branch(self, camera_id: str, branch_name: str) -> bool:
        """Remove camera from branch."""
        with self._lock:
            self._delay()
            cam = self._cameras.get(camera_id)
            if not cam or branch_name not in cam["branch_pads"] or len(cam["branch_pads"]) == 1:
                return False

            try:
                cam["branch_pads"][branch_name].add_probe(
                    Gst.PadProbeType.BLOCK_DOWNSTREAM, lambda *_: Gst.PadProbeReturn.REMOVE
                )
                time.sleep(0.2)
                self._unlink_branch(cam, branch_name, camera_id)
                self._last_op = time.time()
                return True
            except Exception as e:
                logger.error(f"remove_camera_from_branch failed: {e}")
                return False

    def _link_branch(self, bin_elem, tee, camera_id, source_id, branch_name, sync=False) -> Gst.Pad:
        """Link: tee -> nvconv -> caps -> queue -> mux."""
        b = self.branches[branch_name]

        nv = Gst.ElementFactory.make("nvvideoconvert", f"nv_{camera_id}_{branch_name}")
        nv.set_property("nvbuf-memory-type", 0)
        bin_elem.add(nv)

        caps = Gst.ElementFactory.make("capsfilter", f"caps_{camera_id}_{branch_name}")
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))
        bin_elem.add(caps)

        q = Gst.ElementFactory.make("queue", f"q_{camera_id}_{branch_name}")
        q.set_property("max-size-buffers", 30)
        q.set_property("max-size-bytes", 0)
        q.set_property("max-size-time", 0)
        q.set_property("leaky", 2)
        bin_elem.add(q)

        tee_src = tee.request_pad_simple("src_%u")
        tee_src.link(nv.get_static_pad("sink"))
        nv.link(caps)
        caps.link(q)

        mux_sink = b.nvstreammux.request_pad_simple(f"sink_{source_id}")
        self._pad_counter += 1
        ghost = Gst.GhostPad.new(f"g_{branch_name}_{self._pad_counter}", q.get_static_pad("src"))
        bin_elem.add_pad(ghost)
        ghost.link(mux_sink)

        if sync:
            nv.sync_state_with_parent()
            caps.sync_state_with_parent()
            q.sync_state_with_parent()

        return tee_src

    def _unlink_branch(self, cam: dict, branch_name: str, camera_id: str) -> None:
        """Unlink and cleanup branch elements."""
        b = self.branches.get(branch_name)
        if not b:
            return

        # Remove ghost pad
        it = cam["bin"].iterate_pads()
        while True:
            ret, pad = it.next()
            if ret == Gst.IteratorResult.OK and pad.get_name().startswith(f"g_{branch_name}"):
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

        # Remove elements
        for prefix in ["q", "caps", "nv"]:
            elem = cam["bin"].get_by_name(f"{prefix}_{camera_id}_{branch_name}")
            if elem:
                elem.set_state(Gst.State.NULL)
                cam["bin"].remove(elem)

        cam["tee"].release_request_pad(cam["branch_pads"][branch_name])
        del cam["branch_pads"][branch_name]

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
