"""
Common utilities for DeepStream pipeline.

This module consolidates:
- Config loading with environment variable expansion
- DeepStream metadata extractors
- FPS monitoring and interval utilities
"""

import ctypes
import os
import re
import time
from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import yaml

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib


# =============================================================================
# Config Loader
# =============================================================================

def load_config(path: str) -> dict:
    """Load YAML with ${VAR:default} expansion"""
    with open(path) as f:
        text = f.read()

    # Expand ${VAR:default} before parsing
    def replace(m):
        return os.environ.get(m.group(1), m.group(2) or "")

    expanded = re.sub(r'\$\{(\w+):?([^}]*)\}', replace, text)
    return yaml.safe_load(expanded)


# =============================================================================
# DeepStream Metadata Extractors
# =============================================================================

def _get_pyds():
    """Lazy import pyds module"""
    import pyds
    return pyds


class BatchIterator:
    """
    Simple iterator for DeepStream batch metadata.
    
    Examples:
        # Iterate all objects
        for frame, obj in BatchIterator(batch):
            source_id = frame.source_id
            object_id = obj.object_id
            
        # Iterate frames only
        for frame in BatchIterator(batch).frames():
            print(f"Frame {frame.frame_num} from source {frame.source_id}")
            
        # Nested iteration
        it = BatchIterator(batch)
        for frame in it.frames():
            for obj in it.objects(frame):
                process(obj)
    """
    
    def __init__(self, batch):
        self.batch = batch
    
    def __iter__(self) -> Iterator[Tuple]:
        """Iterate all (frame_meta, obj_meta) pairs"""
        for frame in self.frames():
            for obj in self.objects(frame):
                yield frame, obj
    
    def frames(self) -> Iterator:
        """Iterate frame metas only"""
        pyds = _get_pyds()
        l_frame = self.batch.frame_meta_list
        while l_frame:
            try:
                yield pyds.NvDsFrameMeta.cast(l_frame.data)
                l_frame = l_frame.next
            except StopIteration:
                break
    
    def objects(self, frame_meta) -> Iterator:
        """Iterate object metas in a frame"""
        pyds = _get_pyds()
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                yield pyds.NvDsObjectMeta.cast(l_obj.data)
                l_obj = l_obj.next
            except StopIteration:
                break


def extract_embedding(obj_meta, dim: int = 512) -> Optional[np.ndarray]:
    """
    Extract L2-normalized embedding from SGIE tensor output.
    
    Args:
        obj_meta: NvDsObjectMeta with tensor output
        dim: Embedding dimension (default: 512)
        
    Returns:
        Normalized numpy array or None if no tensor found
    """
    pyds = _get_pyds()
    l_user = obj_meta.obj_user_meta_list
    while l_user:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                tensor = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                layer = pyds.get_nvds_LayerInfo(tensor, 0)
                if layer:
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    emb = np.ctypeslib.as_array(ptr, shape=(dim,)).copy()
                    norm = np.linalg.norm(emb)
                    return emb / norm if norm > 0 else emb
            l_user = l_user.next
        except StopIteration:
            break
    return None


def get_batch_meta(buffer):
    """
    Get NvDsBatchMeta from GStreamer buffer.
    
    Args:
        buffer: GstBuffer from probe info
        
    Returns:
        NvDsBatchMeta or None
    """
    if not buffer:
        return None
    pyds = _get_pyds()
    return pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))


# =============================================================================
# FPS Monitor
# =============================================================================

class FPSMonitor:
    """
    Reusable FPS monitor with stats callback support.
    
    Usage:
        fps = FPSMonitor(name="Detection", log_interval=1.0, stats_interval=10.0)
        fps.on_frame()
        if fps.should_log():
            fps.log()
        if fps.should_stats():
            fps.log_stats()
    """
    
    def __init__(
        self,
        name: str,
        log_interval: float = 1.0,
        stats_interval: float = 10.0,
        stats_callback: Optional[Callable[[], dict]] = None,
    ):
        self._name = name
        self._log_interval = log_interval
        self._stats_interval = stats_interval
        self._stats_callback = stats_callback
        
        self._fps_count = 0
        self._fps_start = time.time()
        self._stats_last = time.time()
    
    def on_frame(self) -> None:
        """Call this once per frame to count"""
        self._fps_count += 1
    
    def should_log(self) -> bool:
        """Check if it's time to log FPS"""
        return time.time() - self._fps_start >= self._log_interval
    
    def should_stats(self) -> bool:
        """Check if it's time to log stats"""
        return time.time() - self._stats_last >= self._stats_interval
    
    def log(self) -> float:
        """Log FPS and reset counter. Returns calculated FPS."""
        elapsed = time.time() - self._fps_start
        fps = self._fps_count / elapsed if elapsed > 0 else 0
        print(f"[{self._name} FPS] {fps:.1f}")
        self._fps_start = time.time()
        self._fps_count = 0
        return fps
    
    def log_stats(self) -> Optional[dict]:
        """Log stats using callback. Returns stats dict or None."""
        if self._stats_callback:
            stats = self._stats_callback()
            print(f"[{self._name} STATS] " + ", ".join(f"{k}={v}" for k, v in stats.items()))
            self._stats_last = time.time()
            return stats
        return None
    
    def reset(self) -> None:
        """Reset all counters"""
        self._fps_count = 0
        self._fps_start = time.time()
        self._stats_last = time.time()
    
    @property
    def current_fps(self) -> float:
        """Get current instantaneous FPS"""
        elapsed = time.time() - self._fps_start
        return self._fps_count / elapsed if elapsed > 0 else 0


def fps_probe_factory(
    name: str,
    log_interval: float = 1.0,
    stats_interval: float = 10.0,
    stats_callback: Optional[Callable[[], dict]] = None,
) -> Callable:
    """
    Create a standard FPS probe callback.
    
    Args:
        name: Processor name for logging
        log_interval: Seconds between FPS logs
        stats_interval: Seconds between stats logs
        stats_callback: Callback to get stats dict
    
    Returns:
        Probe callback function
    """
    monitor = FPSMonitor(name, log_interval, stats_interval, stats_callback)
    call_count = [0]
    
    def fps_probe(pad, info, user_data) -> Gst.PadProbeReturn:
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK
        
        batch = get_batch_meta(buffer)
        if not batch:
            return Gst.PadProbeReturn.OK
        
        call_count[0] += 1
        if call_count[0] <= 3:
            print(f"[DEBUG] {name} probe called, batch={batch is not None}, count={call_count[0]}")
        
        monitor.on_frame()
        
        if monitor.should_log():
            monitor.log()
        
        if monitor.should_stats():
            monitor.log_stats()
        
        return Gst.PadProbeReturn.OK
    
    print(f"[fps_probe_factory] Created probe for '{name}'")
    return fps_probe


class IntervalRunner:
    """
    Run callback at fixed interval using GLib timeout.
    
    Usage:
        runner = IntervalRunner(interval_ms=10000, callback=cleanup_func)
        runner.start()
        runner.stop()
    """
    
    def __init__(self, interval_ms: int, callback: Callable[[int], None]):
        self._interval_ms = interval_ms
        self._callback = callback
        self._source_id: Optional[int] = None
        self._frame_count = 0
    
    def start(self) -> None:
        """Start the interval timer"""
        if self._source_id is None:
            self._source_id = GLib.timeout_add(self._interval_ms, self._run)
    
    def stop(self) -> None:
        """Stop the interval timer"""
        if self._source_id is not None:
            GLib.source_remove(self._source_id)
            self._source_id = None
    
    def _run(self) -> bool:
        """Called by GLib timer"""
        self._frame_count += 1
        self._callback(self._frame_count)
        return True
    
    @property
    def is_running(self) -> bool:
        return self._source_id is not None
