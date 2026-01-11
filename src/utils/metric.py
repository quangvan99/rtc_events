"""
FPS Monitor Utilities

Shared FPS calculation and statistics for pipeline processors.
Provides reusable FPS monitoring with configurable intervals.
"""

import time
from typing import Callable, Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib


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
    
    from src.utils.extractors import get_batch_meta
    
    def fps_probe(pad, info, user_data) -> Gst.PadProbeReturn:
        batch = get_batch_meta(info.get_buffer())
        if not batch:
            return Gst.PadProbeReturn.OK
        
        monitor.on_frame()
        
        if monitor.should_log():
            monitor.log()
        
        if monitor.should_stats():
            monitor.log_stats()
        
        return Gst.PadProbeReturn.OK
    
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
