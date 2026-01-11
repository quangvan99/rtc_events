"""
Detection Processor

BranchProcessor implementation for object detection pipeline.
Provides basic detection with FPS monitoring and statistics.

This processor is auto-registered with ProcessorRegistry using the @register decorator.
"""

from typing import Dict, Any, Callable

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.interfaces import BranchProcessor
from src.registry import ProcessorRegistry
from src.sinks.base_sink import BaseSink
from src.utils.extractors import BatchIterator, get_batch_meta
from src.utils.metric import fps_probe_factory


COLOR_TEXT = (1.0, 1.0, 1.0, 1.0)
COLOR_TEXT_BG = (0.0, 0.0, 0.0, 0.7)

FONT_SIZE = 12
FONT_NAME = "Serif"


def update_display(obj_meta, detection_count: int, score: float = 0.0) -> None:
    """Update OSD display for detected objects"""
    rect = obj_meta.rect_params
    obj_w, obj_h = int(rect.width), int(rect.height)

    text = obj_meta.text_params
    display_text = f"Objects: {detection_count} [{obj_w}x{obj_h}]"
    text.display_text = display_text
    text.x_offset = int(rect.left)
    text.y_offset = max(0, int(rect.top) - 25)
    text.font_params.font_name = FONT_NAME
    text.font_params.font_size = FONT_SIZE

    r, g, b, a = COLOR_TEXT
    text.font_params.font_color.red, text.font_params.font_color.green = r, g
    text.font_params.font_color.blue, text.font_params.font_color.alpha = b, a

    text.set_bg_clr = 1
    r, g, b, a = COLOR_TEXT_BG
    text.text_bg_clr.red, text.text_bg_clr.green = r, g
    text.text_bg_clr.blue, text.text_bg_clr.alpha = b, a


@ProcessorRegistry.register("detection")
class DetectionProcessor(BranchProcessor):
    """
    Object detection branch processor implementation.
    
    Handles:
    - FPS monitoring (via fps_probe_factory)
    - Detection statistics
    - Object counting per frame
    
    Config params (from branch YAML):
        params:
            log_interval: seconds between FPS logs (default: 1.0)
            stats_interval: seconds between stats logs (default: 10.0)
    """
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._sink = None
        self._total_frames = 0
        self._total_objects = 0
    
    @property
    def name(self) -> str:
        return "detection"
    
    def setup(self, config: Dict[str, Any], sink: BaseSink) -> None:
        self._config = config
        self._sink = sink
        params = config.get("params", {})
        print(f"[DetectionProcessor] Setup complete (log_interval={params.get('log_interval', 1.0)}s)")
    
    def _get_stats(self) -> dict:
        """Return stats dict for FPSMonitor"""
        return {
            "frames": self._total_frames,
            "objects": self._total_objects,
            "avg_obj/frame": round(self._total_objects / max(self._total_frames, 1), 1),
        }
    
    def get_probes(self) -> Dict[str, Callable]:
        params = self._config.get("params", {})
        return {
            "fps_probe": fps_probe_factory(
                name="Detection",
                log_interval=params.get("log_interval", 1.0),
                stats_interval=params.get("stats_interval", 10.0),
                stats_callback=self._get_stats,
            ),
            "osd_probe": self._osd_probe,
        }
    
    def _osd_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        """OSD display probe for detection info"""
        batch = get_batch_meta(info.get_buffer())
        if not batch:
            return Gst.PadProbeReturn.OK
        
        for frame, obj in BatchIterator(batch):
            self._total_frames += 1
            self._total_objects += 1
            update_display(obj, 1, 0.0)
        
        return Gst.PadProbeReturn.OK
    
    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any) -> None:
        print(f"[DetectionProcessor] Pipeline built, branch: {branch_info.name}")

    def on_start(self) -> None:
        print("[DetectionProcessor] Started")

    def on_stop(self) -> None:
        print(f"[DetectionProcessor] Stopped - frames={self._total_frames}, objects={self._total_objects}")
