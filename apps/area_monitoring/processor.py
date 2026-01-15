from typing import Any, Callable, Dict, List, Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import pyds

from src.processor_registry import ProcessorRegistry
from src.sinks.base_sink import BaseSink
from src.common import get_batch_meta, fps_probe_factory, BatchIterator

# -----------------------------------------------------------------------------
# Main Processor
# -----------------------------------------------------------------------------

@ProcessorRegistry.register("area_monitoring")
class AreaMonitoringProcessor:
    def __init__(self, source_mapper=None):
        self._config: Dict[str, Any] = {}
        self._sink: Optional[BaseSink] = None

    @property
    def name(self) -> str:
        return "area_monitoring"

    def setup(self, config: Dict[str, Any], sink: BaseSink) -> None:
        """Initialize area monitoring components"""
        self._config = config
        self._sink = sink

    def get_probes(self) -> Dict[str, Callable]:
        """Return probe callbacks"""
        params = self._config.get("params", {})
        return {
            "area_fps_probe": fps_probe_factory(
                name="AreaMonitoring",
                log_interval=params.get("log_interval", 1.0),
                stats_interval=params.get("stats_interval", 10.0),
            ),
        }
    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any) -> None:
        """Called when pipeline is built"""
        print(f"[AreaMonitoringProcessor] Pipeline built, branch: {branch_info.name}")

    def on_start(self) -> None:
        """Called when pipeline starts"""
        print("[AreaMonitoringProcessor] Started")

    def on_stop(self) -> None:
        """Called when pipeline stops"""
        pass