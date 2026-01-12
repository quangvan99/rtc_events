"""Fakesink Adapter - No-output sink for testing/benchmarking"""

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.sinks.base_sink import BaseSink


class FakesinkAdapter(BaseSink):
    """Fakesink adapter - discards all data"""

    _counter = 0

    def __init__(self, sync: bool = False):
        self.sync = sync
        self.element = None

    def create(self, pipeline: Gst.Pipeline) -> Gst.Element:
        FakesinkAdapter._counter += 1
        self.element = Gst.ElementFactory.make("fakesink", f"fakesink_{FakesinkAdapter._counter}")
        self.element.set_property("sync", self.sync)
        pipeline.add(self.element)
        return self.element

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
