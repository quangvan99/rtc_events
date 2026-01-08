#!/usr/bin/env python3
"""
Fakesink Adapter - Implements BaseSink for no-output (testing/benchmarking)
"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from sinks.base_sink import BaseSink


class FakesinkAdapter(BaseSink):
    """Fakesink adapter for pipeline testing without output"""

    _counter = 0  # Class-level counter for unique names

    def __init__(self, sync: bool = False, name: str = None):
        """
        Initialize fakesink adapter.

        Args:
            sync: If True, sink synchronizes to clock (default: False for max speed)
            name: Optional custom name for the sink element
        """
        self.sync = sync
        self.name = name
        self.element: Gst.Element | None = None

    def create(self, pipeline: Gst.Pipeline) -> Gst.Element:
        """
        Create fakesink element, add to pipeline.

        Returns: fakesink element for linking
        """
        # Generate unique name if not provided
        if self.name:
            sink_name = self.name
        else:
            FakesinkAdapter._counter += 1
            sink_name = f"fakesink_{FakesinkAdapter._counter}"

        self.element = Gst.ElementFactory.make("fakesink", sink_name)
        if not self.element:
            raise RuntimeError("Cannot create fakesink element")

        self.element.set_property("sync", self.sync)
        pipeline.add(self.element)

        print(f"[Fakesink] Created '{sink_name}' (sync={self.sync})")
        return self.element

    def start(self) -> None:
        """Called before pipeline starts - no-op for fakesink."""
        pass

    def stop(self) -> None:
        """Called when pipeline stops - no-op for fakesink."""
        pass
