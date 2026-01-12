"""Fakesink Adapter - No-output sink for testing/benchmarking"""

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.sinks.base_sink import BaseSink


class FakesinkAdapter(BaseSink):
    """Fakesink adapter - discards all data with proper format handling"""

    _counter = 0

    def __init__(self, sync: bool = False):
        self.sync = sync
        self.element = None
        self.elements = []

    def create(self, pipeline: Gst.Pipeline) -> Gst.Element:
        FakesinkAdapter._counter += 1
        p = f"fake{FakesinkAdapter._counter}"

        chain = [
            self._make("queue", f"{p}_q", {"max-size-buffers": 30, "leaky": 2}),
            self._make("nvvideoconvert", f"{p}_nv", {"compute-hw": 1, "nvbuf-memory-type": 3}),
            # self._make("capsfilter", f"{p}_caps", {"caps": Gst.Caps.from_string("video/x-raw,format=RGBA")}),
            # self._make("videoconvert", f"{p}_vc"),
            # self._make("queue", f"{p}_q2", {"max-size-buffers": 30, "leaky": 2}),
            self._make("fakesink", f"{p}_sink", {"sync": False, "async": False, "qos": False}),
        ]

        for elem in chain:
            pipeline.add(elem)
        for i in range(len(chain) - 1):
            if not chain[i].link(chain[i + 1]):
                raise RuntimeError(f"Failed to link {chain[i].get_name()} -> {chain[i+1].get_name()}")

        self.elements = chain
        return chain[0]

    def _make(self, factory: str, name: str, props: dict = None) -> Gst.Element:
        elem = Gst.ElementFactory.make(factory, name)
        if not elem:
            raise RuntimeError(f"Cannot create element: {factory}")
        for k, v in (props or {}).items():
            elem.set_property(k.replace("-", "_"), v)
        return elem

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
