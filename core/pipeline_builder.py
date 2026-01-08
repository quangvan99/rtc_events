"""Config-driven DeepStream Pipeline Builder"""

import asyncio
import configparser
import os
import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

# DeepStream Python bindings
sys.path.append("/opt/nvidia/deepstream/deepstream/lib")

from sinks.base_sink import BaseSink
from core.probe_registry import ProbeRegistry


class PipelineBuilder:
    """Config-driven DeepStream pipeline construction"""

    def __init__(self, config: dict, sink: BaseSink):
        """
        Initialize pipeline builder

        Args:
            config: Pipeline configuration dict (from load_config)
            sink: Sink adapter for pipeline output
        """
        self.config = config
        self.sink = sink
        self.probe_registry = ProbeRegistry()

        self.pipeline: Gst.Pipeline | None = None
        self.elements: list[Gst.Element] = []
        self.named_elements: dict[str, Gst.Element] = {}
        self._counter = 0

    def _create_element(self, elem_cfg: dict) -> Gst.Element:
        """
        Create GStreamer element from config

        Args:
            elem_cfg: Element configuration dict with 'type', optional 'name', 'properties', etc.

        Returns:
            Created GStreamer element

        Raises:
            RuntimeError: If element creation fails
        """
        elem_type = elem_cfg["type"]

        # Auto-generate name if not provided
        if "name" in elem_cfg:
            name = elem_cfg["name"]
        else:
            self._counter += 1
            name = f"{elem_type}_{self._counter}"

        elem = Gst.ElementFactory.make(elem_type, name)
        if not elem:
            raise RuntimeError(f"Cannot create element: {elem_type}")

        # Set properties
        for key, value in elem_cfg.get("properties", {}).items():
            elem.set_property(key, value)

        # Handle caps
        if "caps" in elem_cfg:
            elem.set_property("caps", Gst.Caps.from_string(elem_cfg["caps"]))

        # Handle config_file (nvinfer, nvtracker)
        if "config_file" in elem_cfg:
            path = elem_cfg["config_file"]
            if elem_type == "nvinfer":
                elem.set_property("config-file-path", path)
            elif elem_type == "nvtracker":
                self._configure_tracker(elem, path)

        self.pipeline.add(elem)
        self.elements.append(elem)

        if "name" in elem_cfg:
            self.named_elements[name] = elem

        return elem

    def _configure_tracker(self, tracker: Gst.Element, cfg_path: str) -> None:
        """
        Configure nvtracker from config file

        Args:
            tracker: nvtracker element
            cfg_path: Path to tracker config file
        """
        if not os.path.exists(cfg_path):
            print(f"[PipelineBuilder] Warning: Tracker config not found: {cfg_path}")
            return

        cfg = configparser.ConfigParser()
        cfg.read(cfg_path)

        if "tracker" not in cfg:
            return

        props_map = {
            "tracker-width": ("tracker-width", int),
            "tracker-height": ("tracker-height", int),
            "gpu-id": ("gpu-id", int),
            "ll-lib-file": ("ll-lib-file", str),
            "ll-config-file": ("ll-config-file", str),
        }

        for key, (prop, typ) in props_map.items():
            if key in cfg["tracker"]:
                val = cfg["tracker"][key]
                tracker.set_property(prop, typ(val))

        print(f"[PipelineBuilder] Tracker configured from: {cfg_path}")

    def register_probe(self, name: str, callback) -> None:
        """
        Register a probe callback

        Args:
            name: Probe name
            callback: Probe callback function
        """
        self.probe_registry.register(name, callback)

    def build(self) -> Gst.Pipeline:
        """
        Build pipeline from configuration

        Returns:
            Constructed GStreamer pipeline

        Raises:
            RuntimeError: If pipeline construction fails
        """
        Gst.init(None)
        cfg = self.config["pipeline"]

        self.pipeline = Gst.Pipeline.new(cfg.get("name", "pipeline"))

        # Create source
        src_cfg = cfg["source"]
        src = self._create_element(src_cfg)

        # Create muxer
        mux_cfg = cfg["muxer"]
        mux = self._create_element(mux_cfg)

        # Create queue between source and muxer
        queue_src = self._create_element({"type": "queue", "name": "queue_src"})
        queue_src.set_property("max-size-buffers", 5)

        # Create all elements in chain
        chain_elements = []
        for elem_cfg in cfg.get("elements", []):
            elem = self._create_element(elem_cfg)
            chain_elements.append(elem)

            # Attach probes
            for pad_name, probe_name in elem_cfg.get("probes", {}).items():
                self.probe_registry.attach(elem, pad_name, probe_name)

        # Link muxer to first element in chain
        if chain_elements:
            if not mux.link(chain_elements[0]):
                raise RuntimeError(
                    f"Failed to link {mux.get_name()} -> {chain_elements[0].get_name()}"
                )

        # Link elements chain
        for i in range(len(chain_elements) - 1):
            if not chain_elements[i].link(chain_elements[i + 1]):
                raise RuntimeError(
                    f"Failed to link {chain_elements[i].get_name()} -> {chain_elements[i + 1].get_name()}"
                )

        # Create and link sink
        sink_elem = self.sink.create(self.pipeline)
        if chain_elements:
            if not chain_elements[-1].link(sink_elem):
                raise RuntimeError(
                    f"Failed to link {chain_elements[-1].get_name()} -> sink"
                )
        else:
            # No chain elements, link mux directly to sink
            if not mux.link(sink_elem):
                raise RuntimeError(f"Failed to link {mux.get_name()} -> sink")

        # Handle dynamic source pad
        mux_sink = mux.request_pad_simple("sink_0")
        queue_src.get_static_pad("src").link(mux_sink)

        def on_pad_added(_, pad, q):
            caps = pad.get_current_caps()
            if caps and caps.get_structure(0).get_name().startswith("video"):
                sink = q.get_static_pad("sink")
                if sink and not sink.is_linked():
                    ret = pad.link(sink)
                    print(f"[PipelineBuilder] Source linked: {ret}")

        src.connect("pad-added", on_pad_added, queue_src)

        # Add bus watch
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()

        print(f"[PipelineBuilder] Pipeline built: {len(self.elements)} elements")
        return self.pipeline

    def get_element(self, name: str) -> Gst.Element | None:
        """
        Get named element from pipeline

        Args:
            name: Element name

        Returns:
            Element if found, None otherwise
        """
        return self.named_elements.get(name)

    def set_bus_callback(self, callback) -> None:
        """
        Set bus message callback

        Args:
            callback: Bus message handler function
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not built yet")

        bus = self.pipeline.get_bus()
        bus.connect("message", callback)
        print("[PipelineBuilder] Bus callback registered")

    def run(self) -> None:
        """Run pipeline with GLib main loop until EOS or error"""
        if not self.pipeline:
            self.build()

        loop = GLib.MainLoop()

        def on_message(bus, message):
            if message.type == Gst.MessageType.EOS:
                print("End of stream")
                loop.quit()
            elif message.type == Gst.MessageType.ERROR:
                err, _ = message.parse_error()
                print(f"Error: {err.message}")
                loop.quit()

        self.set_bus_callback(on_message)

        try:
            self.pipeline.set_state(Gst.State.PLAYING)
            loop.run()
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.pipeline.set_state(Gst.State.NULL)

    async def run_async(self, setup_callback=None) -> None:
        """Run pipeline with asyncio integration for WebRTC"""
        if not self.pipeline:
            self.build()

        stop_event = asyncio.Event()

        def on_message(bus, message):
            if message.type == Gst.MessageType.EOS:
                print("End of stream")
                stop_event.set()
            elif message.type == Gst.MessageType.ERROR:
                err, _ = message.parse_error()
                print(f"Error: {err.message}")
                stop_event.set()

        self.set_bus_callback(on_message)

        try:
            if setup_callback:
                await setup_callback()

            self.pipeline.set_state(Gst.State.PLAYING)
            await stop_event.wait()

        except KeyboardInterrupt:
            print("\nInterrupted")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.pipeline.set_state(Gst.State.NULL)
