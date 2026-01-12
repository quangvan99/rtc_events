"""Config-driven multi-branch DeepStream pipeline builder."""

from __future__ import annotations

import asyncio
import configparser
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

if "/opt/nvidia/deepstream/deepstream/lib" not in sys.path:
    sys.path.append("/opt/nvidia/deepstream/deepstream/lib")

logger = logging.getLogger(__name__)

from src.sinks.base_sink import BaseSink
from src.probe_registry import ProbeRegistry
from src.common import load_config
from src.processor_registry import ProcessorRegistry, BranchProcessor


@dataclass
class BranchInfo:
    """Metadata for a single processing branch."""
    name: str
    nvstreammux: Gst.Element
    elements: list[Gst.Element] = field(default_factory=list)
    sink: Optional[Gst.Element] = None
    max_cameras: int = 8


class PipelineBuilder:
    """Config-driven multi-branch DeepStream pipeline builder."""

    def __init__(self, config: dict, processors: list[BranchProcessor] = None,
                 branch_sinks: dict[str, BaseSink] = None, auto_discover: bool = True):
        self.config = config
        self.probe_registry = ProbeRegistry()
        self.pipeline: Optional[Gst.Pipeline] = None
        self.branches: dict[str, BranchInfo] = {}
        self.processors: dict[str, BranchProcessor] = {}

        if processors:
            for p in processors:
                self.processors[p.name] = p
        elif auto_discover:
            ProcessorRegistry.auto_import("apps")
            for p in ProcessorRegistry.create_for_config(config):
                self.processors[p.name] = p

        self.branch_sinks = branch_sinks if branch_sinks else self._create_sinks()

    def _create_sinks(self) -> dict[str, BaseSink]:
        """Create sinks based on config."""
        from src.sinks.filesink_adapter import FilesinkAdapter
        from src.sinks.fakesink_adapter import FakesinkAdapter

        sinks = {}
        out_cfg = self.config.get("output", {})
        out_dir = out_cfg.get("dir", "data/output")
        out_type = out_cfg.get("type", "fakesink")
        os.makedirs(out_dir, exist_ok=True)

        for name, cfg in self.config.get("pipeline", {}).get("branches", {}).items():
            if isinstance(cfg, str):
                cfg = load_config(cfg)
            sink_cfg = cfg.get("sink", {}) if isinstance(cfg, dict) else {}

            if sink_cfg.get("type") == "filesink" or out_type == "filesink":
                loc = sink_cfg.get("properties", {}).get("location") or f"{out_dir}/output_{name}.avi"
                sinks[name] = FilesinkAdapter(location=loc)
            else:
                sinks[name] = FakesinkAdapter()
        return sinks

    def build(self) -> Gst.Pipeline:
        """Build pipeline from config."""
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new(self.config.get("pipeline", {}).get("name", "pipeline"))

        branches_cfg = self.config.get("pipeline", {}).get("branches", {})
        if not branches_cfg:
            raise RuntimeError("No branches configured")

        # Setup processors
        for name, cfg in branches_cfg.items():
            if name in self.processors:
                if isinstance(cfg, str):
                    cfg = load_config(cfg)
                self.processors[name].setup(cfg, self.branch_sinks[name])
                for probe_name, cb in self.processors[name].get_probes().items():
                    self.probe_registry.register(probe_name, cb)

        # Build branches
        for name, cfg in branches_cfg.items():
            self._build_branch(name, cfg)

        # Notify processors
        for name, proc in self.processors.items():
            if name in self.branches:
                proc.on_pipeline_built(self.pipeline, self.branches[name])

        self.pipeline.get_bus().add_signal_watch()
        logger.info(f"Pipeline built: {len(self.branches)} branches")
        return self.pipeline

    def _build_branch(self, name: str, cfg) -> None:
        """Build single branch: mux -> elements -> sink."""
        if isinstance(cfg, str):
            cfg = load_config(cfg)

        # Create muxer
        mux = Gst.ElementFactory.make("nvstreammux", f"mux_{name}")
        for k, v in cfg.get("muxer", {}).items():
            mux.set_property(k.replace("-", "_"), v)
        mux.set_property("live_source", cfg.get("muxer", {}).get("live-source", 1))
        mux.set_property("nvbuf_memory_type", 0)
        self.pipeline.add(mux)

        # Create element chain
        chain = [mux]
        for elem_cfg in cfg.get("elements", []):
            elem = self._create_element(elem_cfg, name)
            chain.append(elem)
            for pad, probe in elem_cfg.get("probes", {}).items():
                self.probe_registry.attach(elem, pad, probe)

        # Create sink
        sink = self.branch_sinks[name].create(self.pipeline)
        chain.append(sink)

        # Link chain
        for i in range(len(chain) - 1):
            if not chain[i].link(chain[i + 1]):
                raise RuntimeError(f"Failed to link {chain[i].get_name()} -> {chain[i + 1].get_name()}")

        self.branches[name] = BranchInfo(name, mux, chain[1:-1], sink, cfg.get("max_cameras", 8))

    def _create_element(self, cfg: dict, prefix: str) -> Gst.Element:
        """Create GStreamer element from config."""
        elem_type = cfg["type"]
        name = f"{prefix}_{cfg.get('name', elem_type)}"
        elem = Gst.ElementFactory.make(elem_type, name)
        if not elem:
            raise RuntimeError(f"Cannot create: {elem_type}")

        for k, v in cfg.get("properties", {}).items():
            elem.set_property(k, v)

        if "caps" in cfg:
            elem.set_property("caps", Gst.Caps.from_string(cfg["caps"]))

        if "config_file" in cfg:
            path = cfg["config_file"]
            if elem_type == "nvinfer":
                elem.set_property("config-file-path", path)
            elif elem_type == "nvtracker":
                self._configure_tracker(elem, path)

        self.pipeline.add(elem)
        return elem

    def _configure_tracker(self, tracker: Gst.Element, cfg_path: str) -> None:
        """Configure nvtracker from INI file."""
        if not os.path.exists(cfg_path):
            logger.warning(f"Tracker config not found: {cfg_path}")
            return

        cfg = configparser.ConfigParser()
        cfg.read(cfg_path)
        if "tracker" not in cfg:
            return

        cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
        for key in ["tracker-width", "tracker-height", "gpu-id", "ll-lib-file", "ll-config-file"]:
            if key in cfg["tracker"]:
                val = cfg["tracker"][key]
                if key in ("ll-config-file", "ll-lib-file") and not os.path.isabs(val):
                    val = os.path.join(cfg_dir, val)
                if key in ("tracker-width", "tracker-height", "gpu-id"):
                    val = int(val)
                tracker.set_property(key, val)

    def set_bus_callback(self, callback: Callable) -> None:
        """Set bus message callback."""
        self.pipeline.get_bus().connect("message", callback)

    def get_processor(self, name: str) -> Optional[BranchProcessor]:
        return self.processors.get(name)

    def get_branch(self, name: str) -> Optional[BranchInfo]:
        return self.branches.get(name)

    def start_processors(self) -> None:
        for p in self.processors.values():
            p.on_start()

    def stop_processors(self) -> None:
        for p in self.processors.values():
            p.on_stop()

    def run(self) -> None:
        """Run pipeline with GLib main loop."""
        if not self.pipeline:
            self.build()

        loop = GLib.MainLoop()

        def on_msg(_bus, msg):
            if msg.type == Gst.MessageType.EOS:
                loop.quit()
            elif msg.type == Gst.MessageType.ERROR:
                err, _ = msg.parse_error()
                logger.error(f"Pipeline error: {err.message}")
                loop.quit()

        self.set_bus_callback(on_msg)

        try:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.start_processors()
            loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_processors()
            self.pipeline.set_state(Gst.State.NULL)

    async def run_async(self, setup_callback: Callable = None) -> None:
        """Run pipeline with asyncio."""
        if not self.pipeline:
            self.build()

        stop = asyncio.Event()

        def on_msg(_bus, msg):
            if msg.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
                if msg.type == Gst.MessageType.ERROR:
                    err, _ = msg.parse_error()
                    logger.error(f"Pipeline error: {err.message}")
                stop.set()

        self.set_bus_callback(on_msg)

        try:
            if setup_callback:
                await setup_callback()
            self.pipeline.set_state(Gst.State.PLAYING)
            self.start_processors()
            await stop.wait()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_processors()
            self.pipeline.set_state(Gst.State.NULL)


# Backward compatibility
TeeFanoutPipelineBuilder = PipelineBuilder
