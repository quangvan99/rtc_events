"""
TeeFanoutPipelineBuilder - Config-driven multi-branch DeepStream pipeline

Creates pipeline with multiple branches, each having its own nvstreammux.
Cameras added via tee fanout - single decode, zero-copy distribution.
"""

from __future__ import annotations

import asyncio
import configparser
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

# DeepStream Python bindings
sys.path.append("/opt/nvidia/deepstream/deepstream/lib")

from sinks.base_sink import BaseSink
from core.probe_registry import ProbeRegistry


def _coerce_property_value(value):
    """Convert string values to appropriate types for GStreamer properties"""
    if not isinstance(value, str):
        return value
    # Try int
    if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
        return int(value)
    # Try float
    try:
        if '.' in value:
            return float(value)
    except ValueError:
        pass
    # Try bool
    if value.lower() in ('true', 'yes', '1'):
        return True
    if value.lower() in ('false', 'no', '0'):
        return False
    return value


@dataclass
class BranchInfo:
    """Metadata for a single processing branch

    Attributes:
        name: Branch identifier (e.g., 'recognition', 'detection')
        nvstreammux: Muxer element for this branch
        elements: List of processing elements (PGIE, tracker, SGIE, etc.)
        sink: Output sink element
        max_cameras: Maximum cameras allowed on this branch
    """
    name: str
    nvstreammux: "Gst.Element"
    elements: list["Gst.Element"] = field(default_factory=list)
    sink: "Gst.Element" = None
    max_cameras: int = 8


class TeeFanoutPipelineBuilder:
    """Config-driven multi-branch DeepStream pipeline builder

    Creates pipeline with multiple branches (e.g., recognition, detection).
    Each branch has its own nvstreammux for independent batching.
    Cameras added dynamically via tee fanout for zero-copy distribution.

    Usage:
        config = load_config("multi-branch.yaml")
        sinks = {"recognition": FakeSinkAdapter(), "detection": FakeSinkAdapter()}
        builder = TeeFanoutPipelineBuilder(config, sinks)
        pipeline = builder.build()
        pipeline.set_state(Gst.State.PLAYING)
    """

    def __init__(
        self,
        config: dict,
        branch_sinks: dict[str, BaseSink] = None
    ):
        """
        Initialize builder

        Args:
            config: Pipeline configuration dict with 'pipeline.branches' and 'output' sections
            branch_sinks: Optional mapping branch_name -> BaseSink adapter
        """
        from sinks.filesink_adapter import FilesinkAdapter
        from sinks.fakesink_adapter import FakesinkAdapter
        import os

        self.config = config
        self.probe_registry = ProbeRegistry()

        self.pipeline: Optional[Gst.Pipeline] = None
        self.branches: dict[str, BranchInfo] = {}
        self.elements: list[Gst.Element] = []
        self.named_elements: dict[str, Gst.Element] = {}
        self._counter = 0
        self.camera_manager = None

        if branch_sinks:
            self.branch_sinks = branch_sinks
        else:
            output_config = config.get("output", {})
            output_dir = output_config.get("dir", "/home/mq/disk2T/quangnv/face/data")
            prefix = output_config.get("prefix", "output")
            extension = output_config.get("extension", "avi")
            output_type = output_config.get("type", "filesink")
            sync = output_config.get("sync", False)
            os.makedirs(output_dir, exist_ok=True)

            branches_cfg = config.get("pipeline", {}).get("branches", {})
            self.branch_sinks = {}

            for name, branch_cfg in branches_cfg.items():
                sink_cfg = branch_cfg.get("sink", {})

                if sink_cfg:
                    sink_type = sink_cfg.get("type", output_type)
                    properties = sink_cfg.get("properties", {})
                    location = properties.get("location")
                else:
                    sink_type = output_type
                    location = f"{output_dir}/{prefix}_{name}.{extension}" if sink_type == "filesink" else None

                if sink_type == "filesink" and location:
                    self.branch_sinks[name] = FilesinkAdapter(location=location)
                else:
                    self.branch_sinks[name] = FakesinkAdapter()

    def build(self) -> Gst.Pipeline:
        """
        Build multi-branch pipeline from configuration

        Returns:
            Constructed GStreamer pipeline with all branches ready

        Raises:
            RuntimeError: If pipeline construction fails
        """
        Gst.init(None)
        cfg = self.config["pipeline"]

        self.pipeline = Gst.Pipeline.new(cfg.get("name", "multi-branch-pipeline"))

        # Build each branch
        branches_cfg = cfg.get("branches", {})
        if not branches_cfg:
            raise RuntimeError("No branches configured in pipeline.branches")

        for branch_name, branch_cfg in branches_cfg.items():
            if branch_name not in self.branch_sinks:
                raise RuntimeError(f"No sink provided for branch: {branch_name}")
            self._build_branch(branch_name, branch_cfg)

        # Setup bus for error/EOS handling
        self._setup_bus()

        print(f"[TeeFanoutBuilder] Pipeline built: {len(self.branches)} branches, "
              f"{len(self.elements)} elements")
        return self.pipeline

    def _build_branch(self, name: str, cfg: dict) -> None:
        """
        Build single branch: nvstreammux -> elements -> sink

        Args:
            name: Branch identifier
            cfg: Branch configuration dict
        """
        # 1. Create nvstreammux for this branch
        mux = Gst.ElementFactory.make("nvstreammux", f"mux_{name}")
        if not mux:
            raise RuntimeError(f"Cannot create nvstreammux for branch: {name}")

        # Configure muxer
        mux_cfg = cfg.get("muxer", {})
        for key, value in mux_cfg.items():
            prop_name = key.replace("-", "_")
            mux.set_property(prop_name, _coerce_property_value(value))

        # Set default properties if not configured
        if "live-source" not in mux_cfg:
            mux.set_property("live_source", 1)
        if "nvbuf-memory-type" not in mux_cfg:
            mux.set_property("nvbuf_memory_type", 0)

        self.pipeline.add(mux)
        self.elements.append(mux)

        # 2. Create element chain
        chain = [mux]
        for elem_cfg in cfg.get("elements", []):
            elem = self._create_element(elem_cfg, f"{name}")
            chain.append(elem)

            # Attach probes if configured
            for pad_name, probe_name in elem_cfg.get("probes", {}).items():
                self.probe_registry.attach(elem, pad_name, probe_name)

        # 3. Create sink
        sink_elem = self.branch_sinks[name].create(self.pipeline)
        chain.append(sink_elem)

        # 4. Link chain: mux -> elem1 -> elem2 -> ... -> sink
        for i in range(len(chain) - 1):
            if not chain[i].link(chain[i + 1]):
                raise RuntimeError(
                    f"Failed to link {chain[i].get_name()} -> {chain[i + 1].get_name()}"
                )

        # 5. Store branch info
        self.branches[name] = BranchInfo(
            name=name,
            nvstreammux=mux,
            elements=chain[1:-1],  # Exclude mux and sink
            sink=sink_elem,
            max_cameras=cfg.get("max_cameras", 8)
        )

        print(f"[TeeFanoutBuilder] Branch '{name}' built: "
              f"{len(chain) - 2} elements, max_cameras={cfg.get('max_cameras', 8)}")

    def _create_element(self, elem_cfg: dict, branch_prefix: str = "") -> Gst.Element:
        """
        Create GStreamer element from config

        Args:
            elem_cfg: Element configuration dict
            branch_prefix: Prefix for element naming (branch name)

        Returns:
            Created GStreamer element
        """
        elem_type = elem_cfg["type"]

        # Auto-generate name with branch prefix
        if "name" in elem_cfg:
            name = f"{branch_prefix}_{elem_cfg['name']}" if branch_prefix else elem_cfg["name"]
        else:
            self._counter += 1
            name = f"{branch_prefix}_{elem_type}_{self._counter}" if branch_prefix else f"{elem_type}_{self._counter}"

        elem = Gst.ElementFactory.make(elem_type, name)
        if not elem:
            raise RuntimeError(f"Cannot create element: {elem_type}")

        # Set properties
        for key, value in elem_cfg.get("properties", {}).items():
            elem.set_property(key, _coerce_property_value(value))

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
        """Configure nvtracker from config file"""
        if not os.path.exists(cfg_path):
            print(f"[TeeFanoutBuilder] Warning: Tracker config not found: {cfg_path}")
            return

        cfg = configparser.ConfigParser()
        cfg.read(cfg_path)

        if "tracker" not in cfg:
            return

        # Get config file directory for resolving relative paths
        cfg_dir = os.path.dirname(os.path.abspath(cfg_path))

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
                # Resolve relative paths for ll-config-file
                if key == "ll-config-file" and not os.path.isabs(val):
                    val = os.path.join(cfg_dir, val)
                tracker.set_property(prop, typ(val))

        print(f"[TeeFanoutBuilder] Tracker configured from: {cfg_path}")

    def _setup_bus(self) -> None:
        """Setup bus watch for error/EOS handling"""
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        print("[TeeFanoutBuilder] Bus watch enabled")

    def register_probe(self, name: str, callback) -> None:
        """Register a probe callback"""
        self.probe_registry.register(name, callback)

    def get_branch(self, name: str) -> Optional[BranchInfo]:
        """Get branch info by name"""
        return self.branches.get(name)

    def get_element(self, name: str) -> Optional[Gst.Element]:
        """Get named element from pipeline"""
        return self.named_elements.get(name)

    def set_bus_callback(self, callback) -> None:
        """Set custom bus message callback"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not built yet")

        bus = self.pipeline.get_bus()
        bus.connect("message", callback)
        print("[TeeFanoutBuilder] Custom bus callback registered")

    def run(self) -> None:
        """Run pipeline with GLib main loop until EOS or error"""
        if not self.pipeline:
            self.build()

        loop = GLib.MainLoop()

        def on_message(bus, message):
            if message.type == Gst.MessageType.EOS:
                print("[TeeFanoutBuilder] End of stream")
                loop.quit()
            elif message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"[TeeFanoutBuilder] Error: {err.message}")
                if debug:
                    print(f"[TeeFanoutBuilder] Debug: {debug}")
                loop.quit()

        self.set_bus_callback(on_message)

        try:
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING state")
            print("[TeeFanoutBuilder] Pipeline PLAYING")
            loop.run()
        except KeyboardInterrupt:
            print("\n[TeeFanoutBuilder] Interrupted")
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            print("[TeeFanoutBuilder] Pipeline stopped")

    async def run_async(self, setup_callback=None) -> None:
        """Run pipeline with asyncio integration"""
        if not self.pipeline:
            self.build()

        stop_event = asyncio.Event()

        def on_message(bus, message):
            if message.type == Gst.MessageType.EOS:
                print("[TeeFanoutBuilder] End of stream")
                stop_event.set()
            elif message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"[TeeFanoutBuilder] Error: {err.message}")
                stop_event.set()

        self.set_bus_callback(on_message)

        try:
            if setup_callback:
                await setup_callback()

            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING state")
            print("[TeeFanoutBuilder] Pipeline PLAYING")
            await stop_event.wait()

        except KeyboardInterrupt:
            print("\n[TeeFanoutBuilder] Interrupted")
        except Exception as e:
            print(f"[TeeFanoutBuilder] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            print("[TeeFanoutBuilder] Pipeline stopped")
