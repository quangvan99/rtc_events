"""
TeeFanoutPipelineBuilder - Config-driven multi-branch DeepStream pipeline

Creates pipeline with multiple branches, each having its own nvstreammux.
Cameras added via tee fanout - single decode, zero-copy distribution.

Supports dependency injection of BranchProcessor instances for clean architecture.
Can also auto-discover processors via ProcessorRegistry.
"""

from __future__ import annotations

import asyncio
import configparser
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, List, Dict, Any

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

# DeepStream Python bindings
sys.path.append("/opt/nvidia/deepstream/deepstream/lib")

from src.sinks.base_sink import BaseSink
from src.core.probe_registry import ProbeRegistry
from src.utils.config import load_config
from src.interfaces.branch_processor import BranchProcessor
from src.registry.processor_registry import ProcessorRegistry


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

    Supports dependency injection of BranchProcessor instances for clean
    separation of concerns. Processors handle application logic (face recognition,
    detection, etc.) while the builder handles pipeline construction.

    Usage (simplest - auto-discovery via registry):
        # Processors self-register via @ProcessorRegistry.register decorator
        # Just importing triggers registration
        config = load_config("multi-branch.yaml")
        builder = TeeFanoutPipelineBuilder(config, auto_discover=True)
        pipeline = builder.build()
        pipeline.set_state(Gst.State.PLAYING)

    Usage (with explicit processors):
        config = load_config("multi-branch.yaml")
        processors = [FaceRecognitionProcessor(), DetectionProcessor()]
        builder = TeeFanoutPipelineBuilder(config, processors=processors)
        pipeline = builder.build()
        builder.start_processors()
        pipeline.set_state(Gst.State.PLAYING)

    Usage (legacy - without processors):
        config = load_config("multi-branch.yaml")
        sinks = {"recognition": FakeSinkAdapter(), "detection": FakeSinkAdapter()}
        builder = TeeFanoutPipelineBuilder(config, branch_sinks=sinks)
        pipeline = builder.build()
        pipeline.set_state(Gst.State.PLAYING)
    """

    def __init__(
        self,
        config: dict,
        processors: List[BranchProcessor] = None,
        branch_sinks: dict[str, BaseSink] = None,
        auto_discover: bool = True
    ):
        """
        Initialize builder with optional processors.

        Args:
            config: Pipeline configuration dict with 'pipeline.branches' and 'output' sections
            processors: List of BranchProcessor instances to inject (optional)
            branch_sinks: Optional mapping branch_name -> BaseSink adapter (legacy)
            auto_discover: If True and no processors provided, auto-discover from registry (default: True)
        """
        from src.sinks.filesink_adapter import FilesinkAdapter
        from src.sinks.fakesink_adapter import FakesinkAdapter
        import os

        self.config = config
        self.probe_registry = ProbeRegistry()

        self.pipeline: Optional[Gst.Pipeline] = None
        self.branches: dict[str, BranchInfo] = {}
        self.elements: list[Gst.Element] = []
        self.named_elements: dict[str, Gst.Element] = {}
        self._counter = 0
        self.camera_manager = None
        
        # Store processors by name
        self.processors: Dict[str, BranchProcessor] = {}
        
        if processors:
            # Use explicitly provided processors
            for proc in processors:
                self.processors[proc.name] = proc
            print(f"[TeeFanoutBuilder] Using {len(processors)} explicit processors")
        elif auto_discover:
            # Auto-discover processors from registry
            self._auto_discover_processors()

        # Initialize branch sinks
        if branch_sinks:
            self.branch_sinks = branch_sinks
        else:
            self._create_default_sinks()
    
    def _auto_discover_processors(self) -> None:
        """
        Auto-discover and create processors from ProcessorRegistry.
        
        This method:
        1. Auto-imports processor modules from apps/
        2. Creates processor instances for configured branches
        """
        # Auto-import all processor modules (triggers @register decorators)
        ProcessorRegistry.auto_import("apps")
        
        # Create processors for configured branches
        processors = ProcessorRegistry.create_for_config(self.config)
        for proc in processors:
            self.processors[proc.name] = proc
        
        if processors:
            print(f"[TeeFanoutBuilder] Auto-discovered {len(processors)} processors: "
                  f"{list(self.processors.keys())}")
        else:
            print("[TeeFanoutBuilder] No processors auto-discovered")
    
    def _create_default_sinks(self) -> None:
        """Create default sinks based on config."""
        from src.sinks.filesink_adapter import FilesinkAdapter
        from src.sinks.fakesink_adapter import FakesinkAdapter
        
        self.branch_sinks = {}
        
        if hasattr(self, 'branch_sinks') and self.branch_sinks:
            return
        output_config = self.config.get("output", {})
        output_dir = output_config.get("dir", "/home/mq/disk2T/quangnv/face/data")
        prefix = output_config.get("prefix", "output")
        extension = output_config.get("extension", "avi")
        output_type = output_config.get("type", "filesink")
        os.makedirs(output_dir, exist_ok=True)

        branches_cfg = self.config.get("pipeline", {}).get("branches", {})

        for name, branch_cfg_or_path in branches_cfg.items():
            if isinstance(branch_cfg_or_path, str):
                branch_cfg = load_config(branch_cfg_or_path)
            else:
                branch_cfg = branch_cfg_or_path

            sink_cfg = branch_cfg.get("sink", {}) if isinstance(branch_cfg, dict) else {}

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
        Build multi-branch pipeline from configuration.

        If processors are registered, their probes are automatically registered
        before building branches, and on_pipeline_built() is called after.

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

        # Setup processors BEFORE building branches (to register probes)
        for branch_name, branch_cfg_or_path in branches_cfg.items():
            if branch_name in self.processors:
                processor = self.processors[branch_name]
                sink = self.branch_sinks.get(branch_name)
                
                if not sink:
                    raise RuntimeError(f"No sink provided for processor branch: {branch_name}")
                
                # Load config if path
                if isinstance(branch_cfg_or_path, str):
                    branch_cfg = load_config(branch_cfg_or_path)
                else:
                    branch_cfg = branch_cfg_or_path
                
                # Setup processor (initializes databases, models, etc.)
                print(f"[TeeFanoutBuilder] Setting up processor for branch: {branch_name}")
                processor.setup(branch_cfg, sink)
                
                # Register all probes from processor
                for probe_name, callback in processor.get_probes().items():
                    self.probe_registry.register(probe_name, callback)

        # Build all branches
        for branch_name, branch_cfg in branches_cfg.items():
            if branch_name not in self.branch_sinks:
                raise RuntimeError(f"No sink provided for branch: {branch_name}")
            self._build_branch(branch_name, branch_cfg)

        # Notify processors after pipeline is built
        for branch_name, processor in self.processors.items():
            if branch_name in self.branches:
                processor.on_pipeline_built(self.pipeline, self.branches[branch_name])

        # Setup bus for error/EOS handling
        self._setup_bus()

        print(f"[TeeFanoutBuilder] Pipeline built: {len(self.branches)} branches, "
              f"{len(self.elements)} elements, {len(self.processors)} processors")
        return self.pipeline

    def _build_branch(self, name: str, cfg_or_path) -> None:
        """
        Build single branch: nvstreammux -> elements -> sink

        Args:
            name: Branch identifier
            cfg_or_path: Either a dict config or path to YAML config file
        """
        # Load config from file if string path
        if isinstance(cfg_or_path, str):
            cfg = load_config(cfg_or_path)
        else:
            cfg = cfg_or_path

        # Get branch config with defaults
        branch_cfg = cfg if isinstance(cfg, dict) else {}
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
    
    def get_processor(self, name: str) -> Optional[BranchProcessor]:
        """Get processor by branch name"""
        return self.processors.get(name)

    def set_bus_callback(self, callback) -> None:
        """Set custom bus message callback"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not built yet")

        bus = self.pipeline.get_bus()
        bus.connect("message", callback)
        print("[TeeFanoutBuilder] Custom bus callback registered")
    
    def start_processors(self) -> None:
        """
        Call on_start() for all registered processors.
        
        Should be called after pipeline.set_state(PLAYING).
        """
        for name, processor in self.processors.items():
            print(f"[TeeFanoutBuilder] Starting processor: {name}")
            processor.on_start()
    
    def stop_processors(self) -> None:
        """
        Call on_stop() for all registered processors.
        
        Should be called before pipeline.set_state(NULL).
        """
        for name, processor in self.processors.items():
            print(f"[TeeFanoutBuilder] Stopping processor: {name}")
            processor.on_stop()
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all processors.
        
        Returns:
            Dict mapping processor name -> stats dict
        """
        stats = {}
        for name, processor in self.processors.items():
            proc_stats = processor.get_stats()
            if proc_stats:
                stats[name] = proc_stats
        return stats

    def run(self) -> None:
        """Run pipeline with GLib main loop until EOS or error.
        
        Automatically starts and stops processors.
        """
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
            
            # Start processors after pipeline is playing
            self.start_processors()
            
            print("[TeeFanoutBuilder] Pipeline PLAYING")
            loop.run()
        except KeyboardInterrupt:
            print("\n[TeeFanoutBuilder] Interrupted")
        finally:
            # Stop processors before pipeline
            self.stop_processors()
            self.pipeline.set_state(Gst.State.NULL)
            print("[TeeFanoutBuilder] Pipeline stopped")

    async def run_async(self, setup_callback=None) -> None:
        """Run pipeline with asyncio integration.
        
        Automatically starts and stops processors.
        """
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
            
            # Start processors after pipeline is playing
            self.start_processors()
            
            print("[TeeFanoutBuilder] Pipeline PLAYING")
            await stop_event.wait()

        except KeyboardInterrupt:
            print("\n[TeeFanoutBuilder] Interrupted")
        except Exception as e:
            print(f"[TeeFanoutBuilder] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Stop processors before pipeline
            self.stop_processors()
            self.pipeline.set_state(Gst.State.NULL)
            print("[TeeFanoutBuilder] Pipeline stopped")