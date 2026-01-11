"""
BranchProcessor Interface

Abstract base class for all branch processors in the multi-branch pipeline.
Each application (face recognition, detection, tracking, etc.) implements
this interface to integrate with TeeFanoutPipelineBuilder.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
from dataclasses import dataclass

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.sinks.base_sink import BaseSink


@dataclass
class ProbeConfig:
    """Configuration for a probe attachment"""
    element_name: str
    pad_name: str
    probe_name: str


class BranchProcessor(ABC):
    """
    Abstract interface for branch processors.
    
    Each application (face recognition, detection, tracking, etc.)
    implements this interface to integrate with TeeFanoutPipelineBuilder.
    
    Lifecycle:
        1. __init__() - Create processor instance
        2. setup(config, sink) - Initialize with config and sink
        3. get_probes() - Register probes with pipeline
        4. on_pipeline_built() - Optional, called after pipeline is constructed
    
    Example:
        class MyProcessor(BranchProcessor):
            @property
            def name(self) -> str:
                return "my_branch"
            
            def setup(self, config, sink):
                self._config = config
                self._sink = sink
            
            def get_probes(self):
                return {"my_probe": self._my_probe_callback}
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the branch name this processor handles.
        
        Must match the branch name in config (e.g., 'recognition', 'detection').
        This name is used to match processor to branch configuration.
        
        Returns:
            Branch name string
        """
        pass
    
    @abstractmethod
    def setup(self, config: Dict[str, Any], sink: BaseSink) -> None:
        """
        Initialize the processor with configuration and sink.
        
        Called before pipeline build. Processor should initialize:
        - Databases
        - Models
        - Internal state
        - Any required resources
        
        Args:
            config: Branch configuration dict (from YAML)
            sink: Output sink adapter for this branch
            
        Raises:
            RuntimeError: If initialization fails
        """
        pass
    
    @abstractmethod
    def get_probes(self) -> Dict[str, Callable]:
        """
        Return probe callbacks to register.
        
        Called after setup() to get all probe callbacks that need
        to be registered with the ProbeRegistry. The probe names
        must match those referenced in the branch config YAML.
        
        Returns:
            Dict mapping probe_name -> callback function
            
        Example:
            return {
                "tracker_probe": self.tracker_probe,
                "sgie_probe": self.sgie_probe,
                "fps_probe": self.fps_probe,
            }
        """
        pass
    
    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any) -> None:
        """
        Called after pipeline is built, before PLAYING state.
        
        Optional hook for post-build initialization. Use this to:
        - Access pipeline elements
        - Configure additional settings
        - Set up connections between components
        
        Args:
            pipeline: The built GStreamer pipeline
            branch_info: BranchInfo dataclass for this processor's branch
        """
        pass
    
    def on_start(self) -> None:
        """
        Called when pipeline starts playing.
        
        Optional hook for start-time initialization. Use this to:
        - Start timers
        - Begin logging
        """
        pass
    
    def on_stop(self) -> None:
        """
        Called when pipeline stops.
        
        Optional hook for cleanup. Use this to:
        - Stop timers
        - Flush buffers
        """
        pass
