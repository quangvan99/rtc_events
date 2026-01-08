"""
Base Sink Interface

Abstract base class for all output sink adapters.
"""

from abc import ABC, abstractmethod
from typing import Optional

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


class BaseSink(ABC):
    """Abstract base class for output sinks"""

    @abstractmethod
    def create(self, pipeline: Gst.Pipeline) -> Gst.Element:
        """
        Create sink element(s), add to pipeline.

        Returns: first element for linking from upstream
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """Called before pipeline starts."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Called when pipeline stops."""
        pass

    def send_event(self, event: dict) -> None:
        """
        Send event via data channel (optional).

        Default implementation is a no-op.
        Override in sinks that support data channels (e.g., WebRTC).
        """
        pass
