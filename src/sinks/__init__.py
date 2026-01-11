"""Sink adapters for DeepStream outputs"""

from .base_sink import BaseSink
from .fakesink_adapter import FakesinkAdapter
from .filesink_adapter import FilesinkAdapter

__all__ = ["BaseSink", "FakesinkAdapter", "FilesinkAdapter"]
