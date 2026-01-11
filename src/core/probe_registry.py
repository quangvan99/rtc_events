"""
Probe Registry - Manages named probe callbacks for GStreamer elements

Usage:
    registry = ProbeRegistry()
    registry.register("my_probe", my_callback)
    registry.attach(element, "src", "my_probe")
"""

from typing import Callable, Any, Optional, Dict

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# Type alias for probe callbacks
ProbeCallback = Callable[[Gst.Pad, Gst.PadProbeInfo, Any], Gst.PadProbeReturn]


class ProbeRegistry:
    """Registry for named probe callbacks
    
    Allows probes to be registered by name and later attached to
    pipeline elements by referencing the name in config files.
    """

    def __init__(self):
        self._probes: Dict[str, ProbeCallback] = {}

    def register(self, name: str, callback: ProbeCallback) -> None:
        """Register a probe callback by name"""
        self._probes[name] = callback
        print(f"[ProbeRegistry] Registered: {name}")

    def get(self, name: str) -> Optional[ProbeCallback]:
        """Get a probe callback by name"""
        return self._probes.get(name)

    def attach(self, element: Gst.Element, pad_name: str, probe_name: str) -> bool:
        """Attach a registered probe to an element's pad"""
        callback = self._probes.get(probe_name)
        if callback is None:
            print(f"[ProbeRegistry] Warning: probe '{probe_name}' not found")
            return False

        pad = element.get_static_pad(pad_name)
        if pad is None:
            print(f"[ProbeRegistry] Warning: pad '{pad_name}' not found on {element.get_name()}")
            return False

        pad.add_probe(Gst.PadProbeType.BUFFER, callback, None)
        print(f"[ProbeRegistry] Attached '{probe_name}' to {element.get_name()}:{pad_name}")
        return True

    def list_probes(self) -> list:
        """List all registered probe names"""
        return list(self._probes.keys())

    def clear(self) -> None:
        """Clear all registered probes"""
        self._probes.clear()