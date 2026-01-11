"""Probe Registry for DeepStream Pipeline"""

from typing import Callable, Any

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

# Type alias for probe callbacks
ProbeCallback = Callable[[Gst.Pad, Gst.PadProbeInfo, Any], Gst.PadProbeReturn]


class ProbeRegistry:
    """Registry for named probe callbacks"""

    def __init__(self):
        """Initialize empty probe registry"""
        self._probes: dict[str, ProbeCallback] = {}

    def register(self, name: str, callback: ProbeCallback) -> None:
        """
        Register a probe callback by name

        Args:
            name: Unique name for the probe
            callback: Probe callback function
        """
        self._probes[name] = callback
        print(f"[ProbeRegistry] Registered probe: {name}")

    def get(self, name: str) -> ProbeCallback | None:
        """
        Get a probe callback by name

        Args:
            name: Probe name

        Returns:
            Probe callback if found, None otherwise
        """
        return self._probes.get(name)

    def attach(self, element: Gst.Element, pad_name: str, probe_name: str) -> bool:
        """
        Attach a registered probe to an element's pad

        Args:
            element: GStreamer element
            pad_name: Name of the pad to attach to (e.g., 'src', 'sink')
            probe_name: Name of the registered probe

        Returns:
            True if attached successfully, False otherwise
        """
        callback = self._probes.get(probe_name)
        if callback is None:
            print(f"[ProbeRegistry] Warning: probe '{probe_name}' not found")
            return False

        pad = element.get_static_pad(pad_name)
        if pad is None:
            print(f"[ProbeRegistry] Warning: pad '{pad_name}' not found on {element.get_name()}")
            return False

        pad.add_probe(Gst.PadProbeType.BUFFER, callback, None)
        print(f"[ProbeRegistry] Attached probe '{probe_name}' to {element.get_name()}:{pad_name}")
        return True

