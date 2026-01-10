"""
CameraBin - Dataclass for single camera's GStreamer elements

Encapsulates nvurisrcbin + tee for multi-branch fanout.
Each camera decodes once, tee distributes to multiple branches via zero-copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst


@dataclass
class CameraBin:
    """Container for single camera's GStreamer elements

    Attributes:
        camera_id: Unique identifier for camera
        uri: Camera source URI (rtsp://, file://, etc.)
        bin: GstBin containing nvurisrcbin + tee
        nvurisrcbin: Source element for decoding
        tee: Fanout element for multi-branch distribution
        branch_queues: Mapping branch_name -> queue element
        branch_pads: Mapping branch_name -> tee src pad
        source_id: DeepStream source ID for this camera
        pad_added_handler: Signal handler ID for pad-added signal
        active_probes: List of active pad probe IDs
    """
    camera_id: str
    uri: str
    bin: "Gst.Bin"
    nvurisrcbin: "Gst.Element"
    tee: "Gst.Element"
    branch_queues: dict[str, "Gst.Element"] = field(default_factory=dict)
    branch_pads: dict[str, "Gst.Pad"] = field(default_factory=dict)
    source_id: int = 0
    pad_added_handler: int = 0
    active_probes: list = field(default_factory=list)

    def get_branches(self) -> list[str]:
        """Get list of branch names this camera is connected to"""
        return list(self.branch_queues.keys())

    def is_connected_to(self, branch_name: str) -> bool:
        """Check if camera is connected to specified branch"""
        return branch_name in self.branch_queues
