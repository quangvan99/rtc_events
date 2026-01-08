"""FilesinkAdapter - MP4/file recording sink"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from sinks.base_sink import BaseSink


class FilesinkAdapter(BaseSink):
    """Sink adapter for recording to video files with H.264 encoding

    Supports MP4 and AVI formats. AVI recommended for abrupt termination resilience.
    """

    # Class-level counter for unique element naming
    _instance_counter = 0

    def __init__(
        self, location: str = "output.avi", codec: str = "h264", bitrate: int = 4000000
    ):
        """
        Args:
            location: Output file path (default: output.avi)
            codec: Video codec (default: h264)
            bitrate: Encoding bitrate in bps (default: 4000000 = 4 Mbps)
        """
        self.location = location
        self.codec = codec
        self.bitrate = bitrate
        self.elements = []
        # Unique ID for element naming to avoid conflicts
        self._id = FilesinkAdapter._instance_counter
        FilesinkAdapter._instance_counter += 1

    def create(self, pipeline: Gst.Pipeline) -> Gst.Element:
        """
        Create filesink pipeline for DeepStream:
        nvvideoconvert(->I420) -> nvv4l2h264enc -> h264parse -> muxer -> filesink

        Handles NVMM buffer conversion to I420 for hardware encoder.
        Uses AVI muxer for resilience to abrupt termination.
        Returns first element (nvvideoconvert) for linking.
        """
        # Create elements with unique names to support multiple sinks
        prefix = f"filesink{self._id}"

        # nvvideoconvert to convert NVMM -> CPU memory for software encoder
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", f"{prefix}_nvvidconv")

        # Capsfilter to convert to CPU memory (no NVMM) for software encoder
        capsfilter = Gst.ElementFactory.make("capsfilter", f"{prefix}_caps")
        caps = Gst.Caps.from_string("video/x-raw, format=I420")
        capsfilter.set_property("caps", caps)

        # Use x264 software encoder for stability (nvv4l2h264enc has memory issues in containers)
        encoder = Gst.ElementFactory.make("x264enc", f"{prefix}_encoder")
        videoconvert = None  # Not needed, nvvidconv handles conversion
        use_nvenc = False

        parser = Gst.ElementFactory.make("h264parse", f"{prefix}_parser")

        # Use AVI muxer - more resilient to abrupt termination than MP4
        if self.location.endswith(".mp4"):
            muxer = Gst.ElementFactory.make("mp4mux", f"{prefix}_muxer")
            print("[FilesinkAdapter] Warning: MP4 may be corrupted if not properly closed")
        else:
            muxer = Gst.ElementFactory.make("avimux", f"{prefix}_muxer")

        filesink = Gst.ElementFactory.make("filesink", f"{prefix}_sink")

        required = [nvvidconv, capsfilter, encoder, parser, muxer, filesink]
        if not all(required):
            raise RuntimeError("Failed to create filesink elements")

        # Configure encoder based on type
        if use_nvenc:
            # NVIDIA hardware encoder (nvv4l2h264enc)
            encoder.set_property("bitrate", self.bitrate)
            encoder.set_property("iframeinterval", 30)
            encoder.set_property("preset-id", 1)  # P1 = highest performance
        else:
            # x264 software encoder
            encoder.set_property("bitrate", self.bitrate // 1000)  # x264 uses kbps
            encoder.set_property("speed-preset", "ultrafast")
            encoder.set_property("tune", "zerolatency")

        # Configure filesink
        filesink.set_property("location", self.location)
        filesink.set_property("sync", False)

        # Add to pipeline
        pipeline.add(nvvidconv)
        pipeline.add(capsfilter)
        pipeline.add(encoder)
        pipeline.add(parser)
        pipeline.add(muxer)
        pipeline.add(filesink)

        # Link elements: nvvidconv -> capsfilter -> encoder -> parser -> muxer -> filesink
        if not nvvidconv.link(capsfilter):
            raise RuntimeError("Failed to link nvvidconv -> capsfilter")
        if not capsfilter.link(encoder):
            raise RuntimeError("Failed to link capsfilter -> encoder")
        if not encoder.link(parser):
            raise RuntimeError("Failed to link encoder -> parser")
        if not parser.link(muxer):
            raise RuntimeError("Failed to link parser -> muxer")
        if not muxer.link(filesink):
            raise RuntimeError("Failed to link muxer -> filesink")

        # Store elements for cleanup
        self.elements = [nvvidconv, capsfilter, encoder, parser, muxer, filesink]

        mux_type = "AVI" if not self.location.endswith(".mp4") else "MP4"
        print(
            f"[FilesinkAdapter] Using {'NVIDIA hardware' if use_nvenc else 'x264 software'} encoder, {mux_type} muxer"
        )

        # Return first element for linking
        return nvvidconv

    def start(self) -> None:
        """Called before pipeline starts - nothing to do for filesink"""
        print(f"[FilesinkAdapter] Recording to: {self.location}")

    def stop(self) -> None:
        """Called when pipeline stops - nothing to do for filesink"""
        print(f"[FilesinkAdapter] Recording stopped: {self.location}")
