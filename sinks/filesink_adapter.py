"""FilesinkAdapter - MP4/file recording sink"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from sinks.base_sink import BaseSink


class FilesinkAdapter(BaseSink):
    """Sink adapter for recording to video files with H.264 encoding

    Supports MP4 and AVI formats. AVI recommended for abrupt termination resilience.
    """

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

    def create(self, pipeline: Gst.Pipeline) -> Gst.Element:
        """
        Create filesink pipeline: videoconvert -> encoder -> parser -> muxer -> filesink

        Uses AVI muxer for resilience to abrupt termination.
        Returns first element (videoconvert) for linking.
        """
        # Create elements - try hardware encoder first, fallback to software
        videoconvert = Gst.ElementFactory.make("videoconvert", "filesink_convert")

        # Try hardware encoder first, fallback to x264enc
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "filesink_encoder")
        use_nvenc = encoder is not None

        if not encoder:
            encoder = Gst.ElementFactory.make("x264enc", "filesink_encoder")

        parser = Gst.ElementFactory.make("h264parse", "filesink_parser")

        # Use AVI muxer - more resilient to abrupt termination than MP4
        # MP4 requires moov atom at end, AVI doesn't
        if self.location.endswith(".mp4"):
            muxer = Gst.ElementFactory.make("mp4mux", "filesink_muxer")
            print("[FilesinkAdapter] Warning: MP4 may be corrupted if not properly closed")
        else:
            muxer = Gst.ElementFactory.make("avimux", "filesink_muxer")

        filesink = Gst.ElementFactory.make("filesink", "filesink")

        if not all([videoconvert, encoder, parser, muxer, filesink]):
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
        pipeline.add(videoconvert)
        pipeline.add(encoder)
        pipeline.add(parser)
        pipeline.add(muxer)
        pipeline.add(filesink)

        # Link elements
        if not videoconvert.link(encoder):
            raise RuntimeError("Failed to link videoconvert -> encoder")
        if not encoder.link(parser):
            raise RuntimeError("Failed to link encoder -> parser")
        if not parser.link(muxer):
            raise RuntimeError("Failed to link parser -> muxer")
        if not muxer.link(filesink):
            raise RuntimeError("Failed to link muxer -> filesink")

        # Store elements for cleanup
        self.elements = [videoconvert, encoder, parser, muxer, filesink]

        mux_type = "AVI" if not self.location.endswith(".mp4") else "MP4"
        print(
            f"[FilesinkAdapter] Using {'NVIDIA hardware' if use_nvenc else 'x264 software'} encoder, {mux_type} muxer"
        )

        # Return first element for linking
        return videoconvert

    def start(self) -> None:
        """Called before pipeline starts - nothing to do for filesink"""
        print(f"[FilesinkAdapter] Recording to: {self.location}")

    def stop(self) -> None:
        """Called when pipeline stops - nothing to do for filesink"""
        print(f"[FilesinkAdapter] Recording stopped: {self.location}")
