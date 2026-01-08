"""FilesinkAdapter - MP4/file recording sink"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from sinks.base_sink import BaseSink


class FilesinkAdapter(BaseSink):
    """Sink adapter for recording to MP4 files with H.264 encoding"""

    def __init__(
        self, location: str = "output.mp4", codec: str = "h264", bitrate: int = 4000000
    ):
        """
        Args:
            location: Output file path (default: output.mp4)
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

        Returns first element (videoconvert) for linking
        """
        # Create elements - try hardware encoder first, fallback to software
        videoconvert = Gst.ElementFactory.make("videoconvert", "filesink_convert")

        # Try hardware encoder first, fallback to x264enc
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "filesink_encoder")
        use_nvenc = encoder is not None

        if not encoder:
            encoder = Gst.ElementFactory.make("x264enc", "filesink_encoder")

        parser = Gst.ElementFactory.make("h264parse", "filesink_parser")
        muxer = Gst.ElementFactory.make("mp4mux", "filesink_muxer")
        filesink = Gst.ElementFactory.make("filesink", "filesink")

        if not all([videoconvert, encoder, parser, muxer, filesink]):
            raise RuntimeError("Failed to create filesink elements")

        # Configure encoder based on type
        if use_nvenc:
            # NVIDIA hardware encoder
            encoder.set_property("bitrate", self.bitrate)
            encoder.set_property("preset-level", 1)  # UltraFastPreset
            encoder.set_property("insert-sps-pps", True)
            encoder.set_property("bufapi-version", True)
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

        print(
            f"[FilesinkAdapter] Using {'NVIDIA hardware' if use_nvenc else 'x264 software'} encoder"
        )

        # Return first element for linking
        return videoconvert

    def start(self) -> None:
        """Called before pipeline starts - nothing to do for filesink"""
        print(f"[FilesinkAdapter] Recording to: {self.location}")

    def stop(self) -> None:
        """Called when pipeline stops - nothing to do for filesink"""
        print(f"[FilesinkAdapter] Recording stopped: {self.location}")
