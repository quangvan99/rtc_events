"""FilesinkAdapter - MP4/file recording sink for DeepStream pipelines"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from sinks.base_sink import BaseSink


class FilesinkAdapter(BaseSink):
    """Sink adapter for recording to video files with H.264 encoding

    DeepStream compatible: uses nvvideoconvert with compute-hw for proper
    GPU->CPU memory transfer, avoiding memory corruption.

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
        queue -> nvvideoconvert(compute-hw=1) -> caps(RGBA) -> videoconvert -> x264enc -> muxer -> filesink

        Key fix: nvvideoconvert with compute-hw=1 ensures proper NVMM->system memory conversion.
        Uses AVI muxer for resilience to abrupt termination.
        Returns first element (queue) for linking.
        """
        prefix = f"filesink{self._id}"

        # Queue at input to decouple from upstream and buffer frames
        queue_in = Gst.ElementFactory.make("queue", f"{prefix}_queue_in")
        queue_in.set_property("max-size-buffers", 5)
        queue_in.set_property("leaky", 2)  # Drop old buffers if full

        # nvvideoconvert with compute-hw for proper NVMM -> system memory
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", f"{prefix}_nvvidconv")
        # compute-hw=1 uses CUDA for conversion, required for NVMM memory type
        nvvidconv.set_property("compute-hw", 1)
        # nvbuf-memory-type=3 is system memory (not NVMM)
        nvvidconv.set_property("nvbuf-memory-type", 3)

        # Capsfilter to force RGBA system memory output
        # Using memory:SystemMemory explicitly for CPU processing
        capsfilter = Gst.ElementFactory.make("capsfilter", f"{prefix}_caps")
        caps = Gst.Caps.from_string("video/x-raw, format=RGBA")
        capsfilter.set_property("caps", caps)

        # videoconvert to convert RGBA -> I420 for x264enc
        videoconvert = Gst.ElementFactory.make("videoconvert", f"{prefix}_vidconv")

        # Queue before encoder to decouple
        queue_enc = Gst.ElementFactory.make("queue", f"{prefix}_queue_enc")
        queue_enc.set_property("max-size-buffers", 3)
        queue_enc.set_property("leaky", 2)

        # x264 software encoder
        encoder = Gst.ElementFactory.make("x264enc", f"{prefix}_encoder")

        parser = Gst.ElementFactory.make("h264parse", f"{prefix}_parser")

        # Use AVI muxer - more resilient to abrupt termination than MP4
        if self.location.endswith(".mp4"):
            muxer = Gst.ElementFactory.make("mp4mux", f"{prefix}_muxer")
            print("[FilesinkAdapter] Warning: MP4 may be corrupted if not properly closed")
        else:
            muxer = Gst.ElementFactory.make("avimux", f"{prefix}_muxer")

        filesink = Gst.ElementFactory.make("filesink", f"{prefix}_sink")

        required = [queue_in, nvvidconv, capsfilter, videoconvert, queue_enc, encoder, parser, muxer, filesink]
        if not all(required):
            missing = [name for elem, name in zip(required,
                ["queue_in", "nvvidconv", "capsfilter", "videoconvert", "queue_enc", "encoder", "parser", "muxer", "filesink"])
                if elem is None]
            raise RuntimeError(f"Failed to create filesink elements: {missing}")

        # Configure x264 encoder for low latency
        encoder.set_property("bitrate", self.bitrate // 1000)  # x264 uses kbps
        encoder.set_property("speed-preset", "ultrafast")
        encoder.set_property("tune", "zerolatency")
        encoder.set_property("key-int-max", 30)

        # Configure filesink
        filesink.set_property("location", self.location)
        filesink.set_property("sync", False)
        filesink.set_property("async", False)

        # Add all elements to pipeline
        for elem in required:
            pipeline.add(elem)

        # Link elements in chain
        chain = [queue_in, nvvidconv, capsfilter, videoconvert, queue_enc, encoder, parser, muxer, filesink]
        for i in range(len(chain) - 1):
            if not chain[i].link(chain[i + 1]):
                raise RuntimeError(f"Failed to link {chain[i].get_name()} -> {chain[i + 1].get_name()}")

        # Store elements for cleanup
        self.elements = chain

        mux_type = "AVI" if not self.location.endswith(".mp4") else "MP4"
        print(f"[FilesinkAdapter] Created: queue -> nvvidconv(compute-hw=1) -> RGBA -> vidconv -> x264enc -> {mux_type}")

        # Return first element for linking
        return queue_in

    def start(self) -> None:
        """Called before pipeline starts - nothing to do for filesink"""
        print(f"[FilesinkAdapter] Recording to: {self.location}")

    def stop(self) -> None:
        """Called when pipeline stops - nothing to do for filesink"""
        print(f"[FilesinkAdapter] Recording stopped: {self.location}")