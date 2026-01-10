#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/mq/disk2T/quangnv/face")
from core.config import load_config
from core.tee_fanout_builder import TeeFanoutPipelineBuilder
from core.multibranch_camera_manager import MultibranchCameraManager
from sinks.filesink_adapter import FilesinkAdapter
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import time

Gst.init(None)
config = load_config("configs/multi-branch.yaml")
sinks = {
    "recognition": FilesinkAdapter(location="/home/mq/disk2T/quangnv/face/data/output_recognition.avi"),
    "detection": FilesinkAdapter(location="/home/mq/disk2T/quangnv/face/data/output_detection.avi")
}

builder = TeeFanoutPipelineBuilder(config, sinks)
pipeline = builder.build()
manager = MultibranchCameraManager(pipeline, builder.branches)

for sink in sinks.values():
    sink.start()

pipeline.set_state(Gst.State.PLAYING)
print("Pipeline started")

rtsp_uri = "rtsp://192.168.6.14:8554/test"

for i in range(1, 4):
    print(f"=== Add cam{i} ===")
    result = manager.add_camera(f"cam{i}", rtsp_uri, ["recognition", "detection"])
    print(f"cam{i}: {result}")
    time.sleep(8)

print("\n=== All cameras added ===")
print(manager.list_cameras())

print("\n=== Remove cam2 from detection ===")
result = manager.remove_camera_from_branch("cam2", "detection")
print(f"Result: {result}")

print("\n=== TEST PASSED ===")
time.sleep(2)
pipeline.set_state(Gst.State.NULL)
