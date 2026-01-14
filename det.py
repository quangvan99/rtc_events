import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds
import numpy as np
from ops import (PERF_DATA, create_file_sink_bin, make_plugin, bus_call, link, 
                        create_source_bin, link_elements_with_queue, create_file_sink_bin_h264)
import time

def tiler_sink_pad_buffer_probe(pad, info, name_branch):
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame_image = np.array(surface, copy=True, order='C')
        print(frame_image.shape)
        # cv2.imwrite(f"frame_ex.jpg", frame_image)

        stream_index = f"stream_{name_branch}_{frame_meta.batch_id}"
        perf_data.update_fps(stream_index)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    Gst.init(None)
    
    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
    pipeline = Gst.Pipeline()

    # paths = ["srt://mqplayer.ddns.net:10080?streamid=live/camera1"]
    # paths = ["rtsp://admin:MQ123456@192.168.6.128"]
    paths = ["file:///app/data/videos/faceQuangnv3.mp4"]*2
    
    perf_data = PERF_DATA()

    sm = make_plugin("nvstreammux", properties={
        "width": 640,
        "height": 640,
        "batch-size": len(paths),
        # "batched-push-timeout": 4000000,
        "nvbuf-memory-type": mem_type,
        # "live-source": 1
    })

    sources = [create_source_bin(i, p, sm) for i, p in enumerate(paths)]

    nvvidconv = make_plugin("nvvideoconvert", properties={'nvbuf-memory-type': mem_type})
    capfilter = make_plugin("capsfilter", properties={"caps": Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")})
    pgie = make_plugin("nvinfer", properties={
        "config-file-path": "data/face/models/scrfd640/infer.txt",
        "batch-size": len(paths)
    })
    sink = make_plugin("fakesink")
    # sink = create_file_sink_bin_h264("/home/projects/videos/output.mp4")
    
    pipeline.add(*sources, sm, nvvidconv, capfilter, pgie, sink)

    sink_pad = capfilter.get_static_pad("sink")
    sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, "")
    GLib.timeout_add_seconds(5, perf_data.perf_print_callback)

    link_elements_with_queue(pipeline, sm, nvvidconv, capfilter, pgie, sink)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop, pipeline)

    try:
        loop.run()
    except:
        pass
        
    # cleanup
    print("Pipeline stopped and cleaned up.")
    pipeline.set_state(Gst.State.NULL)