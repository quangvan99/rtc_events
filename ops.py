import sys  # Add this import
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import time
from threading import Lock

start_time=time.time()
fps_mutex = Lock()

class GETFPS:
    def __init__(self,stream_id=None):
        global start_time
        self.start_time=start_time
        self.is_first=True
        self.frame_count=0
        self.stream_id=stream_id

    def update_fps(self):
        end_time = time.time()
        if self.is_first:
            self.start_time = end_time
            self.is_first = False
        else:
            global fps_mutex
            with fps_mutex:
                self.frame_count = self.frame_count + 1

    def get_fps(self):
        end_time = time.time()
        with fps_mutex:
            stream_fps = float(self.frame_count/(end_time - self.start_time))
            self.frame_count = 0
        self.start_time = end_time
        return round(stream_fps, 2)

    def print_data(self):
        print('frame_count=',self.frame_count)
        print('start_time=',self.start_time)

class PERF_DATA:
    def __init__(self):
        self.all_stream_fps = {}

    def perf_print_callback(self):
        self.perf_dict = {stream_index:stream.get_fps() for (stream_index, stream) in self.all_stream_fps.items()}
        print ("\n**PERF: ", self.perf_dict, "\n")
        return True
    
    def update_fps(self, name):
        if name not in self.all_stream_fps:
            self.all_stream_fps[name] = GETFPS()

        self.all_stream_fps[name].update_fps()

def link(e1, e2, pad1=None, pad2=None):
    """ Link two pads or two elements together."""
    if pad1 is not None:
        srcpad = e1.get_static_pad(pad1)
    else:
        srcpad = e1
    if not srcpad:
        srcpad = e1.get_request_pad(pad1)
    if pad2 is not None:
        sinkpad = e2.get_static_pad(pad2)
    else:
        sinkpad = e2
    if not sinkpad:
        sinkpad = e2.get_request_pad(pad2)

    ret = srcpad.link(sinkpad)
    if (ret == True) or (ret == Gst.PadLinkReturn.OK):
        return
    print(f"Failed to link {srcpad} to {sinkpad}")
    exit()

def make_plugin(kind, name=None, properties={}):
    plug = Gst.ElementFactory.make(kind, name)
    if not plug:
        sys.stderr.write(f" Unable to create {name} \n")
    for k, v in properties.items():
        plug.set_property(k, v)
    return plug

def link_elements(*args):
    for i in range(len(args) - 1):
        src = args[i]
        sink = args[i+1]
        link(src, sink)
        

def link_elements_with_queue(pipeline, *args):
    for i in range(len(args) - 1):
        src = args[i]
        sink = args[i+1]
        queue = make_plugin("queue")
        pipeline.add(queue)
        link(src, queue)
        link(queue, sink)


def cb_newpad(decodebin, pad, source_id, sm):
    caps = pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()

    if gstname.find("video") != -1:
        pad_name = "sink_%u" % source_id
        sinkpad = sm.get_static_pad(pad_name)
        if not sinkpad:
            sinkpad = sm.get_request_pad(pad_name)
        link(pad, sinkpad)

def decodebin_child_added(child_proxy, Object, name, user_data):
    # Connect callback to internal decodebin signal
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') is not None:
            Object.set_property("drop-on-latency", True)

def create_source_bin(index, uri, sm):
    bin_name = f'src_{index}'
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", bin_name)
    if not uri_decode_bin:
        print(" Unable to create uri decode bin \n")
        exit()
    uri_decode_bin.set_property("uri",uri)
    uri_decode_bin.connect("pad-added", cb_newpad, index, sm)
    uri_decode_bin.connect("child-added", decodebin_child_added, None)
    return uri_decode_bin

def bus_call(bus, message, loop, pipeline):
    
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        loop.quit()
        pipeline.set_state(Gst.State.NULL)
    elif t == Gst.MessageType.STATE_CHANGED:
        if message.src == pipeline:
            old_state, new_state, pending_state = message.parse_state_changed()
            print("Pipeline state changed from {0:s} to {1:s}".format(
                Gst.Element.state_get_name(old_state),
                Gst.Element.state_get_name(new_state)))
    
    return True


def create_file_sink_bin(output_file="output.avi"):
    mp4_sink_bin = Gst.Bin.new("mp4-sink-bin")
    nvvidconv = make_plugin("nvvideoconvert")
    vidconv = make_plugin("videoconvert")
    capsfilter = make_plugin("capsfilter", 
                              properties={"caps": Gst.Caps.from_string("video/x-raw, format=I420")})
    encoder = make_plugin("jpegenc")
    container = make_plugin("avimux")
    filesink = make_plugin("filesink", properties={"location": output_file})

    mp4_sink_bin.add(nvvidconv)
    mp4_sink_bin.add(vidconv)
    mp4_sink_bin.add(capsfilter)
    mp4_sink_bin.add(encoder)
    mp4_sink_bin.add(container)
    mp4_sink_bin.add(filesink)

    mp4_sink_bin.add_pad(Gst.GhostPad.new("sink", nvvidconv.get_static_pad("sink")))
    link_elements(nvvidconv, vidconv, capsfilter, encoder, container, filesink)
    return mp4_sink_bin

def create_file_sink_bin_h264(output_file="output.mp4", bitrate=2000000):
    """
    Sử dụng H.264 encoder với bitrate thấp hơn
    """
    mp4_sink_bin = Gst.Bin.new("mp4-sink-bin")
    nvvidconv = make_plugin("nvvideoconvert")
    capsfilter = make_plugin("capsfilter", 
                              properties={"caps": Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")})
    
    # Sử dụng H.264 encoder thay vì JPEG
    encoder = make_plugin("nvv4l2h264enc", properties={
        "bitrate": bitrate,      # 2Mbps thay vì mặc định
        # "maxperf-enable": True,  # Enable max performance
        "control-rate": 1,       # CBR mode
        "iframeinterval": 30     # I-frame mỗi 30 frames
    })
    
    # Parse H.264 stream
    h264parse = make_plugin("h264parse")
    
    # MP4 container
    container = make_plugin("mp4mux")
    filesink = make_plugin("filesink", properties={"location": output_file})

    # Add elements
    for element in [nvvidconv, capsfilter, encoder, h264parse, container, filesink]:
        mp4_sink_bin.add(element)

    mp4_sink_bin.add_pad(Gst.GhostPad.new("sink", nvvidconv.get_static_pad("sink")))
    link_elements(nvvidconv, capsfilter, encoder, h264parse, container, filesink)
    return mp4_sink_bin