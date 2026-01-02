#!/usr/bin/env python3
"""
WebRTC Face Recognition Stream
Usage:
    1. python signalling.py
    2. Open view.html in browser
    3. python stream_face.py
"""
import asyncio
import json
import ssl
import sys
import platform
import time
import ctypes
import configparser

import numpy as np
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstWebRTC', '1.0')
gi.require_version('GstSdp', '1.0')
from gi.repository import Gst, GstWebRTC, GstSdp
import websockets

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds

# ============================================================================
# CONFIG
# ============================================================================

SERVER = 'wss://192.168.6.16:8443'
STUN_SERVER = 'stun://stun.l.google.com:19302'
PEER_ID = '1'

VIDEO_URI = 'file:///home/mq/disk2T/quangnv/face/testcase/faceQuangnv4.mp4'
WIDTH, HEIGHT = 1920, 1080

PGIE_CONFIG = 'models/scrfd640/infer.txt'
TRACKER_CONFIG = 'models/NvDCF/config_tracker.txt'
SGIE_CONFIG = 'models/arcface/infer.txt'
FEATURES_PATH = 'extract/features_arcface.json'

SKIP_REID = 5
VOTE_THRESHOLD = 5
DISTANCE_THRESHOLD = 1.15

# Encoder: 'gpu' (nvv4l2h264enc) or 'cpu' (x264enc)
ENCODER = 'cpu'

# ============================================================================
# TRACKER CLASSES
# ============================================================================

class Tracker:
    def __init__(self, bbox, class_id, object_id, feature=None, frame_num=-99):
        self.object_id = int(object_id)
        self.class_id = int(class_id)
        self.bbox = bbox
        self.label = None
        self.votes = []
        self.feature = feature
        self.frame_reid = frame_num
        self.age = 0

    def update(self, bbox, frame_num, feature=None):
        self.bbox = bbox
        self.age = 0
        if feature is not None:
            self.feature = feature
            self.frame_reid = frame_num


class TrackerManager:
    def __init__(self, max_age=30):
        self.max_age = max_age
        self.trackers = []

    def update(self, oid, class_id, bbox, feature=None, frame_num=-99):
        for trk in self.trackers:
            if trk.object_id == oid:
                trk.update(bbox, frame_num, feature)
                return trk

        trk = Tracker(bbox, class_id, oid, feature, frame_num)
        self.trackers.append(trk)
        return trk

    def cleanup(self):
        for trk in self.trackers:
            trk.age += 1
        self.trackers = [t for t in self.trackers if t.age <= self.max_age]

    def get(self, oid):
        for trk in self.trackers:
            if trk.object_id == oid:
                return trk
        return None


# ============================================================================
# WEBRTC FACE CLIENT
# ============================================================================

class WebRTCFaceClient:
    def __init__(self, loop):
        self.loop = loop
        self.conn = None
        self.pipe = None
        self.webrtc = None

        # Face recognition data
        self.person_names, self.features = self._load_features()
        self.tracker_mgr = TrackerManager(max_age=30)
        self.fps_count = 0
        self.fps_time = time.time()

    def _load_features(self):
        names, features = [], []
        try:
            with open(FEATURES_PATH, encoding='utf-8') as f:
                data = json.load(f)
            for name, val in data.items():
                names.append(name)
                features.append(np.array(val['feature']))
            features = np.array(features).squeeze(1)
            print(f'Loaded {len(names)} faces')
        except Exception as e:
            print(f'Load features failed: {e}')
            features = np.array([])
        return names, features

    def _make_element(self, factory, name):
        elm = Gst.ElementFactory.make(factory, name)
        if not elm:
            raise RuntimeError(f'Cannot create {factory}')
        self.pipe.add(elm)
        return elm

    def _create_pipeline(self):
        self.pipe = Gst.Pipeline.new('face-pipeline')

        # Source
        src = self._make_element('uridecodebin', 'src')
        src.set_property('uri', VIDEO_URI)

        # Streammux
        mux = self._make_element('nvstreammux', 'mux')
        mux.set_property('width', WIDTH)
        mux.set_property('height', HEIGHT)
        mux.set_property('batch-size', 1)
        mux.set_property('batched-push-timeout', 40000)

        # Convert + Caps
        conv1 = self._make_element('nvvideoconvert', 'conv1')
        if platform.uname()[4] != 'aarch64':
            conv1.set_property('nvbuf-memory-type', int(pyds.NVBUF_MEM_CUDA_UNIFIED))

        caps1 = self._make_element('capsfilter', 'caps1')
        caps1.set_property('caps', Gst.Caps.from_string('video/x-raw(memory:NVMM),format=RGBA'))

        # PGIE (face detection)
        pgie = self._make_element('nvinfer', 'pgie')
        pgie.set_property('config-file-path', PGIE_CONFIG)

        # Tracker
        tracker = self._make_element('nvtracker', 'tracker')
        self._set_tracker_props(tracker)

        # SGIE (face recognition)
        sgie = self._make_element('nvinfer', 'sgie')
        sgie.set_property('config-file-path', SGIE_CONFIG)

        # OSD
        osd = self._make_element('nvdsosd', 'osd')

        # Convert for encoder
        conv2 = self._make_element('nvvideoconvert', 'conv2')
        caps2 = self._make_element('capsfilter', 'caps2')

        # Encoder (GPU or CPU)
        if ENCODER == 'gpu':
            caps2.set_property('caps', Gst.Caps.from_string('video/x-raw(memory:NVMM),format=I420'))
            # Queue trước encoder để buffer
            queue_enc = self._make_element('queue', 'queue_enc')
            queue_enc.set_property('max-size-buffers', 3)
            queue_enc.set_property('leaky', 2)  # downstream

            enc = self._make_element('nvv4l2h264enc', 'enc')
            enc.set_property('bitrate', 4000000)
            enc.set_property('iframeinterval', 30)
            enc.set_property('control-rate', 1)  # constant bitrate
            if platform.uname()[4] == 'aarch64':  # Jetson
                enc.set_property('bufapi-version', 1)

            h264parse = self._make_element('h264parse', 'h264parse')
            h264parse.set_property('config-interval', -1)
        else:
            queue_enc = None
            caps2.set_property('caps', Gst.Caps.from_string('video/x-raw'))
            enc = self._make_element('x264enc', 'enc')
            enc.set_property('tune', 'zerolatency')
            enc.set_property('speed-preset', 'ultrafast')
            enc.set_property('key-int-max', 30)
            h264parse = None

        # RTP
        pay = self._make_element('rtph264pay', 'pay')
        pay.set_property('config-interval', -1)

        rtpcaps = self._make_element('capsfilter', 'rtpcaps')
        rtpcaps.set_property('caps', Gst.Caps.from_string(
            'application/x-rtp,media=video,encoding-name=H264,payload=96'))

        # WebRTC
        self.webrtc = self._make_element('webrtcbin', 'webrtc')
        self.webrtc.set_property('stun-server', STUN_SERVER)
        self.webrtc.set_property('latency', 200)  # giảm latency
        self.webrtc.set_property('bundle-policy', 3)

        # Link static elements
        if ENCODER == 'gpu':
            elements = [conv1, caps1, pgie, tracker, sgie, osd, conv2, caps2, queue_enc, enc, h264parse, pay, rtpcaps]
        else:
            elements = [conv1, caps1, pgie, tracker, sgie, osd, conv2, caps2, enc, pay, rtpcaps]
        for i in range(len(elements) - 1):
            elements[i].link(elements[i + 1])
        rtpcaps.link(self.webrtc)

        # Dynamic link for source
        def on_pad_added(_, pad, mux):
            if pad.get_current_caps().get_structure(0).get_name().startswith('video'):
                pad.link(mux.get_request_pad('sink_0'))
        src.connect('pad-added', on_pad_added, mux)
        mux.link(conv1)

        # Add probes
        tracker.get_static_pad('src').add_probe(
            Gst.PadProbeType.BUFFER, self._probe_skip)
        sgie.get_static_pad('src').add_probe(
            Gst.PadProbeType.BUFFER, self._probe_recognize)
        osd.get_static_pad('src').add_probe(
            Gst.PadProbeType.BUFFER, self._probe_fps)

        # WebRTC signals
        self.webrtc.connect('on-negotiation-needed', self._on_negotiation)
        self.webrtc.connect('on-ice-candidate', self._on_ice)

        self.pipe.set_state(Gst.State.PLAYING)
        print('Pipeline started')
        return True

    def _set_tracker_props(self, tracker):
        cfg = configparser.ConfigParser()
        cfg.read(TRACKER_CONFIG)
        props = {
            'tracker-width': 'tracker-width',
            'tracker-height': 'tracker-height',
            'gpu-id': 'gpu_id',
            'll-lib-file': 'll-lib-file',
            'll-config-file': 'll-config-file',
        }
        for key, prop in props.items():
            if key in cfg['tracker']:
                val = cfg['tracker'][key]
                if key in ['tracker-width', 'tracker-height', 'gpu-id']:
                    tracker.set_property(prop, int(val))
                else:
                    tracker.set_property(prop, val)

    def _probe_fps(self, _, info):
        self.fps_count += 1
        now = time.time()
        if now - self.fps_time > 5:
            print(f'FPS: {self.fps_count / (now - self.fps_time):.1f}')
            self.fps_count = 0
            self.fps_time = now
        self.tracker_mgr.cleanup()
        return Gst.PadProbeReturn.OK

    def _probe_skip(self, _, info):
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
        l_frame = batch.frame_meta_list
        while l_frame:
            try:
                frame = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame.obj_meta_list
            while l_obj:
                try:
                    obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                trk = self.tracker_mgr.get(obj.object_id)
                label = 'Unknown'

                if trk:
                    if frame.frame_num - trk.frame_reid < SKIP_REID:
                        obj.unique_component_id = 100
                    if trk.label:
                        obj.unique_component_id = 100
                        label = trk.label

                # Set text display
                obj.text_params.display_text = label
                obj.text_params.x_offset = int(obj.rect_params.left + obj.rect_params.width + 5)
                obj.text_params.y_offset = int(obj.rect_params.top + 5)
                obj.text_params.font_params.font_size = 12
                obj.text_params.font_params.font_name = 'Serif'
                obj.text_params.font_params.font_color.set(1, 1, 1, 1)
                obj.text_params.text_bg_clr.set(0, 0, 0, 0.6)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    def _probe_recognize(self, _, info):
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
        l_frame = batch.frame_meta_list
        while l_frame:
            try:
                frame = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame.obj_meta_list
            while l_obj:
                try:
                    obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                oid = obj.object_id
                bbox = [int(obj.rect_params.left), int(obj.rect_params.top),
                       int(obj.rect_params.left + obj.rect_params.width),
                       int(obj.rect_params.top + obj.rect_params.height)]

                feature = self._extract_feature(obj)
                trk = self.tracker_mgr.update(oid, obj.class_id, bbox, feature,
                                              frame.frame_num if feature is not None else -99)

                # Match face
                if trk.label is None and trk.feature is not None and len(self.features) > 0:
                    dist = np.linalg.norm(self.features - trk.feature, axis=1)
                    if dist.min() <= DISTANCE_THRESHOLD:
                        trk.votes.append(dist.argmin())

                    if len(trk.votes) >= VOTE_THRESHOLD:
                        trk.label = self.person_names[np.bincount(trk.votes).argmax()]

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    def _extract_feature(self, obj):
        l_user = obj.obj_user_meta_list
        while l_user:
            try:
                user = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if user.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                tensor = pyds.NvDsInferTensorMeta.cast(user.user_meta_data)
                layer = pyds.get_nvds_LayerInfo(tensor, 0)
                ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                feat = np.ctypeslib.as_array(ptr, shape=(512,))
                return feat / np.linalg.norm(feat)

            try:
                l_user = l_user.next
            except StopIteration:
                break
        return None

    # WebRTC handlers
    def _send(self, msg):
        asyncio.run_coroutine_threadsafe(self.conn.send(msg), self.loop)

    def _on_negotiation(self, _):
        print('Negotiation needed')
        promise = Gst.Promise.new_with_change_func(self._on_offer, None, None)
        self.webrtc.emit('create-offer', None, promise)

    def _on_offer(self, promise, _, __):
        reply = promise.get_reply()
        if not reply:
            return
        offer = reply['offer']
        if not offer:
            return
        self.webrtc.emit('set-local-description', offer, Gst.Promise.new())
        self._send(json.dumps({'sdp': {'type': 'offer', 'sdp': offer.sdp.as_text()}}))

    def _on_ice(self, _, idx, candidate):
        if '.local' not in candidate:
            self._send(json.dumps({'ice': {'candidate': candidate, 'sdpMLineIndex': idx}}))

    def _handle_answer(self, sdp):
        _, msg = GstSdp.SDPMessage.new_from_text(sdp)
        answer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.ANSWER, msg)
        self.webrtc.emit('set-remote-description', answer, Gst.Promise.new())

    def _handle_ice(self, ice):
        if '.local' not in ice['candidate']:
            self.webrtc.emit('add-ice-candidate', ice['sdpMLineIndex'], ice['candidate'])

    async def run(self):
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        self.conn = await websockets.connect(SERVER, ssl=ssl_ctx)
        print(f'Connected to {SERVER}')
        await self.conn.send('HELLO sender')

        async for msg in self.conn:
            print(f'<<< {msg[:80]}' if len(msg) > 80 else f'<<< {msg}')

            if msg == 'HELLO':
                await self.conn.send(f'SESSION {PEER_ID}')
            elif msg == 'SESSION_OK':
                self._create_pipeline()
            elif msg.startswith('ERROR'):
                break
            else:
                try:
                    data = json.loads(msg)
                    if 'sdp' in data and data['sdp']['type'] == 'answer':
                        self._handle_answer(data['sdp']['sdp'])
                    elif 'ice' in data:
                        self._handle_ice(data['ice'])
                except:
                    pass

    async def stop(self):
        if self.pipe:
            self.pipe.set_state(Gst.State.NULL)
        if self.conn:
            await self.conn.close()


# ============================================================================
# MAIN
# ============================================================================

async def main():
    Gst.init(None)
    print('=' * 50)
    print('WebRTC Face Recognition')
    print(f'Video: {VIDEO_URI}')
    print('=' * 50)

    client = WebRTCFaceClient(asyncio.get_event_loop())
    try:
        await client.run()
    except KeyboardInterrupt:
        pass
    finally:
        await client.stop()


if __name__ == '__main__':
    asyncio.run(main())
